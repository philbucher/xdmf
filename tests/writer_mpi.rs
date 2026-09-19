//! End-to-end coverage for [`xdmf::TimeSeriesWriter::write_mesh_parallel`]/
//! [`xdmf::ParallelTimeSeriesDataWriter::write_time_step`]: every rank writes its own share of a
//! small chain mesh, then rank 0 reads the result back through the ordinary (serial)
//! [`xdmf::TimeSeriesReader`] and checks it against the *same mesh written serially in one shot*
//! -- a stronger oracle than hand-computed expected values, since it can't share a bug with the
//! implementation the way a hand-derived expectation can.

#![cfg(feature = "mpi")]

use std::path::PathBuf;

use mpi::traits::{Communicator, CommunicatorCollectives, Root};
use mpi_test::mpi_test;
use temp_dir::TempDir;
use xdmf::{
    CellType, DataAttribute, DataStorage, ParallelTimeSeriesWriter, TimeSeriesReader,
    TimeSeriesWriter,
};

/// Every rank owns 4 points in its own contiguous block of global ids, chained by 3 local `Edge`
/// cells within that block. Every rank but 0 also owns one bridging cell back to the previous
/// rank's last point -- connectivity referencing a point *another* rank owns, which is the whole
/// point of writing points as `owned_points`/`owned_global_ids` rather than a full local halo.
const POINTS_PER_RANK: usize = 4;

#[mpi_test(np = [1, 2, 3])]
fn write_mesh_parallel_matches_the_same_mesh_written_serially() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = usize::try_from(world.rank()).unwrap();
    let size = usize::try_from(world.size()).unwrap();

    let start = rank * POINTS_PER_RANK;

    let owned_points: Vec<f64> = (0..POINTS_PER_RANK)
        .flat_map(|i| [(start + i) as f64, 0.0, 0.0])
        .collect();
    let owned_global_ids: Vec<u64> = (start..start + POINTS_PER_RANK)
        .map(|id| id as u64)
        .collect();

    let mut local_connectivity: Vec<u64> = Vec::new();
    if rank > 0 {
        local_connectivity.extend([start as u64 - 1, start as u64]);
    }
    for i in 0..POINTS_PER_RANK - 1 {
        local_connectivity.extend([(start + i) as u64, (start + i + 1) as u64]);
    }
    let num_local_cells = local_connectivity.len() / 2;
    let local_cell_types = vec![CellType::Edge; num_local_cells];

    let owned_temperature: Vec<f64> = owned_global_ids
        .iter()
        .map(|&id| id as f64 * 10.0)
        .collect();
    let local_owner_rank: Vec<f64> = vec![rank as f64; num_local_cells];

    // `TempDir::new()` picks a fresh random path *per process*, so every rank would otherwise
    // try to collectively open a different file -- rank 0 picks the one shared path and
    // broadcasts it, and keeps the `TempDir` (which removes itself on drop) alive until the very
    // end so no rank is left writing into an already-removed directory.
    let root = world.process_at_rank(0);
    let tmp_dir = (rank == 0).then(|| TempDir::new().unwrap());

    let mut path_bytes = tmp_dir
        .as_ref()
        .map(|dir| {
            dir.path()
                .join("mpi_mesh")
                .to_str()
                .unwrap()
                .as_bytes()
                .to_vec()
        })
        .unwrap_or_default();

    let mut len = u32::try_from(path_bytes.len()).unwrap();
    root.broadcast_into(&mut len);
    path_bytes.resize(len as usize, 0);
    root.broadcast_into(&mut path_bytes[..]);

    let file_name = PathBuf::from(String::from_utf8(path_bytes).unwrap());

    // `new_parallel` takes the communicator by value and keeps it for later parallel calls to
    // reuse -- `universe.world()` is called again here rather than moving the `world` above,
    // which is still needed for the broadcast/barrier around it; that's cheap and repeatable for
    // the built-in world communicator (see `TimeSeriesWriter::new_parallel`'s docs).
    let writer = ParallelTimeSeriesWriter::new(
        &file_name,
        DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
        universe.world(),
    )
    .unwrap();

    let mut ts_writer = writer
        .write_mesh(
            &owned_points,
            &owned_global_ids,
            &local_connectivity,
            &local_cell_types,
        )
        .unwrap();

    ts_writer
        .write_time_step("0.0", |step| {
            step.point_data("temperature", DataAttribute::Scalar, &owned_temperature)?;
            step.cell_data("owner_rank", DataAttribute::Scalar, &local_owner_rank)
        })
        .unwrap();

    let xdmf_path = ts_writer.file_name().to_path_buf();

    // closing the writer drops its HDF5 file handle, which is collective under MPI-IO -- every
    // rank must do this before any rank reopens the file independently
    drop(ts_writer);
    world.barrier();

    if rank != 0 {
        return;
    }

    let total_points = size * POINTS_PER_RANK;
    let total_cells = size * (POINTS_PER_RANK - 1) + size.saturating_sub(1);

    // The same global mesh and field data as above, but assembled as a single array in rank
    // order -- exactly what every rank's own `owned_points`/`local_connectivity`/... concatenate
    // to, since the parallel writer places each rank's share at its `Exscan`-computed offset
    // without reordering it.
    let global_points: Vec<f64> = (0..total_points)
        .flat_map(|id| [id as f64, 0.0, 0.0])
        .collect();
    let global_temperature: Vec<f64> = (0..total_points).map(|id| id as f64 * 10.0).collect();

    let mut global_connectivity: Vec<u64> = Vec::new();
    let mut global_owner_rank: Vec<f64> = Vec::new();
    for r in 0..size {
        let start = r * POINTS_PER_RANK;
        if r > 0 {
            global_connectivity.extend([start as u64 - 1, start as u64]);
            global_owner_rank.push(r as f64);
        }
        for i in 0..POINTS_PER_RANK - 1 {
            global_connectivity.extend([(start + i) as u64, (start + i + 1) as u64]);
            global_owner_rank.push(r as f64);
        }
    }
    let global_cell_types = vec![CellType::Edge; total_cells];

    let serial_writer = TimeSeriesWriter::new(
        tmp_dir.as_ref().unwrap().path().join("serial_mesh"),
        DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
    )
    .unwrap();
    let mut serial_ts_writer = serial_writer
        .write_mesh(&global_points, &global_connectivity, &global_cell_types)
        .unwrap();
    serial_ts_writer
        .write_time_step("0.0", |step| {
            step.point_data("temperature", DataAttribute::Scalar, &global_temperature)?;
            step.cell_data("owner_rank", DataAttribute::Scalar, &global_owner_rank)
        })
        .unwrap();
    let serial_path = serial_ts_writer.file_name().to_path_buf();
    drop(serial_ts_writer);

    let parallel_reader = TimeSeriesReader::new(&xdmf_path).unwrap();
    let serial_reader = TimeSeriesReader::new(&serial_path).unwrap();

    assert_eq!(parallel_reader.num_points(), serial_reader.num_points());
    assert_eq!(parallel_reader.num_cells(), serial_reader.num_cells());
    assert_eq!(parallel_reader.times(), serial_reader.times());

    let mut parallel_points = Vec::new();
    let mut serial_points = Vec::new();
    parallel_reader
        .read_points::<f64>(&mut parallel_points)
        .unwrap();
    serial_reader
        .read_points::<f64>(&mut serial_points)
        .unwrap();
    assert_eq!(parallel_points, serial_points);

    let mut parallel_connectivity = Vec::new();
    let mut parallel_cell_types = Vec::new();
    parallel_reader
        .read_topology::<u64>(&mut parallel_connectivity, &mut parallel_cell_types)
        .unwrap();
    let mut serial_connectivity = Vec::new();
    let mut serial_cell_types = Vec::new();
    serial_reader
        .read_topology::<u64>(&mut serial_connectivity, &mut serial_cell_types)
        .unwrap();
    assert_eq!(parallel_connectivity, serial_connectivity);
    assert_eq!(parallel_cell_types, serial_cell_types);

    let mut parallel_temperature = Vec::new();
    let mut serial_temperature = Vec::new();
    parallel_reader
        .read_point_data::<f64>(0, "temperature", &mut parallel_temperature)
        .unwrap();
    serial_reader
        .read_point_data::<f64>(0, "temperature", &mut serial_temperature)
        .unwrap();
    assert_eq!(parallel_temperature, serial_temperature);

    let mut parallel_owner_rank = Vec::new();
    let mut serial_owner_rank = Vec::new();
    parallel_reader
        .read_cell_data::<f64>(0, "owner_rank", &mut parallel_owner_rank)
        .unwrap();
    serial_reader
        .read_cell_data::<f64>(0, "owner_rank", &mut serial_owner_rank)
        .unwrap();
    assert_eq!(parallel_owner_rank, serial_owner_rank);
}

//! End-to-end coverage for [`xdmf::TimeSeriesWriter::write_mesh_parallel`]/
//! [`xdmf::ParallelTimeSeriesDataWriter::write_time_step`]: every rank writes its own share of a
//! small chain mesh, then rank 0 reads the result back through the ordinary (serial)
//! [`xdmf::TimeSeriesReader`] and checks it against the seamless, single-rank-equivalent mesh.

#![cfg(feature = "mpi")]

use std::path::PathBuf;

use mpi::traits::{Communicator, CommunicatorCollectives, Root};
use mpi_test::mpi_test;
use temp_dir::TempDir;
use xdmf::{CellType, DataAttribute, DataStorage, ParallelTimeSeriesWriter, TimeSeriesReader};

/// Every rank owns 4 points in its own contiguous block of global ids, chained by 3 local `Edge`
/// cells within that block. Every rank but 0 also owns one bridging cell back to the previous
/// rank's last point -- connectivity referencing a point *another* rank owns, which is the whole
/// point of writing points as `owned_points`/`owned_global_ids` rather than a full local halo.
const POINTS_PER_RANK: usize = 4;

#[mpi_test(np = [1, 2, 3])]
fn write_mesh_parallel_round_trips_through_a_serial_read() {
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

    let reader = TimeSeriesReader::new(&xdmf_path).unwrap();
    assert_eq!(reader.num_points(), total_points);
    assert_eq!(reader.num_cells(), total_cells);
    assert_eq!(reader.times(), ["0.0"]);

    let mut points = Vec::new();
    reader.read_points::<f64>(&mut points).unwrap();
    let expected_points: Vec<f64> = (0..total_points)
        .flat_map(|id| [id as f64, 0.0, 0.0])
        .collect();
    assert_eq!(points, expected_points);

    let mut connectivity = Vec::new();
    let mut cell_types = Vec::new();
    reader
        .read_topology::<u64>(&mut connectivity, &mut cell_types)
        .unwrap();
    assert_eq!(cell_types, vec![CellType::Edge; total_cells]);
    // every index is a valid global point id -- the real assertion here is that this reads back
    // at all rather than panicking/erroring on an out-of-range value from a wrong per-rank offset
    assert!(
        connectivity
            .iter()
            .all(|&index| index < total_points as u64)
    );

    let mut temperature = Vec::new();
    reader
        .read_point_data::<f64>(0, "temperature", &mut temperature)
        .unwrap();
    let expected_temperature: Vec<f64> = (0..total_points).map(|id| id as f64 * 10.0).collect();
    assert_eq!(temperature, expected_temperature);

    let mut owner_rank = Vec::new();
    reader
        .read_cell_data::<f64>(0, "owner_rank", &mut owner_rank)
        .unwrap();
    assert_eq!(owner_rank.len(), total_cells);
    // every cell's `owner_rank` value came from the rank that wrote it, so the sum recovers how
    // many cells each rank actually contributed: rank 0 gets 3 (no bridge), every other rank 4
    let expected_sum: f64 = (1..size).map(|r| r as f64 * 4.0).sum();
    float_cmp::assert_approx_eq!(f64, owner_rank.iter().sum::<f64>(), expected_sum);
}

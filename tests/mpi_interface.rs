//! Shows how [`xdmf::mpi`] is meant to be used, and pins down that the interface compiles.
//!
//! Nothing behind the interface is implemented yet, so [`write_a_distributed_mesh`] cannot run to
//! completion -- it exists to be *type-checked*: it is the usage this API is being designed for,
//! written out in full, so a change to any signature breaks the build here rather than being
//! noticed once a caller tries it. The test itself only asserts that the flow fails cleanly with
//! [`xdmf::Error::Internal`] instead of panicking or hanging -- construction already works, so it
//! is [`xdmf::mpi::TimeSeriesWriter::write_mesh`] that stops it.

#![cfg(feature = "mpi")]

use std::path::{Path, PathBuf};

use mpi_test::mpi_test;
use xdmf::{
    CellType, DataAttribute, DataStorage, Error,
    mpi::{Communicator, TimeSeriesWriter},
};

/// One rank's share of a chain of `Edge` cells along the x axis, and one step of data on it.
///
/// Every rank owns [`POINTS_PER_RANK`] points in its own contiguous block of global ids, chained
/// by local cells within that block. Every rank but the first also owns one cell bridging back to
/// the previous rank's last point -- connectivity referencing a point *another* rank owns, which
/// is the case `owned_points`/`owned_global_ids` exist for.
fn write_a_distributed_mesh(
    comm: &impl Communicator,
    rank: usize,
    file_name: &Path,
) -> Result<PathBuf, Error> {
    const POINTS_PER_RANK: usize = 4;

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
    let local_cell_types = vec![CellType::Edge; local_connectivity.len() / 2];

    let temperature: Vec<f64> = owned_global_ids
        .iter()
        .map(|&id| id as f64 * 10.0)
        .collect();
    let owner_rank: Vec<f64> = vec![rank as f64; local_cell_types.len()];

    let writer = TimeSeriesWriter::new(
        file_name,
        DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
        comm,
    )?;

    let mut ts_writer = writer.write_mesh(
        &owned_points,
        &owned_global_ids,
        &local_connectivity,
        &local_cell_types,
    )?;

    ts_writer.write_time_step("0.0", |step| {
        step.point_data("temperature", DataAttribute::Scalar, &temperature)?;
        step.cell_data("owner_rank", DataAttribute::Scalar, &owner_rank)
    })?;

    Ok(ts_writer.file_name().to_path_buf())
}

#[mpi_test(np = [1, 2])]
fn the_interface_is_not_implemented_yet() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = usize::try_from(world.rank()).unwrap();

    let result = write_a_distributed_mesh(&world, rank, &PathBuf::from("mpi_interface"));

    std::assert_matches!(
        result.unwrap_err(),
        Error::Internal(message) if message.contains("interface sketch")
    );
}

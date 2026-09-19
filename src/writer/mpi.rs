//! Writing one XDMF file from a mesh distributed over MPI ranks.
//!
//! Every rank holds a piece of the mesh and writes its own share into one shared file, which
//! opens as a single seamless mesh -- indistinguishable from serial output, with no rank
//! boundaries visible in `ParaView`. This mirrors [`crate::TimeSeriesWriter`] step for step
//! ([`TimeSeriesWriter::new`] -> [`TimeSeriesWriter::write_mesh`] ->
//! [`TimeSeriesDataWriter::write_time_step`]); the only difference is that every array passed in
//! is this rank's own share rather than the whole mesh.
//!
//! These are deliberately separate types from the serial ones, rather than a parallel constructor
//! on them: it keeps a caller from reaching [`crate::TimeSeriesDataWriter::write_time_step`] on a
//! mesh written collectively, or [`TimeSeriesDataWriter::write_time_step`] on one written
//! serially, by construction rather than by a runtime check. On the HDF5 backend that mistake is
//! not a clean error -- a non-collective, whole-buffer write against a file opened under the
//! MPI-IO driver hangs rather than failing.
//!
//! # Interface only
//!
//! **Almost nothing here is implemented yet**: [`TimeSeriesWriter::new`] duplicates the
//! communicator and keeps it, and everything past that returns [`Error::Internal`]. No file is
//! created, so a writer obtained from `new` cannot do anything with it yet. This module exists so
//! the shape of the API can be reviewed and written against before the collectives behind it are
//! built. See `07_mpi.md` (on the plans share, outside this repository) for the design.
//!
//! # Version coupling
//!
//! [`Communicator`] -- the one `mpi` item that appears in this module's signatures -- is
//! re-exported here so a caller can name the bound against the exact version this crate's `mpi`
//! feature was built with. A caller still depends on the `mpi` crate itself to *create* a
//! communicator, and must match this crate's version of it, since the two must agree on the type.

use std::{
    fmt,
    path::{Path, PathBuf},
};

use ::mpi::topology::SimpleCommunicator;
pub use ::mpi::traits::Communicator;

use crate::{
    CellType, ConnectivityIndex, Coordinate, DataAttribute, DataStorage, Error, Result, Values,
};

/// What every entry point in this module returns until the implementation lands.
const UNIMPLEMENTED: Error = Error::Internal("xdmf::mpi is an interface sketch, not implemented");

/// Writer for a distributed mesh, the counterpart of [`crate::TimeSeriesWriter`].
pub struct TimeSeriesWriter {
    xdmf_file_name: PathBuf,
    comm: SimpleCommunicator,
}

impl fmt::Debug for TimeSeriesWriter {
    /// Shows this rank's own number rather than the communicator, which has no useful `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeSeriesWriter")
            .field("xdmf_file_name", &self.xdmf_file_name)
            .field("rank", &self.comm.rank())
            .finish()
    }
}

impl TimeSeriesWriter {
    /// Create a writer over `comm`. Every rank must call this together, with the same
    /// `file_name` and `data_storage`, since the underlying file is opened collectively.
    ///
    /// `comm` is duplicated and the copy kept, so later calls need not be handed one again and,
    /// more importantly, so this writer's collectives run in their own communication context.
    /// Collectives match by the order they are called in rather than by a tag, so a writer
    /// sharing the caller's communicator could match one of the caller's own collectives -- a
    /// nonblocking one still in flight while a step is written, say. Duplicating removes that
    /// class of bug, at the cost of one more collective call in a function that is already
    /// collective. Any communicator can be passed: the copy this keeps is freed with the writer.
    ///
    /// Only [`DataStorage::Hdf5SingleFile`] is supported: the whole design rests on one file
    /// opened collectively, and the ascii/binary storages have no parallel story yet.
    pub fn new(
        file_name: impl AsRef<Path>,
        data_storage: DataStorage,
        comm: &impl Communicator,
    ) -> Result<Self> {
        // no file is opened yet, and the storage is not checked yet -- the backend behind both is
        // what is still missing, so this only takes the communicator it was designed to take
        let _unimplemented = data_storage;

        Ok(Self {
            xdmf_file_name: file_name.as_ref().to_path_buf().with_extension("xdmf2"),
            comm: comm.duplicate(),
        })
    }

    /// The XDMF file this writer writes, same as [`crate::TimeSeriesWriter::file_name`].
    pub fn file_name(&self) -> &Path {
        &self.xdmf_file_name
    }

    /// Write this rank's share of the mesh. Every rank must call this together.
    ///
    /// `owned_points` is this rank's owned points only, flat `xyz` -- **no ghost points**. A point
    /// shared between ranks is written by whichever rank owns it and referenced from the other
    /// ranks' `local_connectivity` by its global id, so this crate never needs to decide who owns
    /// what.
    ///
    /// `owned_global_ids` gives each owned point its id in the global mesh, one per point. The
    /// ids must be dense across all ranks (covering `0..N` exactly once) and, on each rank, one
    /// ascending contiguous block -- which is what an exclusive prefix sum of the per-rank owned
    /// counts produces. They are a cross-check on a partitioning the caller has already done, not
    /// a free-form assignment: passing them lets a caller mistake be reported instead of silently
    /// writing a mesh with a hole. A future revision may lift the contiguity requirement by
    /// placing each point at its own id.
    ///
    /// `local_connectivity` and `local_cell_types` are this rank's cells, indexing points by
    /// *global* id. Cells carry no equivalent of `owned_global_ids`: a cell belongs to exactly one
    /// rank by construction, so they are concatenated in rank order. Every rank must pass at least
    /// one cell, all of the same [`CellType`], and every rank must agree on which -- a `Mixed`
    /// topology is not supported here.
    ///
    /// Submeshes have no counterpart here either: a named submesh can span cells owned by many
    /// ranks, which needs a cross-rank merge rather than the existing per-rank split.
    pub fn write_mesh<C: Coordinate, I: ConnectivityIndex>(
        self,
        owned_points: &[C],
        owned_global_ids: &[u64],
        local_connectivity: &[I],
        local_cell_types: &[CellType],
    ) -> Result<TimeSeriesDataWriter> {
        let _unimplemented = (
            owned_points,
            owned_global_ids,
            local_connectivity,
            local_cell_types,
        );
        Err(UNIMPLEMENTED)
    }
}

/// Writer for a distributed mesh's time steps, the counterpart of
/// [`crate::TimeSeriesDataWriter`], obtained from [`TimeSeriesWriter::write_mesh`].
pub struct TimeSeriesDataWriter {
    xdmf_file_name: PathBuf,
    comm: SimpleCommunicator,
}

impl fmt::Debug for TimeSeriesDataWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeSeriesDataWriter")
            .field("xdmf_file_name", &self.xdmf_file_name)
            .field("rank", &self.comm.rank())
            .finish_non_exhaustive()
    }
}

impl TimeSeriesDataWriter {
    /// The XDMF file this writer writes, same as [`crate::TimeSeriesDataWriter::file_name`].
    pub fn file_name(&self) -> &Path {
        &self.xdmf_file_name
    }

    /// Write one time step. Every rank must call this together, with the same `time`, and with a
    /// closure that writes the same names, in the same order, on every rank -- the datasets
    /// behind them are created collectively, so a rank writing a different set of fields cannot
    /// be detected locally.
    ///
    /// Each [`TimeStep::point_data`]/[`TimeStep::cell_data`] call takes this rank's own share, in
    /// the same subset and order as the `owned_points`/`local_connectivity` the mesh was written
    /// with.
    pub fn write_time_step<F, E>(&mut self, time: impl Into<String>, write_step: F) -> Result<(), E>
    where
        F: FnOnce(&mut TimeStep<'_>) -> Result<(), E>,
        E: From<Error>,
    {
        let _unimplemented = (time.into(), write_step);
        Err(UNIMPLEMENTED.into())
    }
}

/// One time step of a distributed mesh, handed to the closure passed to
/// [`TimeSeriesDataWriter::write_time_step`]. The counterpart of [`crate::TimeStep`].
pub struct TimeStep<'a> {
    writer: &'a mut TimeSeriesDataWriter,
    time: String,
}

impl fmt::Debug for TimeStep<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeStep")
            .field("time", &self.time)
            .field("file_name", &self.writer.file_name())
            .finish_non_exhaustive()
    }
}

impl TimeStep<'_> {
    /// Write this rank's share of one point attribute: one value per *owned* point, in the same
    /// order as the `owned_points` the mesh was written with.
    pub fn point_data<'v>(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        data: impl Into<Values<'v>>,
    ) -> Result<()> {
        let _unimplemented = (name, attribute, data.into());
        Err(UNIMPLEMENTED)
    }

    /// Write this rank's share of one cell attribute: one value per local cell, in the same order
    /// as the `local_cell_types` the mesh was written with.
    pub fn cell_data<'v>(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        data: impl Into<Values<'v>>,
    ) -> Result<()> {
        let _unimplemented = (name, attribute, data.into());
        Err(UNIMPLEMENTED)
    }
}

//! Writing one XDMF file from a mesh distributed over MPI ranks.
//!
//! Every rank holds a piece of the mesh and passes only its own share, and the result is one
//! seamless mesh -- the same file a serial [`crate::TimeSeriesWriter`] would have written for the
//! whole mesh, with no rank boundaries visible in `ParaView`. This mirrors the serial API step for
//! step ([`TimeSeriesWriter::new`] -> [`TimeSeriesWriter::write_mesh`] ->
//! [`TimeSeriesDataWriter::write_time_step`]); the difference is that every array passed in is
//! this rank's own share rather than the whole mesh, and that every rank makes every call
//! together, in the same order, with the same names and times.
//!
//! These are deliberately separate types from the serial ones, rather than a parallel constructor
//! on them: it keeps a caller from reaching [`crate::TimeSeriesDataWriter::write_time_step`] on a
//! mesh written collectively, or [`TimeSeriesDataWriter::write_time_step`] on one written
//! serially, by construction rather than by a runtime check.
//!
//! # Storages
//!
//! The two HDF5 storages are written *in place*: every file is opened collectively through HDF5's
//! MPI-IO driver, and every rank writes its own share of every array straight into it, so no
//! rank ever holds more than its own share. Compression works as it does serially.
//!
//! The ascii and binary storages are written by rank 0 alone: each array is *gathered* to rank
//! 0, which writes it through the serial writer. The file is the same, but rank 0 briefly holds
//! the whole of each array it writes -- the mesh once, then one field at a time. Without MPI-IO
//! there is no other way to write one shared ascii or binary file.
//!
//! # Errors
//!
//! A call fails on every rank or on none: every point at which one rank's outcome can differ
//! from another's is agreed on across all of them before any of them goes on. A rank whose own
//! input was at fault returns that error; the others return the same variant, naming the rank.
//! Ranks calling with different arguments (another time, another field name, another order) are
//! told so rather than left waiting on each other. A [`TimeSeriesDataWriter::write_time_step`]
//! closure that fails on one rank discards the step on all of them.
//!
//! What cannot be caught this way is a rank that skips a call altogether -- one that never calls
//! `write_time_step` at all leaves the others waiting, as any MPI collective would.
//!
//! # Version coupling
//!
//! [`Communicator`] -- the one `mpi` item that appears in this module's signatures -- is
//! re-exported here so a caller can name the bound against the exact version this crate's `mpi`
//! feature was built with. A caller still depends on the `mpi` crate itself to *create* a
//! communicator, and must match this crate's version of it, since the two must agree on the type.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt,
    path::{Path, PathBuf},
};

use ::mpi::topology::SimpleCommunicator;
pub use ::mpi::traits::Communicator;
use layout::{CellKind, Layout};
use sync::{agree, converge, phase};

use super::{
    CELL_DATA, DOMAIN_DATA_ITEMS, POINT_DATA, geometry,
    hdf5::{MultipleFilesHdf5Writer, SingleFileHdf5Writer},
    sorted_names,
    submesh::{encode_mixed, poly_cell_points},
    validate_attribute, validate_file_name, validate_time,
};
use crate::{
    CELLS, CellType, ConnectivityIndex, Coordinate, DataAttribute, DataStorage, DataWriter, Error,
    POINTS, Result, Values, paraview,
    values::dimensions_of_len,
    xdmf_elements::{
        attribute::{Attribute, Center},
        data_item::{DataItem, Format},
        dimensions::Dimensions,
        grid::Grid,
        topology::{Topology, TopologyType},
    },
};

mod layout;
mod sync;

/// Writer for a distributed mesh, the counterpart of [`crate::TimeSeriesWriter`].
pub struct TimeSeriesWriter {
    // declared before `comm`, so dropped before it: closing a collectively opened HDF5 file is
    // itself a collective call
    backend: Backend<super::TimeSeriesWriter>,
    data_storage: DataStorage,
    xdmf_file_name: PathBuf,
    comm: SimpleCommunicator,
}

/// How the heavy data reaches the file, wrapping the serial writer type `W` doing the writing.
enum Backend<W> {
    /// Every rank writes its own share into collectively opened files.
    Collective(W),
    /// Rank 0 writes everything, gathered from every rank; `None` on every other rank.
    Gathered(Option<W>),
}

impl fmt::Debug for TimeSeriesWriter {
    /// Shows this rank's own number rather than the communicator, which has no useful `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeSeriesWriter")
            .field("xdmf_file_name", &self.xdmf_file_name)
            .field("data_storage", &self.data_storage)
            .field("rank", &self.comm.rank())
            .finish()
    }
}

impl TimeSeriesWriter {
    /// Create a writer over `comm`. Every rank must call this together, with the same
    /// `file_name` and `data_storage`.
    ///
    /// `comm` is duplicated and the copy kept, so later calls need not be handed one again and,
    /// more importantly, so this writer's collectives run in their own communication context.
    /// Collectives match by the order they are called in rather than by a tag, so a writer
    /// sharing the caller's communicator could match one of the caller's own collectives -- a
    /// nonblocking one still in flight while a step is written, say. Duplicating removes that
    /// class of bug, at the cost of one more collective call in a function that is already
    /// collective. Any communicator can be passed: the copy this keeps is freed with the writer.
    ///
    /// Every [`DataStorage`] is supported; see the [module documentation](self) for how they
    /// differ.
    pub fn new(
        file_name: impl AsRef<Path>,
        data_storage: DataStorage,
        comm: &impl Communicator,
    ) -> Result<Self> {
        let comm = comm.duplicate();
        let file_name = file_name.as_ref();
        let xdmf_file_name = file_name.with_extension("xdmf2");

        let begin = phase(("new", file_name, data_storage));

        let backend = match data_storage {
            DataStorage::Hdf5SingleFile { deflate_level }
            | DataStorage::Hdf5MultipleFiles { deflate_level } => {
                let checked = validate_file_name(&xdmf_file_name)
                    .and_then(|()| crate::validate_deflate_level(deflate_level));
                agree(&comm, begin, checked)?;

                let deflate_level = deflate_level.unwrap_or(super::hdf5::DEFAULT_DEFLATE_LEVEL);
                let writer: Result<Box<dyn DataWriter>> =
                    if matches!(data_storage, DataStorage::Hdf5SingleFile { .. }) {
                        SingleFileHdf5Writer::new_collective(file_name, deflate_level, &comm)
                            .map(|writer| Box::new(writer) as Box<dyn DataWriter>)
                    } else {
                        MultipleFilesHdf5Writer::new_collective(file_name, deflate_level, &comm)
                            .map(|writer| Box::new(writer) as Box<dyn DataWriter>)
                    };
                let writer = agree(&comm, phase(("new: opened", begin)), writer)?;

                Backend::Collective(super::TimeSeriesWriter {
                    xdmf_file_name: xdmf_file_name.clone(),
                    writer,
                })
            }
            DataStorage::Ascii | DataStorage::AsciiInline | DataStorage::Binary => {
                let writer = if comm.rank() == 0 {
                    super::TimeSeriesWriter::new(file_name, data_storage).map(Some)
                } else {
                    validate_file_name(&xdmf_file_name).map(|()| None)
                };

                Backend::Gathered(agree(&comm, begin, writer)?)
            }
        };

        Ok(Self {
            backend,
            data_storage,
            xdmf_file_name,
            comm,
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
    /// counts produces. Since a block is therefore fully described by its first id, the array is
    /// mostly a cross-check on a partitioning the caller has already done: it lets a caller
    /// mistake (a point owned by no rank, or by two) be reported instead of silently writing a
    /// mesh with a hole. It is a whole array rather than one offset so that a future revision can
    /// lift the contiguity requirement by placing each point at its own id, without a breaking
    /// change.
    ///
    /// `local_connectivity` and `local_cell_types` are this rank's cells, indexing points by
    /// *global* id. Cells carry no equivalent of `owned_global_ids`: a cell belongs to exactly one
    /// rank by construction, so they are concatenated in rank order, and cell data is matched to
    /// them in that same order. A rank may pass no points or no cells at all, as long as the mesh
    /// as a whole has both; cells of more than one type, on one rank or across ranks, are written
    /// as a `Mixed` topology as they would be serially.
    ///
    /// Submeshes have no counterpart here: a named submesh can span cells owned by many ranks,
    /// which needs a cross-rank merge rather than the existing per-rank split. A mesh of points
    /// only, without cells, has none either.
    pub fn write_mesh<C: Coordinate, I: ConnectivityIndex>(
        self,
        owned_points: &[C],
        owned_global_ids: &[u64],
        local_connectivity: &[I],
        local_cell_types: &[CellType],
    ) -> Result<TimeSeriesDataWriter> {
        let Self {
            backend,
            data_storage,
            xdmf_file_name,
            comm,
        } = self;
        let format = storage_format(data_storage);

        let begin = phase((
            "write_mesh",
            C::as_values(&[]).type_name(),
            I::as_values(&[]).type_name(),
        ));
        let shapes = check_shapes(
            owned_points.len(),
            owned_global_ids.len(),
            local_connectivity.len(),
            local_cell_types,
        );
        agree(&comm, begin, shapes)?;

        let points = layout::point_shares(&comm, owned_global_ids)?;
        let (cells, num_cells, cell_kind) = layout::cell_shares(&comm, local_cell_types)?;
        let layout = layout::layout(&comm, points, (cells, num_cells));

        let indices = check_indices(local_connectivity, layout.num_points(), format);
        agree(&comm, phase(("write_mesh: indices", begin)), indices)?;

        let mesh = LocalMesh {
            points: owned_points,
            connectivity: local_connectivity,
            cell_types: local_cell_types,
            cell_kind,
        };

        let backend = match backend {
            Backend::Collective(writer) => {
                Backend::Collective(write_mesh_collective(&comm, writer, &mesh, &layout)?)
            }
            Backend::Gathered(writer) => {
                Backend::Gathered(write_mesh_gathered(&comm, writer, &mesh, &layout)?)
            }
        };

        Ok(TimeSeriesDataWriter {
            backend,
            layout,
            written_times: HashMap::new(),
            format,
            xdmf_file_name,
            comm,
        })
    }
}

/// One rank's share of a mesh, as passed to [`TimeSeriesWriter::write_mesh`].
struct LocalMesh<'m, C, I> {
    points: &'m [C],
    connectivity: &'m [I],
    cell_types: &'m [CellType],
    cell_kind: CellKind,
}

/// The `Format` a storage writes, which every rank needs for [`paraview::validate`] -- in a
/// gathered storage, only rank 0 holds the writer that would otherwise say.
const fn storage_format(data_storage: DataStorage) -> Format {
    match data_storage {
        DataStorage::Ascii | DataStorage::AsciiInline => Format::XML,
        DataStorage::Binary => Format::Binary,
        DataStorage::Hdf5SingleFile { .. } | DataStorage::Hdf5MultipleFiles { .. } => Format::HDF,
    }
}

/// The checks on one rank's own arguments that need nothing from the other ranks.
fn check_shapes(
    num_coordinates: usize,
    num_global_ids: usize,
    connectivity_len: usize,
    cell_types: &[CellType],
) -> Result<()> {
    if !num_coordinates.is_multiple_of(3) {
        return Err(Error::InvalidMesh {
            reason: format!(
                "points must have 3 dimensions, but {num_coordinates} is not a multiple of 3"
            ),
        });
    }

    if num_global_ids != num_coordinates / 3 {
        return Err(Error::InvalidMesh {
            reason: format!(
                "owned_global_ids has {num_global_ids} entries, but owned_points has {} points",
                num_coordinates / 3
            ),
        });
    }

    let expected: usize = cell_types.iter().map(CellType::num_points).sum();
    if expected != connectivity_len {
        return Err(Error::InvalidMesh {
            reason: format!(
                "size of connectivity ({connectivity_len}) does not match the number expected \
                 from the cell types ({expected})"
            ),
        });
    }

    Ok(())
}

/// Every index of this rank's connectivity must name a point of the whole mesh, and the
/// connectivity type must be able to name every one of them.
fn check_indices<I: ConnectivityIndex>(
    connectivity: &[I],
    num_points: usize,
    format: Format,
) -> Result<()> {
    let num_points = i128::try_from(num_points).unwrap_or(i128::MAX);

    if num_points - 1 > I::MAX_INDEX {
        return Err(Error::InvalidMesh {
            reason: format!(
                "the mesh has {num_points} points, more than its connectivity type can index"
            ),
        });
    }

    if let Some(index) = connectivity
        .iter()
        .map(|index| index.as_i128())
        .find(|index| !(0..num_points).contains(index))
    {
        return Err(Error::InvalidMesh {
            reason: format!(
                "connectivity index {index} is out of bounds, the mesh only has {num_points} \
                 points"
            ),
        });
    }

    paraview::validate(&I::as_values(connectivity), format)
}

/// Write the mesh into collectively opened files, every rank its own share in place.
fn write_mesh_collective<C: Coordinate, I: ConnectivityIndex>(
    comm: &SimpleCommunicator,
    mut writer: super::TimeSeriesWriter,
    mesh: &LocalMesh<'_, C, I>,
    layout: &Layout,
) -> Result<super::TimeSeriesDataWriter> {
    let format = writer.writer.format();

    let cells = match mesh.cell_kind {
        CellKind::Uniform(_) => Cow::Borrowed(mesh.connectivity),
        CellKind::Mixed => Cow::Owned(encode_mixed(mesh.connectivity, mesh.cell_types)),
    };
    let (connectivity_shares, connectivity_len) = layout::concatenated(comm, cells.len())?;
    let local_connectivity = connectivity_shares
        .get(usize::try_from(comm.rank()).unwrap_or(0))
        .copied()
        .unwrap_or_default();

    let num_coordinates = layout
        .num_points()
        .checked_mul(3)
        .ok_or(Error::Internal("a point count was checked to fit a usize"))?;
    let local_points = layout.local_points();

    let points = C::as_values(mesh.points);
    let written = writer.writer.write_mesh_array_collective(
        POINTS,
        &points,
        local_points.offset() * 3,
        num_coordinates,
    );
    let points_item = DataItem {
        name: Some("coords".to_string()),
        item_type: None,
        dimensions: Some(Dimensions(vec![layout.num_points(), 3])),
        data: agree(comm, phase("write_mesh: points"), written)?,
        number_type: Some(points.number_type()),
        precision: Some(points.precision()),
        format: Some(format),
        endian: format.endian(),
        reference: None,
    };

    let connectivity = I::as_values(&cells);
    let written = writer.writer.write_mesh_array_collective(
        CELLS,
        &connectivity,
        local_connectivity.offset(),
        connectivity_len,
    );
    let connectivity_item = DataItem {
        name: Some("connectivity".to_string()),
        item_type: None,
        dimensions: Some(Dimensions(vec![connectivity_len])),
        data: agree(comm, phase("write_mesh: connectivity"), written)?,
        number_type: Some(connectivity.number_type()),
        precision: Some(connectivity.precision()),
        format: Some(format),
        endian: format.endian(),
        reference: None,
    };

    let (topology_type, nodes_per_element) = match mesh.cell_kind {
        CellKind::Uniform(cell_type) => {
            (TopologyType::from(cell_type), poly_cell_points(cell_type))
        }
        CellKind::Mixed => (TopologyType::Mixed, None),
    };
    let topology = Topology {
        topology_type,
        nodes_per_element,
        number_of_elements: layout.num_cells().to_string(),
        data_item: DataItem::new_reference(&connectivity_item, DOMAIN_DATA_ITEMS),
    };

    let grid = Grid::new_uniform("mesh", geometry(&points_item), topology);

    let built = writer.build_data_writer(
        grid,
        vec![points_item, connectivity_item],
        Vec::new(),
        layout.num_points(),
        layout.num_cells(),
    );
    let mut writer = agree(comm, phase("write_mesh: built"), built)?;

    write_light_data(comm, &mut writer, "write_mesh")?;

    Ok(writer)
}

/// Flush the collectively written heavy data -- collective itself -- and write the light data
/// from rank 0, the only one that may write that file.
fn write_light_data(
    comm: &SimpleCommunicator,
    writer: &mut super::TimeSeriesDataWriter,
    call: &str,
) -> Result<()> {
    let flushed = writer.writer.flush();
    agree(comm, phase((call, "flushed")), flushed)?;

    let written = if comm.rank() == 0 {
        writer.write_light_data_file()
    } else {
        Ok(())
    };
    agree(comm, phase((call, "light data")), written)
}

/// Gather the mesh to rank 0 and write it there through the serial writer.
fn write_mesh_gathered<C: Coordinate, I: ConnectivityIndex>(
    comm: &SimpleCommunicator,
    writer: Option<super::TimeSeriesWriter>,
    mesh: &LocalMesh<'_, C, I>,
    layout: &Layout,
) -> Result<Option<super::TimeSeriesDataWriter>> {
    let (point_shares, num_coordinates) =
        layout::value_shares(layout.points(), layout.num_points(), 3)?;
    let (connectivity_shares, connectivity_len) =
        layout::concatenated(comm, mesh.connectivity.len())?;
    let (connectivity_shares, _) = layout::value_shares(&connectivity_shares, connectivity_len, 1)?;
    let (cell_shares, _) = layout::value_shares(layout.cells(), layout.num_cells(), 1)?;

    let points = layout::gather(
        comm,
        &C::as_values(mesh.points),
        &point_shares,
        num_coordinates,
    );
    let connectivity = layout::gather(
        comm,
        &I::as_values(mesh.connectivity),
        &connectivity_shares,
        connectivity_len,
    );
    let codes: Vec<u8> = mesh
        .cell_types
        .iter()
        .map(|cell_type| *cell_type as u8)
        .collect();
    let codes = layout::gather_slice(comm, &codes, &cell_shares, layout.num_cells());

    let written = match (writer, points, connectivity, codes) {
        (None, ..) => Ok(None),
        (Some(writer), Some(points), Some(connectivity), Some(codes)) => {
            write_gathered_mesh::<C, I>(writer, points, connectivity, &codes).map(Some)
        }
        _ => Err(Error::Internal("rank 0 did not receive the gathered mesh")),
    };

    agree(comm, phase("write_mesh: gathered"), written)
}

/// Rank 0's part of [`write_mesh_gathered`]: the whole mesh, as the serial writer takes it.
fn write_gathered_mesh<C: Coordinate, I: ConnectivityIndex>(
    writer: super::TimeSeriesWriter,
    points: Values<'static>,
    connectivity: Values<'static>,
    codes: &[u8],
) -> Result<super::TimeSeriesDataWriter> {
    let points = C::coordinates_from_values(points)?;
    let connectivity = I::indices_from_values(connectivity)?;
    let cell_types = codes
        .iter()
        .map(|code| {
            CellType::from_code(*code).ok_or(Error::Internal("a gathered cell type is unknown"))
        })
        .collect::<Result<Vec<_>>>()?;

    writer.write_mesh(&points, &connectivity, &cell_types)
}

/// Writer for a distributed mesh's time steps, the counterpart of
/// [`crate::TimeSeriesDataWriter`], obtained from [`TimeSeriesWriter::write_mesh`].
pub struct TimeSeriesDataWriter {
    // declared before `comm`, so dropped before it -- see `TimeSeriesWriter`
    backend: Backend<super::TimeSeriesDataWriter>,
    layout: Layout,
    // kept here as well as in the serial writer: in a gathered storage only rank 0 has one, but
    // every rank must reject a repeated time the same way
    written_times: HashMap<u64, String>,
    format: Format,
    xdmf_file_name: PathBuf,
    comm: SimpleCommunicator,
}

impl fmt::Debug for TimeSeriesDataWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeSeriesDataWriter")
            .field("xdmf_file_name", &self.xdmf_file_name)
            .field("rank", &self.comm.rank())
            .field("num_points", &self.layout.num_points())
            .field("num_cells", &self.layout.num_cells())
            .finish_non_exhaustive()
    }
}

/// An error out of the closure a gathered step runs inside the serial writer's own: either the
/// caller's, passed through untouched, or this crate's -- kept apart so that rank 0 can share the
/// latter with the other ranks.
enum StepError<E> {
    Caller(E),
    Crate(Error),
}

impl<E> From<Error> for StepError<E> {
    fn from(error: Error) -> Self {
        Self::Crate(error)
    }
}

/// What every rank other than `rank` reports for a step whose closure failed on `rank`.
fn failed_elsewhere(time: &str, rank: i32) -> Error {
    Error::InvalidTimeStep {
        time: time.to_string(),
        reason: format!("discarded, since writing it failed on rank {rank}"),
    }
}

impl TimeSeriesDataWriter {
    /// The XDMF file this writer writes, same as [`crate::TimeSeriesDataWriter::file_name`].
    pub fn file_name(&self) -> &Path {
        &self.xdmf_file_name
    }

    /// Write one time step. Every rank must call this together, with the same `time`, and with a
    /// closure that writes the same names, in the same order, on every rank.
    ///
    /// Each [`TimeStep::point_data`]/[`TimeStep::cell_data`] call takes this rank's own share, in
    /// the same subset and order as the `owned_points`/`local_connectivity` the mesh was written
    /// with.
    ///
    /// As serially, `Err` discards the step and leaves its time available for another attempt --
    /// here on every rank, if the closure failed on any one of them. A rank whose closure failed
    /// returns its own error; the others return [`Error::InvalidTimeStep`] naming that rank.
    pub fn write_time_step<F, E>(&mut self, time: impl Into<String>, write_step: F) -> Result<(), E>
    where
        F: FnOnce(&mut TimeStep<'_>) -> Result<(), E>,
        E: From<Error>,
    {
        let time = time.into();
        let begin = phase(("write_time_step", &time));
        let validated = validate_time(time, &self.written_times);
        let (time, time_bits) = agree(&self.comm, begin, validated)?;

        let context = StepContext {
            comm: &self.comm,
            layout: &self.layout,
            format: self.format,
            time: &time,
        };

        // set once the step is part of the document, even if writing the file failed after that:
        // as serially, such a step stays written, and its time cannot be written again
        let mut committed = false;

        let written = match &mut self.backend {
            Backend::Collective(writer) => {
                collective_step(&context, writer, write_step, &mut committed)
            }
            Backend::Gathered(writer) => {
                gathered_step(&context, writer.as_mut(), write_step, &mut committed)
            }
        };

        if committed {
            self.written_times.insert(time_bits, time);
        }

        written?;
        Ok(())
    }
}

/// What every [`TimeStep`] of one step needs to know, whichever backend it writes through.
#[derive(Clone, Copy)]
struct StepContext<'a> {
    comm: &'a SimpleCommunicator,
    layout: &'a Layout,
    format: Format,
    time: &'a str,
}

impl<'c> StepContext<'c> {
    /// Where every rank meets at the end of the step, see [`converge`].
    fn end(&self) -> u64 {
        phase(("end of time step", self.time))
    }

    fn step<'a>(&self, target: Target<'a>) -> TimeStep<'a>
    where
        'c: 'a,
    {
        TimeStep {
            comm: self.comm,
            layout: self.layout,
            format: self.format,
            time: self.time,
            target,
            point_names: HashSet::new(),
            cell_names: HashSet::new(),
            num_fields: 0,
        }
    }
}

/// One step on collectively opened files: every rank writes its share of each field in place,
/// and rank 0 adds the step's `<Grid>` to the light data.
fn collective_step<F, E>(
    context: &StepContext<'_>,
    writer: &mut super::TimeSeriesDataWriter,
    write_step: F,
    committed: &mut bool,
) -> Result<(), E>
where
    F: FnOnce(&mut TimeStep<'_>) -> Result<(), E>,
    E: From<Error>,
{
    let mut step = context.step(Target::Collective {
        writer,
        attributes: Vec::new(),
        initialized: false,
    });
    let result = write_step(&mut step);
    let failed_rank = converge(context.comm, context.end(), result.is_err());

    let Target::Collective {
        writer,
        attributes,
        initialized,
    } = step.target
    else {
        return Err(Error::Internal("a collective step lost its writer").into());
    };

    // every rank has the same `failed_rank`, `initialized` and number of attributes, so every
    // rank takes the same branch here, and a discard -- collective in HDF5 -- is made by all
    if let Some(rank) = failed_rank {
        if initialized {
            let _discard_result = writer.writer.write_data_discard();
        }
        return Err(result
            .err()
            .unwrap_or_else(|| failed_elsewhere(context.time, rank).into()));
    }

    if attributes.is_empty() {
        if initialized {
            let _discard_result = writer.writer.write_data_discard();
        }
        return Err(Error::InvalidTimeStep {
            time: context.time.to_string(),
            reason: format!("no data written, needs at least one {POINT_DATA} or {CELL_DATA}"),
        }
        .into());
    }

    let finalized = writer
        .writer
        .write_data_finalize()
        .and_then(|()| writer.push_step(context.time, attributes, Vec::new()));
    agree(context.comm, phase(("finalized", context.end())), finalized)?;

    writer.step_times.push(context.time.to_string());
    *committed = true;

    write_light_data(context.comm, writer, context.time)?;
    Ok(())
}

/// One step through a serial writer on rank 0: every rank gathers each field to rank 0, which
/// writes it through the serial writer's own step -- inside whose closure the caller's runs.
fn gathered_step<F, E>(
    context: &StepContext<'_>,
    writer: Option<&mut super::TimeSeriesDataWriter>,
    write_step: F,
    committed: &mut bool,
) -> Result<(), E>
where
    F: FnOnce(&mut TimeStep<'_>) -> Result<(), E>,
    E: From<Error>,
{
    let commit = phase(("committed", context.end()));

    let Some(writer) = writer else {
        let mut step = context.step(Target::Other);
        let result = write_step(&mut step);

        if let Some(rank) = converge(context.comm, context.end(), result.is_err()) {
            return Err(result
                .err()
                .unwrap_or_else(|| failed_elsewhere(context.time, rank).into()));
        }

        // rank 0 writes the step after every closure succeeded, which can still fail there
        *committed = sync::shared_flag(context.comm, false);
        return agree(context.comm, commit, Ok(())).map_err(E::from);
    };

    // `None` until the caller's closure has run and every rank has reported back
    let mut converged = None;

    let written = writer.write_time_step(context.time, |serial| -> Result<(), StepError<E>> {
        let mut step = context.step(Target::Root(serial));
        let result = write_step(&mut step);

        let failed_rank = converge(context.comm, context.end(), result.is_err());
        converged = Some(failed_rank);

        match (result, failed_rank) {
            (Err(error), _) => Err(StepError::Caller(error)),
            (Ok(()), Some(rank)) => Err(failed_elsewhere(context.time, rank).into()),
            (Ok(()), None) => Ok(()),
        }
    });

    let as_crate_error = |written: Result<(), StepError<E>>| match written {
        Ok(()) => Ok(()),
        Err(StepError::Crate(error)) => Err(error),
        Err(StepError::Caller(_)) => Err(Error::Internal(
            "a closure that succeeded on every rank failed on rank 0",
        )),
    };

    match converged {
        // every rank's closure succeeded, and the serial writer has finished the step since --
        // or failed to, which every other rank needs to hear about
        Some(None) => {
            // the serial writer records a step before writing the file, which may still fail
            let recorded = writer.step_times.last().map(String::as_str) == Some(context.time);
            *committed = sync::shared_flag(context.comm, recorded);
            agree(context.comm, commit, as_crate_error(written)).map_err(E::from)
        }
        // every rank already knows the step failed
        Some(Some(_)) => written.map_err(|error| match error {
            StepError::Caller(error) => error,
            StepError::Crate(error) => E::from(error),
        }),
        // the serial writer refused the step before running the closure -- unreachable, since
        // every rank validated the time the same way first -- so the other ranks still need to
        // be told, at the point they are waiting at
        None => {
            converge(context.comm, context.end(), true);
            Err(E::from(as_crate_error(written).err().unwrap_or(
                Error::Internal("the serial writer skipped a step's closure"),
            )))
        }
    }
}

/// Rank 0's view of the serial writer's step, which [`TimeStep`] writes each gathered field
/// through. A trait object rather than the serial type itself, which would tie [`TimeStep`]'s
/// lifetime to the serial step's.
trait GatheredStep {
    fn write(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        values: Values<'_>,
        center: Center,
    ) -> Result<()>;
}

impl GatheredStep for super::TimeStep<'_> {
    fn write(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        values: Values<'_>,
        center: Center,
    ) -> Result<()> {
        self.write_attribute(name, attribute, values, center)
    }
}

/// Where a [`TimeStep`] sends each field.
enum Target<'a> {
    /// Into collectively opened files, this rank's share in place.
    Collective {
        writer: &'a mut super::TimeSeriesDataWriter,
        attributes: Vec<Attribute>,
        // whether `write_data_initialize` has run, deferred to the first field as serially
        initialized: bool,
    },
    /// Gathered here, on rank 0, and written through the serial step.
    Root(&'a mut (dyn GatheredStep + 'a)),
    /// Gathered to rank 0, from any other rank.
    Other,
}

/// One time step of a distributed mesh, handed to the closure passed to
/// [`TimeSeriesDataWriter::write_time_step`]. The counterpart of [`crate::TimeStep`].
pub struct TimeStep<'a> {
    comm: &'a SimpleCommunicator,
    layout: &'a Layout,
    format: Format,
    time: &'a str,
    target: Target<'a>,
    // tracked separately since the same name may be used for one of each
    point_names: HashSet<String>,
    cell_names: HashSet<String>,
    // fields written so far, which names each one's heavy data and is part of its phase
    num_fields: usize,
}

impl fmt::Debug for TimeStep<'_> {
    /// Names only, sorted, as serially.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeStep")
            .field("time", &self.time)
            .field("point_data", &sorted_names(&self.point_names))
            .field("cell_data", &sorted_names(&self.cell_names))
            .finish_non_exhaustive()
    }
}

impl TimeStep<'_> {
    /// Write this rank's share of one point attribute: one value (or one tuple) per *owned*
    /// point, in the same order as the `owned_points` the mesh was written with.
    pub fn point_data<'v>(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        data: impl Into<Values<'v>>,
    ) -> Result<()> {
        self.write_attribute(name, attribute, data.into(), Center::Node)
    }

    /// Write this rank's share of one cell attribute: one value (or one tuple) per local cell, in
    /// the same order as the `local_cell_types` the mesh was written with.
    pub fn cell_data<'v>(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        data: impl Into<Values<'v>>,
    ) -> Result<()> {
        self.write_attribute(name, attribute, data.into(), Center::Cell)
    }

    fn write_attribute(
        &mut self,
        name: &str,
        data_attribute: DataAttribute,
        values: Values<'_>,
        center: Center,
    ) -> Result<()> {
        let is_point_data = center == Center::Node;
        let (label, shares, num_entities, local, seen_names) = if is_point_data {
            (
                POINT_DATA,
                self.layout.points(),
                self.layout.num_points(),
                self.layout.local_points(),
                &self.point_names,
            )
        } else {
            (
                CELL_DATA,
                self.layout.cells(),
                self.layout.num_cells(),
                self.layout.local_cells(),
                &self.cell_names,
            )
        };

        // everything every rank must agree on about this field before any collective for it
        let field = phase((
            self.time,
            self.num_fields,
            label,
            name,
            data_attribute,
            values.type_name(),
        ));

        let collective = matches!(self.target, Target::Collective { .. });
        let checked = (|| {
            let stride = validate_attribute(
                name,
                data_attribute,
                values.len(),
                local.len(),
                label,
                seen_names,
            )?;
            paraview::validate(&values, self.format)?;

            // a gathered storage sends each rank's share as one message, whose size is capped
            let (value_shares, len) = if collective {
                let len = num_entities
                    .checked_mul(stride)
                    .ok_or_else(|| Error::InvalidData {
                        reason: format!("{label} '{name}' has more values than a usize can count"),
                    })?;
                (Vec::new(), len)
            } else {
                layout::value_shares(shares, num_entities, stride)?
            };

            Ok((stride, value_shares, len))
        })();
        let (stride, value_shares, len) = agree(self.comm, field, checked)?;

        let index = self.num_fields;
        self.num_fields += 1;

        let written = phase(("written", field));

        match &mut self.target {
            Target::Collective {
                writer,
                attributes,
                initialized,
            } => {
                let attribute = (|| {
                    if !*initialized {
                        writer.writer.write_data_initialize(self.time)?;
                        *initialized = true;
                    }

                    let format = writer.writer.format();
                    let data = writer.writer.write_data_collective(
                        index,
                        &values,
                        local.offset() * stride,
                        len,
                    )?;

                    Ok(Attribute {
                        name: name.to_string(),
                        attribute_type: data_attribute.into(),
                        center,
                        data_items: vec![DataItem {
                            name: None,
                            item_type: None,
                            dimensions: Some(dimensions_of_len(len, data_attribute)),
                            number_type: Some(values.number_type()),
                            format: Some(format),
                            precision: Some(values.precision()),
                            endian: format.endian(),
                            data,
                            reference: None,
                        }],
                    })
                })();

                attributes.push(agree(self.comm, written, attribute)?);
            }
            Target::Root(serial) => {
                let gathered = layout::gather(self.comm, &values, &value_shares, len)
                    .ok_or(Error::Internal("rank 0 did not receive a gathered field"))
                    .and_then(|values| serial.write(name, data_attribute, values, center));
                agree(self.comm, written, gathered)?;
            }
            Target::Other => {
                layout::gather(self.comm, &values, &value_shares, len);
                agree(self.comm, written, Ok(()))?;
            }
        }

        // recorded only once the attribute is actually written, so a rejected call can be
        // retried under the same name
        if is_point_data {
            self.point_names.insert(name.to_string());
        } else {
            self.cell_names.insert(name.to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_format_matches_each_backend() {
        let dir = temp_dir::TempDir::new().unwrap();

        for (storage, expected) in [
            (DataStorage::Ascii, Format::XML),
            (DataStorage::AsciiInline, Format::XML),
            (DataStorage::Binary, Format::Binary),
            (
                DataStorage::Hdf5SingleFile {
                    deflate_level: None,
                },
                Format::HDF,
            ),
            (
                DataStorage::Hdf5MultipleFiles {
                    deflate_level: None,
                },
                Format::HDF,
            ),
        ] {
            let name = format!("{storage:?}").replace(['{', '}', ':', ' '], "");
            let writer = crate::create_writer(&dir.path().join(name), storage).unwrap();

            assert_eq!(storage_format(storage), expected);
            assert_eq!(writer.format(), expected);
        }
    }
}

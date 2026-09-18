//! Sizing and correctness collectives for
//! [`ParallelTimeSeriesWriter::write_mesh`](super::ParallelTimeSeriesWriter::write_mesh).
//!
//! See `07_mpi.md` (on the plans share, outside this repository) for the design these implement.

use std::{collections::HashSet, fmt, path::Path};

use mpi::{
    collective::SystemOperation,
    traits::{Communicator, CommunicatorCollectives},
};

use super::{
    CELL_DATA, DOMAIN_DATA_ITEMS, POINT_DATA, TimeSeriesDataWriter, TimeSeriesWriter, geometry,
    hdf5, is_valid_data_name, sorted_names, validate_file_name,
};
use crate::{
    CellType, ConnectivityIndex, Coordinate, DataAttribute, DataStorage, Error, Result, Values,
    mpi_safe_create_dir_all, paraview,
    xdmf_elements::{
        attribute,
        data_item::DataItem,
        dimensions::Dimensions,
        grid::Grid,
        topology::{Topology, TopologyType},
    },
};

/// This rank's exclusive prefix sum of `local_len` (its offset into the rank-ordered
/// concatenation of every rank's own), and the total across every rank.
///
/// Used for a mesh's cells, which -- unlike its points -- have no ownership question of their
/// own: a rank's local cells are unconditionally its owned cells (see "Cells have no ownership
/// question" in `07_mpi.md`), so they are simply concatenated in rank order rather than placed by
/// an explicit id.
pub(super) fn exclusive_scan_offset(comm: &impl Communicator, local_len: usize) -> (usize, usize) {
    let mut offset = 0_usize;
    comm.exclusive_scan_into(&local_len, &mut offset, SystemOperation::sum());

    let mut total = 0_usize;
    comm.all_reduce_into(&local_len, &mut total, SystemOperation::sum());

    (offset, total)
}

/// Verify that every rank's `owned_global_ids` sits on the single ascending, contiguous block a
/// plain hyperslab write can place correctly, and that the union of every rank's block is exactly
/// `0..declared_total` with no gap and no overlap.
///
/// Returns this rank's own offset (the start of its block, `0` if it owns nothing) and the
/// global point count.
///
/// This is a stricter, and different, check than the design in `07_mpi.md` sketches: that design
/// only verifies the aggregate count/min/max match, which -- on its own -- does not catch a caller
/// whose ids are dense but not laid out as one block per rank in rank order (a real point would
/// land at the wrong row, silently). Requiring a contiguous per-rank block, as the doc's own
/// "prerequisite work outside this crate" section already assumes a caller builds via `Exscan`,
/// closes that gap; a future revision could instead write each point at its own id via HDF5's
/// point-selection API, at the cost of a scattered rather than a hyperslab write, if a caller ever
/// needs ids that are not rank-contiguous.
pub(super) fn owned_ids_sanity_check(
    comm: &impl Communicator,
    owned_global_ids: &[u64],
) -> Result<(usize, usize)> {
    // Every check below is local -- computed here, but not acted on until every rank has made
    // exactly the same sequence of collective calls. Returning early on a purely local result
    // (e.g. this rank's own block is not contiguous) would leave the other ranks waiting on a
    // reduction that this one never issues, which hangs rather than fails.
    let local_contiguous = owned_global_ids
        .windows(2)
        .all(|pair| pair[1] == pair[0] + 1);
    let local_len = owned_global_ids.len();
    let local_min = owned_global_ids.first().copied().unwrap_or(u64::MAX);
    let local_max = owned_global_ids.last().copied().unwrap_or(0);

    let mut all_contiguous = 0_u8;
    comm.all_reduce_into(
        &u8::from(local_contiguous),
        &mut all_contiguous,
        SystemOperation::min(),
    );

    let mut declared_total = 0_usize;
    comm.all_reduce_into(&local_len, &mut declared_total, SystemOperation::sum());

    let mut min_id = u64::MAX;
    comm.all_reduce_into(&local_min, &mut min_id, SystemOperation::min());

    let mut max_id = 0_u64;
    comm.all_reduce_into(&local_max, &mut max_id, SystemOperation::max());

    // every collective above has now run on every rank; branching from here on is safe

    if all_contiguous == 0 {
        return Err(Error::InvalidMesh {
            reason: "owned_global_ids must be sorted, ascending and contiguous on every rank -- \
                     write_mesh_parallel does not support a scattered id assignment yet"
                .to_string(),
        });
    }

    if declared_total == 0 {
        return Err(Error::InvalidMesh {
            reason: "write_mesh_parallel needs at least one owned point across all ranks"
                .to_string(),
        });
    }

    let expected_max = u64::try_from(declared_total - 1)
        .map_err(|_conversion_failed| Error::Internal("a global point count does not fit a u64"))?;

    if min_id != 0 || max_id != expected_max {
        return Err(Error::InvalidMesh {
            reason: format!(
                "owned_global_ids across all ranks must cover exactly 0..{declared_total}, one \
                 contiguous block per rank; saw ids from {min_id} to {max_id} over \
                 {declared_total} points total -- a point is likely owned by zero ranks or by \
                 more than one"
            ),
        });
    }

    let offset = if local_len == 0 {
        0
    } else {
        usize::try_from(local_min).unwrap_or(0)
    };

    Ok((offset, declared_total))
}

/// Verify that every one of `local_connectivity`'s indices is a valid global point id, i.e. less
/// than `global_num_points`.
///
/// Collective, like [`owned_ids_sanity_check`]/[`agree_on_cell_type`]: an out-of-range index is a
/// purely local fact, but every rank must still reach the same collectives after this call
/// regardless of whether its own connectivity passes, so this checks locally and only then
/// agrees on the answer -- an early return here on the failing rank alone, before the ones after
/// it, would leave the other ranks waiting on a reduction/dataset creation this one never reaches.
pub(super) fn validate_connectivity_indices<I: crate::ConnectivityIndex>(
    comm: &impl Communicator,
    local_connectivity: &[I],
    global_num_points: usize,
) -> Result<()> {
    let local_valid = local_connectivity
        .iter()
        .all(|&index| (0..global_num_points as i128).contains(&index.as_i128()));

    let mut all_valid = 0_u8;
    comm.all_reduce_into(
        &u8::from(local_valid),
        &mut all_valid,
        SystemOperation::min(),
    );

    if all_valid == 0 {
        return Err(Error::InvalidMesh {
            reason: format!(
                "local_connectivity has an index out of bounds on at least one rank; the mesh \
                 only has {global_num_points} points"
            ),
        });
    }

    Ok(())
}

/// Verify that every rank passed one, and the same, [`CellType`] -- a `Mixed` topology, whether
/// from more than one type on one rank or ranks disagreeing on their single type, is not
/// supported by [`super::ParallelTimeSeriesWriter::write_mesh`] yet.
pub(super) fn agree_on_cell_type(
    comm: &impl Communicator,
    local_cell_types: &[CellType],
) -> Result<CellType> {
    // local checks, computed but (as in `owned_ids_sanity_check`) not acted on until every rank
    // has made the same collective calls below, whatever its own arguments look like
    let local_is_uniform = local_cell_types
        .first()
        .is_some_and(|first| local_cell_types.iter().all(|cell_type| cell_type == first));
    // `0` never collides with a real cell type's code (every one of those starts at `1`), so an
    // empty rank's sentinel cannot spuriously agree with another rank's real type
    let local_code = local_cell_types.first().map_or(0_u8, |ct| *ct as u8);

    let mut all_uniform = 0_u8;
    comm.all_reduce_into(
        &u8::from(local_is_uniform),
        &mut all_uniform,
        SystemOperation::min(),
    );

    let mut max_code = 0_u8;
    comm.all_reduce_into(&local_code, &mut max_code, SystemOperation::max());
    let mut min_code = u8::MAX;
    comm.all_reduce_into(&local_code, &mut min_code, SystemOperation::min());

    if all_uniform == 0 {
        return Err(Error::InvalidMesh {
            reason: "write_mesh_parallel requires one CellType per rank (at least one cell, and \
                     all of it the same type); a Mixed topology isn't supported for parallel \
                     meshes yet"
                .to_string(),
        });
    }

    if min_code != max_code {
        return Err(Error::InvalidMesh {
            reason: "every rank must write the same CellType; a Mixed topology across ranks \
                     isn't supported for parallel meshes yet"
                .to_string(),
        });
    }

    // `local_is_uniform` held (checked above via `all_uniform`), so indexing the first element is
    // safe on every rank that reaches this line
    Ok(local_cell_types[0])
}

/// Write this rank's own share of the mesh's points and describe it as a named, `Domain`-level
/// `DataItem` sized to the *global* mesh, mirroring `TimeSeriesWriter::points_data_item`.
///
/// A free function taking `writer` explicitly, rather than a second `impl TimeSeriesWriter`
/// block, since a type may not have more than one inherent impl block per crate (clippy enforces
/// this as `multiple_inherent_impl` -- see the crate-level `[lints.clippy]`).
fn points_data_item_parallel(
    writer: &mut TimeSeriesWriter,
    local_points: &Values<'_>,
    point_offset: usize,
    global_num_points: usize,
) -> Result<DataItem> {
    let format = writer.writer.format();

    Ok(DataItem {
        name: Some("coords".to_string()),
        item_type: None,
        dimensions: Some(Dimensions(vec![global_num_points, 3])),
        data: writer.writer.write_mesh_array_parallel(
            crate::POINTS,
            local_points,
            point_offset * 3,
            global_num_points * 3,
        )?,
        number_type: Some(local_points.number_type()),
        precision: Some(local_points.precision()),
        format: Some(format),
        endian: format.endian(),
        reference: None,
    })
}

/// Write this rank's own cells and describe them as a named, `Domain`-level `DataItem` sized to
/// the *global* mesh, mirroring `TimeSeriesWriter::connectivity_data_item`. `offset`/`global_len`
/// are already in units of connectivity entries (this rank's/every rank's cell count times the
/// cell type's point count), not of cells.
fn connectivity_data_item_parallel<I: ConnectivityIndex>(
    writer: &mut TimeSeriesWriter,
    local_cells: &[I],
    offset: usize,
    global_len: usize,
) -> Result<DataItem> {
    let values = I::as_values(local_cells);
    let format = writer.writer.format();

    Ok(DataItem {
        name: Some("connectivity".to_string()),
        item_type: None,
        dimensions: Some(Dimensions(vec![global_len])),
        data: writer
            .writer
            .write_mesh_array_parallel(crate::CELLS, &values, offset, global_len)?,
        number_type: Some(values.number_type()),
        precision: Some(values.precision()),
        format: Some(format),
        endian: format.endian(),
        reference: None,
    })
}

/// Writer for time series data in XDMF format, obtained with [`ParallelTimeSeriesWriter::new`]
/// instead of [`TimeSeriesWriter::new`].
///
/// A distinct type from [`TimeSeriesWriter`], rather than a parallel constructor on it, for the
/// same reason [`ParallelTimeSeriesDataWriter`] is distinct from [`TimeSeriesDataWriter`]: it
/// keeps a caller from reaching [`write_mesh`](TimeSeriesWriter::write_mesh) on a communicator's
/// worth of ranks, or [`write_mesh`](Self::write_mesh) on a single one, by construction rather
/// than by a runtime check.
pub struct ParallelTimeSeriesWriter {
    inner: TimeSeriesWriter,
    comm: mpi::topology::SimpleCommunicator,
}

impl fmt::Debug for ParallelTimeSeriesWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParallelTimeSeriesWriter")
            .field("inner", &self.inner)
            .field("rank", &self.comm.rank())
            .finish()
    }
}

impl ParallelTimeSeriesWriter {
    /// The parallel counterpart of [`TimeSeriesWriter::new`]: every rank must call this together,
    /// with the same `file_name`/`data_storage`, since the underlying HDF5 file is opened
    /// collectively over `comm`.
    ///
    /// Only [`DataStorage::Hdf5SingleFile`] is supported so far -- ascii/binary parallel output is
    /// still an open design question, see `07_mpi.md` on the plans share.
    pub fn new(
        file_name: impl AsRef<Path>,
        data_storage: DataStorage,
        comm: mpi::topology::SimpleCommunicator,
    ) -> Result<Self> {
        let xdmf_file_name = file_name.as_ref().to_path_buf().with_extension("xdmf2");

        validate_file_name(&xdmf_file_name)?;

        let DataStorage::Hdf5SingleFile { deflate_level } = data_storage else {
            return Err(Error::InvalidConfiguration {
                reason: "ParallelTimeSeriesWriter::new only supports the Hdf5SingleFile \
                         DataStorage so far"
                    .to_string(),
            });
        };
        // deflate_level is otherwise unused here: a collectively-written dataset is left
        // uncompressed for now, see `write_values_parallel` in `writer/hdf5.rs`
        crate::validate_deflate_level(deflate_level)?;

        if let Some(parent) = xdmf_file_name.parent() {
            mpi_safe_create_dir_all(parent)?;
        }

        let writer = hdf5::SingleFileHdf5Writer::new_parallel(
            file_name.as_ref(),
            deflate_level.unwrap_or(hdf5::DEFAULT_DEFLATE_LEVEL),
            &comm,
        )?;

        Ok(Self {
            inner: TimeSeriesWriter {
                xdmf_file_name,
                writer: Box::new(writer),
            },
            comm,
        })
    }

    /// The XDMF file this writer writes, same as [`TimeSeriesWriter::file_name`].
    pub fn file_name(&self) -> &Path {
        self.inner.file_name()
    }

    /// The parallel counterpart of [`TimeSeriesWriter::write_mesh`]: every rank passes its own
    /// share of the mesh, and every rank must call this together -- with the same `comm` given to
    /// [`new`](Self::new).
    ///
    /// `owned_points` is this rank's own points only -- no ghost points -- and
    /// `owned_global_ids` gives each one a dense id in `0..N`. A ghost point is never passed here
    /// at all: it is written by whichever rank owns it, and referenced from
    /// `local_connectivity` by that global id. Every rank's own ids must be one ascending,
    /// contiguous block (e.g. built via an exclusive prefix sum of each rank's owned-point count,
    /// then assigned sequentially) -- see "Global node numbering and ownership" in `07_mpi.md`.
    ///
    /// `local_connectivity`/`local_cell_types` are this rank's own cells. Since arotau (the
    /// intended first caller) never has ghost cells, a rank's local cells are unconditionally its
    /// owned cells, concatenated across ranks in rank order -- there is no equivalent of
    /// `owned_global_ids` for cells. Every rank must pass at least one cell, and the same single
    /// [`CellType`]: a `Mixed` topology isn't supported here yet.
    pub fn write_mesh<C: Coordinate, I: ConnectivityIndex>(
        mut self,
        owned_points: &[C],
        owned_global_ids: &[u64],
        local_connectivity: &[I],
        local_cell_types: &[CellType],
    ) -> Result<ParallelTimeSeriesDataWriter> {
        if !self.inner.writer.supports_parallel() {
            return Err(Error::InvalidConfiguration {
                reason: "this DataStorage does not support parallel writing".to_string(),
            });
        }

        // Local, non-collective checks: these validate one rank's own arguments in isolation
        // (shape, not distributed-data correctness), so every rank is expected to either pass or
        // fail them the same way. Unlike the checks after the collectives below, an early return
        // here on a rank whose own arguments are malformed, while another rank's happen to be
        // shaped correctly, is not guarded against -- that rank would return here while others
        // went on to the collectives, which hangs rather than errors cleanly. A caller is
        // expected to build its own/every rank's arguments the same way (e.g. from the same
        // local mesh-partitioning code), which is what makes that mismatch not a realistic
        // failure mode in practice, unlike a legitimate per-rank difference in *data* (a missing
        // owned point, disagreeing cell types), which the collectives below do guard against.
        if !owned_points.len().is_multiple_of(3) {
            return Err(Error::InvalidMesh {
                reason: format!(
                    "points must have 3 dimensions, but {} is not a multiple of 3",
                    owned_points.len()
                ),
            });
        }
        let local_num_points = owned_points.len() / 3;

        if owned_global_ids.len() != local_num_points {
            return Err(Error::InvalidMesh {
                reason: format!(
                    "owned_global_ids has {} entries, but owned_points has {local_num_points} \
                     points",
                    owned_global_ids.len()
                ),
            });
        }

        let exp_conn_len: usize = local_cell_types.iter().map(CellType::num_points).sum();
        if exp_conn_len != local_connectivity.len() {
            return Err(Error::InvalidMesh {
                reason: format!(
                    "size of connectivity ({}) does not match the number expected from the cell \
                     types ({exp_conn_len})",
                    local_connectivity.len()
                ),
            });
        }

        paraview::validate(
            &I::as_values(local_connectivity),
            self.inner.writer.format(),
        )?;

        // collective from here on: every rank must reach every one of the following three calls,
        // in this order, even a rank whose own arguments would fail one of their local checks --
        // see the comments in `owned_ids_sanity_check`/`agree_on_cell_type`
        let (point_offset, global_num_points) =
            owned_ids_sanity_check(&self.comm, owned_global_ids)?;
        let cell_type = agree_on_cell_type(&self.comm, local_cell_types)?;
        let (cell_offset, global_num_cells) =
            exclusive_scan_offset(&self.comm, local_cell_types.len());

        validate_connectivity_indices(&self.comm, local_connectivity, global_num_points)?;

        let points = C::as_values(owned_points);
        let points_item =
            points_data_item_parallel(&mut self.inner, &points, point_offset, global_num_points)?;

        let num_points_per_cell = cell_type.num_points();
        let connectivity_item = connectivity_data_item_parallel(
            &mut self.inner,
            local_connectivity,
            cell_offset * num_points_per_cell,
            global_num_cells * num_points_per_cell,
        )?;

        let topology = Topology {
            topology_type: TopologyType::from(cell_type),
            nodes_per_element: super::submesh::poly_cell_points(cell_type),
            number_of_elements: global_num_cells.to_string(),
            data_item: DataItem::new_reference(&connectivity_item, DOMAIN_DATA_ITEMS),
        };

        let grid = Grid::new_uniform("mesh", geometry(&points_item), topology);

        let layout = ParallelLayout {
            point_offset,
            local_num_points,
            cell_offset,
            local_num_cells: local_cell_types.len(),
        };

        let inner = self.inner.finish_mesh(
            grid,
            vec![points_item, connectivity_item],
            Vec::new(),
            global_num_points,
            global_num_cells,
            self.comm.rank() == 0,
        )?;

        Ok(ParallelTimeSeriesDataWriter {
            inner,
            comm: self.comm,
            layout,
        })
    }
}

/// This rank's own share of a mesh written by [`ParallelTimeSeriesWriter::write_mesh`], cached on
/// [`ParallelTimeSeriesDataWriter`] for
/// [`ParallelTimeSeriesDataWriter::write_time_step`](ParallelTimeSeriesDataWriter::write_time_step)
/// to place each step's attributes with -- sizing is fixed at mesh-write time, so no further
/// collective is needed per step.
#[derive(Clone, Copy, Debug)]
struct ParallelLayout {
    point_offset: usize,
    local_num_points: usize,
    cell_offset: usize,
    local_num_cells: usize,
}

/// Writer for time series data in XDMF format, obtained by writing a mesh with
/// [`ParallelTimeSeriesWriter::write_mesh`].
///
/// A distinct type from [`TimeSeriesDataWriter`], rather than the same type used for both, so
/// that [`TimeSeriesDataWriter::write_time_step`] and [`Self::write_time_step`] are two different
/// methods a caller cannot mix up: writing a parallel mesh's step data through the former (or a
/// non-parallel mesh's through the latter) is a compile error instead of a call into the wrong
/// `DataWriter` methods -- on this backend that would try independent, whole-buffer HDF5 writes
/// against a file opened collectively under MPI-IO, unsafe under the MPI-IO driver rather than
/// merely wrong.
pub struct ParallelTimeSeriesDataWriter {
    inner: TimeSeriesDataWriter,
    comm: mpi::topology::SimpleCommunicator,
    layout: ParallelLayout,
}

impl fmt::Debug for ParallelTimeSeriesDataWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParallelTimeSeriesDataWriter")
            .field("inner", &self.inner)
            .field("rank", &self.comm.rank())
            .finish_non_exhaustive()
    }
}

impl ParallelTimeSeriesDataWriter {
    /// The XDMF file this writer writes, same as
    /// [`TimeSeriesDataWriter::file_name`].
    pub fn file_name(&self) -> &Path {
        self.inner.file_name()
    }

    /// The parallel counterpart of
    /// [`TimeSeriesDataWriter::write_time_step`]: every rank must call this together, with the
    /// same `time`, and with a closure that writes the same
    /// [`point_data`](ParallelTimeStep::point_data)/[`cell_data`](ParallelTimeStep::cell_data)
    /// names in the same order on every rank -- the underlying HDF5 dataset creation is
    /// collective. Unlike [`TimeSeriesDataWriter::write_time_step`], each call takes only this
    /// rank's own share (the same subset/order as the `owned_points`/`local_connectivity` this
    /// writer was built from).
    pub fn write_time_step<F, E>(&mut self, time: impl Into<String>, write_step: F) -> Result<(), E>
    where
        F: FnOnce(&mut ParallelTimeStep<'_>) -> Result<(), E>,
        E: From<Error>,
    {
        let time = time.into();
        let Ok(parsed_time) = time.parse::<f64>() else {
            return Err(Error::InvalidTimeStep {
                time,
                reason: "must be a valid float".to_string(),
            }
            .into());
        };

        if !parsed_time.is_finite() {
            return Err(Error::InvalidTimeStep {
                time,
                reason: "must be a finite float".to_string(),
            }
            .into());
        }

        let time_bits = if parsed_time == 0.0 { 0.0 } else { parsed_time }.to_bits();

        if let Some(existing) = self.inner.written_times.get(&time_bits) {
            let reason = if existing == &time {
                "already written".to_string()
            } else {
                format!("already written (as '{existing}')")
            };
            return Err(Error::InvalidTimeStep { time, reason }.into());
        }

        let mut step = ParallelTimeStep {
            writer: self,
            time,
            time_bits,
            attributes: Vec::new(),
            point_names: HashSet::new(),
            cell_names: HashSet::new(),
            initialized: false,
            next_array_index: 0,
        };

        match write_step(&mut step) {
            Ok(()) => step.finish().map_err(E::from),
            Err(error) => {
                let _discard_result = step.discard();
                Err(error)
            }
        }
    }
}

/// A single time step being written in parallel, handed to the closure passed to
/// [`ParallelTimeSeriesDataWriter::write_time_step`].
///
/// Each [`point_data`](Self::point_data)/[`cell_data`](Self::cell_data) call writes this rank's
/// own share immediately, so one buffer can serve every field of the step. Every rank must call
/// the same names, in the same order, since the underlying HDF5 dataset creation is collective.
pub struct ParallelTimeStep<'a> {
    writer: &'a mut ParallelTimeSeriesDataWriter,
    time: String,
    time_bits: u64,
    attributes: Vec<attribute::Attribute>,
    point_names: HashSet<String>,
    cell_names: HashSet<String>,
    initialized: bool,
    next_array_index: usize,
}

impl fmt::Debug for ParallelTimeStep<'_> {
    /// Names only, since the attributes themselves carry the step's data. Sorted, as a
    /// `HashSet`'s iteration order varies between runs.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParallelTimeStep")
            .field("time", &self.time)
            .field("point_data", &sorted_names(&self.point_names))
            .field("cell_data", &sorted_names(&self.cell_names))
            .finish_non_exhaustive()
    }
}

impl ParallelTimeStep<'_> {
    /// Write this rank's own share of one point attribute, immediately. Must be the same subset
    /// and order as the `owned_points` the mesh was written with.
    pub fn point_data<'v>(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        data: impl Into<Values<'v>>,
    ) -> Result<()> {
        self.write_attribute(name, attribute, data.into(), attribute::Center::Node)
    }

    /// Write this rank's own share of one cell attribute, immediately. Must be the same order as
    /// the `local_connectivity` the mesh was written with.
    pub fn cell_data<'v>(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        data: impl Into<Values<'v>>,
    ) -> Result<()> {
        self.write_attribute(name, attribute, data.into(), attribute::Center::Cell)
    }

    fn write_attribute(
        &mut self,
        name: &str,
        data_attribute: DataAttribute,
        values: Values<'_>,
        center: attribute::Center,
    ) -> Result<()> {
        let layout = self.writer.layout;

        let is_point_data = center == attribute::Center::Node;
        let (label, local_entities, offset, global_entities) = if is_point_data {
            (
                POINT_DATA,
                layout.local_num_points,
                layout.point_offset,
                self.writer.inner.num_points,
            )
        } else {
            (
                CELL_DATA,
                layout.local_num_cells,
                layout.cell_offset,
                self.writer.inner.num_cells,
            )
        };

        if !is_valid_data_name(name) {
            return Err(Error::InvalidData {
                reason: format!(
                    "data name '{name}' of {label} is not valid, must contain a \
                     non-whitespace character and must not contain control characters"
                ),
            });
        }

        let seen_names = if is_point_data {
            &self.point_names
        } else {
            &self.cell_names
        };
        if seen_names.contains(name) {
            return Err(Error::InvalidData {
                reason: format!("name '{name}' of {label} is used more than once"),
            });
        }

        let stride = data_attribute
            .size()
            .filter(|size| *size != 0)
            .ok_or_else(|| Error::InvalidData {
                reason: format!(
                    "attribute type {data_attribute:?} of {label} '{name}' has no usable size: \
                     its number of components must be non-zero and must itself fit a usize"
                ),
            })?;

        let exp_size = local_entities
            .checked_mul(stride)
            .ok_or_else(|| Error::InvalidData {
                reason: format!(
                    "attribute type {data_attribute:?} of {label} '{name}' describes \
                     {local_entities} entities of {stride} components each, whose total does \
                     not fit a usize"
                ),
            })?;
        if values.len() != exp_size {
            return Err(Error::InvalidData {
                reason: format!(
                    "size of {label} '{name}' on this rank must be {exp_size} (its own share, \
                     {local_entities} entities), but is {}",
                    values.len()
                ),
            });
        }

        let global_len = global_entities
            .checked_mul(stride)
            .ok_or_else(|| Error::InvalidData {
                reason: format!(
                    "attribute type {data_attribute:?} of {label} '{name}' describes \
                     {global_entities} entities of {stride} components each, whose total does \
                     not fit a usize"
                ),
            })?;

        // reject values ParaView would read back as different numbers before anything is
        // written, so a caller mistake leaves no partial output behind
        paraview::validate(&values, self.writer.inner.writer.format())?;

        if !self.initialized {
            self.writer.inner.writer.write_data_initialize(&self.time)?;
            self.initialized = true;
        }

        let index = self.next_array_index;
        self.next_array_index += 1;

        let format = self.writer.inner.writer.format();
        let data = self.writer.inner.writer.write_data_parallel(
            index,
            &values,
            offset * stride,
            global_len,
        )?;

        self.attributes.push(attribute::Attribute {
            name: name.to_string(),
            attribute_type: data_attribute.into(),
            center,
            data_items: vec![DataItem {
                name: None,
                item_type: None,
                dimensions: Some(crate::values::dimensions_of_len(global_len, data_attribute)),
                number_type: Some(values.number_type()),
                format: Some(format),
                precision: Some(values.precision()),
                endian: format.endian(),
                data,
                reference: None,
            }],
        });

        // recorded only once the attribute is actually written, so a rejected call can be
        // retried under the same name
        if is_point_data {
            self.point_names.insert(name.to_string());
        } else {
            self.cell_names.insert(name.to_string());
        }

        Ok(())
    }

    /// Complete the time step, adding its `<Grid>` to the XDMF file. Only rank 0 writes the file
    /// to disk -- every rank builds the same in-memory `<Grid>`, but every rank racing to write
    /// the same file would corrupt it.
    fn finish(self) -> Result<()> {
        if self.attributes.is_empty() {
            let time = self.time.clone();
            // an attribute can fail after initializing the backend, and a closure that ignores
            // that error still arrives here -- discarded rather than dropped, or the backend
            // would stay initialized and every later step would fail
            let _discard_result = self.discard();
            return Err(Error::InvalidTimeStep {
                time,
                reason: format!("no data written, needs at least one {POINT_DATA} or {CELL_DATA}"),
            });
        }

        if let Err(error) = self.writer.inner.writer.write_data_finalize() {
            let _discard_result = self.discard();
            return Err(error);
        }

        let ParallelTimeStep {
            writer,
            time,
            time_bits,
            attributes,
            ..
        } = self;

        writer.inner.push_step(&time, attributes, Vec::new())?;

        writer.inner.step_times.push(time.clone());
        writer.inner.written_times.insert(time_bits, time);

        // flushed unconditionally: this is collective under MPI-IO (every rank must call it),
        // while only rank 0 goes on to actually write the light-data file itself
        writer.inner.writer.flush()?;
        if writer.comm.rank() == 0 {
            writer.inner.write_light_data_file()?;
        }

        Ok(())
    }

    /// Abandon the time step, removing the heavy data already written for it.
    fn discard(self) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }

        self.writer.inner.writer.write_data_discard()
    }
}

#[cfg(test)]
mod tests {
    use mpi_test::mpi_test;

    use super::*;

    #[mpi_test(np = [1, 3])]
    fn exclusive_scan_offset_concatenates_in_rank_order() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = usize::try_from(world.rank()).unwrap();

        // rank `r` contributes `r + 1` cells: 1, 2, 3, ...
        let local_len = rank + 1;
        let (offset, total) = exclusive_scan_offset(&world, local_len);

        let expected_offset: usize = (0..rank).map(|r| r + 1).sum();
        let expected_total: usize = (0..world.size() as usize).map(|r| r + 1).sum();

        assert_eq!(offset, expected_offset);
        assert_eq!(total, expected_total);
    }

    #[mpi_test(np = [1, 4])]
    fn owned_ids_sanity_check_accepts_a_dense_rank_ordered_partition() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = usize::try_from(world.rank()).unwrap();
        let size = world.size() as usize;

        // rank `r` owns 3 ids, at `3*r .. 3*r + 3`
        let start = 3 * rank;
        let owned: Vec<u64> = (start..start + 3).map(|id| id as u64).collect();

        let (offset, total) = owned_ids_sanity_check(&world, &owned).unwrap();

        assert_eq!(offset, start);
        assert_eq!(total, 3 * size);
    }

    #[mpi_test(np = [2, 4])]
    fn owned_ids_sanity_check_rejects_a_point_owned_by_nobody() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = usize::try_from(world.rank()).unwrap();
        let size = world.size() as usize;

        // every rank owns 3 ids, except rank 0, which only claims the first 2 of its own block --
        // so id 2 is owned by nobody. Trimmed off the *end* of rank 0's own block, rather than out
        // of its middle, so every rank's own ids stay locally contiguous and it's the aggregate
        // check (declared total one short of `max_id + 1`), not the per-rank contiguity check,
        // that has to catch this.
        let start = 3 * rank;
        let len = if rank == 0 { 2 } else { 3 };
        let owned: Vec<u64> = (start..start + len).map(|id| id as u64).collect();

        let result = owned_ids_sanity_check(&world, &owned);
        let Err(Error::InvalidMesh { reason }) = result else {
            panic!("expected InvalidMesh, got {result:?}, size={size}");
        };
        assert!(reason.contains("owned by zero ranks"));
    }

    #[mpi_test(np = [2, 3])]
    fn owned_ids_sanity_check_rejects_a_non_contiguous_block() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = usize::try_from(world.rank()).unwrap();

        // rank 0 owns a gap-free block except for one hole; every other rank owns nothing. The
        // contiguity check is local to rank 0, but the failure it feeds into `all_contiguous` is
        // collective, so every rank -- not just rank 0 -- sees the same rejection.
        let owned: Vec<u64> = if rank == 0 { vec![0, 1, 3] } else { vec![] };

        let result = owned_ids_sanity_check(&world, &owned);
        let Err(Error::InvalidMesh { reason }) = result else {
            panic!("expected InvalidMesh, got {result:?}");
        };
        assert!(reason.contains("contiguous"));
    }

    #[mpi_test(np = [1, 3])]
    fn agree_on_cell_type_accepts_the_same_type_everywhere() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let cell_type = agree_on_cell_type(&world, &[CellType::Triangle, CellType::Triangle]);
        assert_eq!(cell_type.unwrap(), CellType::Triangle);
    }

    #[mpi_test(np = [2, 3])]
    fn agree_on_cell_type_rejects_ranks_disagreeing() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank();

        let local_cell_type = if rank == 0 {
            CellType::Triangle
        } else {
            CellType::Quadrilateral
        };

        let result = agree_on_cell_type(&world, &[local_cell_type]);
        let Err(Error::InvalidMesh { reason }) = result else {
            panic!("expected InvalidMesh, got {result:?}");
        };
        assert!(reason.contains("every rank must write the same CellType"));
    }
}

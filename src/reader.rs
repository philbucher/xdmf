//! Reading XDMF time series written with the `Hdf5SingleFile`/`Hdf5MultipleFiles` storages, i.e.
//! every `DataItem` with `Format="HDF"`. `Format="XML"`/`"Binary"` are not supported yet and
//! reading one is [`Error::Unsupported`].
//!
//! [`TimeSeriesReader::new`] parses the whole document up front -- mesh/time-step metadata and,
//! for a mesh with submeshes, which points and cells each one holds -- so every read call after it
//! is a plain, independent, repeatable query against an already-parsed document: read the points,
//! read the topology, read one field of one step. There is deliberately no phase a caller has to
//! pass through first (unlike the writer, where writing the mesh is a one-time, irreversible
//! mutation of the file): reading has no such constraint, so nothing here enforces one.
//!
//! A mesh written with [`crate::TimeSeriesWriter::write_mesh_with_submeshes`] has no points or
//! connectivity of its own -- each submesh holds only the points its own cells use, renumbered
//! against them, and submeshes may overlap. [`TimeSeriesReader::read_points`]/
//! [`TimeSeriesReader::read_topology`] put the original mesh back together from the mesh's own
//! coordinates (written once, per direction) plus each submesh's `submesh_cells`/`<Geometry>`
//! selector. This is also why only the HDF5 storages can be read: the ascii and binary storages
//! write only a compacted copy of each submesh's points, so a point no cell uses is not in the
//! file at all and cannot be reconstructed.

mod hdf5_reader;
mod light_data;
mod selection;
mod topology;

use std::path::Path;

use light_data::{Analysis, Document};
use selection::Membership;

use crate::{
    CellType, DataAttribute, DataStorage, Error, Result, SUBMESH_CELLS, Values,
    xdmf_elements::{
        Domain,
        attribute::{self, AttributeType},
        data_item::{DataItem, NumberType},
        grid::Grid,
    },
};

/// Metadata about one field the reader found at a time step, sized so the caller can allocate a
/// buffer before reading the data.
#[derive(Clone, Debug)]
pub struct DataInfo {
    /// Name of the field
    pub name: String,
    /// The field's shape. For a field whose `AttributeType` collapsed several `DataAttribute`
    /// shapes onto `Matrix` when it was written (`Tensor6`, `Matrix(n, m)`, `Generic`, see
    /// `values.rs`), this is always [`DataAttribute::Generic`] with the real component count --
    /// the file does not state which of the three it originally was.
    pub attribute: DataAttribute,
    /// The element type the file holds this field as.
    pub number_type: NumberType,
    /// The element width, in bytes, the file holds this field at.
    pub precision: u8,
    /// Total element count: `num_entities * attribute.size()`.
    pub len: usize,
}

/// A type a [`TimeSeriesReader`] read call can fill a buffer with: `f32`, `f64`, `i32`, `i64`,
/// `u32` or `u64`, mirroring the [`Values`] variants.
///
/// Widening is allowed and is not an error: `f32` file data read as `Vec<f64>` succeeds, as does
/// `u32` read as `Vec<u64>`. Narrowing (`f64` file data read as `Vec<f32>`) is
/// [`Error::NumberTypeMismatch`] instead of silently losing precision.
pub trait ValueType: sealed::SealedValueType {}

impl ValueType for f64 {}
impl ValueType for f32 {}
impl ValueType for i64 {}
impl ValueType for i32 {}
impl ValueType for u64 {}
impl ValueType for u32 {}

pub(crate) mod sealed {
    use super::{Error, Result, Values, value_type_name};

    pub trait SealedValueType: Sized {
        fn from_values(values: Values<'static>) -> Result<Vec<Self>>;
    }

    fn mismatch(requested: &str, found: &Values<'_>) -> Error {
        Error::NumberTypeMismatch {
            reason: format!(
                "requested {requested}, but the file holds {}",
                value_type_name(found)
            ),
        }
    }

    impl SealedValueType for f64 {
        fn from_values(values: Values<'static>) -> Result<Vec<Self>> {
            match values {
                Values::F64(v) => Ok(v.into_owned()),
                Values::F32(v) => Ok(v.iter().map(|&x| Self::from(x)).collect()),
                other => Err(mismatch("f64", &other)),
            }
        }
    }

    impl SealedValueType for f32 {
        fn from_values(values: Values<'static>) -> Result<Vec<Self>> {
            match values {
                Values::F32(v) => Ok(v.into_owned()),
                other => Err(mismatch("f32", &other)),
            }
        }
    }

    impl SealedValueType for i64 {
        fn from_values(values: Values<'static>) -> Result<Vec<Self>> {
            match values {
                Values::I64(v) => Ok(v.into_owned()),
                Values::I32(v) => Ok(v.iter().map(|&x| Self::from(x)).collect()),
                other => Err(mismatch("i64", &other)),
            }
        }
    }

    impl SealedValueType for i32 {
        fn from_values(values: Values<'static>) -> Result<Vec<Self>> {
            match values {
                Values::I32(v) => Ok(v.into_owned()),
                other => Err(mismatch("i32", &other)),
            }
        }
    }

    impl SealedValueType for u64 {
        fn from_values(values: Values<'static>) -> Result<Vec<Self>> {
            match values {
                Values::U64(v) => Ok(v.into_owned()),
                Values::U32(v) => Ok(v.iter().map(|&x| Self::from(x)).collect()),
                other => Err(mismatch("u64", &other)),
            }
        }
    }

    impl SealedValueType for u32 {
        fn from_values(values: Values<'static>) -> Result<Vec<Self>> {
            match values {
                Values::U32(v) => Ok(v.into_owned()),
                other => Err(mismatch("u32", &other)),
            }
        }
    }
}

fn value_type_name(values: &Values<'_>) -> &'static str {
    match values {
        Values::F64(_) => "f64",
        Values::F32(_) => "f32",
        Values::I64(_) => "i64",
        Values::I32(_) => "i32",
        Values::U64(_) => "u64",
        Values::U32(_) => "u32",
    }
}

/// Reads an XDMF time series: the parsed light data, the mesh's metadata (including, for a mesh
/// with submeshes, which points and cells each one holds), and every read call against it.
pub struct TimeSeriesReader {
    document: Document,
    num_points: usize,
    num_cells: usize,
    times: Vec<String>,
    submesh_names: Vec<String>,
    /// One membership per submesh, empty without submeshes: which mesh points/cells each submesh holds.
    points_membership: Vec<Membership>,
    cells_membership: Vec<Membership>,
}

impl std::fmt::Debug for TimeSeriesReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimeSeriesReader")
            .field("num_points", &self.num_points)
            .field("num_cells", &self.num_cells)
            .field("times", &self.times)
            .field("submesh_names", &self.submesh_names)
            .finish_non_exhaustive()
    }
}

impl TimeSeriesReader {
    /// Parse the light data (XML) of an XDMF file and report the mesh's metadata, without
    /// touching any heavy data -- except for a mesh with submeshes, where each submesh's point
    /// and cell membership is being parsed eagerly
    pub fn new(file_name: impl AsRef<Path>) -> Result<Self> {
        let document = Document::open(file_name.as_ref())?;
        let domain = document.domain()?;
        let analysis = Analysis::build(domain)?;

        let submesh_names = analysis.submesh_names.clone();
        let times = analysis.times()?;

        let (num_points, num_cells, points_membership, cells_membership) =
            if submesh_names.is_empty() {
                let (num_points, num_cells) = mesh_size_plain(analysis.submeshes[0][0], domain)?;
                (num_points, num_cells, Vec::new(), Vec::new())
            } else {
                let num_points = mesh_num_points_with_submeshes(analysis.submeshes[0][0], domain)?;
                let points_membership =
                    submesh_points_membership(&analysis, domain, &document.base_dir)?;
                let cells_membership = parse_submesh_cells(&document, domain)?;

                if cells_membership.len() != submesh_names.len() {
                    return Err(Error::InvalidDocument {
                        reason: format!(
                            "'{SUBMESH_CELLS}' lists {} submeshes, but the document has {}",
                            cells_membership.len(),
                            submesh_names.len()
                        ),
                    });
                }

                let num_cells = mesh_num_cells_from_membership(&cells_membership);
                (num_points, num_cells, points_membership, cells_membership)
            };

        Ok(Self {
            document,
            num_points,
            num_cells,
            times,
            submesh_names,
            points_membership,
            cells_membership,
        })
    }

    /// Total number of points in the mesh.
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    /// Total number of cells in the mesh
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    /// Number of time steps written.
    pub fn num_steps(&self) -> usize {
        self.times.len()
    }

    /// The time of every step written so far, in write order. Empty if no step has been written.
    pub fn times(&self) -> &[String] {
        &self.times
    }

    /// The name of every submesh, in the order they were written. Empty when the mesh has no submeshes.
    pub fn submesh_names(&self) -> &[String] {
        &self.submesh_names
    }

    /// The global cell indices the submesh at `submesh` (an index into [`Self::submesh_names`])
    /// holds, in the order the submesh was written with -- what a caller writing this same split
    /// back with
    /// [`TimeSeriesWriter::write_mesh_with_submeshes`](crate::TimeSeriesWriter::write_mesh_with_submeshes)
    /// would pass as that submesh's own index list. Indices are into the `cell_types`/
    /// `connectivity` buffers [`Self::read_topology`] fills.
    ///
    /// Empty (and `submesh` always out of range) for a mesh with no submeshes.
    pub fn submesh_cells(&self, submesh: usize) -> Result<Vec<usize>> {
        self.cells_membership
            .get(submesh)
            .map(|membership| membership.iter().collect())
            .ok_or_else(|| self.submesh_out_of_range(submesh))
    }

    /// The global point indices the submesh at `submesh` holds, ascending -- the points its own
    /// cells use. Indices are into the `points` buffer [`Self::read_points`] fills. See
    /// [`Self::submesh_cells`] for the index and range rules.
    pub fn submesh_points(&self, submesh: usize) -> Result<Vec<usize>> {
        self.points_membership
            .get(submesh)
            .map(|membership| membership.iter().collect())
            .ok_or_else(|| self.submesh_out_of_range(submesh))
    }

    fn submesh_out_of_range(&self, submesh: usize) -> Error {
        Error::InvalidDocument {
            reason: format!(
                "submesh index {submesh} is out of range, the mesh has {} submeshes",
                self.submesh_names.len()
            ),
        }
    }

    /// Read the mesh's points. The buffer is cleared first, so its existing capacity is reused.
    pub fn read_points(&self, points: &mut Vec<f64>) -> Result<()> {
        points.clear();

        let domain = self.document.domain()?;
        let analysis = Analysis::build(domain)?;
        let first_grid = analysis.submeshes[0][0];

        if self.submesh_names.is_empty() {
            read_points_plain(&self.document, domain, first_grid, points)
        } else {
            read_points_with_submeshes(&self.document, domain, first_grid, points)
        }
    }

    /// Read the mesh's connectivity and cell types. The buffers are cleared first, so their
    /// existing capacity is reused.
    pub fn read_topology(
        &self,
        connectivity: &mut Vec<u64>,
        cell_types: &mut Vec<CellType>,
    ) -> Result<()> {
        connectivity.clear();
        cell_types.clear();

        let domain = self.document.domain()?;
        let analysis = Analysis::build(domain)?;

        if self.submesh_names.is_empty() {
            read_topology_plain(
                &self.document,
                domain,
                analysis.submeshes[0][0],
                connectivity,
                cell_types,
            )
        } else {
            read_topology_with_submeshes(
                &self.document,
                &analysis,
                &self.points_membership,
                &self.cells_membership,
                connectivity,
                cell_types,
            )
        }
    }

    /// What point data is present at `step`, so a caller can size a buffer and pick a type before
    /// calling [`Self::read_point_data`].
    pub fn point_data_info(&self, step: usize) -> Result<Vec<DataInfo>> {
        self.data_info(step, attribute::Center::Node, self.num_points)
    }

    /// What cell data is present at `step`, so a caller can size a buffer and pick a type before
    /// calling [`Self::read_cell_data`].
    pub fn cell_data_info(&self, step: usize) -> Result<Vec<DataInfo>> {
        self.data_info(step, attribute::Center::Cell, self.num_cells)
    }

    fn data_info(
        &self,
        step: usize,
        center: attribute::Center,
        num_entities: usize,
    ) -> Result<Vec<DataInfo>> {
        let grid = self.step_grid(step, 0)?;
        let attributes = grid.attributes.as_deref().unwrap_or_default();

        attributes
            .iter()
            .filter(|attribute| attribute.center == center)
            .map(|attribute| build_data_info(attribute, num_entities))
            .collect()
    }

    /// Read one named point attribute of one step into `into` (cleared first, capacity reused).
    ///
    /// `T` may widen the file's element type (see [`ValueType`]) but not narrow it. A caller can
    /// avoid this by checking the field's actual type first.
    pub fn read_point_data<T: ValueType>(
        &self,
        step: usize,
        name: &str,
        into: &mut Vec<T>,
    ) -> Result<()> {
        self.read_data(
            step,
            name,
            attribute::Center::Node,
            self.num_points,
            &self.points_membership,
            into,
        )
    }

    /// Read one named cell attribute of one step into `into` (cleared first, capacity reused).
    /// See [`Self::read_point_data`] for the widening rules.
    pub fn read_cell_data<T: ValueType>(
        &self,
        step: usize,
        name: &str,
        into: &mut Vec<T>,
    ) -> Result<()> {
        self.read_data(
            step,
            name,
            attribute::Center::Cell,
            self.num_cells,
            &self.cells_membership,
            into,
        )
    }

    fn read_data<T: ValueType>(
        &self,
        step: usize,
        name: &str,
        center: attribute::Center,
        num_entities: usize,
        membership: &[Membership],
        into: &mut Vec<T>,
    ) -> Result<()> {
        let domain = self.document.domain()?;
        let base_dir = &self.document.base_dir;

        let values = if self.submesh_names.is_empty() {
            let grid = self.step_grid(step, 0)?;
            let attribute = find_attribute(grid, name, center)?;
            let item = attribute_item(attribute)?;
            selection::read_data_item(item, base_dir, domain)?
        } else {
            self.read_submesh_field(step, name, center, num_entities, membership)?
        };

        let converted = T::from_values(values)?;
        into.clear();
        into.extend(converted);

        Ok(())
    }

    /// One field's values, over the whole mesh, for a mesh with submeshes.
    ///
    /// A field with at least one ascending submesh is written once, whole, and every ascending
    /// submesh's `<Attribute>` selects its own share out of it (`write_data_selected`,
    /// `time_series_writer.rs`) -- so if any submesh's `DataItem` for this field is
    /// selection-shaped, its nested source *is* the whole field and is read directly, exactly as
    /// [`Self::read_points`] takes the mesh's own coordinates from the source rather than from the
    /// submeshes. This is also what correctly recovers an entity that belongs to no submesh at all
    /// (an unused point), since the global array covers every entity and the per-submesh scatter
    /// below cannot.
    ///
    /// Only when *no* submesh holds a selection for this field (every submesh was given a private
    /// copy of its own share, which happens when none of them are ascending) is there no global
    /// array to read, and the field is reassembled by scattering every submesh's own copy back
    /// through its membership -- the field-data counterpart of the connectivity scatter in
    /// [`Self::read_topology`]. An entity in no submesh is then genuinely not in the file.
    fn read_submesh_field(
        &self,
        step: usize,
        name: &str,
        center: attribute::Center,
        num_entities: usize,
        membership: &[Membership],
    ) -> Result<Values<'static>> {
        let domain = self.document.domain()?;
        let base_dir = &self.document.base_dir;
        let mut submesh_values = Vec::with_capacity(self.submesh_names.len());

        for submesh in 0..self.submesh_names.len() {
            let grid = self.step_grid(step, submesh)?;
            let attribute = find_attribute(grid, name, center)?;
            let item = attribute_item(attribute)?;

            if item.item_type.is_some() {
                let (_selector, source) = selection::selection_parts(item)?;
                return selection::read_data_item(source, base_dir, domain);
            }

            submesh_values.push(selection::read_data_item(item, base_dir, domain)?);
        }

        scatter_field(num_entities, &submesh_values, membership)
    }

    /// The grid for `submesh`'s (0 for a mesh with no submeshes) `step`-th time step.
    ///
    /// `step` is bounded by the steps [`Self::num_steps`] reports, not by how many grids the
    /// submesh has: a mesh whose steps were never written still has one grid per submesh, but
    /// that grid is the mesh itself and carries no `Time` (see `Analysis`), so counting it would
    /// hand out a step 0 the document does not have.
    fn step_grid(&self, step: usize, submesh: usize) -> Result<&Grid> {
        if step >= self.times.len() {
            return Err(Error::InvalidDocument {
                reason: format!(
                    "step index {step} is out of range, {} steps were written",
                    self.times.len()
                ),
            });
        }

        let domain = self.document.domain()?;
        let analysis = Analysis::build(domain)?;

        let grids = analysis.submeshes.get(submesh).ok_or(Error::Internal(
            "step_grid called with a submesh index out of range",
        ))?;

        grids
            .get(step)
            .copied()
            .ok_or_else(|| Error::InvalidDocument {
                reason: format!(
                    "the document has {} steps, but submesh {submesh} has only {} of them",
                    self.times.len(),
                    grids.len()
                ),
            })
    }
}

fn attribute_item(attribute: &attribute::Attribute) -> Result<&DataItem> {
    attribute
        .data_items
        .first()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Attribute '{}' has no DataItem", attribute.name),
        })
}

fn find_attribute<'a>(
    grid: &'a Grid,
    name: &str,
    center: attribute::Center,
) -> Result<&'a attribute::Attribute> {
    grid.attributes
        .as_deref()
        .unwrap_or_default()
        .iter()
        .find(|attribute| attribute.name == name && attribute.center == center)
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("no {center:?}-centered attribute named '{name}' at this step"),
        })
}

fn build_data_info(attribute: &attribute::Attribute, num_entities: usize) -> Result<DataInfo> {
    let item = attribute_item(attribute)?;

    let dims = item
        .dimensions
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Attribute '{}' DataItem has no Dimensions", attribute.name),
        })?;
    let component_shape = dims.0.get(1..).ok_or_else(|| Error::InvalidDocument {
        reason: format!(
            "Attribute '{}' DataItem has empty Dimensions",
            attribute.name
        ),
    })?;

    let number_type = item.number_type.ok_or_else(|| Error::InvalidDocument {
        reason: format!("Attribute '{}' DataItem has no NumberType", attribute.name),
    })?;
    let precision = item.precision.ok_or_else(|| Error::InvalidDocument {
        reason: format!("Attribute '{}' DataItem has no Precision", attribute.name),
    })?;

    let components: usize = component_shape.iter().product();
    let len = num_entities
        .checked_mul(components)
        .ok_or(Error::Internal("DataInfo length does not fit a usize"))?;

    Ok(DataInfo {
        name: attribute.name.clone(),
        attribute: reconstruct_data_attribute(attribute.attribute_type, component_shape),
        number_type,
        precision,
        len,
    })
}

/// The inverse of `From<DataAttribute> for AttributeType` -- lossy for `Matrix`, which
/// `Tensor6`, `Matrix(n, m)` and `Generic` all collapse onto in the file, see
/// [`DataInfo::attribute`]'s doc.
fn reconstruct_data_attribute(
    attribute_type: AttributeType,
    component_shape: &[usize],
) -> DataAttribute {
    match attribute_type {
        AttributeType::Scalar => DataAttribute::Scalar,
        AttributeType::Vector => DataAttribute::Vector,
        AttributeType::Tensor => DataAttribute::Tensor,
        AttributeType::Tensor6 => DataAttribute::Tensor6,
        AttributeType::Matrix => {
            DataAttribute::Generic(component_shape.first().copied().unwrap_or(1))
        }
    }
}

/// Points for a mesh with no submeshes: a plain, non-selection `DataItem`.
fn read_points_plain(
    document: &Document,
    domain: &Domain,
    grid: &Grid,
    points: &mut Vec<f64>,
) -> Result<()> {
    let geometry = grid
        .geometry
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Geometry", grid.name),
        })?;
    let points_item = geometry
        .data_items
        .first()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Geometry of Grid '{}' has no DataItem", grid.name),
        })?;

    let values = selection::read_data_item(points_item, &document.base_dir, domain)?;
    *points = values_to_f64(&values)?;

    Ok(())
}

/// Topology for a mesh with no submeshes: a plain, non-selection `DataItem`.
fn read_topology_plain(
    document: &Document,
    domain: &Domain,
    grid: &Grid,
    connectivity: &mut Vec<u64>,
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    let topology = grid
        .topology
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Topology", grid.name),
        })?;
    let raw = selection::read_data_item(&topology.data_item, &document.base_dir, domain)?;
    let decoded = topology::decode(topology, &raw)?;

    *connectivity = decoded.connectivity;
    *cell_types = decoded.cell_types;

    Ok(())
}

/// The mesh's own points, for a mesh with submeshes: read directly out of the source each
/// submesh's `<Geometry>` selects from -- the global, per-direction coordinate arrays, written
/// once regardless of how many submeshes there are.
fn read_points_with_submeshes(
    document: &Document,
    domain: &Domain,
    first_grid: &Grid,
    points: &mut Vec<f64>,
) -> Result<()> {
    let base_dir = &document.base_dir;

    let geometry = first_grid
        .geometry
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Geometry", first_grid.name),
        })?;

    if geometry.data_items.len() != 3 {
        return Err(Error::InvalidDocument {
            reason: format!(
                "a submesh's Geometry must have 3 DataItems (one per direction), found {}",
                geometry.data_items.len()
            ),
        });
    }

    let mut directions: Vec<Vec<f64>> = Vec::with_capacity(3);
    for item in &geometry.data_items {
        let selection_item = light_data::resolve_reference(item, domain)?;
        let (_selector, source) = selection::selection_parts(selection_item)?;
        let values = selection::read_data_item(source, base_dir, domain)?;
        directions.push(values_to_f64(&values)?);
    }

    let num_points_total = directions[0].len();
    if directions[1].len() != num_points_total || directions[2].len() != num_points_total {
        return Err(Error::InvalidDocument {
            reason: "the mesh's per-direction coordinate arrays have different lengths".to_string(),
        });
    }

    points.reserve(num_points_total * 3);
    let [x, y, z] = [&directions[0], &directions[1], &directions[2]];
    for ((&x, &y), &z) in x.iter().zip(y).zip(z) {
        points.push(x);
        points.push(y);
        points.push(z);
    }

    Ok(())
}

/// Reconstruct the mesh's cell types and connectivity from its submeshes: each submesh's
/// topology decoded, then scattered into the mesh's own indexing through `cells_membership` and
/// each submesh's own point membership.
fn read_topology_with_submeshes(
    document: &Document,
    analysis: &Analysis<'_>,
    points_membership: &[Membership],
    cells_membership: &[Membership],
    connectivity: &mut Vec<u64>,
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    let domain = document.domain()?;
    let base_dir = &document.base_dir;
    let num_cells = mesh_num_cells_from_membership(cells_membership);

    if points_membership.len() != analysis.num_submeshes()
        || cells_membership.len() != analysis.num_submeshes()
    {
        return Err(Error::Internal(
            "read_topology_with_submeshes called with membership of the wrong length",
        ));
    }

    let mut decoded = Vec::with_capacity(analysis.num_submeshes());
    for submesh_grids in &analysis.submeshes {
        decoded.push(decode_submesh_topology(submesh_grids[0], domain, base_dir)?);
    }

    let mut global_cell_types = vec![CellType::Vertex; num_cells];
    let mut covered = vec![false; num_cells];

    for (submesh, cells) in decoded.iter().zip(cells_membership) {
        for (local_cell, &cell_type) in submesh.cell_types.iter().enumerate() {
            let global_cell = cell_of_submesh(cells, local_cell)?;
            let slot = covered
                .get_mut(global_cell)
                .ok_or_else(|| Error::InvalidDocument {
                    reason: format!(
                        "'{SUBMESH_CELLS}' names cell {global_cell}, but the mesh only has \
                     {num_cells} cells"
                    ),
                })?;
            *slot = true;
            global_cell_types[global_cell] = cell_type;
        }
    }

    if covered.iter().any(|&is_covered| !is_covered) {
        return Err(Error::InvalidDocument {
            reason: "some mesh cells are not covered by any submesh".to_string(),
        });
    }

    let mut offsets = Vec::with_capacity(num_cells + 1);
    let mut offset = 0_usize;
    for cell_type in &global_cell_types {
        offsets.push(offset);
        offset += cell_type.num_points();
    }
    offsets.push(offset);

    connectivity.resize(offset, 0);

    for ((submesh, cells), points) in decoded.iter().zip(cells_membership).zip(points_membership) {
        let mut local_offset = 0_usize;
        for (local_cell, &cell_type) in submesh.cell_types.iter().enumerate() {
            let stride = cell_type.num_points();
            let global_cell = cell_of_submesh(cells, local_cell)?;
            let global_start = offsets[global_cell];

            for component in 0..stride {
                let local_point = submesh.connectivity[local_offset + component] as usize;
                let global_point = points.get(local_point).ok_or_else(|| {
                    Error::InvalidDocument {
                        reason: format!(
                            "a submesh's connectivity references its point {local_point}, but its \
                             Geometry selects only {} points",
                            points.len()
                        ),
                    }
                })?;
                connectivity[global_start + component] = global_point as u64;
            }

            local_offset += stride;
        }
    }

    *cell_types = global_cell_types;

    Ok(())
}

/// The mesh position of a submesh's `local_cell`, rejecting a `Topology` that holds more cells
/// than `submesh_cells` names for it.
fn cell_of_submesh(cells: &Membership, local_cell: usize) -> Result<usize> {
    cells.get(local_cell).ok_or_else(|| Error::InvalidDocument {
        reason: format!(
            "a submesh's Topology holds more than the {} cells '{SUBMESH_CELLS}' names for it",
            cells.len()
        ),
    })
}

fn decode_submesh_topology(
    grid: &Grid,
    domain: &Domain,
    base_dir: &Path,
) -> Result<topology::DecodedTopology> {
    let topology = grid
        .topology
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Topology", grid.name),
        })?;
    let raw = selection::read_data_item(&topology.data_item, base_dir, domain)?;

    topology::decode(topology, &raw)
}

/// Which mesh points each submesh holds, read from its `<Geometry>` selector.
fn submesh_points_membership(
    analysis: &Analysis<'_>,
    domain: &Domain,
    base_dir: &Path,
) -> Result<Vec<Membership>> {
    analysis
        .submeshes
        .iter()
        .map(|grids| submesh_geometry_membership(grids[0], domain, base_dir))
        .collect()
}

fn submesh_geometry_membership(
    grid: &Grid,
    domain: &Domain,
    base_dir: &Path,
) -> Result<Membership> {
    let geometry = grid
        .geometry
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Geometry", grid.name),
        })?;
    let first_item = geometry
        .data_items
        .first()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Geometry of Grid '{}' has no DataItem", grid.name),
        })?;
    let selection_item = light_data::resolve_reference(first_item, domain)?;
    let (selector, _source) = selection::selection_parts(selection_item)?;

    selection::parse_selector(selector, base_dir, domain)
}

/// Parse `<Information Name="submesh_cells">`'s value into one [`Membership`] per submesh, in
/// submesh order -- see `write_submesh_index_list` (`time_series_writer.rs`) for the two shapes
/// an entry may be.
fn parse_submesh_cells(document: &Document, domain: &Domain) -> Result<Vec<Membership>> {
    let value = document
        .information(SUBMESH_CELLS)
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("the document has submeshes but no '{SUBMESH_CELLS}' Information"),
        })?;

    value
        .split_whitespace()
        .map(|entry| parse_submesh_cells_entry(entry, document, domain))
        .collect()
}

fn parse_submesh_cells_entry(
    entry: &str,
    document: &Document,
    domain: &Domain,
) -> Result<Membership> {
    if let Some((start, len)) = entry.split_once(':') {
        let start = start.parse().map_err(|_source| Error::InvalidDocument {
            reason: format!("'{SUBMESH_CELLS}' entry '{entry}' has an invalid start"),
        })?;
        let len = len.parse().map_err(|_source| Error::InvalidDocument {
            reason: format!("'{SUBMESH_CELLS}' entry '{entry}' has an invalid length"),
        })?;
        return Ok(Membership::Contiguous { start, len });
    }

    let item = light_data::find_by_name(domain, entry).ok_or_else(|| Error::InvalidDocument {
        reason: format!("'{SUBMESH_CELLS}' names a DataItem '{entry}' that does not exist"),
    })?;
    let values = selection::read_data_item(item, &document.base_dir, domain)?;

    Ok(Membership::Explicit(selection::values_to_usize(&values)?))
}

/// Total mesh size for a mesh with no submeshes: `num_points` from the Geometry's own resolved
/// `DataItem`, `num_cells` from the `Topology`'s `NumberOfElements` -- both plain XML metadata,
/// no heavy data touched.
fn mesh_size_plain(grid: &Grid, domain: &Domain) -> Result<(usize, usize)> {
    let geometry = grid
        .geometry
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Geometry", grid.name),
        })?;
    let item = geometry
        .data_items
        .first()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Geometry of Grid '{}' has no DataItem", grid.name),
        })?;
    // the geometry's own DataItem is a `Reference="XML"` to the actual, named coordinate array --
    // resolve it to reach its Dimensions
    let item = light_data::resolve_reference(item, domain)?;
    let dims = item
        .dimensions
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!(
                "Geometry DataItem of Grid '{}' has no Dimensions",
                grid.name
            ),
        })?;
    let num_points = *dims.0.first().ok_or_else(|| Error::InvalidDocument {
        reason: format!(
            "Geometry DataItem of Grid '{}' has empty Dimensions",
            grid.name
        ),
    })?;

    let topology = grid
        .topology
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Topology", grid.name),
        })?;
    let num_cells = topology
        .number_of_elements
        .parse::<usize>()
        .map_err(|_source| Error::InvalidDocument {
            reason: format!(
                "Topology NumberOfElements '{}' of Grid '{}' is not a valid number",
                topology.number_of_elements, grid.name
            ),
        })?;

    Ok((num_points, num_cells))
}

/// `num_points` for a mesh with submeshes: the `Dimensions` of the *source* array a submesh's
/// geometry selects out of, i.e. the mesh's own coordinates.
fn mesh_num_points_with_submeshes(first_grid: &Grid, domain: &Domain) -> Result<usize> {
    let geometry = first_grid
        .geometry
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Geometry", first_grid.name),
        })?;
    let item = geometry
        .data_items
        .first()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Geometry of Grid '{}' has no DataItem", first_grid.name),
        })?;

    let selection_item = light_data::resolve_reference(item, domain)?;
    let (_selector, source) = selection::selection_parts(selection_item)?;
    let dims = source
        .dimensions
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: "the mesh's coordinate source DataItem has no Dimensions".to_string(),
        })?;

    dims.0
        .first()
        .copied()
        .ok_or_else(|| Error::InvalidDocument {
            reason: "the mesh's coordinate source DataItem has empty Dimensions".to_string(),
        })
}

/// `num_cells` for a mesh with submeshes: `1 + max` over every submesh's `submesh_cells` entry,
/// sound because the writer's `check_all_cells_covered` makes every cell belong to at least one
/// submesh.
fn mesh_num_cells_from_membership(cells_membership: &[Membership]) -> usize {
    let max_cell = cells_membership.iter().flat_map(Membership::iter).max();

    max_cell.map_or(0, |max| max + 1)
}

/// Scatter every submesh's own share of a field into the mesh's own indexing, through its
/// membership -- the field-data counterpart of the connectivity scatter in
/// [`read_topology_with_submeshes`]. Works uniformly whether a submesh's `DataItem` was a
/// selection into a shared array or (not emitted by this crate's own writer, but not assumed
/// against either) a private copy: either way `read_data_item` already reduced it to "this
/// submesh's own values, in its own entity order".
fn scatter_field(
    num_entities: usize,
    submesh_values: &[Values<'static>],
    membership: &[Membership],
) -> Result<Values<'static>> {
    if submesh_values.len() != membership.len() {
        return Err(Error::Internal(
            "scatter_field called with mismatched submeshes and membership",
        ));
    }

    let Some(first) = submesh_values.first() else {
        return Err(Error::InvalidDocument {
            reason: "a field with submeshes has no data at all".to_string(),
        });
    };

    let components = component_count(first, &membership[0])?;
    let total = num_entities.checked_mul(components).ok_or(Error::Internal(
        "scattered field length does not fit a usize",
    ))?;

    macro_rules! scatter_arm {
        ($variant:ident) => {{
            let mut entries: Vec<(&[_], &Membership)> = Vec::with_capacity(submesh_values.len());
            for (values, entities) in submesh_values.iter().zip(membership) {
                let Values::$variant(v) = values else {
                    return Err(Error::InvalidDocument {
                        reason: "a submesh's field data has a different type than another's"
                            .to_string(),
                    });
                };
                if v.len() != entities.len() * components {
                    return Err(Error::InvalidDocument {
                        reason: format!(
                            "a submesh's field data has {} values, expected {}",
                            v.len(),
                            entities.len() * components
                        ),
                    });
                }
                entries.push((v.as_ref(), entities));
            }
            Values::from(scatter_typed(total, &entries, components))
        }};
    }

    Ok(match first {
        Values::F64(_) => scatter_arm!(F64),
        Values::F32(_) => scatter_arm!(F32),
        Values::I64(_) => scatter_arm!(I64),
        Values::I32(_) => scatter_arm!(I32),
        Values::U64(_) => scatter_arm!(U64),
        Values::U32(_) => scatter_arm!(U32),
    })
}

fn component_count(values: &Values<'_>, membership: &Membership) -> Result<usize> {
    let len = values.len();
    let entities = membership.len();

    if entities == 0 || !len.is_multiple_of(entities) {
        return Err(Error::InvalidDocument {
            reason: format!(
                "a field's {len} values do not divide evenly over its submesh's {entities} entities"
            ),
        });
    }

    Ok(len / entities)
}

fn scatter_typed<T: Copy + Default>(
    total: usize,
    entries: &[(&[T], &Membership)],
    components: usize,
) -> Vec<T> {
    let mut buffer = vec![T::default(); total];

    for (values, membership) in entries {
        for (local, global_entity) in membership.iter().enumerate() {
            let src = local * components;
            let dst = global_entity * components;
            buffer[dst..dst + components].copy_from_slice(&values[src..src + components]);
        }
    }

    buffer
}

fn values_to_f64(values: &Values<'_>) -> Result<Vec<f64>> {
    match values {
        Values::F64(v) => Ok(v.to_vec()),
        Values::F32(v) => Ok(v.iter().map(|&value| f64::from(value)).collect()),
        other => Err(Error::InvalidDocument {
            reason: format!(
                "geometry data must be floating-point, found {}",
                value_type_name(other)
            ),
        }),
    }
}

fn parse_data_storage(value: &str) -> Option<DataStorage> {
    match value {
        "Ascii" => return Some(DataStorage::Ascii),
        "AsciiInline" => return Some(DataStorage::AsciiInline),
        "Binary" => return Some(DataStorage::Binary),
        _ => {}
    }

    if let Some(rest) = value.strip_prefix("Hdf5SingleFile") {
        return Some(DataStorage::Hdf5SingleFile {
            deflate_level: parse_deflate_level(rest),
        });
    }
    if let Some(rest) = value.strip_prefix("Hdf5MultipleFiles") {
        return Some(DataStorage::Hdf5MultipleFiles {
            deflate_level: parse_deflate_level(rest),
        });
    }

    None
}

/// Best-effort parse of `{ deflate_level: Some(3) }`/`{ deflate_level: None }` out of the `Debug`
/// formatting `new_document` (`time_series_writer.rs`) writes the `Information` with.
fn parse_deflate_level(rest: &str) -> Option<u8> {
    let after_some = rest.split("Some(").nth(1)?;
    let digits = after_some.split(')').next()?;
    digits.trim().parse().ok()
}

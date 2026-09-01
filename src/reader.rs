//! Reading XDMF time series written with the `Hdf5SingleFile`/`Hdf5MultipleFiles` storages, i.e.
//! every `DataItem` with `Format="HDF"`. `Format="XML"`/`"Binary"` are not supported yet: a
//! document that says which storage wrote it is rejected as [`Error::Unsupported`] when it is
//! opened, and a foreign one whose `DataItem`s turn out to hold either as it is read.
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

mod light_data;
mod selection;
mod topology;

cfg_select! {
    feature = "hdf5" => {
        mod hdf5_reader;
    }
    _ => {
        /// What the heavy-data reader is in a build without the `hdf5` feature: the one error
        /// saying so, for every `Format="HDF"` `DataItem` a document turns out to hold.
        ///
        /// [`TimeSeriesReader::new`] already rejects a document that *says* an HDF5 storage wrote
        /// it, so this is reached only for a foreign document, one `DataItem` at a time. Selecting
        /// a whole module rather than gating each item inside it is what `lib.rs` does for
        /// `hdf5_writer`, and it is why `hdf5_reader.rs` itself holds no `cfg`.
        mod hdf5_reader {
            use std::path::Path;

            use crate::{Error, Result, Values, reader::sealed::SealedValueType};

            /// Nothing to cache when no file can be opened, but `Document` holds one either way.
            /// Braced rather than a unit struct so that `FileCache::default()` stays the way it is
            /// written in the build that has a field to fill.
            #[derive(Default)]
            pub(super) struct FileCache {}

            pub(super) fn read(_: &Path, _: &str, _: &FileCache) -> Result<Values<'static>> {
                Err(no_hdf5_feature())
            }

            pub(super) fn read_exact_into<T: SealedValueType>(
                _: &Path,
                _: &str,
                _: &FileCache,
                _: &mut Vec<T>,
            ) -> Result<bool> {
                Err(no_hdf5_feature())
            }

            fn no_hdf5_feature() -> Error {
                Error::Unsupported {
                    reason: "the document holds Format=\"HDF\" data, but this build was compiled \
                             without the 'hdf5' feature"
                        .to_string(),
                }
            }
        }
    }
}

use std::{path::Path, str::FromStr};

use light_data::{Analysis, Document};
use selection::Membership;

use crate::{
    CellType, ConnectivityIndex, Coordinate, DATA_STORAGE, DataAttribute, DataStorage, Error,
    Result, SUBMESH_CELLS, Values,
    xdmf_elements::{
        Domain,
        attribute::{self, AttributeType},
        data_item::{DataItem, NumberType},
        grid::Grid,
        topology::Topology,
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
    /// Components per entity, the product of the dimensions past the first. What [`len`](Self::len)
    /// was sized with, so trust it over [`attribute`](Self::attribute)'s own count.
    pub components: usize,
    /// Total element count: `num_entities * components`.
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
    use super::{Error, Result, Values};

    // `H5Type` is a supertrait only where there is an HDF5 to read: it is what lets
    // `hdf5_reader::read_exact_into` fill a caller's buffer straight from a dataset of this same
    // type, without the intermediate `Values` array. Every type that implements this trait is one
    // of the six primitives, so both bounds hold for all of them either way; the trait is `pub`
    // inside a `pub(crate)` module, so neither is nameable outside the crate.
    cfg_select! {
        feature = "hdf5" => {
            pub trait SealedValueType: Sized + Copy + Default + hdf5::H5Type {
                fn from_values(values: Values<'static>) -> Result<Vec<Self>>;
            }
        }
        _ => {
            pub trait SealedValueType: Sized + Copy + Default {
                fn from_values(values: Values<'static>) -> Result<Vec<Self>>;
            }
        }
    }

    fn mismatch(requested: &str, found: &Values<'_>) -> Error {
        Error::NumberTypeMismatch {
            reason: format!(
                "requested {requested}, but the file holds {}",
                found.type_name()
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

/// Reads an XDMF time series: the parsed light data, the mesh's metadata (including, for a mesh
/// with submeshes, which points and cells each one holds), and every read call against it.
///
/// The light data is parsed once, in [`new`](Self::new), so a reader shows the steps the document
/// had when it was opened; reopen it to pick up steps written since. A read keeps its heavy-data
/// file open afterwards, for the next read that names the same one, so that file stays open until
/// the reader is dropped.
///
/// ```rust,no_run
/// use xdmf::TimeSeriesReader;
///
/// // open a file written with one of the two HDF5 storages
/// let reader = TimeSeriesReader::new("xdmf_writing.xdmf2").expect("failed to open XDMF file");
///
/// // points and topology (connectivity + cell types) are independent reads, each filling a
/// // buffer of whichever element type the caller wants it at
/// let mut points: Vec<f64> = Vec::new();
/// reader
///     .read_points(&mut points)
///     .expect("failed to read points");
///
/// let mut connectivity: Vec<u64> = Vec::new();
/// let mut cell_types = Vec::new();
/// reader
///     .read_topology(&mut connectivity, &mut cell_types)
///     .expect("failed to read topology");
///
/// // if the mesh was written with submeshes, each one's own cells (and points) can be recovered,
/// // as indices into the buffers above -- empty for a mesh with no submeshes
/// for (index, name) in reader.submesh_names().iter().enumerate() {
///     let cells = reader
///         .submesh_cells(index)
///         .expect("failed to read submesh cells");
///     println!("{name}: {} cells", cells.len());
/// }
///
/// // then read each step's data, reusing the same buffers
/// let mut point_data = Vec::new();
/// let mut cell_data = Vec::new();
/// for step in 0..reader.num_steps() {
///     reader
///         .read_point_data::<f64>(step, "point_data", &mut point_data)
///         .expect("failed to read point data");
///     reader
///         .read_cell_data::<f64>(step, "cell_data", &mut cell_data)
///         .expect("failed to read cell data");
/// }
/// ```
pub struct TimeSeriesReader {
    document: Document,
    /// How the document's grids break down, as positions rather than references, so that it can
    /// be built once here rather than rebuilt by every read call.
    analysis: Analysis,
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
    ///
    /// A document written with a storage this reader cannot read is rejected here, rather than at
    /// the first call that reaches heavy data.
    pub fn new(file_name: impl AsRef<Path>) -> Result<Self> {
        let document = Document::open(file_name.as_ref())?;
        check_readable(&document)?;

        let domain = document.domain()?;
        let analysis = Analysis::build(domain)?;

        let submesh_names = analysis.submesh_names().to_vec();
        let times = analysis.times(domain)?;

        let (num_points, num_cells, points_membership, cells_membership) = if submesh_names
            .is_empty()
        {
            let (num_points, num_cells) = mesh_size_plain(analysis.mesh_grid(0, domain)?, domain)?;
            (num_points, num_cells, Vec::new(), Vec::new())
        } else {
            let num_points =
                mesh_num_points_with_submeshes(analysis.mesh_grid(0, domain)?, domain)?;
            let points_membership = submesh_points_membership(&analysis, domain, &document)?;
            check_membership_in_range(&points_membership, num_points, "point")?;
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
            analysis,
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

    /// Read the mesh's points into a buffer of `f32` or `f64` (see [`Coordinate`]). The buffer is
    /// cleared first, so its existing capacity is reused.
    ///
    /// Reading `f32` coordinates into a `Vec<f64>` is allowed; the narrowing direction is
    /// [`Error::NumberTypeMismatch`] rather than a silent loss of precision, the same rule
    /// [`ValueType`] states for field data.
    pub fn read_points<C: Coordinate>(&self, points: &mut Vec<C>) -> Result<()> {
        points.clear();

        let domain = self.document.domain()?;
        let first_grid = self.analysis.mesh_grid(0, domain)?;

        if self.submesh_names.is_empty() {
            read_points_plain(&self.document, domain, first_grid, points)
        } else {
            read_points_with_submeshes(&self.document, domain, first_grid, points)
        }
    }

    /// Read the mesh's connectivity into a buffer of `u32`, `u64`, `i32` or `i64` (see
    /// [`ConnectivityIndex`]), and its cell types. Both buffers are cleared first, so their
    /// existing capacity is reused.
    ///
    /// Unlike the widening rule the field data follows, the index type is checked against the
    /// *values*: what comes back is a position in the mesh this reader put back together, not the
    /// file's own array, so an index the requested type cannot hold is
    /// [`Error::IntegerOutOfRange`] whatever the file was written as.
    pub fn read_topology<I: ConnectivityIndex>(
        &self,
        connectivity: &mut Vec<I>,
        cell_types: &mut Vec<CellType>,
    ) -> Result<()> {
        connectivity.clear();
        cell_types.clear();

        let domain = self.document.domain()?;

        if self.submesh_names.is_empty() {
            return read_topology_plain(
                &self.document,
                domain,
                self.analysis.mesh_grid(0, domain)?,
                connectivity,
                cell_types,
            );
        }

        read_topology_with_submeshes(
            &self.document,
            &self.analysis,
            &self.points_membership,
            &self.cells_membership,
            connectivity,
            cell_types,
        )
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

        if self.submesh_names.is_empty() {
            let grid = self.step_grid(step, 0)?;
            let attribute = find_attribute(grid, name, center)?;
            let item = attribute_item(attribute)?;

            // straight into the caller's buffer: a field written and read back at the same width
            // never needs an array of its own, so a loop over the steps allocates once rather
            // than once per step
            return selection::read_data_item_into(
                item,
                &self.document,
                domain,
                into,
                T::from_values,
            );
        }

        self.read_submesh_field(step, name, center, num_entities, membership, into)
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
    fn read_submesh_field<T: ValueType>(
        &self,
        step: usize,
        name: &str,
        center: attribute::Center,
        num_entities: usize,
        membership: &[Membership],
        into: &mut Vec<T>,
    ) -> Result<()> {
        let domain = self.document.domain()?;
        let mut submesh_values = Vec::with_capacity(self.submesh_names.len());

        for submesh in 0..self.submesh_names.len() {
            let grid = self.step_grid(step, submesh)?;
            let attribute = find_attribute(grid, name, center)?;
            let item = attribute_item(attribute)?;

            if item.item_type.is_some() {
                let (_selector, source) = selection::selection_parts(item)?;
                // the whole field, read like a mesh without submeshes -- straight into the
                // caller's buffer, and the reads already made for earlier submeshes are dropped
                return selection::read_data_item_into(
                    source,
                    &self.document,
                    domain,
                    into,
                    T::from_values,
                );
            }

            submesh_values.push(selection::read_data_item(item, &self.document, domain)?);
        }

        let scattered = scatter_field(num_entities, &submesh_values, membership)?;
        into.clear();
        into.extend(T::from_values(scattered)?);

        Ok(())
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

        self.analysis
            .step_grid(submesh, step, self.document.domain()?)
    }
}

/// Reject a document written with a storage this reader cannot read.
///
/// The `data_storage` `Information` is written with the `Debug` formatting (`new_document`,
/// `time_series_writer.rs`), so the HDF5 variants are followed by the `deflate_level` they were
/// written with; only the variant name matters here. A document that names no storage, or names
/// one this crate does not know, is a foreign file and is let through -- its `DataItem`s are
/// checked for `Format="HDF"` one by one as they are read.
fn check_readable(document: &Document) -> Result<()> {
    let Some(name) = document
        .information(DATA_STORAGE)
        .and_then(|value| value.split_whitespace().next())
    else {
        return Ok(());
    };

    let Ok(storage) = DataStorage::from_str(name) else {
        return Ok(());
    };

    match storage {
        DataStorage::Hdf5SingleFile { .. } | DataStorage::Hdf5MultipleFiles { .. } => Ok(()),
        DataStorage::Ascii | DataStorage::AsciiInline | DataStorage::Binary => {
            Err(Error::Unsupported {
                reason: format!(
                    "the document was written with the {name} storage, which this reader cannot \
                     read -- only Hdf5SingleFile and Hdf5MultipleFiles can be"
                ),
            })
        }
    }
}

/// One reconstructed connectivity index as the type the caller asked for.
///
/// The mesh position a submesh's own point number maps to, which is what
/// [`TimeSeriesReader::read_topology`] checks rather than the file's own array -- a submesh's
/// numbering is local and always fits, its place in the whole mesh may not.
fn to_connectivity_index<I: ConnectivityIndex>(index: usize) -> Result<I> {
    I::from_index(index).ok_or_else(|| Error::IntegerOutOfRange {
        value: index as i128,
        reason: format!(
            "the connectivity index does not fit the requested index type, whose largest is {}",
            I::MAX_INDEX
        ),
    })
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
        components,
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
fn read_points_plain<C: Coordinate>(
    document: &Document,
    domain: &Domain,
    grid: &Grid,
    points: &mut Vec<C>,
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

    selection::read_data_item_into(
        points_item,
        document,
        domain,
        points,
        C::coordinates_from_values,
    )
}

/// Topology for a mesh with no submeshes: a plain, non-selection `DataItem`.
///
/// The file's array is read into the caller's buffer and decoded there, so a mesh whose cells all
/// share one type -- what this crate writes whenever it can -- costs exactly the one allocation
/// the caller asked for.
fn read_topology_plain<I: ConnectivityIndex>(
    document: &Document,
    domain: &Domain,
    grid: &Grid,
    connectivity: &mut Vec<I>,
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    let topology = grid_topology(grid)?;

    selection::read_data_item_into(
        &topology.data_item,
        document,
        domain,
        connectivity,
        I::indices_from_values,
    )?;

    topology::decode_in_place(topology, connectivity, cell_types)
}

/// The mesh's own points, for a mesh with submeshes: read directly out of the source each
/// submesh's `<Geometry>` selects from -- the global, per-direction coordinate arrays, written
/// once regardless of how many submeshes there are.
///
/// One direction at a time, scattered straight into its stride of the interleaved output, so the
/// three whole-mesh arrays are never live together and the second and third reads refill the
/// buffer the first one allocated.
fn read_points_with_submeshes<C: Coordinate>(
    document: &Document,
    domain: &Domain,
    first_grid: &Grid,
    points: &mut Vec<C>,
) -> Result<()> {
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

    let mut direction: Vec<C> = Vec::new();
    let mut num_points_total = 0;

    for (axis, item) in geometry.data_items.iter().enumerate() {
        let selection_item = light_data::resolve_reference(item, domain)?;
        let (_selector, source) = selection::selection_parts(selection_item)?;
        selection::read_data_item_into(
            source,
            document,
            domain,
            &mut direction,
            C::coordinates_from_values,
        )?;

        if axis == 0 {
            num_points_total = direction.len();
            points.clear();
            points.resize(num_points_total * 3, C::default());
        } else if direction.len() != num_points_total {
            return Err(Error::InvalidDocument {
                reason: "the mesh's per-direction coordinate arrays have different lengths"
                    .to_string(),
            });
        }

        for (point, &coordinate) in direction.iter().enumerate() {
            points[point * 3 + axis] = coordinate;
        }
    }

    Ok(())
}

/// Reconstruct the mesh's cell types and connectivity from its submeshes: each submesh's
/// topology decoded, then scattered into the mesh's own indexing through `cells_membership` and
/// each submesh's own point membership.
///
/// Two passes, because the mesh's cell offsets need *every* cell's type before *any* cell's points
/// can be placed -- but only one submesh's decoded topology is live at a time, rather than the
/// whole mesh's connectivity a second time. The first pass reads no heavy data at all for a
/// uniform submesh, which states its one cell type in the light data; only `Mixed` is decoded
/// twice.
fn read_topology_with_submeshes<I: ConnectivityIndex>(
    document: &Document,
    analysis: &Analysis,
    points_membership: &[Membership],
    cells_membership: &[Membership],
    connectivity: &mut Vec<I>,
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    let domain = document.domain()?;
    let num_cells = mesh_num_cells_from_membership(cells_membership);

    if points_membership.len() != analysis.num_submeshes()
        || cells_membership.len() != analysis.num_submeshes()
    {
        return Err(Error::Internal(
            "read_topology_with_submeshes called with membership of the wrong length",
        ));
    }

    let mut scratch_connectivity: Vec<I> = Vec::new();
    let mut scratch_cell_types: Vec<CellType> = Vec::new();
    let mut covered = vec![false; num_cells];

    cell_types.clear();
    cell_types.resize(num_cells, CellType::Vertex);

    for (submesh, cells) in cells_membership.iter().enumerate() {
        let grid = analysis.mesh_grid(submesh, domain)?;

        if let Some(cell_type) = topology::uniform_cell_type(grid_topology(grid)?)? {
            for global_cell in cells.iter() {
                mark_cell(global_cell, cell_type, cell_types, &mut covered)?;
            }
        } else {
            decode_submesh_topology(
                grid,
                domain,
                document,
                cells,
                &mut scratch_connectivity,
                &mut scratch_cell_types,
            )?;

            for (&cell_type, global_cell) in scratch_cell_types.iter().zip(cells.iter()) {
                mark_cell(global_cell, cell_type, cell_types, &mut covered)?;
            }
        }
    }

    if covered.iter().any(|&is_covered| !is_covered) {
        return Err(Error::InvalidDocument {
            reason: "some mesh cells are not covered by any submesh".to_string(),
        });
    }

    let mut offsets = Vec::with_capacity(num_cells + 1);
    let mut offset = 0_usize;
    for cell_type in cell_types.iter() {
        offsets.push(offset);
        offset += cell_type.num_points();
    }
    offsets.push(offset);

    connectivity.clear();
    connectivity.resize(offset, I::default());

    for ((submesh, cells), points) in (0..analysis.num_submeshes())
        .zip(cells_membership)
        .zip(points_membership)
    {
        let grid = analysis.mesh_grid(submesh, domain)?;
        decode_submesh_topology(
            grid,
            domain,
            document,
            cells,
            &mut scratch_connectivity,
            &mut scratch_cell_types,
        )?;

        let mut local_offset = 0_usize;
        for (&cell_type, global_cell) in scratch_cell_types.iter().zip(cells.iter()) {
            // `offsets` was built from the cell types pass 1 recorded, where the last submesh to
            // claim an overlapped cell won. Two submeshes disagreeing about a shared cell's type
            // would make this cell's span here a different width than the slot reserved for it,
            // and write over its neighbour's -- so the disagreement is reported instead.
            if cell_type != cell_types[global_cell] {
                return Err(Error::InvalidDocument {
                    reason: format!(
                        "submeshes disagree about mesh cell {global_cell}: one holds it as \
                         {cell_type:?}, another as {:?}",
                        cell_types[global_cell]
                    ),
                });
            }

            let stride = cell_type.num_points();
            let global_start = offsets[global_cell];

            for component in 0..stride {
                // every value came through `SealedIndex::indices_from_values`, which rejects an
                // entry that is not a position, so this cannot be `None`
                let local_point = scratch_connectivity[local_offset + component]
                    .as_index()
                    .ok_or(Error::Internal(
                        "a decoded connectivity entry is not a position",
                    ))?;
                let global_point = points.get(local_point).ok_or_else(|| {
                    Error::InvalidDocument {
                        reason: format!(
                            "a submesh's connectivity references its point {local_point}, but its \
                             Geometry selects only {} points",
                            points.len()
                        ),
                    }
                })?;
                connectivity[global_start + component] = to_connectivity_index::<I>(global_point)?;
            }

            local_offset += stride;
        }
    }

    Ok(())
}

/// Record one mesh cell's type, and that some submesh covers it.
fn mark_cell(
    global_cell: usize,
    cell_type: CellType,
    cell_types: &mut [CellType],
    covered: &mut [bool],
) -> Result<()> {
    let num_cells = cell_types.len();
    let slot = covered
        .get_mut(global_cell)
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!(
                "'{SUBMESH_CELLS}' names cell {global_cell}, but the mesh only has {num_cells} \
                 cells"
            ),
        })?;

    *slot = true;
    cell_types[global_cell] = cell_type;

    Ok(())
}

/// Decode one submesh's topology into buffers the caller reuses across submeshes, rejecting a
/// `Topology` that holds a different number of cells than `submesh_cells` names for it.
fn decode_submesh_topology<I: ConnectivityIndex>(
    grid: &Grid,
    domain: &Domain,
    document: &Document,
    cells: &Membership,
    connectivity: &mut Vec<I>,
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    let topology = grid_topology(grid)?;

    selection::read_data_item_into(
        &topology.data_item,
        document,
        domain,
        connectivity,
        I::indices_from_values,
    )?;
    topology::decode_in_place(topology, connectivity, cell_types)?;

    if cell_types.len() != cells.len() {
        return Err(Error::InvalidDocument {
            reason: format!(
                "a submesh's Topology holds {} cells, but '{SUBMESH_CELLS}' names {} for it",
                cell_types.len(),
                cells.len()
            ),
        });
    }

    Ok(())
}

/// A grid's `Topology`, which every grid this reader looks at has.
fn grid_topology(grid: &Grid) -> Result<&Topology> {
    grid.topology
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Topology", grid.name),
        })
}

/// Which mesh points each submesh holds, read from its `<Geometry>` selector.
fn submesh_points_membership(
    analysis: &Analysis,
    domain: &Domain,
    document: &Document,
) -> Result<Vec<Membership>> {
    (0..analysis.num_submeshes())
        .map(|submesh| {
            submesh_geometry_membership(analysis.mesh_grid(submesh, domain)?, domain, document)
        })
        .collect()
}

/// Reject a membership naming an entity the mesh does not have, once, when the document is opened.
///
/// The cell lists need no such check -- `mesh_num_cells_from_membership` takes the mesh's cell
/// count *from* them -- but a submesh's points are named by its `<Geometry>` selector while the
/// mesh's point count comes from the array that selector reads out of, so the two are independent
/// statements of the file's and can disagree. Every later use of these indices treats them as
/// positions: `scatter_field` writes at them, and [`TimeSeriesReader::submesh_points`] hands them
/// out as indices into the buffer [`TimeSeriesReader::read_points`] fills.
fn check_membership_in_range(
    membership: &[Membership],
    num_entities: usize,
    entity: &str,
) -> Result<()> {
    for (submesh, entities) in membership.iter().enumerate() {
        if let Some(out_of_range) = entities.iter().find(|&index| index >= num_entities) {
            return Err(Error::InvalidDocument {
                reason: format!(
                    "submesh {submesh} holds {entity} {out_of_range}, but the mesh only has \
                     {num_entities} {entity}s"
                ),
            });
        }
    }

    Ok(())
}

fn submesh_geometry_membership(
    grid: &Grid,
    domain: &Domain,
    document: &Document,
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

    selection::parse_selector(selector, document, domain)
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
    let values = selection::read_data_item(item, document, domain)?;

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

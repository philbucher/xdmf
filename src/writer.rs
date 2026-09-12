//! Writing a series of time steps to XDMF.
//!
//! The mesh is written once and referenced from each time step, rather than repeated per step.
//!
//! Inspired by the `TimeSeriesWriter` of [meshio](https://github.com/nschloe/meshio).

use std::{
    collections::{HashMap, HashSet},
    fmt,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

pub use submesh::SubmeshCells;
use submesh::{
    IndexList, LocalPoints, PreparedMesh, SelectionKey, Submesh, cell_offsets, entities_of,
    extract_connectivity, index_values, poly_cell_points, prepare_cells, prepare_submeshes,
    renumber_connectivity, selected_coordinates, selected_geometry, submesh_index_name,
    submesh_points, submesh_topology, validate_points_and_cells,
};

use crate::{
    CellType, ConnectivityIndex, Coordinate, DATA_STORAGE, DataAttribute, DataStorage, DataWriter,
    Error, Result, SELECTIONS, SUBMESH_CELLS, SUBMESH_POINTS, Values, create_writer,
    error::io_ctx,
    mpi_safe_create_dir_all, paraview,
    values::GatherBuffers,
    xdmf_elements::{
        Information, Xdmf, attribute,
        data_item::{DataContent, DataItem, Format, ItemType, NumberType},
        dimensions::Dimensions,
        geometry::{Geometry, GeometryType},
        grid::{CollectionType, Grid, Time},
        topology::{Topology, TopologyType},
    },
};

pub(crate) mod ascii;
pub(crate) mod binary;
#[cfg(feature = "hdf5")]
pub(crate) mod hdf5;
mod submesh;

/// Writer for time series data in XDMF format.
pub struct TimeSeriesWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
}

impl fmt::Debug for TimeSeriesWriter {
    /// Shows the backend as its `DataStorage`, the only part of it a caller can act on.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeSeriesWriter")
            .field("xdmf_file_name", &self.xdmf_file_name)
            .field("data_storage", &self.writer.data_storage())
            .finish()
    }
}

impl TimeSeriesWriter {
    /// Create a new `TimeSeriesWriter`.
    /// ```rust
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("name_xdmf_file", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    /// ```
    pub fn new(file_name: impl AsRef<Path>, data_storage: DataStorage) -> Result<Self> {
        let xdmf_file_name = file_name.as_ref().to_path_buf().with_extension("xdmf2");

        validate_file_name(&xdmf_file_name)?;

        if let Some(parent) = xdmf_file_name.parent() {
            mpi_safe_create_dir_all(parent)?;
        }

        Ok(Self {
            xdmf_file_name,
            writer: create_writer(file_name.as_ref(), data_storage)?,
        })
    }

    /// The XDMF file this writer writes: the name it was given, with the XDMF extension on it.
    /// The heavy data takes the same base and its own storage's extension.
    pub fn file_name(&self) -> &Path {
        &self.xdmf_file_name
    }

    /// Writes the mesh to the XDMF file, returning a `TimeSeriesDataWriter` for writing time steps.
    ///
    /// Connectivity type caps mesh size: `u32`/`u64` at `u32::MAX` points, `i64` at the full 64
    /// bits but only in the HDF5 storages.
    ///
    /// ```rust
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("xdmf_write_mesh", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    ///
    /// // define 3 points and 2 cells (a line and a triangle)
    /// let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// let connectivity = [0_u32, 1, 0, 2, 1]; // line (0,1) and triangle (0,2,1)
    /// let cell_types = [xdmf::CellType::Edge, xdmf::CellType::Triangle];
    ///
    /// // write the mesh
    /// let mut ts_writer = xdmf_writer.write_mesh(&coords, &connectivity, &cell_types);
    /// # // hidden: doctests run in the crate root, so the example cleans up after itself
    /// # std::fs::remove_file("xdmf_write_mesh.xdmf2").expect("the example writes this file");
    /// ```
    ///
    /// A mesh of points only has no connectivity to infer the index type from, so it has to be
    /// named:
    ///
    /// ```rust
    /// # use xdmf::TimeSeriesWriter;
    /// # let xdmf_writer = TimeSeriesWriter::new("xdmf_write_points", xdmf::DataStorage::AsciiInline)
    /// #     .expect("failed to create XDMF writer");
    /// let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// let mut ts_writer = xdmf_writer.write_mesh(&coords, &[] as &[u32], &[]);
    /// # std::fs::remove_file("xdmf_write_points.xdmf2").expect("the example writes this file");
    /// ```
    pub fn write_mesh<C: Coordinate, I: ConnectivityIndex>(
        mut self,
        points: &[C],
        connectivity: &[I],
        cell_types: &[CellType],
    ) -> Result<TimeSeriesDataWriter> {
        validate_points_and_cells(points.len(), connectivity, cell_types)?;

        let mesh = self.prepare_mesh(points, connectivity, cell_types)?;

        let points_item = self.points_data_item(None, &C::as_values(points))?;
        let connectivity_item = self.connectivity_data_item(None, &mesh.cells)?;

        let topology = Topology {
            topology_type: mesh.topology_type,
            nodes_per_element: mesh.nodes_per_element,
            number_of_elements: mesh.num_cells.to_string(),
            data_item: DataItem::new_reference(&connectivity_item, DOMAIN_DATA_ITEMS),
        };

        let grid = Grid::new_uniform("mesh", geometry(&points_item), topology);

        self.finish_mesh(
            grid,
            vec![points_item, connectivity_item],
            Vec::new(),
            mesh.num_points,
            mesh.num_cells,
        )
    }

    /// Writes the mesh split into named submeshes, returning a `TimeSeriesDataWriter`.
    ///
    /// A submesh is a named subset of the mesh's cells, shown as its own block in `ParaView`'s
    /// Multi-block Inspector. Submeshes may overlap, but every cell must belong to at least one.
    ///
    /// Field data is still passed once per step over the whole mesh; the writer cuts each
    /// submesh's share automatically.
    ///
    /// ```rust
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("xdmf_write_submeshes", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    ///
    /// // define 4 points and 3 cells (a line and two triangles)
    /// let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    /// let connectivity = [0_u32, 1, 0, 2, 1, 1, 2, 3];
    /// let cell_types = [
    ///     xdmf::CellType::Edge,
    ///     xdmf::CellType::Triangle,
    ///     xdmf::CellType::Triangle,
    /// ];
    ///
    /// // cell 0 is the edge, cells 1 and 2 are the surface
    /// let mut ts_writer = xdmf_writer
    ///     .write_mesh_with_submeshes(&coords, &connectivity, &cell_types, [
    ///         ("edge", &[0][..]),
    ///         ("surface", &[1, 2][..]),
    ///     ])
    ///     .expect("failed to write mesh");
    /// # // hidden: doctests run in the crate root, so the example cleans up after itself
    /// # std::fs::remove_file("xdmf_write_submeshes.xdmf2").expect("the example writes this file");
    /// ```
    ///
    /// A submesh that is one block of consecutive cells can be given as a [`std::ops::Range`]
    /// rather than an index list -- see [`SubmeshCells`]:
    ///
    /// ```rust
    /// # use xdmf::TimeSeriesWriter;
    /// # let xdmf_writer = TimeSeriesWriter::new("xdmf_write_submesh_ranges", xdmf::DataStorage::AsciiInline)
    /// #     .expect("failed to create XDMF writer");
    /// # let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    /// # let connectivity = [0_u32, 1, 0, 2, 1, 1, 2, 3];
    /// # let cell_types = [
    /// #     xdmf::CellType::Edge,
    /// #     xdmf::CellType::Triangle,
    /// #     xdmf::CellType::Triangle,
    /// # ];
    /// let mut ts_writer = xdmf_writer
    ///     .write_mesh_with_submeshes(&coords, &connectivity, &cell_types, [
    ///         ("edge", 0..1),
    ///         ("surface", 1..3),
    ///     ])
    ///     .expect("failed to write mesh");
    /// # std::fs::remove_file("xdmf_write_submesh_ranges.xdmf2").expect("the example writes this file");
    /// ```
    ///
    /// The submeshes are taken as an iterator for convenience, but are not consumed lazily: the
    /// whole list is read and validated before anything is written.
    pub fn write_mesh_with_submeshes<'c, C, I, N, B>(
        mut self,
        points: &[C],
        connectivity: &[I],
        cell_types: &[CellType],
        submeshes: impl IntoIterator<Item = (N, B)>,
    ) -> Result<TimeSeriesDataWriter>
    where
        C: Coordinate,
        I: ConnectivityIndex,
        N: AsRef<str>,
        B: Into<SubmeshCells<'c>>,
    {
        validate_points_and_cells(points.len(), connectivity, cell_types)?;

        // validate before preparing the mesh, so a bad submesh list is reported without writing
        // the points out first
        let submeshes = prepare_submeshes(submeshes, num_cells(points.len(), cell_types))?;

        let mesh = self.prepare_mesh(points, connectivity, cell_types)?;

        // where each cell's entries start in the prepared connectivity, needed only to cut it up
        // per submesh below
        let offsets = cell_offsets(cell_types, mesh.topology_type, mesh.num_cells);

        let points = C::as_values(points);

        let mut data_items = Vec::with_capacity(2 * submeshes.len());
        let mut grids = Vec::with_capacity(submeshes.len());
        let mut prepared = Vec::with_capacity(submeshes.len());
        // scratch space reused across submeshes, for cutting out coordinates...
        let mut gather_buffers = GatherBuffers::default();
        // ...and for renumbering connectivity
        let mut local_points = LocalPoints::default();

        // written once, for every submesh to select its points out of, where the storage supports
        // selections; otherwise each submesh gets a copy of its share below
        let mesh_coordinates = if self.writer.supports_selections() {
            Some(self.write_mesh_coordinates(&points)?)
        } else {
            None
        };

        for (index, submesh) in submeshes.into_iter().enumerate() {
            // each submesh holds only the points its own cells use, renumbered against them --
            // never the mesh's coordinates whole, which would duplicate every point field per
            // block
            let points_of_submesh = submesh_points(
                &mesh.cells,
                &offsets,
                cell_types,
                mesh.topology_type,
                &submesh.cells,
            )?;
            let (topology_type, nodes_per_element) = submesh_topology(
                cell_types,
                mesh.topology_type,
                mesh.nodes_per_element,
                &submesh.cells,
            );

            let mut cells = extract_connectivity(
                &mesh.cells,
                &offsets,
                cell_types,
                mesh.topology_type,
                topology_type,
                &submesh.cells,
            );
            renumber_connectivity(
                &mut cells,
                cell_types,
                topology_type,
                &submesh.cells,
                &points_of_submesh,
                &mut local_points,
            )?;

            let geometry = match &mesh_coordinates {
                Some(coordinates) => {
                    let selected = selected_coordinates(
                        index,
                        coordinates,
                        &points_of_submesh,
                        mesh.num_points,
                    );
                    let geometry = selected_geometry(&selected);
                    data_items.extend(selected);
                    geometry
                }
                None => {
                    let submesh_coords = match &points_of_submesh {
                        IndexList::Contiguous { start, len } => points.slice(start * 3, len * 3),
                        IndexList::Scattered(indices) => gather_buffers.gather(&points, 3, indices),
                    };

                    let points_item = self.points_data_item(Some(index), &submesh_coords)?;
                    let geometry = geometry(&points_item);
                    data_items.push(points_item);
                    geometry
                }
            };

            let connectivity_item = self.connectivity_data_item(Some(index), &cells)?;

            let topology = Topology {
                topology_type,
                nodes_per_element,
                number_of_elements: submesh.cells.len().to_string(),
                data_item: DataItem::new_reference(&connectivity_item, DOMAIN_DATA_ITEMS),
            };

            grids.push(Grid::new_uniform(&submesh.name, geometry, topology));
            data_items.push(connectivity_item);
            prepared.push(Submesh {
                name: submesh.name,
                cells: submesh.cells,
                points: points_of_submesh,
            });
        }

        let grid = Grid::new_collection("mesh", CollectionType::Spatial, Some(grids));

        self.finish_mesh(grid, data_items, prepared, mesh.num_points, mesh.num_cells)
    }

    /// Assemble the connectivity and decide the topology it is written as.
    fn prepare_mesh<'c, C: Coordinate, I: ConnectivityIndex>(
        &mut self,
        points: &[C],
        connectivity: &'c [I],
        cell_types: &[CellType],
    ) -> Result<PreparedMesh<'c, I>> {
        let num_cells = num_cells(points.len(), cell_types);
        let points = C::as_values(points);
        let num_points = points.len() / 3;

        let (topology_type, cells) = prepare_cells(connectivity, cell_types, num_points)?;

        // checked on the whole array rather than per submesh, since a submesh holds a subset of
        // these same values
        paraview::validate(&I::as_values(&cells), self.writer.format())?;

        // only `Polyvertex`/`Polyline` carry a per-element node count; `topology_type` is
        // non-`Mixed` only when every cell shares one type, so this one value covers the whole
        // mesh and every submesh of it
        let nodes_per_element = (topology_type != TopologyType::Mixed)
            .then(|| poly_cell_points(cell_types.first().copied().unwrap_or(CellType::Vertex)))
            .flatten();

        Ok(PreparedMesh {
            num_points,
            num_cells,
            topology_type,
            nodes_per_element,
            cells,
        })
    }

    /// Write one array of point coordinates and describe it as a named, `Domain`-level `DataItem`,
    /// so cloning the grid per time step repeats a short reference rather than the coordinates.
    fn points_data_item(
        &mut self,
        submesh: Option<usize>,
        points: &Values<'_>,
    ) -> Result<DataItem> {
        let format = self.writer.format();
        let name = match submesh {
            Some(index) => format!("coords_{index}"),
            None => "coords".to_string(),
        };

        Ok(DataItem {
            name: Some(name),
            item_type: None,
            dimensions: Some(Dimensions(vec![points.len() / 3, 3])),
            data: self.writer.write_points(submesh, points)?,
            number_type: Some(points.number_type()),
            precision: Some(points.precision()),
            format: Some(format),
            endian: format.endian(),
            reference: None,
        })
    }

    /// Write the mesh's own coordinates, once, as one array per direction (an `X_Y_Z` geometry
    /// rather than interleaved `XYZ`), so all three of a submesh's selections share one index
    /// list. The `DataItem`s returned are flat and unnamed, since `ParaView` matches the rank of a
    /// selection against the array it selects out of.
    fn write_mesh_coordinates(&mut self, points: &Values<'_>) -> Result<[DataItem; 3]> {
        // its own buffers: this runs once, before the per-submesh gathers start
        let mut buffers = GatherBuffers::default();

        Ok([
            self.write_coordinate_component(points, 0, &mut buffers)?,
            self.write_coordinate_component(points, 1, &mut buffers)?,
            self.write_coordinate_component(points, 2, &mut buffers)?,
        ])
    }

    /// One of those three arrays: every `component`-th value of the interleaved coordinates.
    fn write_coordinate_component(
        &mut self,
        points: &Values<'_>,
        component: usize,
        buffers: &mut GatherBuffers,
    ) -> Result<DataItem> {
        let format = self.writer.format();
        let coordinates = buffers.component(points, 3, component);

        Ok(DataItem {
            name: None,
            item_type: None,
            dimensions: Some(Dimensions(vec![coordinates.len()])),
            data: self.writer.write_point_component(component, &coordinates)?,
            number_type: Some(coordinates.number_type()),
            precision: Some(coordinates.precision()),
            format: Some(format),
            endian: format.endian(),
            reference: None,
        })
    }

    /// Write one connectivity array and describe it as a named, `Domain`-level `DataItem`, so
    /// cloning the grid per time step repeats a short reference rather than the connectivity.
    fn connectivity_data_item<I: ConnectivityIndex>(
        &mut self,
        submesh: Option<usize>,
        cells: &[I],
    ) -> Result<DataItem> {
        let values = I::as_values(cells);
        let format = self.writer.format();

        // numbered rather than named, since the per-step grids resolve it by XPath -- this keeps
        // a caller's (arbitrary, printable) submesh name out of it
        let name = match submesh {
            Some(index) => format!("connectivity_{index}"),
            None => "connectivity".to_string(),
        };

        Ok(DataItem {
            name: Some(name),
            item_type: None,
            dimensions: Some(Dimensions(vec![cells.len()])),
            data: self.writer.write_connectivity(submesh, &values)?,
            number_type: Some(values.number_type()),
            precision: Some(values.precision()),
            format: Some(format),
            endian: format.endian(),
            reference: None,
        })
    }

    /// Record which cells and which points of the mesh each submesh holds, for reading the file
    /// back -- a side channel for a reader, which `ParaView` does not read either.
    ///
    /// One `Information` names, per submesh, either `<start>:<len>` for a contiguous list or the
    /// `DataItem` holding its indices. Only the cell list needs one when the storage selects
    /// points, since the `<Geometry>` already states them; the point arrays are still written
    /// because that geometry references them by name.
    fn write_submesh_index_lists(
        &mut self,
        submeshes: &[Submesh],
        data_items: &mut Vec<DataItem>,
        selections: &mut HashMap<SelectionKey, DataItem>,
    ) -> Result<Vec<Information>> {
        if submeshes.is_empty() {
            return Ok(Vec::new());
        }

        let cells = self.write_submesh_index_list(
            SUBMESH_CELLS,
            submeshes,
            |submesh| &submesh.cells,
            |writer, index, values| writer.write_submesh_cells(index, values),
            data_items,
            selections,
        )?;
        let points = self.write_submesh_index_list(
            SUBMESH_POINTS,
            submeshes,
            |submesh| &submesh.points,
            |writer, index, values| writer.write_submesh_points(index, values),
            data_items,
            selections,
        )?;

        // the point arrays are written either way: where the geometry is a selection they *are*
        // it, and otherwise a reader has nothing else to go on
        if self.writer.supports_selections() {
            return Ok(vec![cells]);
        }

        Ok(vec![cells, points])
    }

    /// One of those two lists, for every submesh: the `Information` naming them and, for each one
    /// that is not a single run, the `DataItem` holding its indices.
    fn write_submesh_index_list(
        &mut self,
        array: &str,
        submeshes: &[Submesh],
        select: fn(&Submesh) -> &IndexList,
        write: fn(&mut dyn DataWriter, usize, &Values<'_>) -> Result<DataContent>,
        data_items: &mut Vec<DataItem>,
        selections: &mut HashMap<SelectionKey, DataItem>,
    ) -> Result<Information> {
        let format = self.writer.format();
        let mut entries = Vec::with_capacity(submeshes.len());

        for (index, submesh) in submeshes.iter().enumerate() {
            let indices = match select(submesh) {
                // a contiguous list needs no array: two numbers say everything it holds
                IndexList::Contiguous { start, len } => {
                    entries.push(format!("{start}:{len}"));
                    continue;
                }
                IndexList::Scattered(indices) => indices,
            };

            let values = index_values(indices)?;
            let name = submesh_index_name(array, index);

            let item = DataItem {
                name: Some(name.clone()),
                item_type: None,
                dimensions: Some(Dimensions(vec![values.len()])),
                // not passed through `paraview::validate`: these are signed and every storage
                // reads them back at the width it declares
                data: write(self.writer.as_mut(), index, &values)?,
                number_type: Some(values.number_type()),
                precision: Some(values.precision()),
                format: Some(format),
                endian: format.endian(),
                reference: None,
            };

            // one index per entity is also what selecting a scalar field takes, so this list
            // doubles as that selector; shapes with more components get their own array, written
            // at the step that first carries one
            selections.insert(
                SelectionKey {
                    submesh: index,
                    point_data: array == SUBMESH_POINTS,
                    components: 1,
                },
                item.clone(),
            );
            data_items.push(item);

            entries.push(name);
        }

        Ok(Information::new(array, entries.join(" ")))
    }

    /// Build the data writer around the finished mesh and write the initial XDMF file.
    fn finish_mesh(
        mut self,
        grid: Grid,
        mut data_items: Vec<DataItem>,
        submeshes: Vec<Submesh>,
        num_points: usize,
        num_cells: usize,
    ) -> Result<TimeSeriesDataWriter> {
        let mut selections = HashMap::new();
        let submesh_lists =
            self.write_submesh_index_lists(&submeshes, &mut data_items, &mut selections)?;

        let mut xdmf = new_document(grid.clone(), data_items, self.writer.data_storage());
        xdmf.information.extend(submesh_lists);

        let mut ts_writer = TimeSeriesDataWriter {
            xdmf_file_name: self.xdmf_file_name,
            writer: self.writer,
            xdmf,
            grid,
            step_times: Vec::new(),
            submeshes,
            selections,
            next_selection_index: 0,
            gather_buffers: GatherBuffers::default(),
            written_times: HashMap::new(),
            num_points,
            num_cells,
        };

        ts_writer.write_xdmf_file()?;

        Ok(ts_writer)
    }
}

/// Append grids to a collection that may not have any yet.
fn append_to_collection(collection: &mut Grid, grids: Vec<Grid>) {
    collection.grids.get_or_insert_with(Vec::new).extend(grids);
}

/// The document a written mesh starts out as: its `<Grid>`, the `DataItem`s that grid references,
/// and the `Information` naming what wrote it. Time steps are added to it as they are completed.
fn new_document(grid: Grid, data_items: Vec<DataItem>, data_storage: DataStorage) -> Xdmf {
    let mut xdmf = Xdmf {
        information: vec![
            Information::new(DATA_STORAGE, format!("{data_storage:?}")),
            Information::new("version", env!("CARGO_PKG_VERSION")),
        ],
        ..Default::default()
    };
    xdmf.domains[0].grids.push(grid);
    xdmf.domains[0].data_items = data_items;

    xdmf
}

/// How many cells a mesh has: one per cell type, or one per point with no cell types at all, since
/// those are written as a polyvertex topology over the points.
fn num_cells(num_coordinates: usize, cell_types: &[CellType]) -> usize {
    if cell_types.is_empty() {
        num_coordinates / 3
    } else {
        cell_types.len()
    }
}

/// `XPath` the per-grid `DataItem` references resolve against.
const DOMAIN_DATA_ITEMS: &str = "/Xdmf/Domain/DataItem";

/// A grid's geometry: the (short) reference to the coordinates it is written with, which each
/// grid needs its own copy of.
fn geometry(points_item: &DataItem) -> Geometry {
    Geometry {
        geometry_type: GeometryType::XYZ,
        data_items: vec![DataItem::new_reference(points_item, DOMAIN_DATA_ITEMS)],
    }
}

/// Writer for time series data in XDMF format, obtained by writing a mesh with `TimeSeriesWriter`.
pub struct TimeSeriesDataWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
    // kept as state and rewritten after every step rather than rebuilt each time, which would
    // deep-copy the whole history -- and for `DataStorage::AsciiInline`, every attribute's data
    xdmf: Xdmf,
    // the mesh's own grid, cloned once per step to carry that step's attributes
    grid: Grid,
    // time of each completed step, in write order -- not kept by `written_times`, nor spelled out
    // in the document once steps are split into one collection per submesh
    step_times: Vec<String>,
    // empty unless the mesh was written with submeshes, which is what makes `grid` a spatial
    // collection instead of a single uniform grid
    submeshes: Vec<Submesh>,
    // index arrays a scattered submesh selects its field share with, keyed by field shape,
    // written once and referenced by every step after; only read when the storage supports
    // selections, and seeded from the start with the mesh's own submesh_points/submesh_cells
    // items, which a scalar field selects with directly
    selections: HashMap<SelectionKey, DataItem>,
    next_selection_index: usize,
    gather_buffers: GatherBuffers,
    // keyed on `f64::to_bits` of the parsed time, so two spellings of the same instant (e.g.
    // "0.1" and "0.10") are recognized as duplicates
    written_times: HashMap<u64, String>,
    num_points: usize,
    num_cells: usize,
}

impl fmt::Debug for TimeSeriesDataWriter {
    /// A summary rather than the whole state, which grows with every step written and, for
    /// [`DataStorage::AsciiInline`], holds the data itself.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeSeriesDataWriter")
            .field("xdmf_file_name", &self.xdmf_file_name)
            .field("data_storage", &self.writer.data_storage())
            .field("num_points", &self.num_points)
            .field("num_cells", &self.num_cells)
            // names only: a scattered submesh holds one index per cell
            .field(
                "submeshes",
                &self
                    .submeshes
                    .iter()
                    .map(|submesh| submesh.name.as_str())
                    .collect::<Vec<_>>(),
            )
            .field("written_times", &self.step_times)
            .finish_non_exhaustive()
    }
}

impl TimeSeriesDataWriter {
    /// The XDMF file this writer writes, same as
    /// [`TimeSeriesWriter::file_name`](crate::TimeSeriesWriter::file_name) reported.
    pub fn file_name(&self) -> &Path {
        &self.xdmf_file_name
    }

    /// Attach a completed step to the document: one `<Grid>` in the file's temporal collection
    /// without submeshes, or with them one temporal collection per submesh, each named after it
    /// and holding only its own grids.
    ///
    /// The nesting is that way round because `ParaView` makes a grid name unique across the whole
    /// document: a submesh named in every step would come back as `name`, `name[1]`, `name[2]`,
    /// ..., losing whatever the user set for that block in the Multi-block Inspector.
    fn push_step(
        &mut self,
        time: &str,
        shared: Vec<attribute::Attribute>,
        per_submesh: Vec<Vec<attribute::Attribute>>,
    ) -> Result<()> {
        let step_grids = self.build_step_grids(time, shared, per_submesh);

        // the first step replaces the mesh's own grid with the collection(s) holding the steps,
        // keeping a mesh-only file a plain `<Grid>` rather than a collection of none
        if self.step_times.is_empty() {
            self.xdmf.domains[0].grids = vec![self.wrap_first_step(step_grids)];
            return Ok(());
        }

        let root = self.xdmf.domains[0]
            .grids
            .first_mut()
            .ok_or(Error::Internal(
                "the document lost the collection holding the time steps",
            ))?;

        if root.collection_type == Some(CollectionType::Temporal) {
            append_to_collection(root, step_grids);
        } else {
            // one temporal collection per submesh, in submesh order, as `wrap_first_step` built them
            for (collection, grid) in root.grids.iter_mut().flatten().zip(step_grids) {
                append_to_collection(collection, vec![grid]);
            }
        }

        Ok(())
    }

    /// The `<Grid>`s one step contributes: one per submesh, or a single one without them.
    fn build_step_grids(
        &self,
        time: &str,
        shared: Vec<attribute::Attribute>,
        per_submesh: Vec<Vec<attribute::Attribute>>,
    ) -> Vec<Grid> {
        if self.submeshes.is_empty() {
            let mut grid = self.grid.clone();
            grid.name = format!("time_series-t{time}");
            grid.time = Some(Time::new(time));
            grid.attributes = Some(shared);

            return vec![grid];
        }

        // `grid` is a spatial collection holding one uniform grid per submesh in submesh order;
        // each step clones those and adds its own data
        self.grid
            .grids
            .iter()
            .flatten()
            .zip(per_submesh)
            .map(|(submesh_grid, cell_attributes)| {
                let mut grid = submesh_grid.clone();
                // the submesh's name belongs to the collection these are gathered into, so this
                // one only has to be unique
                grid.name = format!("{}-t{time}", submesh_grid.name);
                grid.time = Some(Time::new(time));

                let mut attributes = shared.clone();
                attributes.extend(cell_attributes);
                grid.attributes = Some(attributes);

                grid
            })
            .collect()
    }

    /// Wrap the first step's grids in the collection(s) that every later step is appended to.
    fn wrap_first_step(&self, step_grids: Vec<Grid>) -> Grid {
        if self.submeshes.is_empty() {
            return Grid::new_collection("time_series", CollectionType::Temporal, Some(step_grids));
        }

        let collections = self
            .submeshes
            .iter()
            .zip(step_grids)
            .map(|(submesh, grid)| {
                Grid::new_collection(&submesh.name, CollectionType::Temporal, Some(vec![grid]))
            })
            .collect();

        Grid::new_collection("mesh", CollectionType::Spatial, Some(collections))
    }

    /// Write one time step, passing a [`TimeStep`] to `write_step` to write its data into.
    ///
    /// `time` must parse as a finite `f64`; it takes `impl Into<String>` so a number can't be
    /// passed directly and silently reformatted.
    ///
    /// `Ok` adds the step's `<Grid>`; `Err` discards it and its heavy data instead. A step keeps
    /// only the attributes actually written -- only an empty step is rejected.
    ///
    /// ```rust
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("xdmf_write_data", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    ///
    /// // define 3 points and 2 cells (a line and a triangle)
    /// let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// let connectivity = [0, 1, 0, 2, 1]; // line (0,1) and triangle (0,2,1)
    /// let cell_types = [xdmf::CellType::Edge, xdmf::CellType::Triangle];
    ///
    /// // write the mesh
    /// let mut time_series_writer = xdmf_writer
    ///     .write_mesh(&coords, &connectivity, &cell_types)
    ///     .expect("failed to write mesh");
    ///
    /// // each attribute is written as it is passed, so a single buffer can be refilled and
    /// // reused for every field of every time step
    /// let mut point_values = vec![0.0; 9];
    /// let cell_values = vec![0.0, 1.0];
    ///
    /// // write the data for 10 time steps
    /// for i in 0..10 {
    ///     time_series_writer
    ///         .write_time_step(i.to_string(), |step| {
    ///             step.point_data("point_data", xdmf::DataAttribute::Vector, &point_values)?;
    ///
    ///             point_values.fill(i as f64); // refill the same buffer for the next attribute
    ///             step.point_data(
    ///                 "more_point_data",
    ///                 xdmf::DataAttribute::Vector,
    ///                 &point_values,
    ///             )?;
    ///
    ///             step.cell_data("cell_data", xdmf::DataAttribute::Scalar, &cell_values)
    ///         })
    ///         .expect("failed to write time step");
    /// }
    /// # // hidden: doctests run in the crate root, so the example cleans up after itself
    /// # std::fs::remove_file("xdmf_write_data.xdmf2").expect("the example writes this file");
    /// ```
    pub fn write_time_step<F, E>(&mut self, time: impl Into<String>, write_step: F) -> Result<(), E>
    where
        F: FnOnce(&mut TimeStep<'_>) -> Result<(), E>,
        E: From<Error>,
    {
        let time = time.into();
        let parsed_time = time
            .parse::<f64>()
            .map_err(|_parse_error| Error::InvalidTimeStep {
                time: time.clone(),
                reason: "must be a valid float".to_string(),
            })?;

        // `f64::from_str` accepts "NaN"/"inf"/"infinity" and overflows large literals to infinity,
        // none of which name an instant a reader can place on a time line
        if !parsed_time.is_finite() {
            return Err(Error::InvalidTimeStep {
                time,
                reason: "must be a finite float".to_string(),
            }
            .into());
        }

        // zero is normalized, since -0.0 and 0.0 are the same instant with different bit patterns
        let time_bits = if parsed_time == 0.0 { 0.0 } else { parsed_time }.to_bits();

        // keyed on the parsed value rather than the string, so different spellings of the same
        // instant are caught too (e.g. "0.1" == "0.10")
        if let Some(existing) = self.written_times.get(&time_bits) {
            // naming the earlier spelling is only informative if it differs from this one
            let reason = if existing == &time {
                "already written".to_string()
            } else {
                format!("already written (as '{existing}')")
            };
            return Err(Error::InvalidTimeStep { time, reason }.into());
        }

        let mut step = TimeStep {
            per_submesh: vec![Vec::new(); self.submeshes.len()],
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
                // the caller's error is returned even if cleanup also fails, so it is not hidden
                // behind a "could not remove file" report
                let _discard_result = step.discard();
                Err(error)
            }
        }
    }

    fn write_xdmf_file(&mut self) -> Result<()> {
        self.writer.flush()?;

        // written to a temporary file first, then renamed, to avoid access races
        let temp_xdmf_file_name = self.xdmf_file_name.with_extension("xdmf.tmp");

        let mut xdmf_file = BufWriter::new(
            std::fs::File::create(&temp_xdmf_file_name)
                .map_err(io_ctx("creating XDMF file", &temp_xdmf_file_name))?,
        );
        self.xdmf
            .write_to(&mut xdmf_file)
            .map_err(io_ctx("writing XDMF XML", &temp_xdmf_file_name))?;
        xdmf_file
            .flush()
            .map_err(io_ctx("flushing XDMF file", &temp_xdmf_file_name))?;

        std::fs::rename(&temp_xdmf_file_name, &self.xdmf_file_name)
            .map_err(io_ctx("renaming XDMF file", &temp_xdmf_file_name))
    }
}

/// A single time step being written, handed to the closure passed to
/// [`TimeSeriesDataWriter::write_time_step`].
///
/// Each [`point_data`](Self::point_data)/[`cell_data`](Self::cell_data) call writes its heavy
/// data immediately, so one buffer can serve every field of the step.
pub struct TimeStep<'a> {
    writer: &'a mut TimeSeriesDataWriter,
    time: String,
    time_bits: u64,
    attributes: Vec<attribute::Attribute>,
    // per submesh, in writer order; empty without submeshes, where everything goes into
    // `attributes` instead
    per_submesh: Vec<Vec<attribute::Attribute>>,
    // tracked separately since the same name may be used for one of each
    point_names: HashSet<String>,
    cell_names: HashSet<String>,
    // whether `write_data_initialize` has run, deferred to the first attribute so a step that
    // writes nothing leaves no trace at all
    initialized: bool,
    // arrays handed to the backend so far this step, which is what names them; counted per call
    // rather than derived from the attributes, since one cell attribute becomes one array per
    // submesh
    next_array_index: usize,
}

impl fmt::Debug for TimeStep<'_> {
    /// Names only, since the attributes themselves carry the step's data. Sorted, as a
    /// `HashSet`'s iteration order varies between runs.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeStep")
            .field("time", &self.time)
            .field("point_data", &sorted_names(&self.point_names))
            .field("cell_data", &sorted_names(&self.cell_names))
            .finish_non_exhaustive()
    }
}

/// The names in a set, sorted, since a `HashSet`'s own order varies between runs.
fn sorted_names(names: &HashSet<String>) -> Vec<&str> {
    let mut names: Vec<&str> = names.iter().map(String::as_str).collect();
    names.sort_unstable();
    names
}

impl TimeStep<'_> {
    /// Write one point attribute, immediately.
    pub fn point_data<'v>(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        data: impl Into<Values<'v>>,
    ) -> Result<()> {
        self.write_attribute(name, attribute, data.into(), attribute::Center::Node)
    }

    /// Write one cell attribute, immediately.
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
        let is_point_data = center == attribute::Center::Node;
        let (label, num_entities) = if is_point_data {
            (POINT_DATA, self.writer.num_points)
        } else {
            (CELL_DATA, self.writer.num_cells)
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

        // the component count and the total are both products of caller-supplied numbers, so
        // neither is multiplied out unchecked: a wrapping total that lands back on the real array
        // length would be accepted and written as a mesh-sized lie about the data's shape
        let stride = data_attribute
            .size()
            .filter(|size| *size != 0)
            .ok_or_else(|| Error::InvalidData {
                reason: format!(
                    "attribute type {data_attribute:?} of {label} '{name}' has no usable size: \
                     its number of components must be non-zero and must itself fit a usize"
                ),
            })?;
        let exp_size = num_entities
            .checked_mul(stride)
            .ok_or_else(|| Error::InvalidData {
                reason: format!(
                    "attribute type {data_attribute:?} of {label} '{name}' describes \
                     {num_entities} entities of {stride} components each, whose total does not \
                     fit a usize"
                ),
            })?;
        if values.len() != exp_size {
            return Err(Error::InvalidData {
                reason: format!(
                    "size of {label} '{name}' must be {exp_size}, but is {}",
                    values.len()
                ),
            });
        }

        // reject values ParaView would read back as different numbers before anything is written,
        // so a caller mistake leaves no partial output behind
        paraview::validate(&values, self.writer.writer.format())?;

        if !self.initialized {
            self.writer.writer.write_data_initialize(&self.time)?;
            self.initialized = true;
        }

        // without submeshes there is one grid carrying the attribute as-is; with them, every
        // field is cut per submesh -- point data by its point list, cell data by its cell list
        if self.writer.submeshes.is_empty() {
            let index = self.take_array_index();
            let attribute = build_attribute(
                self.writer.writer.as_mut(),
                index,
                name,
                data_attribute,
                &values,
                center,
            )?;
            self.attributes.push(attribute);
        } else {
            self.write_data_per_submesh(name, data_attribute, stride, &values, center)?;
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

    /// The next array number, which the backends name their heavy data by; unique within the
    /// step, since the file name and HDF5 group already carry the time.
    fn take_array_index(&mut self) -> usize {
        let index = self.next_array_index;
        self.next_array_index += 1;
        index
    }

    /// Give every submesh its share of one field. The `Attribute` keeps the caller's name in all
    /// of them, since `ParaView` matches a field across blocks by that name; only the *storage*
    /// name is made unique. A storage that supports selections is written once and referenced;
    /// the rest get a copy of each submesh's share, gathered here.
    fn write_data_per_submesh(
        &mut self,
        name: &str,
        data_attribute: DataAttribute,
        stride: usize,
        values: &Values<'_>,
        center: attribute::Center,
    ) -> Result<()> {
        // written once for the whole mesh only if some submesh can select out of it; otherwise
        // every submesh needs a copy anyway, so that copy is all that gets written
        let point_data = center == attribute::Center::Node;
        let selects = self.writer.writer.supports_selections()
            && self
                .writer
                .submeshes
                .iter()
                .any(|submesh| entities_of(submesh, point_data).is_ascending());

        if selects {
            return self.write_data_selected(name, data_attribute, stride, values, center);
        }

        // split into disjoint borrows: gathering writes into the writer's scratch space while the
        // backend reads it, and both are fields of the same writer
        let TimeSeriesDataWriter {
            writer,
            submeshes,
            gather_buffers,
            ..
        } = &mut *self.writer;
        let next_array_index = &mut self.next_array_index;

        // collected here and only appended once every submesh succeeded, so a failure partway
        // through does not leave some blocks carrying an attribute the rest lack
        let mut written = Vec::with_capacity(submeshes.len());

        for submesh in submeshes {
            let entities = entities_of(submesh, center == attribute::Center::Node);

            let submesh_values = match entities {
                IndexList::Contiguous { start, len } => values.slice(start * stride, len * stride),
                IndexList::Scattered(indices) => gather_buffers.gather(values, stride, indices),
            };

            let index = *next_array_index;
            *next_array_index += 1;

            written.push(build_attribute(
                writer.as_mut(),
                index,
                name,
                data_attribute,
                &submesh_values,
                center,
            )?);
        }

        for (attributes, attribute) in self.per_submesh.iter_mut().zip(written) {
            attributes.push(attribute);
        }

        Ok(())
    }

    /// Write one field once, whole, and give every submesh a `<DataItem>` selecting its own share
    /// of it -- keeping a step's heavy data independent of the number of submeshes and their
    /// overlap. A submesh whose entities are one run selects with a `HyperSlab`, any other with
    /// `Coordinates` through its index array. Only for a storage whose selections `ParaView`
    /// honours, which is the HDF5 ones; see [`DataWriter::supports_selections`].
    fn write_data_selected(
        &mut self,
        name: &str,
        data_attribute: DataAttribute,
        components: usize,
        values: &Values<'_>,
        center: attribute::Center,
    ) -> Result<()> {
        let index = self.take_array_index();
        let mut source =
            build_data_item(self.writer.writer.as_mut(), index, data_attribute, values)?;
        // the shape a submesh's own share has, which the selection carries instead
        let dimensions = source
            .dimensions
            .replace(Dimensions(vec![values.len()]))
            .ok_or(Error::Internal("a written array has no dimensions"))?
            .0;

        let point_data = center == attribute::Center::Node;

        // split into disjoint borrows: writing a selection array borrows the backend and the
        // document while the submesh it is written for is read
        let TimeSeriesDataWriter {
            writer,
            submeshes,
            selections,
            next_selection_index,
            gather_buffers,
            xdmf,
            ..
        } = &mut *self.writer;
        let next_array_index = &mut self.next_array_index;

        // collected first and appended once every submesh succeeded, so a failure partway leaves
        // no submesh carrying an attribute the others lack
        let mut written = Vec::with_capacity(submeshes.len());

        for (submesh_index, submesh) in submeshes.iter().enumerate() {
            let entities = entities_of(submesh, point_data);

            // a submesh whose cells are not ascending gets a copy of its share, as it would from
            // a storage that cannot be selected out of at all
            if let IndexList::Scattered(unordered) = entities
                && !entities.is_ascending()
            {
                let index = *next_array_index;
                *next_array_index += 1;

                let gathered = gather_buffers.gather(values, components, unordered);
                written.push(build_attribute(
                    writer.as_mut(),
                    index,
                    name,
                    data_attribute,
                    &gathered,
                    center,
                )?);
                continue;
            }

            let selector = match entities {
                IndexList::Contiguous { start, len } => hyper_slab(*start, *len, components),
                IndexList::Scattered(_) => {
                    let key = SelectionKey {
                        submesh: submesh_index,
                        point_data,
                        components,
                    };

                    // written at the step that first carries a field of this width, and
                    // referenced by every step after it
                    let item = match selections.get(&key) {
                        Some(item) => item,
                        None => {
                            let item = write_selection_indices(
                                writer.as_mut(),
                                next_selection_index,
                                entities,
                                components,
                            )?;
                            xdmf.domains[0].data_items.push(item.clone());
                            selections.entry(key).or_insert(item)
                        }
                    };

                    DataItem::new_reference(item, DOMAIN_DATA_ITEMS)
                }
            };

            written.push(attribute::Attribute {
                name: name.to_string(),
                attribute_type: data_attribute.into(),
                center,
                data_items: vec![selection(selector, &source, entities.len(), &dimensions)],
            });
        }

        for (attributes, attribute) in self.per_submesh.iter_mut().zip(written) {
            attributes.push(attribute);
        }

        Ok(())
    }

    /// Complete the time step, adding its `<Grid>` to the XDMF file.
    fn finish(self) -> Result<()> {
        // with submeshes every attribute sits in `per_submesh` instead, so such a step is not
        // empty even though `attributes` is
        if self.attributes.is_empty() && self.per_submesh.iter().all(Vec::is_empty) {
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

        if let Err(error) = self.writer.writer.write_data_finalize() {
            // the step is not recorded, so its heavy data is removed rather than left behind
            // with no `<Grid>` referencing it
            let _discard_result = self.discard();
            return Err(error);
        }

        let TimeStep {
            writer,
            time,
            time_bits,
            attributes,
            per_submesh,
            ..
        } = self;

        // built and attached once here, rather than on every rewrite of the file
        writer.push_step(&time, attributes, per_submesh)?;

        writer.step_times.push(time.clone());
        writer.written_times.insert(time_bits, time);

        writer.write_xdmf_file()
    }

    /// Abandon the time step, removing the heavy data already written for it.
    fn discard(self) -> Result<()> {
        // nothing to undo if no attribute initialized the backend -- and `write_data_discard`
        // would reject the unbalanced call
        if !self.initialized {
            return Ok(());
        }

        self.writer.writer.write_data_discard()
    }
}

/// Write one attribute's values and describe them as an XDMF `Attribute`. The backend names the
/// heavy data by `index`; the caller's `name` goes only into the `Attribute`, keeping any name a
/// caller chooses out of the filesystem.
fn build_attribute(
    writer: &mut dyn DataWriter,
    index: usize,
    name: &str,
    data_attribute: DataAttribute,
    values: &Values<'_>,
    center: attribute::Center,
) -> Result<attribute::Attribute> {
    Ok(attribute::Attribute {
        name: name.to_string(),
        attribute_type: data_attribute.into(),
        center,
        data_items: vec![build_data_item(writer, index, data_attribute, values)?],
    })
}

/// Wrap one submesh's selector and the whole field it selects from into the `DataItem` that
/// submesh's `<Attribute>` reads through.
///
/// The source item is repeated in each submesh rather than referenced, since the path into the
/// heavy storage is shorter than a reference to it, and is written flat since `ParaView` matches
/// the rank of a selection against the dataset it reads.
fn selection(
    selector: DataItem,
    source: &DataItem,
    num_entities: usize,
    dimensions: &[usize],
) -> DataItem {
    let item_type = if selector.reference.is_some() {
        ItemType::Coordinates
    } else {
        ItemType::HyperSlab
    };

    // the submesh's share has the field's shape with its own entity count in front
    let mut selected = Vec::with_capacity(dimensions.len());
    selected.push(num_entities);
    selected.extend_from_slice(&dimensions[1..]);

    DataItem {
        name: None,
        item_type: Some(item_type),
        dimensions: Some(Dimensions(selected)),
        number_type: source.number_type,
        // no `Format`: the nested source says where the values are, this item only says which
        format: None,
        precision: source.precision,
        endian: None,
        data: vec![selector, source.clone()].into(),
        reference: None,
    }
}

/// The selector of a submesh whose entities are one run: a start, a stride and a count, small
/// enough to go into the XML itself rather than into an array.
///
/// Counted in values rather than entities, since the array it selects out of is written flat.
fn hyper_slab(start: usize, len: usize, components: usize) -> DataItem {
    DataItem {
        name: None,
        item_type: None,
        dimensions: Some(Dimensions(vec![3])),
        number_type: Some(NumberType::Int),
        format: Some(Format::XML),
        precision: Some(4),
        endian: None,
        data: format!("{} 1 {}", start * components, len * components).into(),
        reference: None,
    }
}

/// Write the index array a scattered submesh selects fields of one width with, and describe it as
/// a named, `Domain`-level `DataItem` for every step to reference.
///
/// One index per value the submesh holds, naming its position in the whole field. A scalar field
/// reuses the submesh's own index list instead.
fn write_selection_indices(
    writer: &mut dyn DataWriter,
    next_index: &mut usize,
    entities: &IndexList,
    components: usize,
) -> Result<DataItem> {
    let mut indices = Vec::with_capacity(entities.len() * components);

    for entity in entities.iter() {
        indices.extend(entity * components..(entity + 1) * components);
    }

    let values = index_values(&indices)?;
    let index = *next_index;
    *next_index += 1;

    let format = writer.format();

    Ok(DataItem {
        name: Some(format!("{SELECTIONS}_{index}")),
        item_type: None,
        dimensions: Some(Dimensions(vec![values.len()])),
        data: writer.write_selection(index, &values)?,
        number_type: Some(values.number_type()),
        precision: Some(values.precision()),
        format: Some(format),
        endian: format.endian(),
        reference: None,
    })
}

/// Write one array to the heavy storage and describe it as an unnamed `DataItem`.
fn build_data_item(
    writer: &mut dyn DataWriter,
    index: usize,
    data_attribute: DataAttribute,
    values: &Values<'_>,
) -> Result<DataItem> {
    let format = writer.format();

    Ok(DataItem {
        name: None,
        item_type: None,
        dimensions: Some(values.dimensions(data_attribute)),
        number_type: Some(values.number_type()),
        format: Some(format),
        precision: Some(values.precision()),
        endian: format.endian(),
        data: writer.write_data(index, values)?,
        reference: None,
    })
}

// plain-string labels for the data category in error messages, since heavy-data naming doesn't
// go through `attribute::Center` either

/// Label for point data in user-facing error messages, named after [`TimeStep::point_data`].
const POINT_DATA: &str = "point_data";
/// Label for cell data in user-facing error messages, named after [`TimeStep::cell_data`].
const CELL_DATA: &str = "cell_data";

/// Whether a name a caller chose -- for a data field or for a submesh -- can be written.
///
/// Only non-blank and printable is required, since a name is only ever light data and the heavy
/// data is numbered -- so `/`, `:`, `%`, `*` and quotes stay allowed, which matters because solver
/// field names carry them (e.g. FDS's `Quantity('SOOT DENSITY')`). Control characters are
/// rejected rather than escaped, since XML 1.0 cannot represent most of them at all.
fn is_valid_data_name(name: &str) -> bool {
    // blank rather than merely empty: a whitespace-only name labels the array with nothing at all
    if name.trim().is_empty() {
        return false;
    }

    !name.chars().any(char::is_control)
}

/// Characters not allowed in the final path component of an XDMF file name.
const INVALID_FILE_NAME_CHARS: [char; 8] = ['?', '\0', ':', '*', '"', '<', '>', '|'];

/// Validate the file name for the XDMF file.
fn validate_file_name(file_name: &Path) -> Result<()> {
    // only the final path component is validated -- parent directories are not under our
    // control and may legitimately contain characters such as ':' (e.g. Windows drive letters)
    let Some(name) = file_name.file_name() else {
        // e.g. an empty path, or one ending in ".."
        return Err(Error::InvalidFileName {
            path: file_name.to_path_buf(),
            reason: "path has no file name component".to_string(),
        });
    };

    let Some(name) = name.to_str() else {
        return Err(Error::InvalidFileName {
            path: file_name.to_path_buf(),
            reason: "file name component is not valid UTF-8".to_string(),
        });
    };

    if name.chars().any(|c| INVALID_FILE_NAME_CHARS.contains(&c)) {
        return Err(Error::InvalidFileName {
            path: file_name.to_path_buf(),
            reason: format!(
                "file name component must not contain any of the following characters: \
                 {INVALID_FILE_NAME_CHARS:?}"
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DataAttribute,
        xdmf_elements::{
            data_item::{DataContent, Format, NumberType},
            grid::Grid,
        },
    };

    fn with_version(expected: &str) -> String {
        expected.replace("$VERSION", env!("CARGO_PKG_VERSION"))
    }

    // The document a hand-built test writer starts from: what `finish_mesh` makes of a mesh grid
    // before any step has been written. The test backends all report `AsciiInline`.
    fn document_for(grid: &Grid) -> Xdmf {
        new_document(grid.clone(), Vec::new(), DataStorage::AsciiInline)
    }

    // The `<Grid>`s the step written last contributed: one per submesh, or a single one for a
    // mesh written without them. Each lives at the end of the collection it was appended to.
    fn last_step_grids(writer: &TimeSeriesDataWriter) -> Vec<&Grid> {
        let root = writer.xdmf.domains[0]
            .grids
            .first()
            .expect("a step was written");

        let collections: Vec<&Grid> = if root.collection_type == Some(CollectionType::Temporal) {
            vec![root]
        } else {
            root.grids.iter().flatten().collect()
        };

        collections
            .into_iter()
            .map(|collection| {
                collection
                    .grids
                    .iter()
                    .flatten()
                    .next_back()
                    .expect("a step was written")
            })
            .collect()
    }

    // The names of a grid's attributes, in order.
    fn attribute_names(grid: &Grid) -> Vec<&str> {
        grid.attributes
            .iter()
            .flatten()
            .map(|attribute| attribute.name.as_str())
            .collect()
    }

    #[test]
    fn time_series_writer_create_folder() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let subfolder = Path::new("out/xdmf"); // deliberately not creating this folder
        let xdmf_folder = tmp_dir.path().join(subfolder);
        let xdmf_file_path = xdmf_folder.join("test_output");

        assert!(!xdmf_folder.exists());

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        assert!(xdmf_folder.exists());
        assert_eq!(
            writer.xdmf_file_name,
            xdmf_file_path.with_extension("xdmf2")
        );
    }

    #[test]
    fn mpi_safe_create_dir_all_works() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let dirs_to_create = tmp_dir.path().join("out/xdmf/test/folder/random/testing");

        // Try to create dirs from 100 threads concurrently
        let handles: Vec<_> = (0..100)
            .map(|_| {
                std::thread::spawn({
                    let dir_thread_local = dirs_to_create.clone();
                    move || mpi_safe_create_dir_all(dir_thread_local).unwrap()
                })
            })
            .collect();

        // join threads, will propagate errors if any
        for handle in handles {
            handle.join().unwrap();
        }

        // Check that the directory was created
        assert!(dirs_to_create.exists());
    }

    #[test]
    fn test_validate_data() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 10;

        // write mesh
        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let values = vec![5.0; NUM_POINTS];

        // Valid time step
        writer
            .write_time_step("0.1", |step| {
                step.point_data("point_data1", DataAttribute::Scalar, &values)
            })
            .unwrap();

        // no data at all provided
        let res = writer.write_time_step("1.0", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "1.0" && reason.contains("no data written")
        );

        // Invalid time step (already exists)
        let res = writer.write_time_step("0.1", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "0.1" && reason == "already written"
        );

        // Invalid time step (not a float)
        let res = writer.write_time_step("invalid_time", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "invalid_time" && reason.contains("must be a valid float")
        );

        // Invalid time step (empty)
        let res = writer.write_time_step("", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time.is_empty() && reason.contains("must be a valid float")
        );
    }

    #[test]
    fn write_time_step_rejects_non_finite_times() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("non_finite_times.xdmf2"), None, None);

        // all of these parse as a float, the last one by overflowing to infinity
        for time in ["NaN", "inf", "-infinity", "1e400"] {
            let res = writer.write_time_step(time, |step| {
                step.point_data("data", DataAttribute::Scalar, vec![0.0; 0])
            });
            std::assert_matches!(
                res.unwrap_err(),
                Error::InvalidTimeStep { time: rejected, reason }
                    if rejected == time && reason == "must be a finite float"
            );
        }

        assert!(writer.step_times.is_empty());
    }

    #[test]
    fn write_time_step_treats_negative_zero_as_the_time_already_written() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("negative_zero.xdmf2"), None, None);

        writer
            .write_time_step("0.0", |step| {
                step.point_data("data", DataAttribute::Scalar, vec![0.0; 0])
            })
            .unwrap();

        // -0.0 is the same instant as 0.0, despite the two having different bit patterns
        let res = writer.write_time_step("-0.0", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "-0.0" && reason == "already written (as '0.0')"
        );
    }

    #[test]
    fn write_time_step_erroring_out_discards_the_data_it_already_wrote() {
        // A caller's own error type, as `write_time_step`'s closure may use: this crate's errors
        // convert into it, and it comes back out of `write_time_step` unchanged.
        #[derive(Debug)]
        enum CallerError {
            ChangedItsMind,
            Xdmf(Error),
        }

        impl From<Error> for CallerError {
            fn from(error: Error) -> Self {
                Self::Xdmf(error)
            }
        }

        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        // Ascii, so the heavy data of the abandoned step is a file that can be checked for
        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::Ascii).unwrap();

        const NUM_POINTS: usize = 10;

        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let values = vec![5.0; NUM_POINTS];

        // one attribute written, then the closure gives up -- with an error of its own, which
        // comes back unchanged rather than squeezed into this crate's `Error`
        let res = writer.write_time_step("0.1", |step| {
            step.point_data("abandoned", DataAttribute::Scalar, &values)?;
            Err(CallerError::ChangedItsMind)
        });
        std::assert_matches!(res.unwrap_err(), CallerError::ChangedItsMind);

        // the heavy data of the abandoned attribute was removed again, rather than being left
        // behind with nothing in the XDMF file referencing it
        let txt_dir = xdmf_file_path.with_extension("txt");
        assert!(!txt_dir.join("data_t_0.1_0.txt").exists());

        // the backing writer is not poisoned: the same time can be used again (the abandoned
        // step never consumed it), and so can a different one
        writer
            .write_time_step("0.1", |step| {
                step.point_data("kept", DataAttribute::Scalar, &values)
            })
            .unwrap();
        writer
            .write_time_step("0.2", |step| {
                step.point_data("kept", DataAttribute::Scalar, &values)
            })
            .unwrap();

        // a rejected attribute reaches the closure's own error type through `From`, so `?` is
        // all it takes to mix the caller's errors with this crate's
        let res = writer.write_time_step("0.3", |step| {
            step.point_data("wrong_size", DataAttribute::Scalar, &[1.0])?;
            Err(CallerError::ChangedItsMind)
        });
        std::assert_matches!(
            res.unwrap_err(),
            CallerError::Xdmf(Error::InvalidData { reason })
                if reason == "size of point_data 'wrong_size' must be 10, but is 1"
        );

        // the abandoned step left no trace in the light data either
        let xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
        assert!(!xdmf.contains("abandoned"));
        assert_eq!(xdmf.matches("<Grid Name=\"time_series-t").count(), 2);
    }

    #[test]
    fn write_time_step_erroring_out_before_writing_anything_needs_no_cleanup() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::Ascii).unwrap();

        let mut writer = writer
            .write_mesh(&[0.0; 3], &[0], &[CellType::Vertex])
            .unwrap();

        // the closure fails before any attribute is written, so `write_data_initialize` never
        // ran -- discarding must not report the unbalanced call as an internal error
        let res = writer.write_time_step("0.1", |_step| {
            Err(Error::InvalidData {
                reason: "nothing to write".to_string(),
            })
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidData { reason } if reason == "nothing to write"
        );

        writer
            .write_time_step("0.1", |step| {
                step.point_data("data", DataAttribute::Scalar, &[1.0])
            })
            .unwrap();
    }

    #[test]
    fn write_time_step_mixes_value_types_within_one_step() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 4;

        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 1, 2, 3],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        // f64 and u64 attributes in the same step -- the case a single generic type parameter
        // on the old list-shaped `write_data` could not express
        let floats = vec![1.5; NUM_POINTS];
        let ids: Vec<u64> = (0..NUM_POINTS as u64).collect();

        writer
            .write_time_step("0.0", |step| {
                step.point_data("floats", DataAttribute::Scalar, &floats)?;
                step.cell_data("ids", DataAttribute::Scalar, &ids)
            })
            .unwrap();

        let xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
        assert!(xdmf.contains(r#"Name="floats" AttributeType="Scalar" Center="Node""#));
        assert!(xdmf.contains(r#"Name="ids" AttributeType="Scalar" Center="Cell""#));
        assert!(xdmf.contains(r#"NumberType="UInt""#));
    }

    #[test]
    fn write_time_step_reuses_a_single_buffer_across_attributes() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 3;

        let mut writer = writer
            .write_mesh(&[0.0; NUM_POINTS * 3], &[0, 1, 2], &[CellType::Vertex; 3])
            .unwrap();

        // the point of the whole builder: one allocation, refilled between attributes
        let mut buf = vec![0.0; NUM_POINTS];

        writer
            .write_time_step("0.0", |step| {
                buf.fill(1.0);
                step.point_data("first", DataAttribute::Scalar, &buf)?;

                buf.fill(2.0);
                step.point_data("second", DataAttribute::Scalar, &buf)
            })
            .unwrap();

        let xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
        let one = "1e0";
        let two = "2e0";
        assert!(xdmf.contains(&format!(">{one} {one} {one}<")));
        assert!(xdmf.contains(&format!(">{two} {two} {two}<")));
    }

    #[test]
    fn test_validate_data_dedup_is_numeric_not_textual() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 10;

        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let values = vec![5.0; NUM_POINTS];
        let write_step = |writer: &mut TimeSeriesDataWriter, time: &str| -> Result<()> {
            writer.write_time_step(time, |step| {
                step.point_data("point_data1", DataAttribute::Scalar, &values)
            })
        };

        write_step(&mut writer, "0.1").unwrap();

        // a different spelling of the same numeric value is still a duplicate
        let res = write_step(&mut writer, "0.10");
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "0.10" && reason == "already written (as '0.1')"
        );

        // a genuinely different value is accepted
        write_step(&mut writer, "0.2").unwrap();
    }

    #[test]
    fn test_validate_data_duplicate_names() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 10;

        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let values = vec![5.0; NUM_POINTS];

        let res = writer.write_time_step("0.0", |step| {
            step.point_data("duplicate", DataAttribute::Scalar, &values)?;
            step.point_data("duplicate", DataAttribute::Scalar, &values)
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidData { reason }
                if reason.contains("name 'duplicate' of point_data is used more than once")
        );

        // the same name for point_data and cell_data is allowed, they are separate entities
        let cell_values = vec![5.0; 4];
        writer
            .write_time_step("0.0", |step| {
                step.point_data("data", DataAttribute::Scalar, &values)?;
                step.cell_data("data", DataAttribute::Scalar, &cell_values)
            })
            .unwrap();
    }

    #[test]
    fn test_validate_data_wrong_point_data_sizes() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 10;

        // write mesh
        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let mut err_for = |name: &str, attribute: DataAttribute, len: usize| -> Error {
            writer
                .write_time_step("0.0", |step| {
                    step.point_data(name, attribute, vec![5.0; len])
                })
                .unwrap_err()
        };

        std::assert_matches!(
            err_for("point_data_sca", DataAttribute::Scalar, NUM_POINTS - 1),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_sca' must be 10, but is 9"
        );
        std::assert_matches!(
            err_for("point_data_vec", DataAttribute::Vector, NUM_POINTS * 2),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_vec' must be 30, but is 20"
        );
        std::assert_matches!(
            err_for("point_data_ten", DataAttribute::Tensor, NUM_POINTS * 3),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_ten' must be 90, but is 30"
        );
        std::assert_matches!(
            err_for("point_data_ten6", DataAttribute::Tensor6, NUM_POINTS * 3),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_ten6' must be 60, but is 30"
        );
        std::assert_matches!(
            err_for(
                "point_data_mat",
                DataAttribute::Matrix(2, 1),
                NUM_POINTS * 3 - 1
            ),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_mat' must be 20, but is 29"
        );
    }

    #[test]
    fn test_validate_data_wrong_cell_data_sizes() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_CELLS: usize = 4;

        // write mesh
        let mut writer = writer
            .write_mesh(
                &[0.0; 10 * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; NUM_CELLS],
            )
            .unwrap();

        let mut err_for = |name: &str, attribute: DataAttribute, len: usize| -> Error {
            writer
                .write_time_step("0.0", |step| {
                    step.cell_data(name, attribute, vec![5.0; len])
                })
                .unwrap_err()
        };

        std::assert_matches!(
            err_for("cell_data_sca", DataAttribute::Scalar, NUM_CELLS - 1),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_sca' must be 4, but is 3"
        );
        std::assert_matches!(
            err_for("cell_data_vec", DataAttribute::Vector, NUM_CELLS * 2),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_vec' must be 12, but is 8"
        );
        std::assert_matches!(
            err_for("cell_data_ten", DataAttribute::Tensor, NUM_CELLS * 3),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_ten' must be 36, but is 12"
        );
        std::assert_matches!(
            err_for("cell_data_ten6", DataAttribute::Tensor6, NUM_CELLS * 3),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_ten6' must be 24, but is 12"
        );
        std::assert_matches!(
            err_for(
                "cell_data_mat",
                DataAttribute::Matrix(2, 1),
                NUM_CELLS * 3 - 1
            ),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_mat' must be 8, but is 11"
        );
    }

    #[test]
    fn test_validate_data_names() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();
        let mut writer = writer
            .write_mesh(&[0.0; 3], &[0], &[CellType::Vertex])
            .unwrap();

        let res = writer.write_time_step("0.0", |step| {
            step.cell_data("cell_data_ten", DataAttribute::Scalar, vec![0.0; 1])?;
            // Only control characters are rejected now that no name reaches the filesystem --
            // brackets, slashes and spaces all became legal, so this needs a tab.
            step.point_data("cell\u{9}data_ten", DataAttribute::Scalar, vec![0.0; 1])
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidData { reason }
                if reason.contains("of point_data is not valid")
                    && reason.contains("control characters")
        );
    }

    #[test]
    fn test_is_valid_data_name() {
        assert!(is_valid_data_name("valid_name"));
        assert!(is_valid_data_name("valid-name"));
        assert!(is_valid_data_name("valid_name_123"));

        // names as they occur in real solver output
        assert!(is_valid_data_name("Quantity('SOOT DENSITY')"));
        assert!(is_valid_data_name("U.component_0"));
        assert!(is_valid_data_name("stress [Pa]"));
        assert!(is_valid_data_name("T_max, avg"));
        assert!(is_valid_data_name("\u{394}\u{3b8}")); // non-ASCII

        // accepted because a name is only ever light data, reaching an XML attribute and nothing
        // else -- these would all have had to be rejected if the heavy data were named after it
        assert!(is_valid_data_name("a/b"));
        assert!(is_valid_data_name("a\\b"));
        assert!(is_valid_data_name("a:b"));
        assert!(is_valid_data_name("a#b"));
        assert!(is_valid_data_name("a%b"));
        assert!(is_valid_data_name("a*b"));
        assert!(is_valid_data_name("a?b"));
        assert!(is_valid_data_name("a\"b"));
        assert!(is_valid_data_name("a<b>c"));
        assert!(is_valid_data_name("a|b"));

        // surrounding whitespace is kept, it still leaves something to read
        assert!(is_valid_data_name(" padded name "));

        // only a blank name and the characters XML cannot represent at all
        assert!(!is_valid_data_name(""));
        assert!(!is_valid_data_name(" ")); // blank
        assert!(!is_valid_data_name("   ")); // blank
        assert!(!is_valid_data_name("\u{a0}")); // blank, non-ASCII whitespace
        assert!(!is_valid_data_name("invalid\0name")); // null-char
        assert!(!is_valid_data_name("invalid\nname")); // control character
        assert!(!is_valid_data_name("invalid\tname")); // control character
        assert!(!is_valid_data_name("invalid\u{7f}name")); // delete
    }

    #[test]
    fn test_validate_file_name() {
        validate_file_name(Path::new("asdf.txt")).unwrap();
        validate_file_name(Path::new("valid-name.txt")).unwrap();
        validate_file_name(Path::new("valid_name.txt")).unwrap();
        validate_file_name(Path::new("valid_name-123.txt")).unwrap();

        // only the final component is validated, a parent may legitimately contain ':'
        validate_file_name(Path::new("C:/some:dir/valid_name.txt")).unwrap();

        let res = validate_file_name(Path::new("valid_name:123.txt"));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidFileName { path, reason }
                if path == Path::new("valid_name:123.txt")
                    && reason.contains("file name component must not contain any of")
        );

        let res = validate_file_name(Path::new(""));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidFileName { path, reason }
                if path == Path::new("") && reason == "path has no file name component"
        );
    }

    fn dummy_geometry() -> Geometry {
        Geometry {
            geometry_type: GeometryType::XYZ,
            data_items: vec![DataItem {
                dimensions: Some(Dimensions(vec![5, 3])),
                data: "0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0".into(),
                number_type: Some(NumberType::Float),
                ..Default::default()
            }],
        }
    }

    fn dummy_topology() -> Topology {
        Topology {
            topology_type: TopologyType::Triangle,
            nodes_per_element: None,
            number_of_elements: "2".into(),
            data_item: DataItem {
                dimensions: Some(Dimensions(vec![6])),
                number_type: Some(NumberType::Int),
                data: "0 1 2 2 3 4".into(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_write_data_preserve_order() {
        struct DummyWriter;

        impl DataWriter for DummyWriter {
            fn format(&self) -> Format {
                Format::XML
            }

            fn data_storage(&self) -> DataStorage {
                DataStorage::AsciiInline
            }

            fn write_points(
                &mut self,
                _submesh: Option<usize>,
                _points: &Values<'_>,
            ) -> Result<DataContent> {
                Ok(DataContent::Raw("points".to_string()))
            }

            fn write_connectivity(
                &mut self,
                _submesh: Option<usize>,
                _cells: &Values<'_>,
            ) -> Result<DataContent> {
                Ok(DataContent::Raw("cells".to_string()))
            }

            fn write_submesh_cells(
                &mut self,
                submesh: usize,
                _cells: &Values<'_>,
            ) -> Result<DataContent> {
                Ok(DataContent::Raw(format!("submesh_cells_{submesh}")))
            }

            fn write_submesh_points(
                &mut self,
                submesh: usize,
                _points: &Values<'_>,
            ) -> Result<DataContent> {
                Ok(DataContent::Raw(format!("submesh_points_{submesh}")))
            }

            fn write_data(&mut self, index: usize, _data: &Values<'_>) -> Result<DataContent> {
                Ok(DataContent::Raw(format!("data_for_{index}")))
            }
        }

        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_write_data_preserve_order.xdmf2");

        let grid = Grid::new_uniform("test", dummy_geometry(), dummy_topology());
        let mut writer = TimeSeriesDataWriter {
            xdmf_file_name: xdmf_file_path.clone(),
            writer: Box::new(DummyWriter),
            xdmf: document_for(&grid),
            grid,
            step_times: Vec::new(),
            num_points: 0,
            num_cells: 0,
            submeshes: Vec::new(),
            selections: HashMap::new(),
            next_selection_index: 0,
            gather_buffers: GatherBuffers::default(),
            written_times: HashMap::new(),
        };

        let write_step = |writer: &mut TimeSeriesDataWriter, time: &str| {
            writer
                .write_time_step(time, |step| {
                    step.point_data("scalar_data", DataAttribute::Scalar, vec![0.0; 0])
                })
                .unwrap();
        };

        write_step(&mut writer, "0.0");
        write_step(&mut writer, "1.0");
        write_step(&mut writer, "2.0");
        write_step(&mut writer, "10.0");

        // Check that the data are in the correct order

        let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">
            <Grid Name="time_series-t0.0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="2">
                    <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                </Topology>
                <Time Value="0.0"/>
                <Attribute Name="scalar_data" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_0</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t1.0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="2">
                    <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                </Topology>
                <Time Value="1.0"/>
                <Attribute Name="scalar_data" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_0</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t2.0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="2">
                    <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                </Topology>
                <Time Value="2.0"/>
                <Attribute Name="scalar_data" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_0</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t10.0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="2">
                    <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                </Topology>
                <Time Value="10.0"/>
                <Attribute Name="scalar_data" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_0</DataItem>
                </Attribute>
            </Grid>
        </Grid>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="$VERSION"/>
</Xdmf>"#;

        let xdmf_file = xdmf_file_path.with_extension("xdmf2");
        let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

        // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
        // std::fs::copy(xdmf_file, "time_series_writer_only_mesh.xdmf").unwrap();

        pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
    }

    // a backend that fails on demand, to exercise a time step's failure paths without a real
    // storage format: writing the array numbered `fail_array` fails, as does finalizing the time
    // given as `fail_finalize_at` -- keyed on the array index, which is also how a failure
    // partway through a submesh loop is reached
    struct FlakyWriter {
        write_time: Option<String>,
        fail_finalize_at: Option<&'static str>,
        fail_array: Option<usize>,
    }

    impl DataWriter for FlakyWriter {
        fn format(&self) -> Format {
            Format::XML
        }

        fn data_storage(&self) -> DataStorage {
            DataStorage::AsciiInline
        }

        fn write_points(
            &mut self,
            _submesh: Option<usize>,
            _points: &Values<'_>,
        ) -> Result<DataContent> {
            Ok(DataContent::Raw("points".to_string()))
        }

        fn write_connectivity(
            &mut self,
            _submesh: Option<usize>,
            _cells: &Values<'_>,
        ) -> Result<DataContent> {
            Ok(DataContent::Raw("cells".to_string()))
        }

        fn write_submesh_cells(
            &mut self,
            submesh: usize,
            _cells: &Values<'_>,
        ) -> Result<DataContent> {
            Ok(DataContent::Raw(format!("submesh_cells_{submesh}")))
        }

        fn write_submesh_points(
            &mut self,
            submesh: usize,
            _points: &Values<'_>,
        ) -> Result<DataContent> {
            Ok(DataContent::Raw(format!("submesh_points_{submesh}")))
        }

        fn write_data(&mut self, index: usize, _data: &Values<'_>) -> Result<DataContent> {
            if self.fail_array == Some(index) {
                // cleared as it fires, so retrying the same array succeeds -- which is what the
                // discard-and-retry tests below need after the failure they provoke
                self.fail_array = None;
                return Err(Error::Io {
                    operation: "writing data (simulated)",
                    path: PathBuf::from("boom"),
                    source: std::io::Error::other("simulated mid-write failure"),
                });
            }
            Ok(DataContent::Raw(format!("data_for_{index}")))
        }

        fn write_data_initialize(&mut self, time: &str) -> Result<()> {
            if self.write_time.is_some() {
                return Err(Error::Internal("writing data was already initialized"));
            }
            self.write_time = Some(time.to_string());
            Ok(())
        }

        fn write_data_finalize(&mut self) -> Result<()> {
            let Some(time) = self.write_time.as_deref() else {
                return Err(Error::Internal("writing data was not initialized"));
            };

            // the step stays open on this failure, like a backend that could not complete it
            if self.fail_finalize_at == Some(time) {
                return Err(Error::Io {
                    operation: "finalizing data (simulated)",
                    path: PathBuf::from("finalize"),
                    source: std::io::Error::other("simulated finalize failure"),
                });
            }

            self.write_time = None;
            Ok(())
        }

        fn write_data_discard(&mut self) -> Result<()> {
            // mirrors the real backends: the step is dropped and the writer is ready for the
            // next one (this backend has no heavy data to remove)
            if self.write_time.is_none() {
                return Err(Error::Internal("writing data was not initialized"));
            }
            self.write_time = None;
            Ok(())
        }
    }

    // A `TimeSeriesDataWriter` on top of `FlakyWriter`, assembled directly rather than through
    // `TimeSeriesWriter::write_mesh`, since only the time-step handling is under test.
    fn flaky_writer(
        xdmf_file_name: PathBuf,
        fail_finalize_at: Option<&'static str>,
        fail_array: Option<usize>,
    ) -> TimeSeriesDataWriter {
        let grid = Grid::new_uniform("test", dummy_geometry(), dummy_topology());

        TimeSeriesDataWriter {
            xdmf_file_name,
            writer: Box::new(FlakyWriter {
                write_time: None,
                fail_finalize_at,
                fail_array,
            }),
            xdmf: document_for(&grid),
            grid,
            step_times: Vec::new(),
            num_points: 0,
            num_cells: 0,
            submeshes: Vec::new(),
            selections: HashMap::new(),
            next_selection_index: 0,
            gather_buffers: GatherBuffers::default(),
            written_times: HashMap::new(),
        }
    }

    // the same backend behind three single-cell submeshes, failing on array 1 -- the second
    // submesh's share of the first field, so the first submesh's share already went out
    fn flaky_writer_with_submeshes(xdmf_file_name: PathBuf) -> TimeSeriesDataWriter {
        let mut writer = flaky_writer(xdmf_file_name, None, Some(1));

        writer.num_cells = 3;
        writer.submeshes = ["first", "mid", "last"]
            .into_iter()
            .enumerate()
            .map(|(index, name)| Submesh {
                name: name.to_string(),
                cells: IndexList::Contiguous {
                    start: index,
                    len: 1,
                },
                // one point per submesh as well, so a point field is cut the same way a cell
                // field is -- what this fixture exercises is the failure partway through
                points: IndexList::Contiguous {
                    start: index,
                    len: 1,
                },
            })
            .collect();
        writer.grid = Grid::new_collection(
            "mesh",
            CollectionType::Spatial,
            Some(
                writer
                    .submeshes
                    .iter()
                    .map(|submesh| {
                        Grid::new_uniform(&submesh.name, dummy_geometry(), dummy_topology())
                    })
                    .collect(),
            ),
        );

        writer
    }

    #[test]
    fn a_failure_partway_through_the_submeshes_writes_no_attribute_at_all() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer_with_submeshes(tmp_dir.path().join("partial.xdmf2"));

        // the closure swallows the failure; every submesh's share is discarded rather than the
        // earlier ones being kept, so the step holds no data and is rejected as empty
        let result = writer.write_time_step("0.0", |step| {
            let _swallowed = step.cell_data("boom", DataAttribute::Scalar, &[1.0, 2.0, 3.0]);
            Ok::<(), Error>(())
        });

        std::assert_matches!(
            result.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "0.0" && reason.contains("no data written")
        );
        assert!(writer.step_times.is_empty());
    }

    #[test]
    fn a_failure_partway_through_the_submeshes_leaves_later_attributes_aligned() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer_with_submeshes(tmp_dir.path().join("aligned.xdmf2"));

        // "boom" fails on the second of three submeshes; the field written afterwards must still
        // land exactly once on every block, not twice on the first from a leftover failed share
        writer
            .write_time_step("0.0", |step| {
                let _swallowed = step.cell_data("boom", DataAttribute::Scalar, &[1.0, 2.0, 3.0]);
                step.cell_data("fine", DataAttribute::Scalar, &[1.0, 2.0, 3.0])
            })
            .unwrap();

        let sub_grids = last_step_grids(&writer);
        assert_eq!(sub_grids.len(), 3);
        for sub_grid in sub_grids {
            assert_eq!(attribute_names(sub_grid), ["fine"]);
        }
    }

    #[test]
    fn write_data_survives_a_mid_write_failure() {
        // fails while writing one attribute after an earlier one already succeeded -- a
        // backend-agnostic regression test for the discard-on-error handling
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(
            tmp_dir.path().join("mid_write_failure.xdmf2"),
            None,
            Some(1),
        );

        // "ok" succeeds before "boom" fails, so this genuinely fails partway through the step
        let res = writer.write_time_step("0.0", |step| {
            step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])?;
            step.point_data("boom", DataAttribute::Scalar, vec![0.0; 0])
        });
        std::assert_matches!(res.unwrap_err(), Error::Io { .. });

        // the failed step must not have consumed the time slot, nor left the backend poisoned
        writer
            .write_time_step("0.0", |step| {
                step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])
            })
            .unwrap();
    }

    #[test]
    fn write_time_step_discards_when_the_closure_swallows_an_attribute_error() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("swallowed_error.xdmf2"), None, Some(0));

        // the closure ignores "boom"'s failure and returns `Ok`, leaving an empty step that must
        // be discarded (not just dropped), even though the backend was already initialized
        let res = writer.write_time_step("0.0", |step| {
            let _write_result = step.point_data("boom", DataAttribute::Scalar, vec![0.0; 0]);
            Ok(())
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "0.0" && reason.contains("no data written")
        );

        assert!(writer.step_times.is_empty());
        assert!(writer.written_times.is_empty());

        // the backing writer is not poisoned: `FlakyWriter` would fail with
        // `Error::Internal("writing data was already initialized")` here otherwise
        writer
            .write_time_step("0.0", |step| {
                step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])
            })
            .unwrap();
    }

    #[test]
    fn write_time_step_keeps_a_step_whose_closure_swallowed_an_attribute_error() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("swallowed_error.xdmf2"), None, Some(1));

        // as above, but "ok" made it -- a step holds exactly what succeeded, so it is written
        // without "boom" rather than failing
        writer
            .write_time_step("0.0", |step| {
                step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])?;
                let _write_result = step.point_data("boom", DataAttribute::Scalar, vec![0.0; 0]);
                // annotated because nothing else in this closure pins the error type
                Ok::<(), Error>(())
            })
            .unwrap();

        let [step_grid] = last_step_grids(&writer)[..] else {
            panic!("a mesh without submeshes contributes one grid per step")
        };
        assert_eq!(
            step_grid.time.as_ref().map(|time| time.value.as_str()),
            Some("0.0")
        );
        assert_eq!(attribute_names(step_grid), ["ok"]);
    }

    #[test]
    fn write_time_step_discards_when_finalizing_fails() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(
            tmp_dir.path().join("finalize_failure.xdmf2"),
            Some("0.0"),
            None,
        );

        // every attribute is written, but completing the step fails
        let res = writer.write_time_step("0.0", |step| {
            step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::Io {
                operation: "finalizing data (simulated)",
                ..
            }
        );

        // the step is not recorded, so its heavy data must not be kept either -- and the time
        // stays available
        assert!(writer.step_times.is_empty());
        assert!(writer.written_times.is_empty());

        // the step was discarded rather than left open, so a following step still works
        writer
            .write_time_step("1.0", |step| {
                step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])
            })
            .unwrap();
    }
}

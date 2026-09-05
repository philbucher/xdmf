//! Writing a series of time steps to XDMF.
//!
//! The mesh is written once and referenced from each time step, rather than repeated per step.
//!
//! Inspired by the `TimeSeriesWriter` of [meshio](https://github.com/nschloe/meshio).

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt,
    io::{BufWriter, Write},
    ops::Range,
    path::{Path, PathBuf},
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
    /// Coordinates go in as `f32` or `f64`, connectivity as `u32`, `i32`, `u64` or `i64`, each at
    /// the width it was passed. The connectivity type caps the mesh size: `u32` and `u64` at
    /// `u32::MAX` points, since `ParaView` decodes `UInt` at 32 bits whatever precision is
    /// declared, `i64` at the full 64 bits but only in the HDF5 storages.
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
    /// A submesh is a named subset of the mesh's cells (or, for a mesh of points only, of the
    /// points), and becomes a separately selectable block in `ParaView`'s Multi-block Inspector.
    /// Submeshes may overlap, but every cell must belong to at least one -- otherwise it would
    /// silently disappear from the visualization rather than fail. A name may be any non-blank
    /// string without control characters.
    ///
    /// Field data is still passed once per step over the whole mesh; the writer cuts each
    /// submesh's share automatically. The HDF5 storages write the coordinates and each field once
    /// and let every submesh select its share, so the heavy data does not grow with the number of
    /// submeshes. The ascii and binary storages -- and any submesh whose cells are not listed in
    /// ascending order, on any storage -- get a copy per submesh instead.
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
    /// A submesh that is one block of consecutive cells can be given as a [`Range`] rather than an
    /// index list:
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

/// A submesh's cell or point indices, as the narrowest integer type that holds all of them.
/// Narrowing is allowed here where it is nowhere else, since these indices are the crate's own.
/// Signed rather than unsigned, even though never negative, because `ParaView` decodes a
/// `NumberType="UInt"` array at 32 bits whatever `Precision` says.
fn index_values(indices: &[usize]) -> Result<Values<'static>> {
    if let Some(indices) = indices
        .iter()
        .map(|&index| i32::try_from(index).ok())
        .collect::<Option<Vec<i32>>>()
    {
        return Ok(Values::from(indices));
    }

    let indices = indices
        .iter()
        .map(|&index| i64::try_from(index).ok())
        .collect::<Option<Vec<i64>>>()
        .ok_or(Error::Internal("an index does not fit into 64 bits"))?;

    Ok(Values::from(indices))
}

/// The parts of a written mesh that do not depend on how its cells are split into submeshes.
struct PreparedMesh<'c, I: Clone> {
    num_points: usize,
    num_cells: usize,
    topology_type: TopologyType,
    /// Per-element node count, set only for the `Polyvertex`/`Polyline` topologies that carry one.
    nodes_per_element: Option<u8>,
    /// Connectivity of the whole mesh, with the cell types prepended for a `Mixed` mesh.
    ///
    /// Borrowed from the caller's array unless something had to be prepended (a `Mixed` mesh, or
    /// one of points only).
    cells: Cow<'c, [I]>,
}

/// One submesh's geometry, as a selection out of the mesh's coordinates: one item per direction,
/// selecting that submesh's points out of that direction's array. Named and `Domain`-level so
/// cloning the grid per time step repeats a short reference.
///
/// All three share the mesh's own `submesh_points` list as their selector for a scattered submesh.
fn selected_coordinates(
    submesh: usize,
    coordinates: &[DataItem; 3],
    points: &IndexList,
    num_points: usize,
) -> Vec<DataItem> {
    coordinates
        .iter()
        .zip(["x", "y", "z"])
        .map(|(source, direction)| {
            let selector = match points {
                IndexList::Contiguous { start, len } => hyper_slab(*start, *len, 1),
                IndexList::Scattered(_) => submesh_index_reference(SUBMESH_POINTS, submesh),
            };

            let mut item = selection(selector, source, points.len(), &[num_points]);
            item.name = Some(format!("coords_{submesh}_{direction}"));

            item
        })
        .collect()
}

/// A grid's geometry over coordinates split by direction: the (short) references to the three
/// selections that cut its own points out of them.
fn selected_geometry(coordinate_items: &[DataItem]) -> Geometry {
    Geometry {
        geometry_type: GeometryType::XYZSeparate,
        data_items: coordinate_items
            .iter()
            .map(|item| DataItem::new_reference(item, DOMAIN_DATA_ITEMS))
            .collect(),
    }
}

/// Name of the `DataItem` holding one submesh's cell or point index list.
fn submesh_index_name(array: &str, submesh: usize) -> String {
    format!("{array}_{submesh}")
}

/// A reference to that list, for a submesh that has one -- which is every submesh whose entities
/// are not one run, and only those.
fn submesh_index_reference(array: &str, submesh: usize) -> DataItem {
    DataItem::new_reference(
        &DataItem {
            name: Some(submesh_index_name(array, submesh)),
            ..Default::default()
        },
        DOMAIN_DATA_ITEMS,
    )
}

/// A grid's geometry: the (short) reference to the coordinates it is written with, which each
/// grid needs its own copy of.
fn geometry(points_item: &DataItem) -> Geometry {
    Geometry {
        geometry_type: GeometryType::XYZ,
        data_items: vec![DataItem::new_reference(points_item, DOMAIN_DATA_ITEMS)],
    }
}

/// A named subset of a mesh's cells as it comes out of validation, before the mesh has been walked
/// to find the points those cells use.
#[derive(Debug)]
struct PreparedSubmesh {
    name: String,
    cells: IndexList,
}

/// A submesh as the data writer keeps it: its name, plus the cell and point lists its share of
/// every time step is cut with.
#[derive(Debug)]
struct Submesh {
    name: String,
    cells: IndexList,
    points: IndexList,
}

/// Which of a submesh's two index lists cuts a field: a caller passes a field over the whole
/// mesh, so what a submesh's share is depends on which of them the field is indexed by.
fn entities_of(submesh: &Submesh, point_data: bool) -> &IndexList {
    if point_data {
        &submesh.points
    } else {
        &submesh.cells
    }
}

/// Which index array a scattered submesh selects fields of one width with: a selector names the
/// position of every value it picks, so it depends on the entity's width as well as the submesh.
/// One array per (submesh, centering, component count) is written once and reused after.
#[derive(Debug, Eq, Hash, PartialEq)]
struct SelectionKey {
    submesh: usize,
    /// point data is cut by the submesh's points, cell data by its cells
    point_data: bool,
    /// how many values one entity has, which is the `DataAttribute`'s component count
    components: usize,
}

/// An ascending list of a submesh's cells or points, as positions in the mesh.
///
/// Mesh generators usually produce element blocks, material zones and boundary patches grouped, so
/// a cell list is one ascending run. That case collapses to two numbers, after which every
/// per-step slice of a field is a borrow of the caller's array rather than a gather.
#[derive(Debug)]
enum IndexList {
    Contiguous { start: usize, len: usize },
    Scattered(Vec<usize>),
}

impl IndexList {
    fn len(&self) -> usize {
        match self {
            Self::Contiguous { len, .. } => *len,
            Self::Scattered(indices) => indices.len(),
        }
    }

    /// Whether the list is ascending, which a `Coordinates` selection needs: `ParaView` hands the
    /// values back in array order rather than the order they were named. A submesh listing its
    /// cells any other way gets a copy of its share instead.
    fn is_ascending(&self) -> bool {
        match self {
            Self::Contiguous { .. } => true,
            Self::Scattered(indices) => indices.windows(2).all(|pair| pair[0] < pair[1]),
        }
    }

    /// The indices themselves, for a scattered list in the order it holds them.
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let (run, indices) = match self {
            Self::Contiguous { start, len } => (*start..*start + *len, [].as_slice()),
            Self::Scattered(indices) => (0..0, indices.as_slice()),
        };

        run.chain(indices.iter().copied())
    }
}

/// The cells of one submesh, as `write_mesh_with_submeshes` takes them.
///
/// Built with `.into()` from a slice, a `Vec`, an array, or a [`Range`], so a submesh of
/// consecutive cells can be given as `start..end` without building an index list at all.
#[derive(Clone, Debug)]
pub enum SubmeshCells<'a> {
    /// A block of consecutive cells, `start..end`.
    Range(Range<usize>),
    /// One index per cell, in the order the submesh holds them.
    Indices(Cow<'a, [usize]>),
}

impl SubmeshCells<'_> {
    /// The cheapest internal form that holds these cells, taking the caller's own allocation
    /// where it is already the right shape.
    fn into_index_list(self) -> IndexList {
        match self {
            Self::Range(range) => IndexList::Contiguous {
                start: range.start,
                len: range.end.saturating_sub(range.start),
            },
            Self::Indices(Cow::Borrowed(indices)) => collapse_indices(indices),
            Self::Indices(Cow::Owned(indices)) => {
                if is_contiguous(&indices) {
                    return IndexList::Contiguous {
                        start: indices.first().copied().unwrap_or(0),
                        len: indices.len(),
                    };
                }

                IndexList::Scattered(indices)
            }
        }
    }
}

impl From<Range<usize>> for SubmeshCells<'_> {
    fn from(range: Range<usize>) -> Self {
        Self::Range(range)
    }
}

impl<'a> From<&'a [usize]> for SubmeshCells<'a> {
    fn from(indices: &'a [usize]) -> Self {
        Self::Indices(Cow::Borrowed(indices))
    }
}

// The `&Vec<usize>` and `&[usize; N]` impls are not redundant with the `&[usize]` one, for the
// same reason `Values`' are not: an `impl Into<...>` argument is resolved by trait matching, which
// does not deref-coerce.
impl<'a> From<&'a Vec<usize>> for SubmeshCells<'a> {
    fn from(indices: &'a Vec<usize>) -> Self {
        Self::Indices(Cow::Borrowed(indices))
    }
}

impl<'a, const N: usize> From<&'a [usize; N]> for SubmeshCells<'a> {
    fn from(indices: &'a [usize; N]) -> Self {
        Self::Indices(Cow::Borrowed(indices))
    }
}

/// Moves the caller's own index list in, so a scattered submesh needs no copy of it.
impl From<Vec<usize>> for SubmeshCells<'_> {
    fn from(indices: Vec<usize>) -> Self {
        Self::Indices(Cow::Owned(indices))
    }
}

impl<const N: usize> From<[usize; N]> for SubmeshCells<'_> {
    fn from(indices: [usize; N]) -> Self {
        Self::Indices(Cow::Owned(indices.to_vec()))
    }
}

/// Validate the submeshes and collapse each one's index list to the cheapest form that holds it.
fn prepare_submeshes<'c, N: AsRef<str>, B: Into<SubmeshCells<'c>>>(
    submeshes: impl IntoIterator<Item = (N, B)>,
    num_cells: usize,
) -> Result<Vec<PreparedSubmesh>> {
    let mut prepared: Vec<PreparedSubmesh> = Vec::new();
    let mut names = HashSet::new();

    // which cells any submesh has claimed, for the coverage check below
    let mut covered = CellBitSet::new(num_cells);
    // which cells the submesh being read has claimed, to tell a cell repeated within one submesh
    // apart from two submeshes overlapping on it; cleared per submesh, not per mesh
    let mut claimed_here = CellBitSet::new(num_cells);

    for (name, cells) in submeshes {
        let name = name.as_ref();
        let cells = cells.into().into_index_list();

        if !is_valid_data_name(name) {
            return Err(Error::InvalidMesh {
                reason: format!(
                    "submesh name '{name}' is not valid, must contain a non-whitespace character \
                     and must not contain control characters"
                ),
            });
        }

        // compared verbatim: the name reaches only the `<Grid>` element that carries it
        if !names.insert(name.to_string()) {
            return Err(Error::InvalidMesh {
                reason: format!("submesh name '{name}' is used more than once"),
            });
        }

        if cells.len() == 0 {
            return Err(Error::InvalidMesh {
                reason: format!("submesh '{name}' is empty, it must contain at least one cell"),
            });
        }

        match &cells {
            // A run needs neither the duplicate check (it has none by construction) nor a walk
            // over its own indices to bound it -- which is what lets a caller hand over a block of
            // a huge mesh as a range without ever materialising one index per cell.
            IndexList::Contiguous { start, len } => {
                let end = start.checked_add(*len).ok_or(Error::Internal(
                    "a submesh's cell range does not fit a usize",
                ))?;

                if end > num_cells {
                    return Err(Error::InvalidMesh {
                        reason: format!(
                            "submesh '{name}' references cell {}, but the mesh only has \
                             {num_cells} cells",
                            end - 1
                        ),
                    });
                }

                for index in *start..end {
                    covered.insert(index);
                }
            }
            IndexList::Scattered(indices) => {
                for &index in indices {
                    if index >= num_cells {
                        return Err(Error::InvalidMesh {
                            reason: format!(
                                "submesh '{name}' references cell {index}, but the mesh only has \
                                 {num_cells} cells"
                            ),
                        });
                    }

                    if claimed_here.contains(index) {
                        return Err(Error::InvalidMesh {
                            reason: format!(
                                "submesh '{name}' contains cell {index} more than once"
                            ),
                        });
                    }

                    claimed_here.insert(index);
                    covered.insert(index);
                }

                for &index in indices {
                    claimed_here.remove(index);
                }
            }
        }

        prepared.push(PreparedSubmesh {
            name: name.to_string(),
            cells,
        });
    }

    if prepared.is_empty() {
        return Err(Error::InvalidMesh {
            reason: "at least one submesh is required".to_string(),
        });
    }

    check_all_cells_covered(&covered)?;

    Ok(prepared)
}

/// One bit per cell, for `prepare_submeshes`'s two membership questions.
///
/// A `Vec<usize>` naming which submesh last claimed each cell would cost 8 bytes per cell (800 MB
/// on a 100M-cell mesh); two bit sets cost a quarter of a byte between them.
struct CellBitSet {
    words: Vec<u64>,
    len: usize,
}

impl CellBitSet {
    const BITS: usize = u64::BITS as usize;

    fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(Self::BITS)],
            len,
        }
    }

    // every index reaching these was already checked against the cell count, keeping the word
    // lookup in bounds
    fn contains(&self, index: usize) -> bool {
        self.words[index / Self::BITS] & (1 << (index % Self::BITS)) != 0
    }

    fn insert(&mut self, index: usize) {
        self.words[index / Self::BITS] |= 1 << (index % Self::BITS);
    }

    fn remove(&mut self, index: usize) {
        self.words[index / Self::BITS] &= !(1 << (index % Self::BITS));
    }

    /// The cells whose bit is unset, in ascending order.
    fn missing(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&index| !self.contains(index))
    }
}

/// Whether a list is one ascending run of consecutive indices, which needs no per-index storage.
/// An empty list counts; `prepare_submeshes` rejects it right after.
fn is_contiguous(cells: &[usize]) -> bool {
    cells.first().is_none_or(|&start| {
        cells
            .iter()
            .enumerate()
            .all(|(offset, &index)| index == start + offset)
    })
}

/// Recognize such a run, borrowing the indices only if it is not one.
fn collapse_indices(cells: &[usize]) -> IndexList {
    if is_contiguous(cells) {
        IndexList::Contiguous {
            start: cells.first().copied().unwrap_or(0),
            len: cells.len(),
        }
    } else {
        IndexList::Scattered(cells.to_vec())
    }
}

/// Reject a mesh with cells in no submesh: such a cell reaches none of the grids, so it would
/// vanish from the visualization rather than fail.
fn check_all_cells_covered(covered: &CellBitSet) -> Result<()> {
    const MAX_LISTED: usize = 10;

    let mut uncovered = covered.missing();

    // only a handful are collected; the rest are just counted, so reporting the mistake on a huge
    // mesh does not itself allocate an array as large as the mesh
    let listed_indices: Vec<usize> = uncovered.by_ref().take(MAX_LISTED).collect();

    if listed_indices.is_empty() {
        return Ok(());
    }

    let num_not_listed = uncovered.count();
    let num_uncovered = listed_indices.len() + num_not_listed;

    let listed = listed_indices
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let ellipsis = if num_not_listed > 0 {
        format!(", ... ({num_not_listed} more)")
    } else {
        String::new()
    };

    Err(Error::InvalidMesh {
        reason: format!(
            "{num_uncovered} of {} cells belong to no submesh: {listed}{ellipsis}. Every cell \
             must be in at least one submesh; leave the others out of the mesh instead",
            covered.len
        ),
    })
}

/// Where each cell's entries start in the prepared connectivity, with a final entry for its end.
///
/// `Mixed` connectivity prepends the cell type (and, for a poly-cell, the point count) to each
/// cell's points; a uniform topology stores only the points.
fn cell_offsets(
    cell_types: &[CellType],
    topology_type: TopologyType,
    num_cells: usize,
) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(num_cells + 1);
    let mut offset = 0;

    for cell in 0..num_cells {
        offsets.push(offset);
        offset += cell_span(cell_types, topology_type, cell);
    }
    offsets.push(offset);

    offsets
}

/// How many entries one cell takes in a connectivity written as `topology_type`: its point ids,
/// behind whatever `Mixed` puts in front of them.
fn cell_span(cell_types: &[CellType], topology_type: TopologyType, cell: usize) -> usize {
    // the polyvertex fallback for a mesh of points only numbers each point one entry
    let Some(cell_type) = cell_types.get(cell) else {
        return 1;
    };

    leading_entries(cell_types, topology_type, cell) + cell_type.num_points()
}

/// A submesh's share of the prepared connectivity, in the order it lists its cells.
///
/// Owned even for one contiguous run, because the copy is renumbered into the submesh's own
/// points afterwards. A submesh written uniformly drops the type codes a `Mixed` mesh carries.
fn extract_connectivity<I: ConnectivityIndex>(
    cells: &[I],
    offsets: &[usize],
    cell_types: &[CellType],
    mesh_topology: TopologyType,
    submesh_topology: TopologyType,
    submesh: &IndexList,
) -> Vec<I> {
    // the exact size is one pass over the cells away, cheaper than growing while gathering
    let size = submesh
        .iter()
        .map(|cell| cell_span(cell_types, submesh_topology, cell))
        .sum();

    let mut extracted = Vec::with_capacity(size);
    for cell in submesh.iter() {
        // what the mesh puts ahead of this cell's points, less what the submesh keeps of it (never
        // negative: a submesh is `Mixed` only where the mesh is, and then keeps all of it)
        let dropped = leading_entries(cell_types, mesh_topology, cell)
            - leading_entries(cell_types, submesh_topology, cell);

        extracted.extend_from_slice(&cells[offsets[cell] + dropped..offsets[cell + 1]]);
    }

    extracted
}

/// The topology one submesh's own cells are written as: the `CellType` all of them share, or the
/// mesh's own where they do not. Decided per submesh, since a `Mixed` mesh's blocks (e.g. a
/// hexahedra volume beside a quadrilateral boundary) are often individually uniform, saving one
/// index per cell. The second value is the per-element node count only `Polyvertex`/`Polyline`
/// carry.
fn submesh_topology(
    cell_types: &[CellType],
    mesh_topology: TopologyType,
    mesh_nodes_per_element: Option<u8>,
    cells: &IndexList,
) -> (TopologyType, Option<u8>) {
    let mesh = (mesh_topology, mesh_nodes_per_element);

    // every cell of the mesh already shares one type, so every submesh's cells share that one too
    if mesh_topology != TopologyType::Mixed {
        return mesh;
    }

    let mut cells = cells.iter();
    // `prepare_submeshes` rejects an empty submesh, so the mesh's own is only a fallback here
    let Some(first) = cells.next().map(|cell| cell_types[cell]) else {
        return mesh;
    };

    if cells.any(|cell| cell_types[cell] != first) {
        return mesh;
    }

    (TopologyType::from(first), poly_cell_points(first))
}

/// How many entries of a cell's span come before its point ids: the cell type and, for a
/// poly-cell, its point count, which only `Mixed` connectivity carries.
fn leading_entries(cell_types: &[CellType], topology_type: TopologyType, cell: usize) -> usize {
    if topology_type != TopologyType::Mixed {
        return 0;
    }

    1 + usize::from(poly_cell_points(cell_types[cell]).is_some())
}

/// The mesh points one submesh's cells use, ascending.
///
/// The submesh's coordinates are cut out of the mesh's with this list, and its connectivity is
/// renumbered against it.
fn submesh_points<I: ConnectivityIndex>(
    cells: &[I],
    offsets: &[usize],
    cell_types: &[CellType],
    topology_type: TopologyType,
    submesh: &IndexList,
) -> Result<IndexList> {
    let mut points = Vec::new();

    for cell in submesh.iter() {
        let start = offsets[cell] + leading_entries(cell_types, topology_type, cell);
        for entry in &cells[start..offsets[cell + 1]] {
            points.push(index_as_usize(*entry)?);
        }
    }

    // sorted rather than kept in the order the cells mention them, so that the numbering a
    // submesh's connectivity is remapped to follows the mesh's own, and a submesh cut out of one
    // region of the mesh collapses to a run
    points.sort_unstable();
    points.dedup();

    Ok(collapse_indices(&points))
}

/// Where each point of the mesh sits in the submesh currently being renumbered.
///
/// A lookup array rather than a binary search of the submesh's own point list, which cost 6-28% of
/// the whole mesh write when measured on a 4M-point mesh. Never cleared between submeshes: each
/// one writes every entry it goes on to read.
#[derive(Default)]
struct LocalPoints {
    of_point: Vec<usize>,
}

impl LocalPoints {
    fn fill(&mut self, points: &[usize]) {
        // ascending, so the last is the largest -- nothing past it is ever looked up
        let needed = points.last().map_or(0, |last| last + 1);
        if self.of_point.len() < needed {
            self.of_point.resize(needed, 0);
        }

        for (local, &point) in points.iter().enumerate() {
            self.of_point[point] = local;
        }
    }
}

/// Renumber a submesh's connectivity from the mesh's point ids to its own, in place.
fn renumber_connectivity<I: ConnectivityIndex>(
    cells: &mut [I],
    cell_types: &[CellType],
    submesh_topology: TopologyType,
    submesh: &IndexList,
    points: &IndexList,
    local_points: &mut LocalPoints,
) -> Result<()> {
    if let IndexList::Scattered(points) = points {
        local_points.fill(points);
    }

    let mut position = 0;

    for cell in submesh.iter() {
        // the extracted array's own layout, not the mesh's: it already dropped whatever the
        // submesh's topology does not carry
        let leading = leading_entries(cell_types, submesh_topology, cell);
        let span = cell_span(cell_types, submesh_topology, cell);

        for entry in &mut cells[position + leading..position + span] {
            let point = index_as_usize(*entry)?;
            let local = match points {
                IndexList::Contiguous { start, .. } => point - start,
                IndexList::Scattered(_) => local_points.of_point[point],
            };

            *entry = I::from_index(local).ok_or(Error::Internal(
                "a point index does not fit the connectivity type",
            ))?;
        }

        position += span;
    }

    Ok(())
}

/// One connectivity entry as a position into the points, which every index in a written mesh is:
/// `validate_points_and_cells` rejected a negative or out-of-range one before this can run.
fn index_as_usize<I: ConnectivityIndex>(index: I) -> Result<usize> {
    usize::try_from(index.as_i128())
        .ok()
        .ok_or(Error::Internal("a connectivity entry is not a point index"))
}

fn validate_points_and_cells<I: ConnectivityIndex>(
    num_coordinates: usize,
    connectivity: &[I],
    cell_types: &[CellType],
) -> Result<()> {
    if num_coordinates == 0 {
        return Err(Error::InvalidMesh {
            reason: "at least one point is required".to_string(),
        });
    }

    if !num_coordinates.is_multiple_of(3) {
        return Err(Error::InvalidMesh {
            reason: format!(
                "points must have 3 dimensions, but {num_coordinates} is not a multiple of 3"
            ),
        });
    }

    // checked before anything is built, so a mesh too large for its index type is reported
    // without assembling its connectivity first; holds even with no connectivity passed at all,
    // since the polyvertex fallback below numbers the points itself
    let num_points = num_coordinates / 3;
    if num_points as i128 - 1 > I::MAX_INDEX {
        return Err(Error::InvalidMesh {
            reason: format!(
                "the mesh has {num_points} points, but its connectivity type can only index up \
                 to {}; a wider one is needed to write it",
                I::MAX_INDEX
            ),
        });
    }

    for index in connectivity {
        let index = index.as_i128();

        if index < 0 {
            return Err(Error::InvalidMesh {
                reason: format!("connectivity index {index} is negative"),
            });
        }
        if index >= num_points as i128 {
            return Err(Error::InvalidMesh {
                reason: format!(
                    "connectivity index {index} is out of bounds, the mesh only has \
                     {num_points} points"
                ),
            });
        }
    }

    let exp_num_points: usize = cell_types.iter().map(|ct| ct.num_points()).sum();
    if exp_num_points != connectivity.len() {
        return Err(Error::InvalidMesh {
            reason: format!(
                "size of connectivity ({}) does not match the number expected from the cell types ({exp_num_points})",
                connectivity.len()
            ),
        });
    }

    Ok(())
}

/// The point count a poly-cell (`Polyvertex`/`Polyline`) must additionally specify.
fn poly_cell_points(cell_type: CellType) -> Option<u8> {
    match cell_type {
        CellType::Vertex => Some(1),
        CellType::Edge => Some(2),
        _ => None,
    }
}

/// Prepare cells/connectivity for writing. When every cell shares one `CellType`, that type is
/// written once as a uniform `TopologyType` instead of being prepended per cell; otherwise each
/// cell gets its type (and, for a poly-cell, its point count) prepended as `Mixed` requires.
fn prepare_cells<'c, I: ConnectivityIndex>(
    connectivity: &'c [I],
    cell_types: &[CellType],
    num_points: usize,
) -> Result<(TopologyType, Cow<'c, [I]>)> {
    // every index fits by the time this runs: the point count was already checked against
    // `I::MAX_INDEX`
    let index_fits = || Error::Internal("a point index does not fit the connectivity type");

    if cell_types.is_empty() {
        // no cells: fall back to polyvertex on the points, which ParaView requires to visualize
        // points alone
        let indices = (0..num_points)
            .map(|index| I::from_index(index).ok_or_else(index_fits))
            .collect::<Result<Vec<_>>>()?;

        return Ok((TopologyType::Polyvertex, Cow::Owned(indices)));
    }

    if let [first, rest @ ..] = cell_types
        && rest.iter().all(|cell_type| cell_type == first)
    {
        // borrowed, not copied: a uniform topology stores the caller's indices as they are, so
        // the array that reaches the backend can be the caller's own however large the mesh is
        return Ok((TopologyType::from(*first), Cow::Borrowed(connectivity)));
    }

    let mut cells_with_types = Vec::with_capacity(connectivity.len() + cell_types.len());
    let mut index = 0_usize;

    for cell_type in cell_types {
        let num_points = cell_type.num_points();
        cells_with_types.push(I::from_u8(*cell_type as u8));

        if let Some(n_points_poly) = poly_cell_points(*cell_type) {
            cells_with_types.push(I::from_u8(n_points_poly));
        }

        cells_with_types.extend_from_slice(&connectivity[index..index + num_points]);

        index += num_points;
    }

    Ok((TopologyType::Mixed, Cow::Owned(cells_with_types)))
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
    /// `DataStorage::AsciiInline`, holds the data itself.
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
    /// The XDMF file this writer writes, same as `TimeSeriesWriter::file_name` reported.
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
    /// On `Ok` the step's `<Grid>` is added to the XDMF file; on `Err` the step is discarded and
    /// its heavy data removed again (the caller's error is still the one reported even if that
    /// removal itself fails).
    ///
    /// `write_step` may fail with any error type this crate's [`Error`] converts into, returned
    /// unchanged; a closure mixing error types needs it stated explicitly, most readably as
    /// `|step| -> Result<(), MyError> { ... }`.
    ///
    /// A step contains exactly the attributes that were written successfully, so a closure that
    /// swallows a rejected attribute's error and returns `Ok` gets a step without it -- only a
    /// step with no attributes at all is rejected.
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
    ///         .write_time_step(&i.to_string(), |step| {
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
    pub fn write_time_step<F, E>(&mut self, time: &str, write_step: F) -> Result<(), E>
    where
        F: FnOnce(&mut TimeStep<'_>) -> Result<(), E>,
        E: From<Error>,
    {
        let parsed_time = time
            .parse::<f64>()
            .map_err(|_parse_error| Error::InvalidTimeStep {
                time: time.to_string(),
                reason: "must be a valid float".to_string(),
            })?;

        // `f64::from_str` accepts "NaN"/"inf"/"infinity" and overflows large literals to infinity,
        // none of which name an instant a reader can place on a time line
        if !parsed_time.is_finite() {
            return Err(Error::InvalidTimeStep {
                time: time.to_string(),
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
            let reason = if existing == time {
                "already written".to_string()
            } else {
                format!("already written (as '{existing}')")
            };
            return Err(Error::InvalidTimeStep {
                time: time.to_string(),
                reason,
            }
            .into());
        }

        let mut step = TimeStep {
            per_submesh: vec![Vec::new(); self.submeshes.len()],
            writer: self,
            time: time.to_string(),
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

/// A single time step being written, handed to the closure passed to `write_time_step`.
///
/// Each `point_data`/`cell_data` call writes its heavy data before returning, so one buffer can
/// serve every field of the step; the light data (XML) is written once, after the closure
/// returns.
///
/// A step needs at least one attribute; returning from the closure without writing any is an
/// error. Returning an error discards the step: no `<Grid>` reaches the XDMF file, its heavy data
/// is removed, and the time stays available.
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
    /// `Coordinates` through its index array. Only for the HDF5 storages, whose selections
    /// `ParaView` honours.
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

/// Label for point data in user-facing error messages.
const POINT_DATA: &str = "point_data";
/// Label for cell data in user-facing error messages.
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

    #[test]
    fn test_poly_cell_points() {
        assert_eq!(poly_cell_points(CellType::Vertex), Some(1));
        assert_eq!(poly_cell_points(CellType::Edge), Some(2));
        assert_eq!(poly_cell_points(CellType::Triangle), None);
        assert_eq!(poly_cell_points(CellType::Quadrilateral), None);
        assert_eq!(poly_cell_points(CellType::Tetrahedron), None);
        assert_eq!(poly_cell_points(CellType::Pyramid), None);
        assert_eq!(poly_cell_points(CellType::Wedge), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron), None);
        assert_eq!(poly_cell_points(CellType::Edge3), None);
        assert_eq!(poly_cell_points(CellType::Quadrilateral9), None);
        assert_eq!(poly_cell_points(CellType::Triangle6), None);
        assert_eq!(poly_cell_points(CellType::Quadrilateral8), None);
        assert_eq!(poly_cell_points(CellType::Tetrahedron10), None);
        assert_eq!(poly_cell_points(CellType::Pyramid13), None);
        assert_eq!(poly_cell_points(CellType::Wedge15), None);
        assert_eq!(poly_cell_points(CellType::Wedge18), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron20), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron24), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron27), None);
    }

    /// `prepare_cells`, with the connectivity taken as a `Vec` so an expected value can be
    /// spelled `vec![..]` whether the real one borrows or owns its array.
    fn prepare_cells_vec<I: ConnectivityIndex>(
        connectivity: &[I],
        cell_types: &[CellType],
        num_points: usize,
    ) -> Result<(TopologyType, Vec<I>)> {
        let (topology_type, cells) = prepare_cells(connectivity, cell_types, num_points)?;

        Ok((topology_type, cells.into_owned()))
    }

    #[test]
    fn test_prepare_cells() {
        // mixed cell types can't be written as a uniform `TopologyType`, so the type is
        // prepended to every cell, as `Mixed` topology requires
        let (topo_type, cells_prep) = prepare_cells_vec(
            &[0_u64, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
            0,
        )
        .unwrap();

        assert_eq!(topo_type, TopologyType::Mixed);
        assert_eq!(
            cells_prep,
            vec![1, 1, 0, 2, 2, 1, 2, 4, 3, 4, 5, 5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn prepare_cells_by_celltype() {
        // when every cell shares the same type, no per-cell type code is written -- the type is
        // carried once as a uniform `TopologyType`, and the connectivity is written as-is
        assert_eq!(
            prepare_cells_vec(&[5_u64], &[CellType::Vertex], 0).unwrap(),
            (TopologyType::Polyvertex, vec![5])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6], &[CellType::Edge], 0).unwrap(),
            (TopologyType::Polyline, vec![5, 6])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7], &[CellType::Triangle], 0).unwrap(),
            (TopologyType::Triangle, vec![5, 6, 7])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8], &[CellType::Quadrilateral], 0).unwrap(),
            (TopologyType::Quadrilateral, vec![5, 6, 7, 8])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8], &[CellType::Tetrahedron], 0).unwrap(),
            (TopologyType::Tetrahedron, vec![5, 6, 7, 8])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8, 9], &[CellType::Pyramid], 0).unwrap(),
            (TopologyType::Pyramid, vec![5, 6, 7, 8, 9])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8, 9, 10], &[CellType::Wedge], 0).unwrap(),
            (TopologyType::Wedge, vec![5, 6, 7, 8, 9, 10])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8, 9, 10, 11, 12], &[CellType::Hexahedron], 0)
                .unwrap(),
            (TopologyType::Hexahedron, vec![5, 6, 7, 8, 9, 10, 11, 12])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7], &[CellType::Edge3], 0).unwrap(),
            (TopologyType::Edge3, vec![5, 6, 7])
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12, 13],
                &[CellType::Quadrilateral9],
                0
            )
            .unwrap(),
            (
                TopologyType::Quadrilateral9,
                vec![5, 6, 7, 8, 9, 10, 11, 12, 13]
            )
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8, 9, 10], &[CellType::Triangle6], 0).unwrap(),
            (TopologyType::Triangle6, vec![5, 6, 7, 8, 9, 10])
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12],
                &[CellType::Quadrilateral8],
                0
            )
            .unwrap(),
            (
                TopologyType::Quadrilateral8,
                vec![5, 6, 7, 8, 9, 10, 11, 12]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                &[CellType::Tetrahedron10],
                0
            )
            .unwrap(),
            (
                TopologyType::Tetrahedron10,
                vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                &[CellType::Pyramid13],
                0
            )
            .unwrap(),
            (
                TopologyType::Pyramid13,
                vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                &[CellType::Wedge15],
                0
            )
            .unwrap(),
            (
                TopologyType::Wedge15,
                vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[
                    5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
                ],
                &[CellType::Wedge18],
                0
            )
            .unwrap(),
            (
                TopologyType::Wedge18,
                vec![
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
                ]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[
                    5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                ],
                &[CellType::Hexahedron20],
                0
            )
            .unwrap(),
            (
                TopologyType::Hexahedron20,
                vec![
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                ]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[
                    5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                    25, 26, 27, 28
                ],
                &[CellType::Hexahedron24],
                0
            )
            .unwrap(),
            (
                TopologyType::Hexahedron24,
                vec![
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28
                ]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[
                    5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                    25, 26, 27, 28, 29, 30, 31
                ],
                &[CellType::Hexahedron27],
                0
            )
            .unwrap(),
            (
                TopologyType::Hexahedron27,
                vec![
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31
                ]
            )
        );
    }

    #[test]
    fn prepare_cells_borrows_a_uniform_connectivity() {
        // the point of the `Cow`: a uniform topology writes the caller's indices as they are, so
        // no copy of them is made however large the mesh is
        let connectivity = [0_u64, 1, 2, 1, 2, 3];
        let (_topology_type, cells) =
            prepare_cells(&connectivity, &[CellType::Triangle; 2], 4).unwrap();

        std::assert_matches!(cells, Cow::Borrowed(borrowed) if borrowed.as_ptr() == connectivity.as_ptr());
    }

    #[test]
    fn prepare_cells_mixed_when_types_differ() {
        // more than one cell of the same repeated type still can't use a uniform `TopologyType`
        // once a different type is mixed in
        let (topo_type, cells_prep) = prepare_cells_vec(
            &[0_u64, 1, 2, 3, 4, 5, 6, 7],
            &[CellType::Triangle, CellType::Triangle, CellType::Edge],
            0,
        )
        .unwrap();

        assert_eq!(topo_type, TopologyType::Mixed);
        assert_eq!(cells_prep, vec![4, 0, 1, 2, 4, 3, 4, 5, 2, 2, 6, 7]);
    }

    #[test]
    fn test_prepare_cells_no_cells() {
        let (topo_type, cells_prep) = prepare_cells_vec(&[] as &[u64], &[], 5).unwrap();

        assert_eq!(topo_type, TopologyType::Polyvertex);
        assert_eq!(cells_prep, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_validate_points_and_cells() {
        // valid input, must not return an error
        validate_points_and_cells(
            33,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        )
        .unwrap();
    }

    // a mesh whose points cannot all be indexed by the connectivity type is rejected; exercised
    // through the validation helper, which only needs the coordinate count, since a mesh this
    // big cannot be allocated in a test
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn validate_points_and_cells_too_many_points() {
        // one point more than the type can index, since the last point is index `num_points - 1`
        let too_many_u32 = usize::try_from(u32::MAX).unwrap() + 2;
        let too_many_i32 = usize::try_from(i32::MAX).unwrap() + 2;

        std::assert_matches!(
            validate_points_and_cells(3 * too_many_u32, &[] as &[u32], &[]).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("can only index up to 4294967295")
        );
        std::assert_matches!(
            validate_points_and_cells(3 * too_many_i32, &[] as &[i32], &[]).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("can only index up to 2147483647")
        );

        // one point less is exactly what each type reaches, and is still addressable
        validate_points_and_cells(3 * (too_many_u32 - 1), &[] as &[u32], &[]).unwrap();
        validate_points_and_cells(3 * (too_many_i32 - 1), &[] as &[i32], &[]).unwrap();

        // the 64-bit types hold any index a mesh can have, so this helper lets both through --
        // the lower cap ParaView puts on `u64` is checked on the connectivity values instead
        validate_points_and_cells(3 * too_many_u32, &[] as &[u64], &[]).unwrap();
        validate_points_and_cells(3 * too_many_u32, &[] as &[i64], &[]).unwrap();
    }

    // the mesh that would trip this needs over 4 billion points, which cannot be built in a
    // test, so this checks the prepared connectivity directly instead
    #[test]
    fn connectivity_above_the_paraview_uint_cap_is_rejected() {
        let too_large = Values::from(vec![u64::from(u32::MAX) + 1]);

        std::assert_matches!(
            paraview::validate(&too_large, Format::XML).unwrap_err(),
            Error::IntegerOutOfRange { value, reason }
                if value == i128::from(u32::MAX) + 1 && reason.contains("no DataStorage avoids this")
        );

        // ...and the largest index it does allow is accepted
        paraview::validate(&Values::from(vec![u64::from(u32::MAX)]), Format::XML).unwrap();
    }

    #[test]
    fn validate_points_and_cells_negative_index() {
        std::assert_matches!(
            validate_points_and_cells(9, &[0_i32, -1, 2], &[CellType::Triangle]).unwrap_err(),
            Error::InvalidMesh { reason } if reason == "connectivity index -1 is negative"
        );
    }

    #[test]
    fn validate_points_and_cells_only_points() {
        // valid input, must not return an error
        validate_points_and_cells(33, &[] as &[u64], &[]).unwrap();
    }

    #[test]
    fn validate_points_and_cells_points_empty() {
        let res = validate_points_and_cells(
            0,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("at least one point")
        );
    }

    #[test]
    fn validate_points_and_cells_points_not_3d() {
        let res = validate_points_and_cells(
            22,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("22 is not a multiple of 3")
        );
    }

    #[test]
    fn validate_points_and_cells_conn_index_out_of_bounds() {
        let res = validate_points_and_cells(
            33,
            &[0, 1, 2, 3, 4, 5, 6, 70],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("connectivity index 70")
                    && reason.contains("only has 11 points")
        );
    }

    #[test]
    fn validate_points_and_cells_conn_mismatch() {
        let res = validate_points_and_cells(
            33,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("connectivity (8)") && reason.contains("cell types (10)")
        );
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

    // A submesh list in the shape `prepare_submeshes` takes, with `&str` names and index slices.
    fn submeshes<'a>(entries: &'a [(&'a str, &'a [usize])]) -> Vec<(&'a str, &'a [usize])> {
        entries.to_vec()
    }

    #[test]
    fn prepare_submeshes_collapses_an_ascending_run() {
        let prepared = prepare_submeshes(submeshes(&[("all", &[0, 1, 2, 3])]), 4).unwrap();

        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].name, "all");
        std::assert_matches!(
            prepared[0].cells,
            IndexList::Contiguous { start: 0, len: 4 }
        );
    }

    #[test]
    fn prepare_submeshes_collapses_a_run_that_does_not_start_at_zero() {
        let prepared =
            prepare_submeshes(submeshes(&[("low", &[0, 1]), ("high", &[2, 3, 4])]), 5).unwrap();

        std::assert_matches!(
            prepared[1].cells,
            IndexList::Contiguous { start: 2, len: 3 }
        );
    }

    #[test]
    fn prepare_submeshes_keeps_a_scattered_list_in_the_given_order() {
        // descending, so it is a permutation of a run rather than one: the order the caller gave
        // is the order the submesh's cells (and its share of every cell field) are written in
        let prepared = prepare_submeshes(submeshes(&[("all", &[2, 0, 1])]), 3).unwrap();

        std::assert_matches!(
            &prepared[0].cells,
            IndexList::Scattered(indices) if indices == &[2, 0, 1]
        );
    }

    #[test]
    fn prepare_submeshes_takes_a_range_without_materialising_its_indices() {
        let prepared = prepare_submeshes([("lower", 0..2), ("upper", 2..3)], 3).unwrap();

        std::assert_matches!(
            prepared[0].cells,
            IndexList::Contiguous { start: 0, len: 2 }
        );
        std::assert_matches!(
            prepared[1].cells,
            IndexList::Contiguous { start: 2, len: 1 }
        );
    }

    #[test]
    fn prepare_submeshes_rejects_a_range_past_the_end_of_the_mesh() {
        let res = prepare_submeshes([("all", 0..4)], 3);

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("references cell 3")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_an_empty_range() {
        let res = prepare_submeshes([("all", 0..3), ("none", 1..1)], 3);

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("submesh 'none' is empty")
        );
    }

    #[test]
    fn prepare_submeshes_moves_an_owned_scattered_list_in() {
        let prepared = prepare_submeshes([("all", vec![2, 0, 1])], 3).unwrap();

        std::assert_matches!(
            &prepared[0].cells,
            IndexList::Scattered(indices) if indices == &[2, 0, 1]
        );
    }

    #[test]
    fn prepare_submeshes_allows_overlapping_submeshes() {
        let prepared = prepare_submeshes(
            submeshes(&[("left", &[0, 1]), ("right", &[1, 2]), ("all", &[0, 1, 2])]),
            3,
        )
        .unwrap();

        assert_eq!(prepared.len(), 3);
    }

    #[test]
    fn prepare_submeshes_rejects_no_submeshes() {
        let empty: Vec<(&str, &[usize])> = Vec::new();

        std::assert_matches!(
            prepare_submeshes(empty, 3).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("at least one submesh is required")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_an_invalid_name() {
        // a space is fine now -- the name only labels the block -- so it takes a control character,
        // which XML cannot represent at all
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("has space", &[0])]), 1),
            Ok(_)
        );
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("has\u{9}tab", &[0])]), 1).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("is not valid")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_a_duplicate_name() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0]), ("part", &[1])]), 2).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("submesh name 'part' is used more than once")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_an_empty_submesh() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("empty", &[])]), 1).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("submesh 'empty' is empty")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_an_out_of_range_cell() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0, 5])]), 3).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("submesh 'part' references cell 5")
                    && reason.contains("only has 3 cells")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_a_cell_repeated_within_one_submesh() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0, 1, 0])]), 2).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("submesh 'part' contains cell 0 more than once")
        );
    }

    #[test]
    fn prepare_submeshes_spans_the_bit_sets_words() {
        // every other test here fits within `CellBitSet`'s first word, hiding word-arithmetic
        // bugs; this one straddles three words instead
        let low: Vec<usize> = (0..70).collect();
        let high: Vec<usize> = (64..150).collect();

        // 150 cells covered by the two, which overlap on 64..70
        let prepared = prepare_submeshes(submeshes(&[("low", &low), ("high", &high)]), 150);
        assert_eq!(prepared.unwrap().len(), 2);

        // cell 149 is claimed twice by the same submesh, two words in
        let repeated: Vec<usize> = high.iter().copied().chain([149]).collect();
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("low", &low), ("high", &repeated)]), 150).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("submesh 'high' contains cell 149 more than once")
        );

        // and cell 130 is in none of them, which the coverage pass has to find past the boundary
        let gapped: Vec<usize> = high.iter().copied().filter(|index| *index != 130).collect();
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("low", &low), ("high", &gapped)]), 150).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("1 of 150 cells belong to no submesh: 130")
        );
    }

    // a `usize` only exceeds `i32::MAX` where it is wider than 32 bits
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn cell_indices_are_written_at_the_narrowest_type_that_holds_them() {
        let small = index_values(&[0, 7, 12]).unwrap();
        std::assert_matches!(&small, Values::I32(indices) if **indices == [0, 7, 12]);

        // one index past what an `i32` holds widens the whole array, since the type is the
        // `DataItem`'s and not each value's
        let large = index_values(&[1, usize::try_from(i32::MAX).unwrap() + 1]).unwrap();
        std::assert_matches!(
            &large,
            Values::I64(indices) if **indices == [1, i64::from(i32::MAX) + 1]
        );
    }

    #[test]
    fn prepare_submeshes_rejects_cells_in_no_submesh() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0, 2])]), 4).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("2 of 4 cells belong to no submesh: 1, 3")
        );
    }

    #[test]
    fn prepare_submeshes_truncates_a_long_list_of_uncovered_cells() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0])]), 20).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("19 of 20 cells belong to no submesh")
                    && reason.contains("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... (9 more)")
        );
    }

    #[test]
    fn cell_offsets_of_mixed_cells() {
        // a triangle takes 1 + 3 entries, an edge 1 + 1 (its point count) + 2, a vertex 1 + 1 + 1
        let offsets = cell_offsets(
            &[CellType::Triangle, CellType::Edge, CellType::Vertex],
            TopologyType::Mixed,
            3,
        );

        assert_eq!(offsets, vec![0, 4, 8, 11]);
    }

    #[test]
    fn cell_offsets_of_uniform_cells() {
        // every cell shares one type, so no per-cell metadata is stored: each quad takes exactly
        // its 4 points
        let offsets = cell_offsets(
            &[CellType::Quadrilateral; 3],
            TopologyType::Quadrilateral,
            3,
        );

        assert_eq!(offsets, vec![0, 4, 8, 12]);
    }

    #[test]
    fn cell_offsets_of_a_point_mesh() {
        // no cell types means the polyvertex fallback, one index per point
        assert_eq!(
            cell_offsets(&[], TopologyType::Polyvertex, 3),
            vec![0, 1, 2, 3]
        );
    }

    // three quads as `prepare_cells` lays them out: the cell type code (5, XDMF's own code for
    // `Quadrilateral`, not VTK's 9) then its four points -- the code collides with a point index
    // here (points are 0..11), which is why offsets say where each cell starts
    const QUAD_CELLS: [u32; 15] = [5, 0, 1, 2, 3, 5, 4, 5, 6, 7, 5, 8, 9, 10, 11];
    const QUAD_OFFSETS: [usize; 4] = [0, 5, 10, 15];

    /// The three cell types `QUAD_CELLS` is laid out for.
    const QUAD_TYPES: [CellType; 3] = [CellType::Quadrilateral; 3];

    #[test]
    fn extract_connectivity_takes_a_contiguous_submesh() {
        let extracted = extract_connectivity(
            &QUAD_CELLS,
            &QUAD_OFFSETS,
            &QUAD_TYPES,
            TopologyType::Mixed,
            TopologyType::Mixed,
            &IndexList::Contiguous { start: 1, len: 2 },
        );

        assert_eq!(extracted, &[5, 4, 5, 6, 7, 5, 8, 9, 10, 11]);
    }

    #[test]
    fn extract_connectivity_gathers_a_scattered_submesh() {
        let extracted = extract_connectivity(
            &QUAD_CELLS,
            &QUAD_OFFSETS,
            &QUAD_TYPES,
            TopologyType::Mixed,
            TopologyType::Mixed,
            &IndexList::Scattered(vec![2, 0]),
        );

        // gathered in the order the submesh names its cells, not in ascending order
        assert_eq!(extracted, &[5, 8, 9, 10, 11, 5, 0, 1, 2, 3]);
    }

    /// A submesh whose cells share one type is written uniformly: the per-cell type code a
    /// `Mixed` mesh carries is dropped, since its `<Topology>` states the type once instead.
    #[test]
    fn extract_connectivity_drops_the_type_codes_of_a_uniform_submesh() {
        let contiguous = extract_connectivity(
            &QUAD_CELLS,
            &QUAD_OFFSETS,
            &QUAD_TYPES,
            TopologyType::Mixed,
            TopologyType::Quadrilateral,
            &IndexList::Contiguous { start: 1, len: 2 },
        );

        assert_eq!(contiguous, &[4, 5, 6, 7, 8, 9, 10, 11]);

        let scattered = extract_connectivity(
            &QUAD_CELLS,
            &QUAD_OFFSETS,
            &QUAD_TYPES,
            TopologyType::Mixed,
            TopologyType::Quadrilateral,
            &IndexList::Scattered(vec![2, 0]),
        );

        assert_eq!(scattered, &[8, 9, 10, 11, 0, 1, 2, 3]);
    }

    #[test]
    fn submesh_topology_is_the_type_its_own_cells_share() {
        let cell_types = [
            CellType::Hexahedron,
            CellType::Quadrilateral,
            CellType::Quadrilateral,
        ];

        // the block of quads is uniform even though the mesh it is cut out of is not
        assert_eq!(
            submesh_topology(
                &cell_types,
                TopologyType::Mixed,
                None,
                &IndexList::Contiguous { start: 1, len: 2 }
            ),
            (TopologyType::Quadrilateral, None)
        );

        // one that spans both types stays Mixed, as does one cell of each
        assert_eq!(
            submesh_topology(
                &cell_types,
                TopologyType::Mixed,
                None,
                &IndexList::Contiguous { start: 0, len: 3 }
            ),
            (TopologyType::Mixed, None)
        );
        assert_eq!(
            submesh_topology(
                &cell_types,
                TopologyType::Mixed,
                None,
                &IndexList::Scattered(vec![2, 0])
            ),
            (TopologyType::Mixed, None)
        );

        // a poly-cell block carries its node count, as the mesh's own topology would
        assert_eq!(
            submesh_topology(
                &[CellType::Hexahedron, CellType::Edge],
                TopologyType::Mixed,
                None,
                &IndexList::Contiguous { start: 1, len: 1 }
            ),
            (TopologyType::Polyline, Some(2))
        );

        // and a mesh that is uniform to begin with hands its own topology to every submesh
        assert_eq!(
            submesh_topology(
                &[CellType::Edge; 2],
                TopologyType::Polyline,
                Some(2),
                &IndexList::Contiguous { start: 0, len: 1 }
            ),
            (TopologyType::Polyline, Some(2))
        );
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

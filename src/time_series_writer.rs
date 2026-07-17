//! This module contains functionalities for writing a series of time steps to XDMF.
//!
//! The mesh is written only once and then referenced in each time step.
//! This is a significant advantage over VTK based formats, making it more efficient both in terms of storage size as well as write speed.
//!
//! The concept is insipred by the `TimeSeriesWriter` of [meshio](https://github.com/nschloe/meshio)

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    io::{BufWriter, Error as IoError, ErrorKind::InvalidInput, Result as IoResult, Write},
    path::{Path, PathBuf},
};

use crate::{
    CellType, DataAttribute, DataStorage, DataWriter, Values, create_writer,
    mpi_safe_create_dir_all,
    xdmf_elements::{
        Information, Xdmf, attribute,
        data_item::{DataItem, Format, NumberType},
        dimensions::Dimensions,
        geometry::{Geometry, GeometryType},
        grid::{CollectionType, Grid, GridType, Time},
        topology::{Topology, TopologyType},
    },
};

/// Writer for time series data in XDMF format.
pub struct TimeSeriesWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
}

impl TimeSeriesWriter {
    /// Create a new `TimeSeriesWriter`.
    /// ```rust
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("name_xdmf_file", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    /// ```
    pub fn new(file_name: impl AsRef<Path>, data_storage: DataStorage) -> IoResult<Self> {
        let xdmf_file_name = file_name.as_ref().to_path_buf().with_extension("xdmf2");

        validate_file_name(&xdmf_file_name)?;

        // create the parent directory if it does not exist
        if let Some(parent) = xdmf_file_name.parent() {
            mpi_safe_create_dir_all(parent)?;
        }

        Ok(Self {
            xdmf_file_name,
            writer: create_writer(file_name.as_ref(), data_storage)?,
        })
    }

    /// Writes the mesh to the XDMF file, returning a `TimeSeriesDataWriter` for writing time steps.
    ///
    /// Sizes of the inputs are validated to ensure consistency with the mesh and defined cell types.
    /// ```rust
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("xdmf_write_mesh", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    ///
    /// // define 3 points and 2 cells (a line and a triangle)
    /// let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// let connectivity = [0, 1, 0, 2, 1]; // line (0,1) and triangle (0,2,1)
    /// let cell_types = [xdmf::CellType::Edge, xdmf::CellType::Triangle];
    ///
    /// // write the mesh
    /// let mut ts_writer = xdmf_writer.write_mesh(&coords, (&connectivity, &cell_types));
    /// ```
    pub fn write_mesh(
        mut self,
        points: &[f64],
        cells: (&[u64], &[CellType]),
    ) -> IoResult<TimeSeriesDataWriter> {
        validate_points_and_cells(points, cells)?;

        let num_points = points.len() / 3;
        let num_cells = if cells.1.is_empty() {
            num_points
        } else {
            cells.1.len()
        };

        let (topo_type, prepared_cells, _cell_windows) = prepare_cells(cells, num_points);

        let (points_data, cells_data) = self.writer.write_mesh(points, &prepared_cells)?;

        let format = self.writer.format();

        let data_item_coords = DataItem {
            name: Some("coords".to_string()),
            dimensions: Some(Dimensions(vec![num_points, 3])),
            data: points_data,
            number_type: Some(NumberType::Float),
            precision: Some(8),
            endian: format.endian(),
            format: Some(format),
            reference: None,
        };

        let data_item_connectivity = DataItem {
            name: Some("connectivity".to_string()),
            dimensions: Some(Dimensions(vec![prepared_cells.len()])),
            number_type: Some(NumberType::UInt),
            data: cells_data,
            format: Some(format),
            precision: Some(format.uint_precision()),
            endian: format.endian(),
            reference: None,
        };

        let data_item_coords_ref =
            DataItem::new_reference(&data_item_coords, "/Xdmf/Domain/DataItem");
        let data_item_connectivity_ref =
            DataItem::new_reference(&data_item_connectivity, "/Xdmf/Domain/DataItem");

        let geometry = Geometry {
            geometry_type: GeometryType::XYZ,
            data_item: data_item_coords_ref,
        };
        let topology = Topology {
            topology_type: topo_type,
            number_of_elements: num_cells.to_string(),
            data_item: data_item_connectivity_ref,
        };

        let mut ts_writer = TimeSeriesDataWriter {
            xdmf_file_name: self.xdmf_file_name,
            writer: self.writer,
            grid: Grid::new_uniform("mesh", geometry, topology),
            data_items: vec![data_item_coords, data_item_connectivity],
            attributes: vec![],
            block_attributes: vec![],
            blocks: None,
            writen_times: HashSet::new(),
            num_points,
            num_cells,
        };

        ts_writer.write()?;

        Ok(ts_writer)
    }

    /// Writes the mesh as a collection of named, possibly overlapping, blocks (submeshes),
    /// rendered by [Paraview](https://www.paraview.org/) as a multiblock dataset.
    ///
    /// Each block is given as a name plus a set of indices into `cells` (0-based, in the same
    /// order as `cells.1`) identifying the cells it contains. Indices need not be contiguous and
    /// may repeat across blocks (a cell can belong to more than one block); each block simply
    /// gets its own copy of the connectivity entries for the cells it references. Node coordinates
    /// are always written once and shared by every block via an XDMF reference, regardless of
    /// which nodes each block's cells actually use.
    /// ```rust
    /// use std::collections::BTreeSet;
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("xdmf_write_mesh_with_blocks", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    ///
    /// // 4 points, 2 triangles sharing an edge
    /// let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    /// let connectivity = [0, 1, 2, 0, 2, 3];
    /// let cell_types = [xdmf::CellType::Triangle, xdmf::CellType::Triangle];
    ///
    /// // one block per triangle
    /// let triangle_a: BTreeSet<usize> = [0].into();
    /// let triangle_b: BTreeSet<usize> = [1].into();
    /// let blocks = [("triangle_a", &triangle_a), ("triangle_b", &triangle_b)];
    ///
    /// let mut ts_writer = xdmf_writer
    ///     .write_mesh_with_blocks(&coords, (&connectivity, &cell_types), &blocks)
    ///     .expect("failed to write mesh with blocks");
    /// ```
    pub fn write_mesh_with_blocks(
        mut self,
        points: &[f64],
        cells: (&[u64], &[CellType]),
        blocks: &[(&str, &BTreeSet<usize>)],
    ) -> IoResult<TimeSeriesDataWriter> {
        validate_points_and_cells(points, cells)?;

        let num_points = points.len() / 3;
        let num_cells = if cells.1.is_empty() {
            num_points
        } else {
            cells.1.len()
        };

        validate_blocks(blocks, num_cells)?;

        let (topo_type, prepared_cells, cell_windows) = prepare_cells(cells, num_points);

        // only points are shared across blocks; each block writes its own connectivity below
        let (points_data, _unused_cells_data) = self.writer.write_mesh(points, &[])?;

        let format = self.writer.format();

        let data_item_coords = DataItem {
            name: Some("coords".to_string()),
            dimensions: Some(Dimensions(vec![num_points, 3])),
            data: points_data,
            number_type: Some(NumberType::Float),
            precision: Some(8),
            endian: format.endian(),
            format: Some(format),
            reference: None,
        };

        let data_item_coords_ref =
            DataItem::new_reference(&data_item_coords, "/Xdmf/Domain/DataItem");

        let mut block_grids = Vec::with_capacity(blocks.len());
        let mut block_infos = Vec::with_capacity(blocks.len());
        let mut data_items = vec![data_item_coords];

        for &(block_name, cell_indices) in blocks {
            let mut block_cells = Vec::new();
            for &idx in cell_indices {
                let (start, len) = cell_windows[idx];
                block_cells.extend_from_slice(&prepared_cells[start..start + len]);
            }

            let block_cells_data = self.writer.write_mesh_block(block_name, &block_cells)?;

            // Written once at the Domain level (like `coords`) and referenced below, so that
            // cloning the block grid once per time step (in `write()`) only repeats a small
            // reference, not the connectivity data itself.
            let data_item_block_connectivity = DataItem {
                name: Some(format!("connectivity_{block_name}")),
                dimensions: Some(Dimensions(vec![block_cells.len()])),
                number_type: Some(NumberType::UInt),
                data: block_cells_data,
                format: Some(format),
                precision: Some(format.uint_precision()),
                endian: format.endian(),
                reference: None,
            };

            let data_item_block_connectivity_ref =
                DataItem::new_reference(&data_item_block_connectivity, "/Xdmf/Domain/DataItem");

            let topology = Topology {
                topology_type: topo_type,
                number_of_elements: cell_indices.len().to_string(),
                data_item: data_item_block_connectivity_ref,
            };

            let geometry = Geometry {
                geometry_type: GeometryType::XYZ,
                data_item: data_item_coords_ref.clone(),
            };

            block_grids.push(Grid::new_uniform(block_name, geometry, topology));
            block_infos.push(MeshBlock {
                name: block_name.to_string(),
                cell_indices: cell_indices.clone(),
            });
            data_items.push(data_item_block_connectivity);
        }

        let blocks_grid = Grid::new_collection("blocks", CollectionType::Spatial, Some(block_grids));

        let mut ts_writer = TimeSeriesDataWriter {
            xdmf_file_name: self.xdmf_file_name,
            writer: self.writer,
            grid: blocks_grid,
            data_items,
            attributes: vec![],
            block_attributes: vec![],
            blocks: Some(block_infos),
            writen_times: HashSet::new(),
            num_points,
            num_cells,
        };

        ts_writer.write()?;

        Ok(ts_writer)
    }
}

/// A named submesh: the set of (0-based) indices into the original `cells` list it contains.
#[derive(Clone, Debug)]
struct MeshBlock {
    name: String,
    cell_indices: BTreeSet<usize>,
}

fn validate_blocks(blocks: &[(&str, &BTreeSet<usize>)], num_cells: usize) -> IoResult<()> {
    if blocks.is_empty() {
        return Err(IoError::new(InvalidInput, "At least one block is required"));
    }

    let mut seen_names = HashSet::new();
    let mut covered = vec![false; num_cells];

    for &(name, cell_indices) in blocks {
        if !is_valid_data_name(name) {
            return Err(IoError::new(
                InvalidInput,
                format!(
                    "Block name '{name}' is not valid, must be non-empty and contain only alphanumeric characters, underscores or dashes",
                ),
            ));
        }

        if !seen_names.insert(name) {
            return Err(IoError::new(
                InvalidInput,
                format!("Block name '{name}' is used more than once"),
            ));
        }

        if cell_indices.is_empty() {
            return Err(IoError::new(
                InvalidInput,
                format!("Block '{name}' must contain at least one cell"),
            ));
        }

        if let Some(&max_idx) = cell_indices.last()
            && max_idx >= num_cells
        {
            return Err(IoError::new(
                InvalidInput,
                format!(
                    "Block '{name}' references cell index {max_idx}, but the mesh only has {num_cells} cells",
                ),
            ));
        }

        for &idx in cell_indices {
            covered[idx] = true;
        }
    }

    const MAX_SHOWN_UNCOVERED: usize = 10;
    let uncovered: Vec<usize> = covered
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_covered)| (!is_covered).then_some(idx))
        .collect();

    if !uncovered.is_empty() {
        let shown = uncovered
            .iter()
            .take(MAX_SHOWN_UNCOVERED)
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let suffix = if uncovered.len() > MAX_SHOWN_UNCOVERED {
            format!(", ... ({} more)", uncovered.len() - MAX_SHOWN_UNCOVERED)
        } else {
            String::new()
        };

        return Err(IoError::new(
            InvalidInput,
            format!(
                "{} of {num_cells} cells are not part of any block: {shown}{suffix}. \
                 If this is intentional, exclude them from the cells passed to write_mesh_with_blocks",
                uncovered.len()
            ),
        ));
    }

    Ok(())
}

// Validate that the points and cells are valid
fn validate_points_and_cells(points: &[f64], cells: (&[u64], &[CellType])) -> IoResult<()> {
    // at least one point is required
    if points.is_empty() {
        return Err(IoError::new(InvalidInput, "At least one point is required"));
    }

    // check that points are a multiple of 3 (x, y, z)
    if !points.len().is_multiple_of(3) {
        return Err(IoError::new(InvalidInput, "Points must have 3 dimensions"));
    }

    // check cells connectivity indices
    let max_connectivity_index = cells.0.iter().max();

    if let Some(&max_index) = max_connectivity_index
        && max_index as usize >= points.len() / 3
    {
        return Err(IoError::new(
            InvalidInput,
            format!(
                "Connectivity indices out of bounds for the given points, max index: {}, but number of points is {}",
                max_index,
                points.len() / 3
            ),
        ));
    }

    // check that the number of connectivities matches the expected number based on the cell types
    let exp_num_points: usize = cells.1.iter().map(|ct| ct.num_points()).sum();
    if exp_num_points != cells.0.len() {
        return Err(IoError::new(
            InvalidInput,
            format!(
                "Size of connectivities not match the expected number based on the cell types: {} != {}",
                cells.0.len(),
                exp_num_points
            ),
        ));
    }

    Ok(())
}

// Poly-cells need to additionally specify the number of points
fn poly_cell_points(cell_type: CellType) -> Option<u64> {
    // For polyvertex and polyline, need to add the number of points
    match cell_type {
        CellType::Vertex => {
            // polyvertex with one point
            Some(1)
        }
        CellType::Edge => {
            // polyline with two points
            Some(2)
        }
        _ => None,
    }
}

/// Prepare cells / connectivity for writing. The cell type is prepended to the connectivity list,
/// and for poly-cells, the number of points is also added.
///
/// Also returns, for each original cell (or point, in the no-cells/Polyvertex case), the
/// `(start, len)` window locating that cell's entry within the flat returned buffer. This is
/// used to slice out an arbitrary subset of cells (e.g. for [`TimeSeriesWriter::write_mesh_with_blocks`])
/// without re-deriving the encoding logic.
/// TODO if all cells are the same, then the type information can be stored as `TopologyType`
fn prepare_cells(
    cells: (&[u64], &[CellType]),
    num_points: usize,
) -> (TopologyType, Vec<u64>, Vec<(usize, usize)>) {
    if cells.1.is_empty() {
        // if there are no cells, use polyvertex on nodes
        // this is required by paraview to visualize only points
        let windows = (0..num_points).map(|i| (i, 1)).collect();
        return (
            TopologyType::Polyvertex,
            (0..num_points as u64).collect(),
            windows,
        );
    }

    let mut cells_with_types = Vec::with_capacity(cells.0.len() + cells.1.len());
    let mut windows = Vec::with_capacity(cells.1.len());
    let mut index = 0_usize;

    for cell_type in cells.1 {
        let num_points = cell_type.num_points();
        let start = cells_with_types.len();

        cells_with_types.push(*cell_type as u64);

        if let Some(n_points_poly) = poly_cell_points(*cell_type) {
            // poly-cells need to specify the number of points
            cells_with_types.push(n_points_poly);
        }

        cells_with_types.extend_from_slice(&cells.0[index..index + num_points]);

        windows.push((start, cells_with_types.len() - start));

        index += num_points; // move index to the next cell
    }

    (TopologyType::Mixed, cells_with_types, windows)
}

/// Writer for time series data in XDMF format. Can be used after writing the mesh with `TimeSeriesWriter::write_mesh`.
pub struct TimeSeriesDataWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
    grid: Grid,
    data_items: Vec<DataItem>,
    attributes: Vec<(String, Vec<attribute::Attribute>)>,
    // Only populated in block mode (`TimeSeriesWriter::write_mesh_with_blocks`): per time step,
    // one attribute list per block, indexed the same as `blocks`.
    block_attributes: Vec<(String, Vec<Vec<attribute::Attribute>>)>,
    // Some(..) iff the mesh was written via `TimeSeriesWriter::write_mesh_with_blocks`.
    blocks: Option<Vec<MeshBlock>>,
    writen_times: HashSet<String>,
    num_points: usize,
    num_cells: usize,
}

impl TimeSeriesDataWriter {
    /// Write point and cell data for a specific time step.
    ///
    /// Accepts str for time to avoid dealing with formatting, thus leaving it to the user.
    /// Sizes of the data arrays are validated to ensure consistency with the mesh and defined dat types.
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
    ///     .write_mesh(&coords, (&connectivity, &cell_types))
    ///     .expect("failed to write mesh");
    ///
    /// // define some point and cell data for time step 0.0
    /// let point_vals: xdmf::Values = vec![0.0; 9].into();
    /// let point_data = [("point_data", xdmf::DataAttribute::Vector, &point_vals)];
    ///
    /// let cell_vals: xdmf::Values = vec![0.0, 1.0].into();
    /// let cell_data = [("cell_data", xdmf::DataAttribute::Scalar, &cell_vals)];
    ///
    /// // write the data for 10 time steps
    /// for i in 0..10 {
    ///     time_series_writer
    ///         .write_data(&i.to_string(), point_data, cell_data)
    ///         .expect("failed to write time step data");
    /// }
    /// ```
    pub fn write_data<'a>(
        &mut self,
        time: &str,
        point_data: impl IntoIterator<Item = (&'a str, DataAttribute, &'a Values)>,
        cell_data: impl IntoIterator<Item = (&'a str, DataAttribute, &'a Values)>,
    ) -> IoResult<()> {
        // Collected into `BTreeMap`s (keyed on name) so that output ordering is deterministic
        // regardless of the order the caller happens to iterate in.
        let point_data = collect_data(point_data, "point")?;
        let cell_data = collect_data(cell_data, "cell")?;

        self.validate_data(time, &point_data, &cell_data)?;

        self.writer.write_data_initialize(time)?;
        let format = self.writer.format();

        // Point data is always shared verbatim, regardless of block mode.
        let mut shared_attributes = Vec::new();
        for (data_name, data) in &point_data {
            shared_attributes.push(build_attribute(
                self.writer.as_mut(),
                format,
                data_name,
                data_name,
                attribute::Center::Node,
                data.0,
                data.1,
            )?);
        }

        if let Some(blocks) = self.blocks.clone() {
            let mut per_block_attributes: Vec<Vec<attribute::Attribute>> =
                vec![shared_attributes; blocks.len()];

            for (data_name, data) in &cell_data {
                let stride = data.0.size();

                for (block, block_attrs) in blocks.iter().zip(per_block_attributes.iter_mut()) {
                    let sliced_vals = data.1.gather(stride, &block.cell_indices);
                    let unique_name = format!("{data_name}__{}", block.name);

                    block_attrs.push(build_attribute(
                        self.writer.as_mut(),
                        format,
                        &unique_name,
                        data_name,
                        attribute::Center::Cell,
                        data.0,
                        &sliced_vals,
                    )?);
                }
            }

            self.block_attributes
                .push((time.to_string(), per_block_attributes));
        } else {
            let mut new_attributes = shared_attributes;

            for (data_name, data) in &cell_data {
                new_attributes.push(build_attribute(
                    self.writer.as_mut(),
                    format,
                    data_name,
                    data_name,
                    attribute::Center::Cell,
                    data.0,
                    data.1,
                )?);
            }

            self.attributes.push((time.to_string(), new_attributes));
        }

        self.writen_times.insert(time.to_string());

        self.writer.write_data_finalize()?;

        self.write()
    }

    fn write(&mut self) -> IoResult<()> {
        self.writer.flush()?;

        let grid_has_data = if self.blocks.is_some() {
            !self.block_attributes.is_empty()
        } else {
            !self.attributes.is_empty()
        };

        // If there are no attributes aka time-data, write the grid directly
        let grid_to_write = if !grid_has_data {
            self.grid.clone()
        } else if self.blocks.is_some() {
            let time_grids = self
                .block_attributes
                .iter()
                .map(|(time, per_block_attributes)| {
                    let mut grid = self.grid.clone();
                    grid.name = format!("time_series-t{time}");
                    grid.time = Some(Time::new(time));

                    // `grid.grids` is always `Some` here: it was built by `write_mesh_with_blocks`
                    // as a `Grid::new_collection` with an explicit list of block sub-grids.
                    if let Some(sub_grids) = grid.grids.as_mut() {
                        for (sub_grid, attrs) in
                            sub_grids.iter_mut().zip(per_block_attributes.iter())
                        {
                            sub_grid.attributes = Some(attrs.clone());
                        }
                    }

                    grid
                })
                .collect();

            Grid::new_collection("time_series", CollectionType::Temporal, Some(time_grids))
        } else {
            let time_grids = self
                .attributes
                .iter()
                .map(|(time, attributes)| {
                    let mut grid = self.grid.clone();

                    match grid.grid_type {
                        GridType::Uniform => {
                            grid.name = format!("time_series-t{time}");
                            grid.time = Some(Time::new(time));
                            grid.attributes = Some(attributes.clone());
                            grid
                        }
                        _ => unimplemented!("Only Uniform grids are supported for time series"),
                    }
                })
                .collect();

            Grid::new_collection("time_series", CollectionType::Temporal, Some(time_grids))
        };

        let mut xdmf = Xdmf {
            information: vec![
                Information::new("data_storage", format!("{:?}", self.writer.data_storage())),
                Information::new("version", env!("CARGO_PKG_VERSION")),
            ],
            ..Default::default()
        };
        xdmf.domains[0].grids.push(grid_to_write);
        xdmf.domains[0].data_items.extend(self.data_items.clone());

        // Write the XDMF file to a temporary file first to avoid access races
        let temp_xdmf_file_name = self.xdmf_file_name.with_extension("xdmf.tmp");

        let mut xdmf_file = BufWriter::new(std::fs::File::create(&temp_xdmf_file_name)?);
        xdmf.write_to(&mut xdmf_file)?;
        xdmf_file.flush()?;

        std::fs::rename(&temp_xdmf_file_name, &self.xdmf_file_name)
    }

    fn validate_data(
        &self,
        time: &str,
        point_data: &BTreeMap<&str, (DataAttribute, &Values)>,
        cell_data: &BTreeMap<&str, (DataAttribute, &Values)>,
    ) -> IoResult<()> {
        // check if time can be parsed as a float
        if time.parse::<f64>().is_err() {
            return Err(IoError::new(
                InvalidInput,
                format!("Time must be a valid float, and not '{time}'"),
            ));
        }

        // check if the time step has already been written
        if self.writen_times.contains(time) {
            return Err(IoError::new(
                InvalidInput,
                format!("Time step '{time}' has already been written"),
            ));
        }

        // check if some data is provided
        if point_data.len() + cell_data.len() == 0 {
            return Err(IoError::new(
                InvalidInput,
                "At least one of point_data or cell_data must be provided",
            ));
        }

        check_data_size(point_data, self.num_points, "point")?;
        check_data_size(cell_data, self.num_cells, "cell")?;

        // check that names do not contain forbidden characters
        validate_data_name(point_data, "point")?;
        validate_data_name(cell_data, "cell")
    }
}

// Build a single `Attribute` (with its own `DataItem`), writing `vals` out via the given writer
// under `storage_name` (must be unique per writer call, e.g. suffixed per block), while the
// Attribute's own (user-facing) `Name` stays `display_name`.
fn build_attribute(
    writer: &mut dyn DataWriter,
    format: Format,
    storage_name: &str,
    display_name: &str,
    center: attribute::Center,
    data_attr: DataAttribute,
    vals: &Values,
) -> IoResult<attribute::Attribute> {
    let data_item = DataItem {
        name: None,
        dimensions: Some(vals.dimensions(data_attr)),
        number_type: Some(vals.number_type()),
        format: Some(format),
        precision: Some(vals.precision(format)),
        endian: format.endian(),
        data: writer.write_data(storage_name, center, vals)?,
        reference: None,
    };

    Ok(attribute::Attribute {
        name: display_name.to_string(),
        attribute_type: data_attr.into(),
        center,
        data_items: vec![data_item],
    })
}

// check sizes of point_data and cell_data
fn check_data_size(
    data_input: &BTreeMap<&str, (DataAttribute, &Values)>,
    num_entities: usize,
    label: &str,
) -> IoResult<()> {
    for (name, data) in data_input {
        let exp_size = num_entities * data.0.size();
        if data.1.len() != exp_size {
            return Err(IoError::new(
                InvalidInput,
                format!(
                    "Size of {label}-data '{name}' must be {}, but is {}",
                    exp_size,
                    data.1.len()
                ),
            ));
        }
    }
    Ok(())
}

// Collect a flat iterator of (name, attribute, values) into a `BTreeMap` keyed on name,
// rejecting a name used more than once instead of silently keeping the last occurrence.
fn collect_data<'a>(
    data: impl IntoIterator<Item = (&'a str, DataAttribute, &'a Values)>,
    label: &str,
) -> IoResult<BTreeMap<&'a str, (DataAttribute, &'a Values)>> {
    let mut map = BTreeMap::new();

    for (name, attr, values) in data {
        if map.insert(name, (attr, values)).is_some() {
            return Err(IoError::new(
                InvalidInput,
                format!("{label}-data name '{name}' is used more than once"),
            ));
        }
    }

    Ok(map)
}

fn validate_data_name(
    data_input: &BTreeMap<&str, (DataAttribute, &Values)>,
    label: &str,
) -> IoResult<()> {
    for name in data_input.keys() {
        if !is_valid_data_name(name) {
            return Err(IoError::new(
                InvalidInput,
                format!(
                    "Data name '{name}' of {label}-data is not valid, must be non-empty and contain only alphanumeric characters, underscores or dashes",
                ),
            ));
        };
    }
    Ok(())
}

fn is_valid_data_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Validate the file name for the XDMF file.
fn validate_file_name(file_name: &Path) -> IoResult<()> {
    // Ensure it's valid UTF-8
    let Some(name) = file_name.to_str() else {
        return Err(IoError::new(InvalidInput, "File name must be valid UTF-8"));
    };

    if name.is_empty() {
        return Err(IoError::new(InvalidInput, "File name must not be empty"));
    }

    let invalid_chars = ['?', '\0', ':', '*', '"', '<', '>', '|'];

    // Check for invalid characters
    if name.chars().any(|c| invalid_chars.contains(&c)) {
        return Err(IoError::new(
            InvalidInput,
            format!(
                "File name '{name}' cannot contain the following characters: {invalid_chars:?}"
            ),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DataAttribute,
        xdmf_elements::{
            data_item::{DataContent, Format},
            grid::Grid,
        },
    };

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

    #[test]
    fn test_prepare_cells() {
        let (topo_type, cells_prep, windows) = prepare_cells(
            (
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                &[
                    CellType::Vertex,
                    CellType::Edge,
                    CellType::Triangle,
                    CellType::Quadrilateral,
                ],
            ),
            0,
        );

        assert_eq!(topo_type, TopologyType::Mixed);
        assert_eq!(
            cells_prep,
            vec![1, 1, 0, 2, 2, 1, 2, 4, 3, 4, 5, 5, 6, 7, 8, 9]
        );
        // Vertex: [1, 1, 0] (start=0, len=3), Edge: [2, 2, 1, 2] (start=3, len=4),
        // Triangle: [4, 3, 4, 5] (start=7, len=4), Quadrilateral: [5, 6, 7, 8, 9] (start=11, len=5)
        assert_eq!(windows, vec![(0, 3), (3, 4), (7, 4), (11, 5)]);
        for &(start, len) in &windows {
            assert!(start + len <= cells_prep.len());
        }
    }

    #[test]
    fn prepare_cells_by_celltype() {
        assert_eq!(
            prepare_cells((&[5], &[CellType::Vertex]), 0).1,
            vec![1, 1, 5]
        );

        assert_eq!(
            prepare_cells((&[5, 6], &[CellType::Edge]), 0).1,
            vec![2, 2, 5, 6]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7], &[CellType::Triangle]), 0).1,
            vec![4, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8], &[CellType::Quadrilateral]), 0).1,
            vec![5, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8], &[CellType::Tetrahedron]), 0).1,
            vec![6, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9], &[CellType::Pyramid]), 0).1,
            vec![7, 5, 6, 7, 8, 9]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10], &[CellType::Wedge]), 0).1,
            vec![8, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Hexahedron]), 0).1,
            vec![9, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7], &[CellType::Edge3]), 0).1,
            vec![34, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells(
                (
                    &[5, 6, 7, 8, 9, 10, 11, 12, 13],
                    &[CellType::Quadrilateral9]
                ),
                0
            )
            .1,
            vec![35, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10], &[CellType::Triangle6]), 0).1,
            vec![36, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells(
                (&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Quadrilateral8]),
                0
            )
            .1,
            vec![37, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells(
                (
                    &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                    &[CellType::Tetrahedron10]
                ),
                0
            )
            .1,
            vec![38, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        );

        assert_eq!(
            prepare_cells(
                (
                    &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                    &[CellType::Pyramid13]
                ),
                0
            )
            .1,
            vec![39, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        );

        assert_eq!(
            prepare_cells(
                (
                    &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    &[CellType::Wedge15]
                ),
                0
            )
            .1,
            vec![40, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );

        assert_eq!(
            prepare_cells(
                (
                    &[
                        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
                    ],
                    &[CellType::Wedge18]
                ),
                0
            )
            .1,
            vec![
                41, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
            ]
        );

        assert_eq!(
            prepare_cells(
                (
                    &[
                        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                    ],
                    &[CellType::Hexahedron20]
                ),
                0
            )
            .1,
            vec![
                48, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
            ]
        );

        assert_eq!(
            prepare_cells(
                (
                    &[
                        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                        25, 26, 27, 28
                    ],
                    &[CellType::Hexahedron24]
                ),
                0
            )
            .1,
            vec![
                49, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28
            ]
        );

        assert_eq!(
            prepare_cells(
                (
                    &[
                        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                        25, 26, 27, 28, 29, 30, 31
                    ],
                    &[CellType::Hexahedron27]
                ),
                0
            )
            .1,
            vec![
                50, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31
            ]
        );
    }

    #[test]
    fn test_prepare_cells_no_cells() {
        let (topo_type, cells_prep, windows) = prepare_cells((&[], &[]), 5);

        assert_eq!(topo_type, TopologyType::Polyvertex);
        assert_eq!(cells_prep, vec![0, 1, 2, 3, 4]);
        assert_eq!(windows, vec![(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]);
    }

    #[test]
    fn test_validate_points_and_cells() {
        // valid input, must not return an error
        validate_points_and_cells(
            &[0.0; 33],
            (
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[
                    CellType::Vertex,
                    CellType::Triangle,
                    CellType::Quadrilateral,
                ],
            ),
        )
        .unwrap();
    }

    #[test]
    fn validate_points_and_cells_only_points() {
        // valid input, must not return an error
        validate_points_and_cells(&[0.0; 33], (&[], &[])).unwrap();
    }

    #[test]
    fn validate_points_and_cells_points_empty() {
        let res = validate_points_and_cells(
            &[],
            (
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[
                    CellType::Vertex,
                    CellType::Triangle,
                    CellType::Quadrilateral,
                ],
            ),
        );

        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err().to_string(),
            "At least one point is required"
        );
    }

    #[test]
    fn validate_points_and_cells_points_not_3d() {
        let res = validate_points_and_cells(
            &[0.0; 22],
            (
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[
                    CellType::Vertex,
                    CellType::Triangle,
                    CellType::Quadrilateral,
                ],
            ),
        );

        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err().to_string(),
            "Points must have 3 dimensions"
        );
    }

    #[test]
    fn validate_points_and_cells_conn_index_out_of_bounds() {
        let res = validate_points_and_cells(
            &[0.0; 33],
            (
                &[0, 1, 2, 3, 4, 5, 6, 70],
                &[
                    CellType::Vertex,
                    CellType::Triangle,
                    CellType::Quadrilateral,
                ],
            ),
        );

        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err().to_string(),
            "Connectivity indices out of bounds for the given points, max index: 70, but number of points is 11"
        );
    }

    #[test]
    fn validate_points_and_cells_conn_mismatch() {
        let res = validate_points_and_cells(
            &[0.0; 33],
            (
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[
                    CellType::Vertex,
                    CellType::Edge,
                    CellType::Triangle,
                    CellType::Quadrilateral,
                ],
            ),
        );

        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of connectivities not match the expected number based on the cell types: 8 != 10"
        );
    }

    fn set(items: &[usize]) -> BTreeSet<usize> {
        items.iter().copied().collect()
    }

    #[test]
    fn validate_blocks_no_blocks() {
        let res = validate_blocks(&[], 3);
        assert_eq!(
            res.unwrap_err().to_string(),
            "At least one block is required"
        );
    }

    #[test]
    fn validate_blocks_invalid_name() {
        let cells = set(&[0]);
        let res = validate_blocks(&[("bad name", &cells)], 1);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Block name 'bad name' is not valid, must be non-empty and contain only alphanumeric characters, underscores or dashes"
        );
    }

    #[test]
    fn validate_blocks_duplicate_name() {
        let cells_a = set(&[0]);
        let cells_b = set(&[1]);
        let res = validate_blocks(&[("a", &cells_a), ("a", &cells_b)], 2);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Block name 'a' is used more than once"
        );
    }

    #[test]
    fn validate_blocks_empty_cell_indices() {
        let cells = set(&[]);
        let res = validate_blocks(&[("a", &cells)], 1);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Block 'a' must contain at least one cell"
        );
    }

    #[test]
    fn validate_blocks_index_out_of_bounds() {
        let cells = set(&[0, 5]);
        let res = validate_blocks(&[("a", &cells)], 3);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Block 'a' references cell index 5, but the mesh only has 3 cells"
        );
    }

    #[test]
    fn validate_blocks_uncovered_cells() {
        let cells = set(&[0]);
        let res = validate_blocks(&[("a", &cells)], 3);
        assert_eq!(
            res.unwrap_err().to_string(),
            "2 of 3 cells are not part of any block: 1, 2. If this is intentional, exclude them from the cells passed to write_mesh_with_blocks"
        );
    }

    #[test]
    fn validate_blocks_uncovered_cells_truncated() {
        let cells = set(&[0]);
        let res = validate_blocks(&[("a", &cells)], 15);
        assert_eq!(
            res.unwrap_err().to_string(),
            "14 of 15 cells are not part of any block: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... (4 more). If this is intentional, exclude them from the cells passed to write_mesh_with_blocks"
        );
    }

    #[test]
    fn validate_blocks_fully_covered_ok() {
        // overlapping blocks together covering every cell is fine
        let cells_a = set(&[0, 1]);
        let cells_b = set(&[1, 2]);
        validate_blocks(&[("a", &cells_a), ("b", &cells_b)], 3).unwrap();
    }

    #[test]
    fn validate_blocks_duplicate_indices_within_block_collapse() {
        // a `BTreeSet` structurally cannot contain a duplicate index, so passing e.g. `[0, 0, 1]`
        // just collapses to `{0, 1}`, covering cells 0 and 1 exactly once each.
        let cells = set(&[0, 0, 1]);
        assert_eq!(cells.len(), 2);
        validate_blocks(&[("a", &cells)], 2).unwrap();
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
                (&[0, 2, 3, 4], &[CellType::Vertex; 4]),
            )
            .unwrap();

        let point_vals: Values = vec![5.0; NUM_POINTS].into();
        let point_data = [("point_data1", DataAttribute::Scalar, &point_vals)];

        // Valid time step
        writer.write_data("0.1", point_data, []).unwrap();

        // Missing data
        let exp_err_missing_data = "At least one of point_data or cell_data must be provided";

        // neither point_data nor cell_data provided
        let res = writer.write_data("1.0", [], []);
        assert_eq!(res.unwrap_err().to_string(), exp_err_missing_data);

        // Invalid time step (already exists)
        let res = writer.write_data("0.1", point_data, []);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Time step '0.1' has already been written"
        );

        // Invalid time step (not a float)
        let res = writer.write_data("invalid_time", [], []);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Time must be a valid float, and not 'invalid_time'"
        );

        // Invalid time step (empty)
        let res = writer.write_data("", [], []);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Time must be a valid float, and not ''"
        );
    }

    #[test]
    fn test_write_data_duplicate_point_data_name() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        let mut writer = writer
            .write_mesh(&[0.0; 10 * 3], (&[0, 2, 3, 4], &[CellType::Vertex; 4]))
            .unwrap();

        let vals_a: Values = vec![5.0; 10].into();
        let vals_b: Values = vec![6.0; 10].into();
        let res = writer.write_data(
            "0.0",
            [
                ("dup", DataAttribute::Scalar, &vals_a),
                ("dup", DataAttribute::Scalar, &vals_b),
            ],
            [],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "point-data name 'dup' is used more than once"
        );
    }

    #[test]
    fn test_write_data_duplicate_cell_data_name() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        let mut writer = writer
            .write_mesh(&[0.0; 10 * 3], (&[0, 2, 3, 4], &[CellType::Vertex; 4]))
            .unwrap();

        let vals_a: Values = vec![5.0; 4].into();
        let vals_b: Values = vec![6.0; 4].into();
        let res = writer.write_data(
            "0.0",
            [],
            [
                ("dup", DataAttribute::Scalar, &vals_a),
                ("dup", DataAttribute::Scalar, &vals_b),
            ],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "cell-data name 'dup' is used more than once"
        );
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
                (&[0, 2, 3, 4], &[CellType::Vertex; 4]),
            )
            .unwrap();

        // scalar point data
        let scalar_vals: Values = vec![5.0; NUM_POINTS - 1].into();
        let res = writer.write_data(
            "0.0",
            [("point_data_sca", DataAttribute::Scalar, &scalar_vals)],
            [],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point-data 'point_data_sca' must be 10, but is 9"
        );

        // vector point data
        let vector_vals: Values = vec![5.0; NUM_POINTS * 2].into();
        let res = writer.write_data(
            "0.0",
            [("point_data_vec", DataAttribute::Vector, &vector_vals)],
            [],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point-data 'point_data_vec' must be 30, but is 20"
        );

        // Tensor point data
        let tensor_vals: Values = vec![5.0; NUM_POINTS * 3].into();
        let res = writer.write_data(
            "0.0",
            [("point_data_ten", DataAttribute::Tensor, &tensor_vals)],
            [],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point-data 'point_data_ten' must be 90, but is 30"
        );

        // Tensor6 point data
        let tensor6_vals: Values = vec![5.0; NUM_POINTS * 3].into();
        let res = writer.write_data(
            "0.0",
            [("point_data_ten6", DataAttribute::Tensor6, &tensor6_vals)],
            [],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point-data 'point_data_ten6' must be 60, but is 30"
        );

        // Matrix point data
        let matrix_vals: Values = vec![5.0; NUM_POINTS * 3 - 1].into();
        let res = writer.write_data(
            "0.0",
            [(
                "point_data_mat",
                DataAttribute::Matrix(2, 1),
                &matrix_vals,
            )],
            [],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point-data 'point_data_mat' must be 20, but is 29"
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
                (&[0, 2, 3, 4], &[CellType::Vertex; NUM_CELLS]),
            )
            .unwrap();

        // scalar cell data
        let scalar_vals: Values = vec![5.0; NUM_CELLS - 1].into();
        let res = writer.write_data(
            "0.0",
            [],
            [("cell_data_sca", DataAttribute::Scalar, &scalar_vals)],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell-data 'cell_data_sca' must be 4, but is 3"
        );

        // vector cell data
        let vector_vals: Values = vec![5.0; NUM_CELLS * 2].into();
        let res = writer.write_data(
            "0.0",
            [],
            [("cell_data_vec", DataAttribute::Vector, &vector_vals)],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell-data 'cell_data_vec' must be 12, but is 8"
        );

        // Tensor cell data
        let tensor_vals: Values = vec![5.0; NUM_CELLS * 3].into();
        let res = writer.write_data(
            "0.0",
            [],
            [("cell_data_ten", DataAttribute::Tensor, &tensor_vals)],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell-data 'cell_data_ten' must be 36, but is 12"
        );

        // Tensor6 cell data
        let tensor6_vals: Values = vec![5.0; NUM_CELLS * 3].into();
        let res = writer.write_data(
            "0.0",
            [],
            [("cell_data_ten6", DataAttribute::Tensor6, &tensor6_vals)],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell-data 'cell_data_ten6' must be 24, but is 12"
        );

        // Matrix cell data
        let matrix_vals: Values = vec![5.0; NUM_CELLS * 3 - 1].into();
        let res = writer.write_data(
            "0.0",
            [],
            [(
                "cell_data_mat",
                DataAttribute::Matrix(2, 1),
                &matrix_vals,
            )],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell-data 'cell_data_mat' must be 8, but is 11"
        );
    }

    #[test]
    fn test_validate_data_names() {
        let vals: Values = vec![0.0; 1].into();
        let data: BTreeMap<&str, (DataAttribute, &Values)> =
            [("cell_data_ten", (DataAttribute::Scalar, &vals))]
                .into_iter()
                .collect();

        validate_data_name(&data, "cell").unwrap();

        let data_invalid_name: BTreeMap<&str, (DataAttribute, &Values)> =
            [("cell[_data]_ten", (DataAttribute::Scalar, &vals))]
                .into_iter()
                .collect();

        let res = validate_data_name(&data_invalid_name, "point");
        assert_eq!(
            res.unwrap_err().to_string(),
            "Data name 'cell[_data]_ten' of point-data is not valid, must be non-empty and contain only alphanumeric characters, underscores or dashes"
        );
    }

    #[test]
    fn test_is_valid_data_name() {
        assert!(is_valid_data_name("valid_name"));
        assert!(is_valid_data_name("valid-name"));
        assert!(is_valid_data_name("valid_name_123"));
        assert!(!is_valid_data_name("")); // empty name
        assert!(!is_valid_data_name("invalid name")); // space
        assert!(!is_valid_data_name("invalid@name")); // special character
        assert!(!is_valid_data_name("invalid#name")); // special character
        assert!(!is_valid_data_name("invalid$name")); // special character
        assert!(!is_valid_data_name("invalid%name")); // special character
        assert!(!is_valid_data_name("invalid^name")); // special character
        assert!(!is_valid_data_name("invalid&name")); // special character
        assert!(!is_valid_data_name("invalid*name")); // special character
        assert!(!is_valid_data_name("invalid(name")); // special character
        assert!(!is_valid_data_name("invalid)name")); // special character
        assert!(!is_valid_data_name("invalid+name")); // special character
        assert!(!is_valid_data_name("invalid=name")); // special character
        assert!(!is_valid_data_name("invalid{name")); // special character
        assert!(!is_valid_data_name("invalid}name")); // special character
        assert!(!is_valid_data_name("invalid[name")); // special character
        assert!(!is_valid_data_name("invalid]name")); // special character
        assert!(!is_valid_data_name("invalid|name")); // special character
        assert!(!is_valid_data_name("invalid:name")); // special character
        assert!(!is_valid_data_name("invalid;name")); // special character
        assert!(!is_valid_data_name("invalid'")); // single quote
        assert!(!is_valid_data_name("invalid\"name")); // double quote
        assert!(!is_valid_data_name("invalid,name")); // comma
        assert!(!is_valid_data_name("invalid.name")); // dot
        assert!(!is_valid_data_name("invalid?name")); // question mark
        assert!(!is_valid_data_name("invalid/name")); // forward slash
        assert!(!is_valid_data_name("invalid\\name")); // backslash
        assert!(!is_valid_data_name("invalid\0name")); // null-char
    }

    #[test]
    fn test_validate_file_name() {
        validate_file_name(Path::new("asdf.txt")).unwrap();
        validate_file_name(Path::new("valid-name.txt")).unwrap();
        validate_file_name(Path::new("valid_name.txt")).unwrap();
        validate_file_name(Path::new("valid_name-123.txt")).unwrap();

        let res = validate_file_name(Path::new("valid_name:123.txt"));
        assert_eq!(
            res.unwrap_err().to_string(),
            "File name 'valid_name:123.txt' cannot contain the following characters: ['?', '\\0', ':', '*', '\"', '<', '>', '|']"
        );
    }

    #[test]
    fn test_write_data_preserve_order() {
        fn dummy_geometry() -> Geometry {
            Geometry {
                geometry_type: GeometryType::XYZ,
                data_item: DataItem {
                    dimensions: Some(Dimensions(vec![5, 3])),
                    data: "0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0".into(),
                    number_type: Some(NumberType::Float),
                    ..Default::default()
                },
            }
        }

        fn dummy_topology() -> Topology {
            Topology {
                topology_type: TopologyType::Triangle,
                number_of_elements: "2".into(),
                data_item: DataItem {
                    dimensions: Some(Dimensions(vec![6])),
                    number_type: Some(NumberType::Int),
                    data: "0 1 2 2 3 4".into(),
                    ..Default::default()
                },
            }
        }

        struct DummyWriter;

        impl DataWriter for DummyWriter {
            fn format(&self) -> Format {
                Format::XML
            }

            fn data_storage(&self) -> DataStorage {
                DataStorage::AsciiInline
            }

            fn write_mesh(
                &mut self,
                _points: &[f64],
                _cells: &[u64],
            ) -> IoResult<(DataContent, DataContent)> {
                Ok((
                    DataContent::Raw("points".to_string()),
                    DataContent::Raw("cells".to_string()),
                ))
            }

            fn write_mesh_block(&mut self, name: &str, _cells: &[u64]) -> IoResult<DataContent> {
                Ok(DataContent::Raw(format!("block_for_{name}")))
            }

            fn write_data(
                &mut self,
                name: &str,
                _center: attribute::Center,
                _data: &crate::values::Values,
            ) -> IoResult<DataContent> {
                Ok(DataContent::Raw(format!("data_for_{name}")))
            }
        }

        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_write_data_preserve_order.xdmf2");

        let mut writer = TimeSeriesDataWriter {
            xdmf_file_name: xdmf_file_path.clone(),
            writer: Box::new(DummyWriter),
            grid: Grid::new_uniform("test", dummy_geometry(), dummy_topology()),
            data_items: Vec::new(),
            num_points: 0,
            num_cells: 0,
            attributes: Vec::new(),
            block_attributes: Vec::new(),
            blocks: None,
            writen_times: HashSet::new(),
        };

        let scalar_vals: Values = vec![0.0; 0].into();
        let point_data = [("scalar_data", DataAttribute::Scalar, &scalar_vals)];

        writer.write_data("0.0", point_data, []).unwrap();
        writer.write_data("1.0", point_data, []).unwrap();
        writer.write_data("2.0", point_data, []).unwrap();
        writer.write_data("10.0", point_data, []).unwrap();

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
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_scalar_data</DataItem>
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
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_scalar_data</DataItem>
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
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_scalar_data</DataItem>
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
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_scalar_data</DataItem>
                </Attribute>
            </Grid>
        </Grid>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="0.1.3"/>
</Xdmf>"#;

        let xdmf_file = xdmf_file_path.with_extension("xdmf2");
        let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

        // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
        // std::fs::copy(xdmf_file, "time_series_writer_only_mesh.xdmf").unwrap();

        pretty_assertions::assert_eq!(expected_xdmf, read_xdmf);
    }
}

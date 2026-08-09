//! This module contains functionalities for writing a series of time steps to XDMF.
//!
//! The mesh is written only once and then referenced in each time step.
//! This is a significant advantage over VTK based formats, making it more efficient both in terms of storage size as well as write speed.
//!
//! The concept is inspired by the `TimeSeriesWriter` of [meshio](https://github.com/nschloe/meshio)

use std::{
    collections::{HashMap, HashSet},
    io::{BufWriter, Error as IoError, ErrorKind::InvalidInput, Result as IoResult, Write},
    path::{Path, PathBuf},
};

use crate::{
    CellType, DataAttribute, DataStorage, DataWriter, Values, create_writer,
    mpi_safe_create_dir_all,
    xdmf_elements::{
        Information, Xdmf, attribute,
        data_item::{DataItem, NumberType},
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
    /// let mut ts_writer = xdmf_writer.write_mesh(&coords, &connectivity, &cell_types);
    /// ```
    pub fn write_mesh(
        mut self,
        points: &[f64],
        connectivity: &[u64],
        cell_types: &[CellType],
    ) -> IoResult<TimeSeriesDataWriter> {
        validate_points_and_cells(points, connectivity, cell_types)?;

        let num_points = points.len() / 3;
        let num_cells = if cell_types.is_empty() {
            num_points
        } else {
            cell_types.len()
        };

        let (topo_type, prepared_cells) = prepare_cells(connectivity, cell_types, num_points);

        let (points_data, cells_data) = self.writer.write_mesh(points, &prepared_cells)?;

        let format = self.writer.format();

        let data_item_coords = DataItem {
            name: Some("coords".to_string()),
            dimensions: Some(Dimensions(vec![num_points, 3])),
            data: points_data,
            number_type: Some(NumberType::Float),
            precision: Some(8),
            format: Some(format),
            endian: format.endian(),
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
            written_times: HashMap::new(),
            num_points,
            num_cells,
        };

        ts_writer.write()?;

        Ok(ts_writer)
    }
}

// Validate that the points and cells are valid
fn validate_points_and_cells(
    points: &[f64],
    connectivity: &[u64],
    cell_types: &[CellType],
) -> IoResult<()> {
    // at least one point is required
    if points.is_empty() {
        return Err(IoError::new(InvalidInput, "At least one point is required"));
    }

    // check that points are a multiple of 3 (x, y, z)
    if !points.len().is_multiple_of(3) {
        return Err(IoError::new(InvalidInput, "Points must have 3 dimensions"));
    }

    // check cells connectivity indices
    let max_connectivity_index = connectivity.iter().max();

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
    let exp_num_points: usize = cell_types.iter().map(|ct| ct.num_points()).sum();
    if exp_num_points != connectivity.len() {
        return Err(IoError::new(
            InvalidInput,
            format!(
                "Size of connectivities not match the expected number based on the cell types: {} != {}",
                connectivity.len(),
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
/// TODO if all cells are the same, then the type information can be stored as `TopologyType`
fn prepare_cells(
    connectivity: &[u64],
    cell_types: &[CellType],
    num_points: usize,
) -> (TopologyType, Vec<u64>) {
    if cell_types.is_empty() {
        // if there are no cells, use polyvertex on nodes
        // this is required by paraview to visualize only points
        return (TopologyType::Polyvertex, (0..num_points as u64).collect());
    }

    let mut cells_with_types = Vec::with_capacity(connectivity.len() + cell_types.len());
    let mut index = 0_usize;

    for cell_type in cell_types {
        let num_points = cell_type.num_points();
        cells_with_types.push(*cell_type as u64);

        if let Some(n_points_poly) = poly_cell_points(*cell_type) {
            // poly-cells need to specify the number of points
            cells_with_types.push(n_points_poly);
        }

        cells_with_types.extend_from_slice(&connectivity[index..index + num_points]);

        index += num_points; // move index to the next cell
    }

    (TopologyType::Mixed, cells_with_types)
}

/// Writer for time series data in XDMF format. Can be used after writing the mesh with `TimeSeriesWriter::write_mesh`.
pub struct TimeSeriesDataWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
    grid: Grid,
    data_items: Vec<DataItem>,
    attributes: Vec<(String, Vec<attribute::Attribute>)>,
    // Keyed on `f64::to_bits` of the parsed time, not the caller's string, so two spellings of
    // the same instant (e.g. "0.1" and "0.10") are recognized as the same duplicate. The value
    // is the spelling first used, for the error message.
    written_times: HashMap<u64, String>,
    num_points: usize,
    num_cells: usize,
}

impl TimeSeriesDataWriter {
    /// Write point and cell data for a specific time step.
    ///
    /// Each entry is a `(name, attribute, values)` triple; pass an empty iterator (e.g. `[]`) to
    /// write no data of that kind. Attributes appear in the output in the order given here.
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
    ///     .write_mesh(&coords, &connectivity, &cell_types)
    ///     .expect("failed to write mesh");
    ///
    /// // buffers are reused across time steps, `Values` only borrows them
    /// let point_values = vec![0.0; 9];
    /// let cell_values = vec![0.0, 1.0];
    ///
    /// // write the data for 10 time steps
    /// for i in 0..10 {
    ///     time_series_writer
    ///         .write_data(
    ///             &i.to_string(),
    ///             [(
    ///                 "point_data",
    ///                 xdmf::DataAttribute::Vector,
    ///                 point_values.as_slice().into(),
    ///             )],
    ///             [(
    ///                 "cell_data",
    ///                 xdmf::DataAttribute::Scalar,
    ///                 cell_values.as_slice().into(),
    ///             )],
    ///         )
    ///         .expect("failed to write time step data");
    /// }
    /// ```
    pub fn write_data<'a>(
        &mut self,
        time: &str,
        point_data: impl IntoIterator<Item = (&'a str, DataAttribute, Values<'a>)>,
        cell_data: impl IntoIterator<Item = (&'a str, DataAttribute, Values<'a>)>,
    ) -> IoResult<()> {
        let point_data = collect_data(point_data, "point")?;
        let cell_data = collect_data(cell_data, "cell")?;

        let time_bits = self.validate_data(time, &point_data, &cell_data)?;

        self.writer.write_data_initialize(time)?;

        // Run `write_data_finalize` regardless of whether writing the attributes below
        // succeeds, so a mid-write failure leaves the backing writer's `write_data_initialize`/
        // `write_data_finalize` pairing balanced instead of poisoning every later time step.
        // The write error (if any) must win over a finalize error, not be masked by it.
        let write_result = self.write_attributes(&point_data, &cell_data);
        let finalize_result = self.writer.write_data_finalize();

        let new_attributes = write_result?;
        finalize_result?;

        self.attributes.push((time.to_string(), new_attributes));
        self.written_times.insert(time_bits, time.to_string());

        self.write()
    }

    fn write_attributes(
        &mut self,
        point_data: &[(&str, DataAttribute, Values<'_>)],
        cell_data: &[(&str, DataAttribute, Values<'_>)],
    ) -> IoResult<Vec<attribute::Attribute>> {
        let format = self.writer.format();
        let mut new_attributes = Vec::with_capacity(point_data.len() + cell_data.len());

        for (data, center) in [
            (point_data, attribute::Center::Node),
            (cell_data, attribute::Center::Cell),
        ] {
            for (data_name, data_attribute, vals) in data {
                let data_item = DataItem {
                    name: None,
                    dimensions: Some(vals.dimensions(*data_attribute)),
                    number_type: Some(vals.number_type()),
                    format: Some(format),
                    precision: Some(vals.precision(format)),
                    endian: format.endian(),
                    data: self.writer.write_data(data_name, center, vals)?,
                    reference: None,
                };

                new_attributes.push(attribute::Attribute {
                    name: (*data_name).to_string(),
                    attribute_type: (*data_attribute).into(),
                    center,
                    data_items: vec![data_item],
                });
            }
        }

        Ok(new_attributes)
    }

    fn write(&mut self) -> IoResult<()> {
        self.writer.flush()?;

        // create the XDMF structure
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

        let temporal_grid =
            Grid::new_collection("time_series", CollectionType::Temporal, Some(time_grids));

        // If there are no attributes aka time-data, write the grid directly
        let grid_to_write = if self.attributes.is_empty() {
            self.grid.clone()
        } else {
            temporal_grid
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

    // Returns the bit pattern of the parsed time on success, so the caller does not have to
    // re-parse (and re-unwrap) a value already known to be a valid float.
    fn validate_data(
        &self,
        time: &str,
        point_data: &[(&str, DataAttribute, Values<'_>)],
        cell_data: &[(&str, DataAttribute, Values<'_>)],
    ) -> IoResult<u64> {
        let parsed_time = time.parse::<f64>().map_err(|_parse_error| {
            IoError::new(
                InvalidInput,
                format!("Time must be a valid float, and not '{time}'"),
            )
        })?;
        let time_bits = parsed_time.to_bits();

        // check if the time step has already been written, keyed on the parsed value rather
        // than the string so different spellings of the same instant are caught too
        if let Some(existing) = self.written_times.get(&time_bits) {
            let message = if existing == time {
                format!("Time step '{time}' has already been written")
            } else {
                format!("Time step '{time}' has already been written (as '{existing}')")
            };
            return Err(IoError::new(InvalidInput, message));
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
        validate_data_name(cell_data, "cell")?;

        // reject values the backend's format cannot represent (e.g. binary's u64->u32 range)
        // up front, before write_data_initialize runs, so a caller mistake here can never leave
        // the writer poisoned for the next call
        for (_, _, vals) in point_data.iter().chain(cell_data.iter()) {
            self.writer.validate_values(vals)?;
        }

        Ok(time_bits)
    }
}

// Collect the caller's iterator, keeping its order but rejecting a name used more than once
// (which would otherwise produce two attributes of the same name in the same grid).
fn collect_data<'a>(
    data: impl IntoIterator<Item = (&'a str, DataAttribute, Values<'a>)>,
    label: &str,
) -> IoResult<Vec<(&'a str, DataAttribute, Values<'a>)>> {
    let collected: Vec<_> = data.into_iter().collect();

    let mut seen_names = HashSet::with_capacity(collected.len());
    for (name, _, _) in &collected {
        if !seen_names.insert(name) {
            return Err(IoError::new(
                InvalidInput,
                format!("Name '{name}' of {label}-data is used more than once"),
            ));
        }
    }

    Ok(collected)
}

// check sizes of point_data and cell_data
fn check_data_size(
    data_input: &[(&str, DataAttribute, Values<'_>)],
    num_entities: usize,
    label: &str,
) -> IoResult<()> {
    for (name, data_attribute, vals) in data_input {
        let exp_size = num_entities * data_attribute.size();
        if vals.len() != exp_size {
            return Err(IoError::new(
                InvalidInput,
                format!(
                    "Size of {label}-data '{name}' must be {}, but is {}",
                    exp_size,
                    vals.len()
                ),
            ));
        }
    }
    Ok(())
}

fn validate_data_name(
    data_input: &[(&str, DataAttribute, Values<'_>)],
    label: &str,
) -> IoResult<()> {
    for (name, _, _) in data_input {
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
    // Only validate the final path component, the parent directories are not under our control
    // and may legitimately contain characters such as ':' (e.g. Windows drive letters).
    let Some(name) = file_name.file_name().and_then(|name| name.to_str()) else {
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
        let (topo_type, cells_prep) = prepare_cells(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
            0,
        );

        assert_eq!(topo_type, TopologyType::Mixed);
        assert_eq!(
            cells_prep,
            vec![1, 1, 0, 2, 2, 1, 2, 4, 3, 4, 5, 5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn prepare_cells_by_celltype() {
        assert_eq!(prepare_cells(&[5], &[CellType::Vertex], 0).1, vec![1, 1, 5]);

        assert_eq!(
            prepare_cells(&[5, 6], &[CellType::Edge], 0).1,
            vec![2, 2, 5, 6]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7], &[CellType::Triangle], 0).1,
            vec![4, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8], &[CellType::Quadrilateral], 0).1,
            vec![5, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8], &[CellType::Tetrahedron], 0).1,
            vec![6, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9], &[CellType::Pyramid], 0).1,
            vec![7, 5, 6, 7, 8, 9]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9, 10], &[CellType::Wedge], 0).1,
            vec![8, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Hexahedron], 0).1,
            vec![9, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7], &[CellType::Edge3], 0).1,
            vec![34, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells(
                &[5, 6, 7, 8, 9, 10, 11, 12, 13],
                &[CellType::Quadrilateral9],
                0
            )
            .1,
            vec![35, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9, 10], &[CellType::Triangle6], 0).1,
            vec![36, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Quadrilateral8], 0).1,
            vec![37, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells(
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                &[CellType::Tetrahedron10],
                0
            )
            .1,
            vec![38, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        );

        assert_eq!(
            prepare_cells(
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                &[CellType::Pyramid13],
                0
            )
            .1,
            vec![39, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        );

        assert_eq!(
            prepare_cells(
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                &[CellType::Wedge15],
                0
            )
            .1,
            vec![40, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );

        assert_eq!(
            prepare_cells(
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
                ],
                &[CellType::Wedge18],
                0
            )
            .1,
            vec![
                41, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
            ]
        );

        assert_eq!(
            prepare_cells(
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                ],
                &[CellType::Hexahedron20],
                0
            )
            .1,
            vec![
                48, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
            ]
        );

        assert_eq!(
            prepare_cells(
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28
                ],
                &[CellType::Hexahedron24],
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
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31
                ],
                &[CellType::Hexahedron27],
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
        let (topo_type, cells_prep) = prepare_cells(&[], &[], 5);

        assert_eq!(topo_type, TopologyType::Polyvertex);
        assert_eq!(cells_prep, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_validate_points_and_cells() {
        // valid input, must not return an error
        validate_points_and_cells(
            &[0.0; 33],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        )
        .unwrap();
    }

    #[test]
    fn validate_points_and_cells_only_points() {
        // valid input, must not return an error
        validate_points_and_cells(&[0.0; 33], &[], &[]).unwrap();
    }

    #[test]
    fn validate_points_and_cells_points_empty() {
        let res = validate_points_and_cells(
            &[],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        assert_eq!(
            res.unwrap_err().to_string(),
            "At least one point is required"
        );
    }

    #[test]
    fn validate_points_and_cells_points_not_3d() {
        let res = validate_points_and_cells(
            &[0.0; 22],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        assert_eq!(
            res.unwrap_err().to_string(),
            "Points must have 3 dimensions"
        );
    }

    #[test]
    fn validate_points_and_cells_conn_index_out_of_bounds() {
        let res = validate_points_and_cells(
            &[0.0; 33],
            &[0, 1, 2, 3, 4, 5, 6, 70],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        assert_eq!(
            res.unwrap_err().to_string(),
            "Connectivity indices out of bounds for the given points, max index: 70, but number of points is 11"
        );
    }

    #[test]
    fn validate_points_and_cells_conn_mismatch() {
        let res = validate_points_and_cells(
            &[0.0; 33],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of connectivities not match the expected number based on the cell types: 8 != 10"
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
        let point_data = || {
            [(
                "point_data1",
                DataAttribute::Scalar,
                values.as_slice().into(),
            )]
        };

        // Valid time step
        writer.write_data("0.1", point_data(), []).unwrap();

        // no data at all provided
        let res = writer.write_data("1.0", [], []);
        assert_eq!(
            res.unwrap_err().to_string(),
            "At least one of point_data or cell_data must be provided"
        );

        // Invalid time step (already exists)
        let res = writer.write_data("0.1", point_data(), []);
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
        let point_data = || {
            [(
                "point_data1",
                DataAttribute::Scalar,
                values.as_slice().into(),
            )]
        };

        writer.write_data("0.1", point_data(), []).unwrap();

        // a different spelling of the same numeric value is still a duplicate
        let res = writer.write_data("0.10", point_data(), []);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Time step '0.10' has already been written (as '0.1')"
        );

        // a genuinely different value is accepted
        writer.write_data("0.2", point_data(), []).unwrap();
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

        let res = writer.write_data(
            "0.0",
            [
                ("duplicate", DataAttribute::Scalar, values.as_slice().into()),
                ("duplicate", DataAttribute::Scalar, values.as_slice().into()),
            ],
            [],
        );
        assert_eq!(
            res.unwrap_err().to_string(),
            "Name 'duplicate' of point-data is used more than once"
        );

        // the same name for point- and cell-data is allowed, they are separate entities
        let cell_values = vec![5.0; 4];
        writer
            .write_data(
                "0.0",
                [("data", DataAttribute::Scalar, values.as_slice().into())],
                [("data", DataAttribute::Scalar, cell_values.as_slice().into())],
            )
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

        let mut err_for = |name: &str, attribute: DataAttribute, len: usize| -> String {
            writer
                .write_data("0.0", [(name, attribute, vec![5.0; len].into())], [])
                .unwrap_err()
                .to_string()
        };

        assert_eq!(
            err_for("point_data_sca", DataAttribute::Scalar, NUM_POINTS - 1),
            "Size of point-data 'point_data_sca' must be 10, but is 9"
        );
        assert_eq!(
            err_for("point_data_vec", DataAttribute::Vector, NUM_POINTS * 2),
            "Size of point-data 'point_data_vec' must be 30, but is 20"
        );
        assert_eq!(
            err_for("point_data_ten", DataAttribute::Tensor, NUM_POINTS * 3),
            "Size of point-data 'point_data_ten' must be 90, but is 30"
        );
        assert_eq!(
            err_for("point_data_ten6", DataAttribute::Tensor6, NUM_POINTS * 3),
            "Size of point-data 'point_data_ten6' must be 60, but is 30"
        );
        assert_eq!(
            err_for(
                "point_data_mat",
                DataAttribute::Matrix(2, 1),
                NUM_POINTS * 3 - 1
            ),
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
                &[0, 2, 3, 4],
                &[CellType::Vertex; NUM_CELLS],
            )
            .unwrap();

        let mut err_for = |name: &str, attribute: DataAttribute, len: usize| -> String {
            writer
                .write_data("0.0", [], [(name, attribute, vec![5.0; len].into())])
                .unwrap_err()
                .to_string()
        };

        assert_eq!(
            err_for("cell_data_sca", DataAttribute::Scalar, NUM_CELLS - 1),
            "Size of cell-data 'cell_data_sca' must be 4, but is 3"
        );
        assert_eq!(
            err_for("cell_data_vec", DataAttribute::Vector, NUM_CELLS * 2),
            "Size of cell-data 'cell_data_vec' must be 12, but is 8"
        );
        assert_eq!(
            err_for("cell_data_ten", DataAttribute::Tensor, NUM_CELLS * 3),
            "Size of cell-data 'cell_data_ten' must be 36, but is 12"
        );
        assert_eq!(
            err_for("cell_data_ten6", DataAttribute::Tensor6, NUM_CELLS * 3),
            "Size of cell-data 'cell_data_ten6' must be 24, but is 12"
        );
        assert_eq!(
            err_for(
                "cell_data_mat",
                DataAttribute::Matrix(2, 1),
                NUM_CELLS * 3 - 1
            ),
            "Size of cell-data 'cell_data_mat' must be 8, but is 11"
        );
    }

    #[test]
    fn test_validate_data_names() {
        let data = [("cell_data_ten", DataAttribute::Scalar, vec![0.0; 1].into())];

        validate_data_name(&data, "cell").unwrap();

        let data_invalid_name = [(
            "cell[_data]_ten",
            DataAttribute::Scalar,
            vec![0.0; 1].into(),
        )];

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

            fn write_data(
                &mut self,
                name: &str,
                _center: attribute::Center,
                _data: &Values<'_>,
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
            written_times: HashMap::new(),
        };

        let point_data = || [("scalar_data", DataAttribute::Scalar, vec![0.0; 0].into())];

        writer.write_data("0.0", point_data(), []).unwrap();
        writer.write_data("1.0", point_data(), []).unwrap();
        writer.write_data("2.0", point_data(), []).unwrap();
        writer.write_data("10.0", point_data(), []).unwrap();

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

    #[test]
    fn write_data_survives_a_mid_write_failure() {
        // A backend that fails while writing one attribute, after already having written an
        // earlier one -- reproducing the shape of the original binary-backend bug (now caught
        // upfront for that specific case by `DataWriter::validate_values`, but the
        // `write_data_finalize`-always-runs fix in `TimeSeriesDataWriter::write_data` is generic
        // and backend-agnostic, so it needs its own backend-agnostic regression test).
        struct FlakyWriter {
            write_time: Option<String>,
        }

        impl DataWriter for FlakyWriter {
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

            fn write_data(
                &mut self,
                name: &str,
                _center: attribute::Center,
                _data: &Values<'_>,
            ) -> IoResult<DataContent> {
                if name == "boom" {
                    return Err(IoError::other("simulated mid-write failure"));
                }
                Ok(DataContent::Raw(format!("data_for_{name}")))
            }

            fn write_data_initialize(&mut self, time: &str) -> IoResult<()> {
                if self.write_time.is_some() {
                    return Err(IoError::other("Writing data was already initialized"));
                }
                self.write_time = Some(time.to_string());
                Ok(())
            }

            fn write_data_finalize(&mut self) -> IoResult<()> {
                if self.write_time.is_none() {
                    return Err(IoError::other("Writing data was not initialized"));
                }
                self.write_time = None;
                Ok(())
            }
        }

        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("mid_write_failure.xdmf2");

        let mut writer = TimeSeriesDataWriter {
            xdmf_file_name: xdmf_file_path,
            writer: Box::new(FlakyWriter { write_time: None }),
            grid: Grid::new_uniform("test", dummy_geometry(), dummy_topology()),
            data_items: Vec::new(),
            num_points: 0,
            num_cells: 0,
            attributes: Vec::new(),
            written_times: HashMap::new(),
        };

        // "ok" is written successfully before "boom" fails, so this genuinely fails partway
        // through the attribute loop, after `write_data_initialize` already ran.
        let res = writer.write_data(
            "0.0",
            [
                ("ok", DataAttribute::Scalar, vec![0.0; 0].into()),
                ("boom", DataAttribute::Scalar, vec![0.0; 0].into()),
            ],
            [],
        );
        assert_eq!(res.unwrap_err().to_string(), "simulated mid-write failure");

        // The failed step must not have consumed the time slot ("0.0" is retried, not a new
        // time) and must not have left the backing writer poisoned: `FlakyWriter` itself would
        // fail with "Writing data was already initialized" here if `write_data_finalize` had
        // been skipped on the error path above.
        writer
            .write_data(
                "0.0",
                [("ok", DataAttribute::Scalar, vec![0.0; 0].into())],
                [],
            )
            .unwrap();
    }
}

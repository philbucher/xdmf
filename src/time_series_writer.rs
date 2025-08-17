use std::{
    collections::BTreeMap,
    io::{BufWriter, Error as IoError, ErrorKind::InvalidInput, Result as IoResult, Write},
    path::{Path, PathBuf},
};

use crate::{
    CellType, DataMap, DataStorage, DataWriter, create_writer, mpi_safe_create_dir_all,
    xdmf_elements::{
        Information, Xdmf, attribute,
        data_item::{DataItem, NumberType},
        dimensions::Dimensions,
        geometry::{Geometry, GeometryType},
        grid::{CollectionType, Grid, GridType, Time},
        topology::{Topology, TopologyType},
    },
};

pub struct TimeSeriesWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
}

impl TimeSeriesWriter {
    /// # Errors
    ///
    /// TODO
    pub fn new(file_name: impl AsRef<Path>, data_storage: DataStorage) -> IoResult<Self> {
        let xdmf_file_name = file_name.as_ref().to_path_buf().with_extension("xdmf2");

        // create the parent directory if it does not exist
        if let Some(parent) = xdmf_file_name.parent() {
            mpi_safe_create_dir_all(parent)?;
        }

        Ok(Self {
            xdmf_file_name,
            writer: create_writer(file_name.as_ref(), data_storage)?,
        })
    }

    /// # Errors
    ///
    /// TODO
    pub fn write_mesh(
        mut self,
        points: &[f64],
        cells: (&[u64], &[CellType]),
    ) -> IoResult<TimeSeriesDataWriter> {
        validate_points_and_cells(points, cells)?;

        let num_cells = cells.1.len();

        let prepared_cells = prepare_cells(cells);

        let (points_data, cells_data) = self.writer.write_mesh(points, &prepared_cells)?;

        let data_item_coords = DataItem {
            name: Some("coords".to_string()),
            dimensions: Some(Dimensions(vec![points.len() / 3, 3])),
            data: points_data,
            number_type: Some(NumberType::Float),
            precision: Some(8),
            format: Some(self.writer.format()),
            reference: None,
        };

        let data_item_connectivity = DataItem {
            name: Some("connectivity".to_string()),
            dimensions: Some(Dimensions(vec![prepared_cells.len()])),
            number_type: Some(NumberType::UInt),
            data: cells_data,
            format: Some(self.writer.format()),
            precision: Some(8),
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
            topology_type: TopologyType::Mixed,
            number_of_elements: num_cells.to_string(),
            data_item: data_item_connectivity_ref,
        };

        let mut ts_writer = TimeSeriesDataWriter {
            xdmf_file_name: self.xdmf_file_name,
            writer: self.writer,
            grid: Grid::new_uniform("mesh", geometry, topology),
            data_items: vec![data_item_coords, data_item_connectivity],
            attributes: BTreeMap::new(),
            num_points: points.len() / 3,
            num_cells,
        };

        ts_writer.write()?;

        Ok(ts_writer)
    }

    // TODO check if indices are within bounds of points and cells
    // TODO use SpatialCollection when submeshes are used
    // TODO each tolologytype can only appear once, otherwise indexing for submeshes will be wrong
    #[cfg(feature = "unstable-submesh-api")]
    pub fn write_mesh_and_submeshes(
        self,
        points: &[f64],
        cells: (&[u64], &[CellType]),
        submeshes: &BTreeMap<String, SubMesh>,
    ) -> IoResult<TimeSeriesDataWriter> {
        let mut ts = self.write_mesh(points, cells)?;

        let format = ts.writer.format();

        for (submesh_name, submesh) in submeshes {
            let name_points = format!("{submesh_name}_points");
            let name_cells = format!("{submesh_name}_cells");

            let (points_data, cells_data) = ts.writer.write_submesh(
                submesh_name,
                &submesh.point_indices,
                &submesh.cell_indices,
            )?;

            ts.xdmf.domains[0].data_items.push(DataItem {
                data: points_data,
                name: Some(name_points),
                dimensions: Some(Dimensions(vec![submesh.point_indices.len()])),
                number_type: Some(NumberType::UInt),
                format: Some(format),
                precision: Some(8),
                reference: None,
            });

            ts.xdmf.domains[0].data_items.push(DataItem {
                data: cells_data,
                name: Some(name_cells),
                dimensions: Some(Dimensions(vec![submesh.cell_indices.len()])),
                number_type: Some(NumberType::UInt),
                format: Some(format),
                precision: Some(8),
                reference: None,
            });
        }

        Ok(ts)
    }
}

#[cfg(feature = "unstable-submesh-api")]
pub struct SubMesh {
    pub point_indices: Vec<u64>,
    pub cell_indices: Vec<u64>,
}

// Validate that the points and cells are valid
fn validate_points_and_cells(points: &[f64], cells: (&[u64], &[CellType])) -> IoResult<()> {
    // at least one point is required
    if points.is_empty() {
        return Err(IoError::new(InvalidInput, "At least one point is required"));
    }

    // check that points are a multiple of 3 (x, y, z)
    if points.len() % 3 != 0 {
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
/// TODO if all cells are the same, then the type information can be stored as `TopologyType`
fn prepare_cells(cells: (&[u64], &[CellType])) -> Vec<u64> {
    let mut cells_with_types = Vec::with_capacity(cells.0.len() + cells.1.len());
    let mut index = 0_usize;

    for cell_type in cells.1 {
        let num_points = cell_type.num_points();
        cells_with_types.push(*cell_type as u64);

        if let Some(n_points_poly) = poly_cell_points(*cell_type) {
            // poly-cells need to specify the number of points
            cells_with_types.push(n_points_poly);
        }

        cells_with_types.extend_from_slice(&cells.0[index..index + num_points]);

        index += num_points; // move index to the next cell
    }

    cells_with_types
}

pub struct TimeSeriesDataWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
    grid: Grid,
    data_items: Vec<DataItem>,
    attributes: BTreeMap<String, Vec<attribute::Attribute>>,
    num_points: usize,
    num_cells: usize,
}

impl TimeSeriesDataWriter {
    /// Write data for a specific time step.
    /// Accepts str for time to avoid dealing with formatting, thus leaving it to the user.
    // TODOs:
    // - maybe write data as ref in attribute, to make cloning cheaper. Really only matters for XML format, so unsure if worth it.
    /// # Errors
    ///
    /// TODO
    pub fn write_data(
        &mut self,
        time: &str,
        point_data: Option<&DataMap>,
        cell_data: Option<&DataMap>,
    ) -> IoResult<()> {
        self.validate_data(time, point_data, cell_data)?;

        self.writer.write_data_initialize(time)?;
        let format = self.writer.format();

        let mut new_attributes = Vec::new();

        let mut create_attributes =
            |data_map: Option<&DataMap>, center: attribute::Center| -> IoResult<()> {
                for (data_name, data) in data_map.unwrap_or(&BTreeMap::new()) {
                    let vals = &data.1;

                    let data_item = DataItem {
                        name: None,
                        dimensions: Some(vals.dimensions(data.0)),
                        number_type: Some(vals.number_type()),
                        format: Some(format),
                        precision: Some(vals.precision()),
                        data: self.writer.write_data(data_name, center, vals)?,
                        reference: None,
                    };

                    let attribute = attribute::Attribute {
                        name: data_name.clone(),
                        attribute_type: data.0.into(),
                        center,
                        data_items: vec![data_item],
                    };

                    new_attributes.push(attribute);
                }

                Ok(())
            };

        create_attributes(point_data, attribute::Center::Node)?;
        create_attributes(cell_data, attribute::Center::Cell)?;

        self.attributes
            .entry(time.to_string())
            .or_default()
            .extend(new_attributes);

        self.writer.write_data_finalize()?;

        self.write()
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

    fn validate_data(
        &self,
        time: &str,
        point_data: Option<&DataMap>,
        cell_data: Option<&DataMap>,
    ) -> IoResult<()> {
        // check if time can be parsed as a float
        if time.parse::<f64>().is_err() {
            return Err(IoError::new(
                InvalidInput,
                format!("Time must be a valid float, and not '{time}'"),
            ));
        }

        // check if the time step has already been written
        if self.attributes.contains_key(time) {
            return Err(IoError::new(
                InvalidInput,
                format!("Time step '{time}' has already been written"),
            ));
        }

        // check if some data is provided
        if (point_data.unwrap_or(&BTreeMap::new()).len()
            + cell_data.unwrap_or(&BTreeMap::new()).len())
            == 0
        {
            return Err(IoError::new(
                InvalidInput,
                "At least one of point_data or cell_data must be provided",
            ));
        }

        // check sizes of point_data and cell_data
        fn check_data_size(
            data_input: Option<&DataMap>,
            num_entities: usize,
            label: &str,
        ) -> IoResult<()> {
            if let Some(data_map) = data_input {
                for (name, data) in data_map {
                    // attribute has a fixed size per entity, e.g. scalar, vector, tensor
                    let exp_size = num_entities * data.0.size();
                    if data.1.len() != exp_size {
                        return Err(IoError::new(
                            InvalidInput,
                            format!(
                                "Size of {label} data '{name}' must be {}, but is {}",
                                exp_size,
                                data.1.len()
                            ),
                        ));
                    }
                }
            }
            Ok(())
        }

        check_data_size(point_data, self.num_points, "point")?;
        check_data_size(cell_data, self.num_cells, "cell")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataAttribute;

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
        let cells_prep = prepare_cells((
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        ));

        assert_eq!(
            cells_prep,
            vec![1, 1, 0, 2, 2, 1, 2, 4, 3, 4, 5, 5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn prepare_cells_by_celltype() {
        assert_eq!(prepare_cells((&[5], &[CellType::Vertex])), vec![1, 1, 5]);

        assert_eq!(
            prepare_cells((&[5, 6], &[CellType::Edge])),
            vec![2, 2, 5, 6]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7], &[CellType::Triangle])),
            vec![4, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8], &[CellType::Quadrilateral])),
            vec![5, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8], &[CellType::Tetrahedron])),
            vec![6, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9], &[CellType::Pyramid])),
            vec![7, 5, 6, 7, 8, 9]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10], &[CellType::Wedge])),
            vec![8, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Hexahedron])),
            vec![9, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7], &[CellType::Edge3])),
            vec![34, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells((
                &[5, 6, 7, 8, 9, 10, 11, 12, 13],
                &[CellType::Quadrilateral9]
            )),
            vec![35, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10], &[CellType::Triangle6])),
            vec![36, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Quadrilateral8])),
            vec![37, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells((
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                &[CellType::Tetrahedron10]
            )),
            vec![38, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        );

        assert_eq!(
            prepare_cells((
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                &[CellType::Pyramid13]
            )),
            vec![39, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        );

        assert_eq!(
            prepare_cells((
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                &[CellType::Wedge15]
            )),
            vec![40, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );

        assert_eq!(
            prepare_cells((
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
                ],
                &[CellType::Wedge18]
            )),
            vec![
                41, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
            ]
        );

        assert_eq!(
            prepare_cells((
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                ],
                &[CellType::Hexahedron20]
            )),
            vec![
                48, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
            ]
        );

        assert_eq!(
            prepare_cells((
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28
                ],
                &[CellType::Hexahedron24]
            )),
            vec![
                49, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28
            ]
        );

        assert_eq!(
            prepare_cells((
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31
                ],
                &[CellType::Hexahedron27]
            )),
            vec![
                50, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31
            ]
        );
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

        let point_data = vec![(
            "point_data1".to_string(),
            (DataAttribute::Scalar, vec![5.0; NUM_POINTS].into()),
        )]
        .into_iter()
        .collect();

        // Valid time step
        writer.write_data("0.1", Some(&point_data), None).unwrap();

        // Missing data
        let exp_err_missing_data = "At least one of point_data or cell_data must be provided";

        // neither point_data nor cell_data provided
        let res = writer.write_data("1.0", None, None);
        assert_eq!(res.unwrap_err().to_string(), exp_err_missing_data);

        // (empty) point_data provided, but cell_data is None
        let res = writer.write_data("1.0", Some(&BTreeMap::new()), None);
        assert_eq!(res.unwrap_err().to_string(), exp_err_missing_data);

        // (empty) cell_data provided, but point_data is None
        let res = writer.write_data("1.0", None, Some(&BTreeMap::new()));
        assert_eq!(res.unwrap_err().to_string(), exp_err_missing_data);

        // Invalid time step (already exists)
        let res = writer.write_data("0.1", Some(&point_data), None);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Time step '0.1' has already been written"
        );

        // Invalid time step (not a float)
        let res = writer.write_data("invalid_time", None, None);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Time must be a valid float, and not 'invalid_time'"
        );

        // Invalid time step (empty)
        let res = writer.write_data("", None, None);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Time must be a valid float, and not ''"
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
        let point_data_scalar = vec![(
            "point_data_sca".to_string(),
            (DataAttribute::Scalar, vec![5.0; NUM_POINTS - 1].into()),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", Some(&point_data_scalar), None);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point data 'point_data_sca' must be 10, but is 9"
        );

        // vector point data
        let point_data_vector = vec![(
            "point_data_vec".to_string(),
            (DataAttribute::Vector, vec![5.0; NUM_POINTS * 2].into()),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", Some(&point_data_vector), None);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point data 'point_data_vec' must be 30, but is 20"
        );

        // Tensor point data
        let point_data_tensor = vec![(
            "point_data_ten".to_string(),
            (DataAttribute::Tensor, vec![5.0; NUM_POINTS * 3].into()),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", Some(&point_data_tensor), None);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point data 'point_data_ten' must be 90, but is 30"
        );

        // Tensor6 point data
        let point_data_tensor6 = vec![(
            "point_data_ten6".to_string(),
            (DataAttribute::Tensor6, vec![5.0; NUM_POINTS * 3].into()),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", Some(&point_data_tensor6), None);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point data 'point_data_ten6' must be 60, but is 30"
        );

        // Matrix point data
        let point_data_matrix = vec![(
            "point_data_mat".to_string(),
            (
                DataAttribute::Matrix(2, 1),
                vec![5.0; NUM_POINTS * 3 - 1].into(),
            ),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", Some(&point_data_matrix), None);
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of point data 'point_data_mat' must be 20, but is 29"
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
        let cell_data_scalar = vec![(
            "cell_data_sca".to_string(),
            (DataAttribute::Scalar, vec![5.0; NUM_CELLS - 1].into()),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", None, Some(&cell_data_scalar));
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell data 'cell_data_sca' must be 4, but is 3"
        );

        // vector cell data
        let cell_data_vector = vec![(
            "cell_data_vec".to_string(),
            (DataAttribute::Vector, vec![5.0; NUM_CELLS * 2].into()),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", None, Some(&cell_data_vector));
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell data 'cell_data_vec' must be 12, but is 8"
        );

        // Tensor cell data
        let cell_data_tensor = vec![(
            "cell_data_ten".to_string(),
            (DataAttribute::Tensor, vec![5.0; NUM_CELLS * 3].into()),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", None, Some(&cell_data_tensor));
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell data 'cell_data_ten' must be 36, but is 12"
        );

        // Tensor6 cell data
        let cell_data_tensor6 = vec![(
            "cell_data_ten6".to_string(),
            (DataAttribute::Tensor6, vec![5.0; NUM_CELLS * 3].into()),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", None, Some(&cell_data_tensor6));
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell data 'cell_data_ten6' must be 24, but is 12"
        );

        // Matrix cell data
        let cell_data_matrix = vec![(
            "cell_data_mat".to_string(),
            (
                DataAttribute::Matrix(2, 1),
                vec![5.0; NUM_CELLS * 3 - 1].into(),
            ),
        )]
        .into_iter()
        .collect();
        let res = writer.write_data("0.0", None, Some(&cell_data_matrix));
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of cell data 'cell_data_mat' must be 8, but is 11"
        );
    }
}

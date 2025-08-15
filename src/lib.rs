use std::{
    collections::BTreeMap,
    io::{BufWriter, Result as IoResult, Write},
    path::{Path, PathBuf},
};

use serde::Serialize;
use xdmf_elements::{
    Information, Xdmf, attribute,
    data_item::{DataContent, DataItem, NumberType},
    dimensions::Dimensions,
    geometry::{Geometry, GeometryType},
    grid::{CollectionType, Grid, GridType, Time},
    topology::{Topology, TopologyType},
};

mod ascii_writer;
#[cfg(feature = "hdf5")]
mod hdf5_writer;

mod values;
pub mod xdmf_elements;

// Re-export types used in the public API
pub use values::Values;
pub use xdmf_elements::{CellType, attribute::AttributeType, data_item::Format};

pub type DataMap = BTreeMap<String, (AttributeType, Values)>;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum DataStorage {
    Ascii,
    AsciiInline,
    Hdf5SingleFile,
    Hdf5MultipleFiles,
}

pub(crate) trait DataWriter {
    fn format(&self) -> Format;

    fn data_storage(&self) -> DataStorage;

    fn write_mesh(&mut self, points: &[f64], cells: &[u64])
    -> IoResult<(DataContent, DataContent)>;

    #[cfg(feature = "unstable-submesh-api")]
    fn write_submesh(
        &mut self,
        name: &str,
        point_indices: &[u64],
        cell_indices: &[u64],
    ) -> IoResult<(String, String)>;

    fn write_data(
        &mut self,
        name: &str,
        center: attribute::Center,
        data: &Values,
    ) -> IoResult<DataContent>;

    fn write_data_initialize(&mut self, _time: &str) -> IoResult<()> {
        Ok(())
    }

    fn write_data_finalize(&mut self) -> IoResult<()> {
        Ok(())
    }

    // flush the writer, if applicable
    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

/// Check if the hdf5 feature is enabled.
pub const fn is_hdf5_enabled() -> bool {
    #[cfg(feature = "hdf5")]
    {
        true
    }
    #[cfg(not(feature = "hdf5"))]
    {
        false
    }
}

pub struct TimeSeriesWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
}

impl TimeSeriesWriter {
    /// # Errors
    ///
    /// TODO
    pub fn new(file_name: impl AsRef<Path>, data_storage: DataStorage) -> IoResult<Self> {
        let xdmf_file_name = file_name.as_ref().to_path_buf().with_extension("xdmf");

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
            precision: Some(8), // Default precision for f64
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
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "At least one point is required",
        ));
    }

    // check that points are a multiple of 3 (x, y, z)
    if points.len() % 3 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Points must have 3 dimensions",
        ));
    }

    // check cells connectivity indices
    let max_connectivity_index = cells.0.iter().max();

    if let Some(&max_index) = max_connectivity_index {
        if max_index as usize >= points.len() / 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Connectivity indices out of bounds for the given points, max index: {}, but number of points is {}",
                    max_index,
                    points.len() / 3
                ),
            ));
        }
    }

    // check that the number of connectivities matches the expected number based on the cell types
    let exp_num_points: usize = cells.1.iter().map(|ct| ct.num_points()).sum();
    if exp_num_points != cells.0.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
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
}

impl TimeSeriesDataWriter {
    /// Write data for a specific time step.
    /// Accepts str for time to avoid dealing with formatting, thus leaving it to the user.
    // TODOs:
    // - check for unique time steps
    // - assert dimensions of points and cells match
    // - check that the data is not empty
    // - maybe write data as ref in attribute, to make cloning cheaper. Really only matters for XML format, so unsure if worth it.
    // - check if time is not "", and it can be converted to a float
    /// # Errors
    ///
    /// TODO
    pub fn write_data(
        &mut self,
        time: &str,
        point_data: Option<&DataMap>,
        cell_data: Option<&DataMap>,
    ) -> IoResult<()> {
        self.writer.write_data_initialize(time)?;
        let format = self.writer.format();

        let mut new_attributes = Vec::new();

        let mut create_attributes =
            |data_map: Option<&BTreeMap<String, (AttributeType, Values)>>,

             center: attribute::Center|
             -> IoResult<()> {
                for (data_name, data) in data_map.unwrap_or(&BTreeMap::new()) {
                    let vals = &data.1;

                    let data_item = DataItem {
                        name: None,
                        dimensions: Some(vals.dimensions()),
                        number_type: Some(vals.number_type()),
                        format: Some(format),
                        precision: Some(vals.precision()),
                        data: self.writer.write_data(data_name, center, vals)?,
                        reference: None,
                    };

                    let attribute = attribute::Attribute {
                        name: data_name.clone(),
                        attribute_type: data.0,
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
}

/// Create directories in a way that is safe for MPI applications.
/// This function will create the directory if it does not exist, and wait for it to appear
/// This is particularly needed on systems such as clusters with slow filesystems, to ensure that
/// all processes can see the created directory before proceeding.
/// See <https://github.com/KratosMultiphysics/Kratos/pull/9247> where this was taken from
/// Its a battle-tested solution tested with > 1000 processes
/// # Errors
///
/// TODO
pub fn mpi_safe_create_dir_all(path: impl AsRef<Path> + std::fmt::Debug) -> IoResult<()> {
    if !&path.as_ref().exists() {
        std::fs::create_dir_all(&path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to create directory {path:?}: {e}"),
            )
        })?;
    }

    if !path.as_ref().exists() {
        // wait for the path to appear in the filesystem
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    Ok(())
}

fn create_writer(file_name: &Path, data_storage: DataStorage) -> IoResult<Box<dyn DataWriter>> {
    match data_storage {
        DataStorage::Ascii => Ok(Box::new(ascii_writer::AsciiWriter::new(file_name)?)),
        DataStorage::AsciiInline => Ok(Box::new(ascii_writer::AsciiInlineWriter::new())),
        DataStorage::Hdf5SingleFile => {
            #[cfg(feature = "hdf5")]
            {
                Ok(Box::new(hdf5_writer::SingleFileHdf5Writer::new(file_name)?))
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(std::io::Error::other(
                    "Using Hdf5SingleFile DataStorage requires the hdf5 feature.",
                ))
            }
        }
        DataStorage::Hdf5MultipleFiles => {
            #[cfg(feature = "hdf5")]
            {
                Ok(Box::new(hdf5_writer::MultipleFilesHdf5Writer::new(
                    file_name,
                )?))
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(std::io::Error::other(
                    "Using Hdf5MultipleFiles DataStorage requires the hdf5 feature.",
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly_cell_points_works() {
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
    fn prepare_cells_works() {
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
    fn validate_points_and_cells_ok() {
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
        assert_eq!(writer.xdmf_file_name, xdmf_file_path.with_extension("xdmf"));
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
}

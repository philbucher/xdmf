use std::collections::BTreeMap;
use std::io::Result as IoResult;
use std::path::{Path, PathBuf};

use xdmf_elements::{
    Xdmf, attribute,
    data_item::{DataItem, NumberType},
    dimensions::Dimensions,
    geometry::{Geometry, GeometryType},
    grid::{Grid, GridType, TimeSeriesGrid, Uniform},
    topology::{Topology, TopologyType},
};

#[cfg(feature = "hdf5")]
mod hdf5_writer;
mod xml_writer;

mod values;
pub mod xdmf_elements;

// Re-export types used in the public API
pub use values::Values;
pub use xdmf_elements::CellType;
pub use xdmf_elements::attribute::AttributeType;
pub use xdmf_elements::data_item::Format;

pub(crate) trait DataWriter {
    fn format(&self) -> Format;

    fn write_mesh(&mut self, points: &[f64], cells: &[u64]) -> IoResult<(String, String)>;

    #[cfg(feature = "unstable-submesh-api")]
    fn write_submesh(
        &mut self,
        name: &str,
        point_indices: &[u64],
        cell_indices: &[u64],
    ) -> IoResult<(String, String)>;

    fn write_data(&mut self, time: &str, data: &Values) -> IoResult<String>;

    // flush the writer, if applicable
    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

pub struct TimeSeriesWriterOptions {
    format: Format,
    multiple_files: bool,
}

impl TimeSeriesWriterOptions {
    pub fn format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    pub fn multiple_files(mut self, multiple_files: bool) -> Self {
        self.multiple_files = multiple_files;
        self
    }

    fn create_writer(&self, file_name: &Path) -> Box<dyn DataWriter> {
        match self.format {
            Format::XML => Box::new(xml_writer::XmlWriter::new()),

            Format::HDF => {
                #[cfg(feature = "hdf5")]
                match self.multiple_files {
                    true => Box::new(hdf5_writer::MultipleFilesHdf5Writer::new(file_name).unwrap()),
                    false => Box::new(hdf5_writer::SingleFileHdf5Writer::new(file_name).unwrap()),
                }

                #[cfg(not(feature = "hdf5"))]
                panic!("HDF5 feature is not enabled. Please enable it in Cargo.toml.");
            }
            _ => unimplemented!("Unsupported format"),
        }
    }
}

impl Default for TimeSeriesWriterOptions {
    fn default() -> Self {
        let default_format = if cfg!(feature = "hdf5") {
            Format::HDF
        } else {
            Format::XML
        };
        Self {
            format: default_format,
            multiple_files: false,
        }
    }
}

pub struct TimeSeriesWriter {
    xdmf_file_name: PathBuf,
    xdmf: Xdmf,
    writer: Box<dyn DataWriter>,
}

impl TimeSeriesWriter {
    pub fn options() -> TimeSeriesWriterOptions {
        TimeSeriesWriterOptions::default()
    }
    pub fn new(file_name: impl AsRef<Path>) -> Self {
        Self::new_with_options(file_name, &TimeSeriesWriter::options())
    }

    pub fn new_with_options(
        file_name: impl AsRef<Path>,
        options: &TimeSeriesWriterOptions,
    ) -> Self {
        // TODO create folder if it does not exist

        Self {
            xdmf_file_name: file_name.as_ref().to_path_buf().with_extension("xdmf"),
            xdmf: Xdmf::default(),
            writer: options.create_writer(file_name.as_ref()),
        }
    }

    // TODO check bounds of connectivity indices
    pub fn write_mesh(
        mut self,
        points: &[f64],
        cells: (&[u64], &[CellType]),
    ) -> IoResult<TimeSeriesDataWriter> {
        validate_points_and_cells(points, cells)?;

        let num_cells = cells.1.len();

        let prepared_cells = prepare_cells(cells)?;

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
            DataItem::new_reference(&data_item_coords, "/Xdmf/Domain/DataItem".to_string());
        let data_item_connectivity_ref =
            DataItem::new_reference(&data_item_connectivity, "/Xdmf/Domain/DataItem".to_string());

        let mesh_grid = Uniform {
            name: "mesh".to_string(),
            grid_type: GridType::Uniform,
            geometry: Geometry {
                geometry_type: GeometryType::XYZ,
                data_item: data_item_coords_ref,
            },
            topology: Topology {
                topology_type: TopologyType::Mixed,
                number_of_elements: num_cells.to_string(),
                data_item: data_item_connectivity_ref,
            },
            time: None,
            attributes: None,
            indices: None,
        };

        self.xdmf.domains[0]
            .grids
            .push(Grid::new_time_series("time_series", mesh_grid));

        self.xdmf.domains[0].data_items.push(data_item_coords);
        self.xdmf.domains[0].data_items.push(data_item_connectivity);

        let mut ts_writer = TimeSeriesDataWriter {
            xdmf_file_name: self.xdmf_file_name,
            xdmf: self.xdmf,
            writer: self.writer,
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

// Validate that the number of points and cells match the expected sizes
fn validate_points_and_cells(points: &[f64], cells: (&[u64], &[CellType])) -> IoResult<()> {
    if points.len() % 3 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Points must have 3 dimensions",
        ));
    }

    let max_connectivity_index = cells.0.iter().max();

    if let Some(&max_index) = max_connectivity_index {
        if max_index as usize >= points.len() / 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Connectivity indices out of bounds for the given points",
            ));
        }
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
/// TODO if all cells are the same, then the type information can be stored as TopologyType
fn prepare_cells(cells: (&[u64], &[CellType])) -> IoResult<Vec<u64>> {
    // validate input
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

    Ok(cells_with_types)
}

pub struct TimeSeriesDataWriter {
    xdmf_file_name: PathBuf,
    xdmf: Xdmf,
    writer: Box<dyn DataWriter>,
}

impl TimeSeriesDataWriter {
    fn temporal_grid(&mut self) -> &mut TimeSeriesGrid {
        let temp_grid = self.xdmf.domains[0]
            .grids
            .last_mut()
            .expect("No grids found");

        match temp_grid {
            Grid::TimeSeriesGrid(gr) => gr,
            _ => panic!("Last grid is not a collection"),
        }
    }

    /// Write data for a specific time step.
    // TODOs:
    // - make sure that names for data location (aka Center) are unique (Paraview just ignores duplicate names)
    // - check for unique time steps
    // - assert dimensions of points and cells match
    pub fn write_data(
        &mut self,
        time: &str,
        point_data: Option<&BTreeMap<String, (AttributeType, Values)>>,
        cell_data: Option<&BTreeMap<String, (AttributeType, Values)>>,
    ) -> IoResult<()> {
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
                        data: self.writer.write_data(time, vals)?,
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

        self.temporal_grid().create_new_time(time, &new_attributes);

        self.write()
    }

    fn write(&mut self) -> IoResult<()> {
        self.writer.flush()?;

        // Write the XDMF file to a temporary file first to avoid access races
        let temp_xdmf_file_name = self.xdmf_file_name.with_extension("xdmf.tmp");

        self.xdmf
            .write_to(&mut std::fs::File::create(&temp_xdmf_file_name)?)?;

        std::fs::rename(&temp_xdmf_file_name, &self.xdmf_file_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        ))
        .unwrap();

        assert_eq!(
            cells_prep,
            vec![1, 1, 0, 2, 2, 1, 2, 4, 3, 4, 5, 5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn test_prepare_cells_by_celltype() {
        assert_eq!(
            prepare_cells((&[5], &[CellType::Vertex])).unwrap(),
            vec![1, 1, 5]
        );

        assert_eq!(
            prepare_cells((&[5, 6], &[CellType::Edge])).unwrap(),
            vec![2, 2, 5, 6]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7], &[CellType::Triangle])).unwrap(),
            vec![4, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8], &[CellType::Quadrilateral])).unwrap(),
            vec![5, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8], &[CellType::Tetrahedron])).unwrap(),
            vec![6, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9], &[CellType::Pyramid])).unwrap(),
            vec![7, 5, 6, 7, 8, 9]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10], &[CellType::Wedge])).unwrap(),
            vec![8, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Hexahedron])).unwrap(),
            vec![9, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7], &[CellType::Edge3])).unwrap(),
            vec![34, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells((
                &[5, 6, 7, 8, 9, 10, 11, 12, 13],
                &[CellType::Quadrilateral9]
            ))
            .unwrap(),
            vec![35, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10], &[CellType::Triangle6])).unwrap(),
            vec![36, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells((&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Quadrilateral8])).unwrap(),
            vec![37, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells((
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                &[CellType::Tetrahedron10]
            ))
            .unwrap(),
            vec![38, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        );

        assert_eq!(
            prepare_cells((
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                &[CellType::Pyramid13]
            ))
            .unwrap(),
            vec![39, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        );

        assert_eq!(
            prepare_cells((
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                &[CellType::Wedge15]
            ))
            .unwrap(),
            vec![40, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );

        assert_eq!(
            prepare_cells((
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
                ],
                &[CellType::Wedge18]
            ))
            .unwrap(),
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
            ))
            .unwrap(),
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
            ))
            .unwrap(),
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
            ))
            .unwrap(),
            vec![
                50, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31
            ]
        );
    }

    #[test]
    fn test_prepare_cells_mismatch() {
        let res = prepare_cells((
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        ));

        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err().to_string(),
            "Size of connectivities not match the expected number based on the cell types: 8 != 10"
        );
    }
}

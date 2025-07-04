use std::collections::BTreeMap;
use std::io::Result as IoResult;
use std::path::{Path, PathBuf};
use std::vec;

use xdmf_elements::{
    Xdmf, attribute,
    data_item::{DataItem, NumberType},
    dimensions::Dimensions,
    geometry::{Geometry, GeometryType},
    grid::{Grid, GridType, TimeSeriesGrid, Uniform},
    topology::Topology,
};

#[cfg(feature = "hdf5")]
mod hdf5_writer;
mod xml_writer;

mod data;
mod values;

// Re-export types used in the public API
pub use data::Data;
pub use values::Values;
pub use xdmf_elements::attribute::AttributeType;
pub use xdmf_elements::data_item::Format;
pub use xdmf_elements::topology::TopologyType;

pub(crate) trait DataWriter {
    fn format(&self) -> Format;

    fn write_mesh(&mut self, points: &Vec<f64>, cells: &Vec<u64>) -> IoResult<(String, String)>;

    fn write_submesh(
        &mut self,
        name: &str,
        point_indices: &Vec<u64>,
        cell_indices: &Vec<u64>,
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
    pub fn write_mesh<'a, C>(
        mut self,
        points: &Vec<f64>,
        cells: &(Vec<u64>, Vec<u8>),
    ) -> IoResult<TimeSeriesDataWriter> {
        if points.len() % 3 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Points must have 3 dimensions",
            ));
        }

        let num_cells = cells.1.len();

        // Concatenate all arrays along axis 0
        let cells_flat = concatenate_cells(cells);

        let (points_data, cells_data) = self.writer.write_mesh(points, &cells_flat)?;

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
            dimensions: Some(Dimensions(vec![num_cells])),
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
                topology_type: TopologyType::Triangle,
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
    // TODO check if submesh names are unique
    // TODO use SpatialCollection when submeshes are used
    // TODO each tolologytype can only appear once, otherwise indexing for submeshes will be wrong
    pub fn write_mesh_and_submeshes<'a, C, M>(
        self,
        points: &Vec<f64>,
        cells: &(Vec<u64>, Vec<u8>),
        submeshes: &BTreeMap<String, SubMesh>,
    ) -> IoResult<TimeSeriesDataWriter> {
        let mut ts = self.write_mesh(points, cells)?;

        let format = ts.writer.format();

        for (submesh_name, submesh) in submeshes {
            let name_points = format!("{}_points", submesh_name);
            let name_cells = format!("{}_cells", submesh_name);

            ts.xdmf.domains[0].data_items.push(DataItem {
                data: ts
                    .writer
                    .write_submesh(&name_points, &submesh.point_indices)?,
                name: Some(name_points),
                dimensions: Some(Dimensions(submesh.point_indices.shape().into())),
                number_type: Some(NumberType::UInt),
                format: Some(format),
                precision: Some(8),
                reference: None,
            });

            ts.xdmf.domains[0].data_items.push(DataItem {
                data: ts
                    .writer
                    .write_submesh(&name_cells, &submesh.cell_indices)?,
                name: Some(name_cells),
                dimensions: Some(Dimensions(submesh.cell_indices.shape().into())),
                number_type: Some(NumberType::UInt),
                format: Some(format),
                precision: Some(8),
                reference: None,
            });
        }

        Ok(ts)
    }
}

pub struct SubMesh {
    pub point_indices: Vec<u64>,
    pub cell_indices: Vec<u64>,
}

fn concatenate_cells(cells: &(Vec<u64>, Vec<u8>)) -> Vec<u64> {
    let concatenated_iter = cells
        .into_iter()
        .sorted_by(|(k1, _), (k2, _)| k1.cmp(k2))
        .flat_map(|(_, view)| view.iter().copied());

    Array1::from_iter(concatenated_iter)
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
    pub fn write_data<D>(
        &mut self,
        time: &str,
        point_data: Option<&BTreeMap<String, Data>>,
        cell_data: Option<&BTreeMap<String, Data>>,
    ) -> IoResult<()> {
        let format = self.writer.format();
        let mut new_attributes = Vec::new();

        let mut create_attributes = |data_map: Option<&BTreeMap<String, Data>>,
                                     center: attribute::Center|
         -> IoResult<()> {
            for (data_name, data) in data_map.unwrap_or(&BTreeMap::new()) {
                let vals = data.values();
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
                    attribute_type: data.attribute_type(),
                    center: center,
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

        let temp_xdmf_file_name = self.xdmf_file_name.with_extension("xdmf.tmp");

        self.xdmf
            .write_to(&mut std::fs::File::create(&temp_xdmf_file_name)?)?;

        std::fs::rename(&temp_xdmf_file_name, &self.xdmf_file_name)
    }
}

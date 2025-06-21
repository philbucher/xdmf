use std::collections::HashMap;
use std::io::Result as IoResult;
use std::path::{Path, PathBuf};

use xdmf_elements::{
    Xdmf, attribute,
    data_item::{DataItem, NumberType},
    dimensions::Dimensions,
    geometry::{Geometry, GeometryType},
    grid::{Grid, TimeSeriesGrid, Uniform},
    topology::Topology,
};

#[cfg(feature = "hdf5")]
mod hdf5_writer;

mod values;

// Re-export types used in the public API
pub use values::Values;
pub use xdmf_elements::data_item::Format;
pub use xdmf_elements::topology::TopologyType;

pub(crate) trait DataWriter {
    fn format(&self) -> Format;
    fn write_mesh(
        &mut self,
        points: &[f64; 3],
        cells: HashMap<TopologyType, Vec<usize>>,
    ) -> IoResult<(String, String)>;
    fn write_data(&mut self, time: &str, data: &Values) -> IoResult<String>;
    fn flush(&mut self) -> IoResult<()> {
        // flush the writer, if applicable
        Ok(())
    }
    fn close(self) -> IoResult<()>;
}

struct XmlWriter {}

impl XmlWriter {
    pub fn new() -> Self {
        XmlWriter {}
    }

    fn data_to_string(&self, data: &Values) -> String {
        // using default implementation for now
        // maybe use custom formatting later
        match data {
            Values::View1Df64(view) => view.to_string(),
            Values::View2Df64(view) => view.to_string(),
            Values::ViewDynf64(view) => view.to_string(),
        }
    }
}

impl DataWriter for XmlWriter {
    fn format(&self) -> Format {
        Format::XML
    }
    fn write_mesh(
        &mut self,
        _points: &[f64; 3],
        _cells: HashMap<TopologyType, Vec<usize>>,
    ) -> IoResult<(String, String)> {
        // Implementation for writing mesh data to XML
        Ok((
            "<DataItem Dimensions=\"4 3\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\">0 0 0 0 1 0 1 1 0 1 0 0.5</DataItem>".to_string(),
            "<DataItem Dimensions=\"6\" NumberType=\"Int\" Format=\"XML\" Precision=\"4\">0 1 2 0 2 3</DataItem>".to_string(),
        ))
    }

    fn write_data(&mut self, _time: &str, data: &Values) -> IoResult<String> {
        Ok(self.data_to_string(data))
    }

    fn close(self) -> Result<(), std::io::Error> {
        // nothing to do here since XML writer does not hold any resources
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
            Format::XML => Box::new(XmlWriter::new()),

            Format::HDF => {
                #[cfg(feature = "hdf5")]
                match self.multiple_files {
                    true => {
                        // Create HDF5 writer for multiple files
                        Box::new(hdf5_writer::MultipleFilesHdf5Writer::new(file_name).unwrap())
                    }
                    false => {
                        // Create HDF5 writer for a single file
                        Box::new(hdf5_writer::SingleFileHdf5Writer::new(file_name).unwrap())
                    }
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
            Format::XML // Default to XML if no feature is enabled
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
        // probably write Mesh right away so it does not need to be stored

        Self {
            xdmf_file_name: file_name.as_ref().to_path_buf().with_extension("xdmf"),
            xdmf: Xdmf::default(),
            writer: options.create_writer(file_name.as_ref()),
        }
    }

    pub fn add_mesh(
        mut self,
        points: &[f64; 3],
        cells: HashMap<TopologyType, Vec<usize>>,
    ) -> IoResult<TimeSeriesDataWriter> {
        let (points_data, cells_data) = self.writer.write_mesh(points, cells)?;

        let mesh_grid = Uniform {
            name: "mesh".to_string(),
            grid_type: xdmf_elements::grid::GridType::Uniform,
            geometry: Geometry {
                geometry_type: GeometryType::XYZ,
                data_item: DataItem {
                    dimensions: Dimensions(vec![4, 3]),
                    data: points_data,
                    number_type: NumberType::Float,
                    ..Default::default()
                },
            },
            topology: Topology {
                topology_type: TopologyType::Triangle,
                number_of_elements: "2".into(),
                data_item: DataItem {
                    dimensions: Dimensions(vec![6]),
                    number_type: NumberType::Int,
                    data: cells_data,
                    ..Default::default()
                },
            },
        };

        self.xdmf.domains[0]
            .grids
            .push(Grid::new_time_series("time_series", mesh_grid));

        // For now, we just return self
        let mut ts_writer = TimeSeriesDataWriter {
            xdmf_file_name: self.xdmf_file_name,
            xdmf: self.xdmf,
            writer: self.writer,
            flushed: false,
        };

        ts_writer.write()?;

        Ok(ts_writer)
    }
}

pub struct TimeSeriesDataWriter {
    xdmf_file_name: PathBuf,
    xdmf: Xdmf,
    writer: Box<dyn DataWriter>,
    flushed: bool, // Indicates if the data has been flushed to the file
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

    /// Add data
    /// Depending on the format, data will either be written directly (hdf), or buffered (xml)
    pub fn add_data(
        &mut self,
        time: &str,
        point_data: &HashMap<String, Values>,
        cell_data: &HashMap<String, Values>,
    ) -> IoResult<()> {
        let format = self.writer.format();
        let mut new_attributes = Vec::new();

        for (name, data) in point_data {
            let data_item = DataItem {
                dimensions: data.dimensions(),
                number_type: data.number_type(),
                format: format,
                precision: data.precision(),
                data: self.writer.write_data(time, data)?,
            };

            let attribute = attribute::Attribute {
                name: name.to_string(),
                attribute_type: attribute::AttributeType::Scalar,
                center: attribute::Center::Node,
                data_item: data_item,
            };
            new_attributes.push(attribute);
        }

        for (name, data) in cell_data {
            let data_item = DataItem {
                dimensions: data.dimensions(),
                number_type: data.number_type(),
                format: format,
                precision: data.precision(),
                data: self.writer.write_data(time, data)?,
            };

            let attribute = attribute::Attribute {
                name: name.to_string(),
                attribute_type: attribute::AttributeType::Scalar,
                center: attribute::Center::Cell,
                data_item: data_item,
            };
            new_attributes.push(attribute);
        }

        let time_grid = self.temporal_grid().create_new_time(time);
        time_grid.attributes.extend(new_attributes);

        // TODO check if time already exists???
        // Maybe this function should right away write the data to the file?
        self.flushed = false;
        Ok(())
    }

    pub fn write(&mut self) -> IoResult<()> {
        self.writer.flush()?;

        // after it was written, drop it from the map (and only keep metadata?)

        let temp_xdmf_file_name = self.xdmf_file_name.with_extension("xdmf.tmp");

        self.xdmf
            .write_to(&mut std::fs::File::create(&temp_xdmf_file_name)?)?;

        std::fs::rename(&temp_xdmf_file_name, &self.xdmf_file_name)?;
        self.flushed = true;
        Ok(())
    }
}

impl Drop for TimeSeriesDataWriter {
    fn drop(&mut self) {
        if !self.flushed {
            // If the data was not flushed, we should flush it before dropping
            self.write();
        }
    }
}

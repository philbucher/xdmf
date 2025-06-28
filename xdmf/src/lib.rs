use std::io::Result as IoResult;
use std::path::{Path, PathBuf};

use itertools::Itertools;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis, concatenate};

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
    fn write_mesh(
        &mut self,
        points: &ArrayView2<f64>,
        cells: &ArrayView1<usize>,
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

    pub fn write_mesh<'a, C>(
        mut self,
        points: &ArrayView2<f64>,
        cells: &'a C,
    ) -> IoResult<TimeSeriesDataWriter>
    where
        &'a C: IntoIterator<Item = (&'a TopologyType, &'a ArrayView2<'a, usize>)>,
    {
        let geom_type = match points.shape()[1] {
            2 => GeometryType::XY,
            3 => GeometryType::XYZ,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Points must have 2 or 3 dimensions",
                ));
            }
        };

        let dim_points = points.shape()[1];
        if dim_points != 2 && dim_points != 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Points must have 2 or 3 dimensions",
            ));
        }

        let num_cells = cells.into_iter().map(|(_, v)| v.shape()[1]).sum::<usize>();

        // Concatenate all arrays along axis 0
        let cells_flat = concatenate_cells(cells);

        let (points_data, cells_data) = self.writer.write_mesh(points, &cells_flat.view())?;

        let mesh_grid = Uniform {
            name: "mesh".to_string(),
            grid_type: GridType::Uniform,
            geometry: Geometry {
                geometry_type: geom_type,
                data_item: DataItem {
                    dimensions: Dimensions(points.shape().into()),
                    data: points_data,
                    number_type: NumberType::Float,
                    precision: 8, // Default precision for f64
                    format: self.writer.format(),
                },
            },
            topology: Topology {
                topology_type: TopologyType::Triangle,
                number_of_elements: num_cells.to_string(),
                data_item: DataItem {
                    dimensions: Dimensions(cells_flat.shape().into()),
                    number_type: NumberType::UInt,
                    data: cells_data,
                    format: self.writer.format(),
                    precision: 8,
                },
            },
        };

        self.xdmf.domains[0]
            .grids
            .push(Grid::new_time_series("time_series", mesh_grid));

        let mut ts_writer = TimeSeriesDataWriter {
            xdmf_file_name: self.xdmf_file_name,
            xdmf: self.xdmf,
            writer: self.writer,
        };

        ts_writer.write()?;

        Ok(ts_writer)
    }

    pub fn write_mesh_and_submeshes<'a, C, M, S, I>(
        mut self,
        _points: &ArrayView2<f64>,
        _cells: &'a C,
        _submeshes: &M,
    ) -> IoResult<TimeSeriesDataWriter>
    where
        &'a C: IntoIterator<Item = (&'a TopologyType, &'a ArrayView2<'a, usize>)>,
        M: IntoIterator<Item = (S, I, I)>,
        S: ToString,
        I: IntoIterator<Item = usize>,
    {
        unimplemented!("Submeshes are not yet implemented");
    }
}

fn concatenate_cells<'a, M>(cells: &'a M) -> Array1<usize>
where
    &'a M: IntoIterator<Item = (&'a TopologyType, &'a ArrayView2<'a, usize>)>,
{
    let views: Vec<ArrayView2<'a, usize>> = cells
        .into_iter()
        .sorted_by(|(k1, _), (k2, _)| k1.cmp(k2))
        .map(|(_, v)| *v)
        .collect();

    concatenate(Axis(0), &views)
        .expect("Concatenation failed")
        .flatten()
        .to_owned()
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

    pub fn write_data<'a, D>(&mut self, time: &str, data: &'a D) -> IoResult<()>
    where
        &'a D: IntoIterator<Item = &'a Data<'a>>,
    {
        let format = self.writer.format();
        let mut new_attributes = Vec::new();

        for d in data {
            let v = d.values();
            let data_item = DataItem {
                dimensions: v.dimensions(),
                number_type: v.number_type(),
                format,
                precision: v.precision(),
                data: self.writer.write_data(time, d.values())?,
            };

            let attribute = attribute::Attribute {
                name: d.name(),
                attribute_type: d.attribute_type(),
                center: d.center(),
                data_item: vec![data_item],
            };
            new_attributes.push(attribute);
        }

        let time_grid = self.temporal_grid().create_new_time(time);
        time_grid.attributes.extend(new_attributes);

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

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use xdmf_elements::Xdmf;

pub use xdmf_elements::data_item::Format;
pub use xdmf_elements::topology::TopologyType;

pub mod values;

pub use values::Values;

pub struct TimeSeriesWriterOptions {
    format: Format,
    multiple_files: bool, // only for hdf5
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
}

impl Default for TimeSeriesWriterOptions {
    fn default() -> Self {
        Self {
            format: Format::HDF,
            multiple_files: false,
        }
    }
}

pub struct TimeSeriesWriter {
    file_name: PathBuf,
    format: Format,
}

impl TimeSeriesWriter {
    pub fn options() -> TimeSeriesWriterOptions {
        TimeSeriesWriterOptions::default()
    }
    pub fn new(file_name: &impl AsRef<Path>) -> Self {
        Self::new_with_options(file_name, TimeSeriesWriter::options())
    }

    pub fn new_with_options(file_name: impl AsRef<Path>, options: TimeSeriesWriterOptions) -> Self {
        // TODO create folder if it does not exist?
        // probably write Mesh right away so it does not need to be stored

        Self {
            file_name: file_name.as_ref().to_path_buf().with_extension("xdmf"),
            format: options.format,
        }
    }

    pub fn add_mesh(
        self,
        points: &[f64; 3],
        cells: HashMap<TopologyType, Vec<usize>>,
    ) -> std::io::Result<TimeSeriesDataWriter> {
        // Here we would write the mesh to the file
        // For now, we just return self
        Ok(TimeSeriesDataWriter {
            file_name: self.file_name,
            format: self.format,
            flushed: false,
        })
    }
}

pub struct TimeSeriesDataWriter {
    file_name: PathBuf,
    format: Format,
    flushed: bool, // Indicates if the data has been flushed to the file
}

impl TimeSeriesDataWriter {
    /// Add data
    /// Depending on the format, data will either be written directly (hdf), or buffered (xml)
    pub fn add_data(
        &mut self,
        time: &str,
        point_data: &HashMap<String, Values>,
        cell_data: &HashMap<String, Values>,
    ) -> std::io::Result<()> {
        for (name, data) in point_data {
            let data_item = xdmf_elements::data_item::DataItem {
                dimensions: data.dimensions(),
                number_type: data.number_type(),
                format: self.format,
                precision: data.precision(),
                data: "asdf".to_string(), // This should be replaced with actual data
            };
        }

        // TODO check if time already exists???
        // Maybe this function should right away write the data to the file?
        self.flushed = false;
        Ok(())
    }

    pub fn write(&mut self) -> std::io::Result<()> {
        // check what data has already beed written to hdf5
        // after it was written, drop it from the map (and only keep metadata?)

        let temp_xdmf_file_name = self.file_name.with_extension("xdmf.tmp");
        let mut file = std::fs::File::create(temp_xdmf_file_name)?;
        // xdmf.write_to(&mut file)?;
        // std::fs::rename(&temp_xdmf_file_name, &self.file_name)?;
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

fn write_data<T: std::fmt::Debug>(
    format: Format,
    data: &[T],
    hdf_file: Option<&std::fs::File>,
) -> std::io::Result<String> {
    if format == Format::XML {
        // For XML or other formats, we might buffer the data
        return Ok(format!("{:?}", data));
    }
    // HDF format

    // Here we would write the data to the file
    // For now, we just print the data
    // writeln!(file, "Time: {}, Name: {}, Data: {:?}", time, name, data)?;
    Ok("DDDDD".to_string())
}

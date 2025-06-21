use std::io::Result as IoResult;
use std::path::{Path, PathBuf};

use hdf5::File as H5File;

use crate::DataWriter;
use xdmf_elements::data_item::Format;

pub(crate) struct SingleFileHdf5Writer {
    h5_file: H5File,
}

impl SingleFileHdf5Writer {
    pub(crate) fn new(file_name: impl AsRef<Path>) -> IoResult<Self> {
        let h5_file = H5File::create(file_name.as_ref().to_path_buf().with_extension("h5"))
            .map_err(|e| std::io::Error::other(e))?;
        Ok(Self { h5_file })
    }
}

impl DataWriter for SingleFileHdf5Writer {
    fn format(&self) -> Format {
        Format::HDF
    }

    fn write_mesh(
        &mut self,
        points: &[f64; 3],
        cells: std::collections::HashMap<xdmf_elements::topology::TopologyType, Vec<usize>>,
    ) -> IoResult<(String, String)> {
        // Implementation for writing mesh data to a single HDF5 file
        unimplemented!()
    }

    fn write_data(&mut self, time: &str, data: &crate::Values) -> IoResult<String> {
        // Implementation for writing data to a single HDF5 file
        unimplemented!()
    }

    fn flush(&mut self) -> IoResult<()> {
        // Flush the HDF5 file
        self.h5_file.flush().map_err(|e| std::io::Error::other(e))
    }

    fn close(self) -> IoResult<()> {
        // Close the HDF5 file
        self.h5_file.close()?;
        Ok(())
    }
}

pub(crate) struct MultipleFilesHdf5Writer {
    base_file_name: PathBuf,
}

impl MultipleFilesHdf5Writer {
    pub(crate) fn new(base_file_name: impl AsRef<Path>) -> IoResult<Self> {
        let base_file_name = base_file_name.as_ref().to_path_buf();
        Ok(Self { base_file_name })
    }
}

impl DataWriter for MultipleFilesHdf5Writer {
    fn format(&self) -> Format {
        Format::HDF
    }

    fn write_mesh(
        &mut self,
        points: &[f64; 3],
        cells: std::collections::HashMap<xdmf_elements::topology::TopologyType, Vec<usize>>,
    ) -> IoResult<(String, String)> {
        // Implementation for writing mesh data to multiple HDF5 files
        unimplemented!()
    }

    fn write_data(&mut self, time: &str, data: &crate::Values) -> IoResult<String> {
        // Implementation for writing data to multiple HDF5 files
        unimplemented!()
    }

    fn flush(&mut self) -> IoResult<()> {
        // Flush the HDF5 files
        Ok(())
    }

    fn close(self) -> IoResult<()> {
        // do nothing since since each file is only opened once and closed after writing
        Ok(())
    }
}

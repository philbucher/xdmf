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
            .map_err(std::io::Error::other)?;
        Ok(Self { h5_file })
    }
}

impl DataWriter for SingleFileHdf5Writer {
    fn format(&self) -> Format {
        Format::HDF
    }

    fn write_mesh(&mut self, _points: &[f64], _cells: &[u64]) -> IoResult<(String, String)> {
        unimplemented!()
    }

    fn write_submesh(
        &mut self,
        _name: &str,
        _point_indices: &[u64],
        _cell_indices: &[u64],
    ) -> IoResult<(String, String)> {
        unimplemented!()
    }

    fn write_data(&mut self, _time: &str, _data: &crate::Values) -> IoResult<String> {
        unimplemented!()
    }

    fn flush(&mut self) -> IoResult<()> {
        // Flush the HDF5 file
        self.h5_file.flush().map_err(std::io::Error::other)
    }
}

pub(crate) struct MultipleFilesHdf5Writer {
    #[allow(dead_code)] // remove this, only temp to silence clippy
    base_file_name: PathBuf,
}

impl MultipleFilesHdf5Writer {
    pub(crate) fn new(base_file_name: impl AsRef<Path>) -> IoResult<Self> {
        // TODO since multiple h5 files will be created, write them in a folder => TODO create this folder
        let base_file_name = base_file_name.as_ref().to_path_buf();
        Ok(Self { base_file_name })
    }
}

impl DataWriter for MultipleFilesHdf5Writer {
    fn format(&self) -> Format {
        Format::HDF
    }

    fn write_mesh(&mut self, _points: &[f64], _cells: &[u64]) -> IoResult<(String, String)> {
        unimplemented!()
    }

    fn write_submesh(
        &mut self,
        _name: &str,
        _point_indices: &[u64],
        _cell_indices: &[u64],
    ) -> IoResult<(String, String)> {
        unimplemented!()
    }

    fn write_data(&mut self, _time: &str, _data: &crate::Values) -> IoResult<String> {
        unimplemented!()
    }
}

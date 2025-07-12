use std::{
    io::Result as IoResult,
    path::{Path, PathBuf},
};

use hdf5::File as H5File;

use crate::{DataWriter, xdmf_elements::data_item::Format};

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

    #[cfg(feature = "unstable-submesh-api")]
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
    h5_files_dir: PathBuf,
}

impl MultipleFilesHdf5Writer {
    pub(crate) fn new(base_file_name: impl AsRef<Path>) -> IoResult<Self> {
        let h5_files_dir = base_file_name.as_ref().to_path_buf().with_extension("h5");

        crate::mpi_safe_create_dir_all(&h5_files_dir)?;

        Ok(Self { h5_files_dir })
    }
}

impl DataWriter for MultipleFilesHdf5Writer {
    fn format(&self) -> Format {
        Format::HDF
    }

    fn write_mesh(&mut self, _points: &[f64], _cells: &[u64]) -> IoResult<(String, String)> {
        let _file_name = self.h5_files_dir.join("mesh.h5");
        unimplemented!()
    }

    #[cfg(feature = "unstable-submesh-api")]
    fn write_submesh(
        &mut self,
        _name: &str,
        _point_indices: &[u64],
        _cell_indices: &[u64],
    ) -> IoResult<(String, String)> {
        unimplemented!()
    }

    fn write_data(&mut self, time: &str, _data: &crate::Values) -> IoResult<String> {
        let _file_name = self.h5_files_dir.join(format!("t_{time}.h5"));
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutliple_files_hdf5_writer_create_dir() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let writer = MultipleFilesHdf5Writer::new(&file_name).unwrap();
        let exp_dir_name = file_name.with_extension("h5");
        assert_eq!(writer.h5_files_dir, exp_dir_name);
        assert!(writer.h5_files_dir.exists());
        assert!(writer.h5_files_dir.is_dir());
    }
}

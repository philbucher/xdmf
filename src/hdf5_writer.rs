use std::{
    io::Result as IoResult,
    path::{Path, PathBuf},
};

use hdf5::File as H5File;

use crate::{DataMap, DataWriter, WrittenData, xdmf_elements::data_item::Format};

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

    fn write_data(
        &mut self,
        _time: &str,
        _point_data: Option<&DataMap>,
        _cell_data: Option<&DataMap>,
    ) -> IoResult<WrittenData> {
        unimplemented!()
    }

    fn flush(&mut self) -> IoResult<()> {
        // Flush the HDF5 file
        self.h5_file.flush().map_err(std::io::Error::other)
    }
}

pub(crate) struct MultipleFilesHdf5Writer {
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

    fn write_mesh(&mut self, points: &[f64], cells: &[u64]) -> IoResult<(String, String)> {
        let file_name = self.h5_files_dir.join("mesh.h5");
        let h5_file = H5File::create(&file_name).map_err(std::io::Error::other)?;

        h5_file
            .new_dataset::<f64>()
            .shape(points.len())
            .create("points")
            .map_err(std::io::Error::other)?
            .write(points)
            .map_err(std::io::Error::other)?;

        h5_file
            .new_dataset::<u64>()
            .shape(cells.len())
            .create("cells")
            .map_err(std::io::Error::other)?
            .write(cells)
            .map_err(std::io::Error::other)?;

        Ok((
            file_name.to_string_lossy().to_string() + ":points",
            file_name.to_string_lossy().to_string() + ":cells",
        ))
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

    fn write_data(
        &mut self,
        _time: &str,
        _point_data: Option<&DataMap>,
        _cell_data: Option<&DataMap>,
    ) -> IoResult<WrittenData> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn test_mutliple_files_hdf5_writer_new() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let writer = MultipleFilesHdf5Writer::new(&file_name).unwrap();
        let exp_dir_name = file_name.with_extension("h5");
        assert_eq!(writer.h5_files_dir, exp_dir_name);
        assert!(writer.h5_files_dir.exists());
        assert!(writer.h5_files_dir.is_dir());
    }

    #[test]
    fn test_mutliple_files_hdf5_writer_write_mesh() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = MultipleFilesHdf5Writer::new(file_name).unwrap();
        let mesh_file = writer.h5_files_dir.join("mesh.h5");
        assert!(!mesh_file.exists());

        let points = vec![0.0, 1.0, 2.0];
        let cells = vec![0, 1, 2];
        let (points_path, cells_path) = writer.write_mesh(&points, &cells).unwrap();
        assert!(mesh_file.exists());

        assert_eq!(
            points_path,
            mesh_file.to_string_lossy().to_string() + ":points"
        );
        assert_eq!(
            cells_path,
            mesh_file.to_string_lossy().to_string() + ":cells"
        );

        // read back the data to verify
        let h5_file = H5File::open(&mesh_file).unwrap();
        let points_data: Vec<f64> = h5_file.dataset("points").unwrap().read().unwrap().to_vec();
        let cells_data: Vec<u64> = h5_file.dataset("cells").unwrap().read().unwrap().to_vec();

        assert_approx_eq!(&[f64], &points, &points_data);
        assert_eq!(&cells, &cells_data);
    }
}

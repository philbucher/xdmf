use std::{
    io::Result as IoResult,
    path::{Path, PathBuf},
};

use hdf5::{File as H5File, Group as H5Group};

use crate::{
    DataWriter, Values,
    xdmf_elements::{attribute, data_item::Format},
};

const MESH: &str = "mesh";
const DATA: &str = "data";
const POINTS: &str = "points";
const CELLS: &str = "cells";

pub(crate) struct SingleFileHdf5Writer {
    h5_file: H5File,
    write_time: Option<String>,
}

/// TODO show file hierarchy, and how data is structured
impl SingleFileHdf5Writer {
    pub(crate) fn new(file_name: impl AsRef<Path>) -> IoResult<Self> {
        let h5_file = H5File::create(file_name.as_ref().to_path_buf().with_extension("h5"))
            .map_err(std::io::Error::other)?;

        Ok(Self {
            h5_file,
            write_time: None,
        })
    }
}

impl DataWriter for SingleFileHdf5Writer {
    fn format(&self) -> Format {
        Format::HDF
    }

    fn write_mesh(&mut self, points: &[f64], cells: &[u64]) -> IoResult<(String, String)> {
        if self.h5_file.link_exists(MESH) {
            return Err(std::io::Error::other("Mesh was already written"));
        }

        let mesh_group = self
            .h5_file
            .create_group(MESH)
            .map_err(std::io::Error::other)?;

        write_mesh(points, cells, &mesh_group)?;

        Ok((
            self.h5_file.filename() + &format!(":{MESH}/{POINTS}"),
            self.h5_file.filename() + &format!(":{MESH}/{CELLS}"),
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
        name: &str,
        center: attribute::Center,
        data: &Values,
    ) -> IoResult<String> {
        let time = self
            .write_time
            .as_ref()
            .ok_or_else(|| std::io::Error::other("Writing data was not initialized"))?;

        let group_name = &format!("{}/t_{time}/{}", DATA, attribute_center_to_hdf5(center));

        // Create the group if it does not exist
        if !self.h5_file.link_exists(group_name) {
            self.h5_file
                .create_group(group_name)
                .map_err(std::io::Error::other)?;
        }

        write_values(
            &self
                .h5_file
                .group(group_name)
                .map_err(std::io::Error::other)?,
            name,
            data,
        )?;

        Ok(self.h5_file.filename() + &format!(":{group_name}/{name}"))
    }

    fn write_data_initialize(&mut self, time: &str) -> IoResult<()> {
        if self.write_time.is_some() {
            return Err(std::io::Error::other(
                "Writing data was already initialized",
            ));
        }

        self.write_time = Some(time.to_string());
        Ok(())
    }
    fn write_data_finalize(&mut self) -> IoResult<()> {
        if self.write_time.is_none() {
            return Err(std::io::Error::other("Writing data was not initialized"));
        }

        self.write_time = None;
        Ok(())
    }

    fn flush(&mut self) -> IoResult<()> {
        // Flush the HDF5 file
        self.h5_file.flush().map_err(std::io::Error::other)
    }
}

/// TODO show file hierarchy, and how data is structured
pub(crate) struct MultipleFilesHdf5Writer {
    h5_files_dir: PathBuf,
    h5_data_file: Option<H5File>,
}

impl MultipleFilesHdf5Writer {
    pub(crate) fn new(base_file_name: impl AsRef<Path>) -> IoResult<Self> {
        let h5_files_dir = base_file_name.as_ref().to_path_buf().with_extension("h5");

        crate::mpi_safe_create_dir_all(&h5_files_dir)?;

        Ok(Self {
            h5_files_dir,
            h5_data_file: None,
        })
    }
}

impl DataWriter for MultipleFilesHdf5Writer {
    fn format(&self) -> Format {
        Format::HDF
    }

    fn write_mesh(&mut self, points: &[f64], cells: &[u64]) -> IoResult<(String, String)> {
        let file_name = self.h5_files_dir.join(format!("{MESH}.h5"));
        let h5_file = H5File::create(&file_name).map_err(std::io::Error::other)?;

        write_mesh(points, cells, &h5_file)?;

        Ok((
            file_name.to_string_lossy().to_string() + ":" + POINTS,
            file_name.to_string_lossy().to_string() + ":" + CELLS,
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
        name: &str,
        center: attribute::Center,
        data: &Values,
    ) -> IoResult<String> {
        // also double check that the name does not already exist

        let data_file = self
            .h5_data_file
            .as_ref()
            .ok_or_else(|| std::io::Error::other("Writing data was not initialized"))?;

        let group_name = attribute_center_to_hdf5(center);

        // Create the group if it does not exist
        if !data_file.link_exists(group_name) {
            data_file
                .create_group(group_name)
                .map_err(std::io::Error::other)?;
        }

        write_values(
            &data_file.group(group_name).map_err(std::io::Error::other)?,
            name,
            data,
        )?;

        Ok(data_file.filename() + &format!(":{group_name}/{name}"))
    }

    fn write_data_initialize(&mut self, time: &str) -> IoResult<()> {
        if self.h5_data_file.is_some() {
            return Err(std::io::Error::other(
                "Writing data was already initialized",
            ));
        }

        let file_name = self.h5_files_dir.join(format!("data_t_{time}.h5"));
        self.h5_data_file = Some(H5File::create(&file_name).map_err(std::io::Error::other)?);

        Ok(())
    }

    fn write_data_finalize(&mut self) -> IoResult<()> {
        if self.h5_data_file.is_none() {
            return Err(std::io::Error::other("Writing data was not initialized"));
        }

        // TODO check if this flushes the file etc
        self.h5_data_file = None;
        Ok(())
    }
}

fn write_mesh(points: &[f64], cells: &[u64], group: &H5Group) -> IoResult<()> {
    group
        .new_dataset::<f64>()
        .shape(points.len())
        .create(POINTS)
        .map_err(std::io::Error::other)?
        .write(points)
        .map_err(std::io::Error::other)?;

    group
        .new_dataset::<u64>()
        .shape(cells.len())
        .create(CELLS)
        .map_err(std::io::Error::other)?
        .write(cells)
        .map_err(std::io::Error::other)
}

fn write_values(group: &H5Group, dataset_name: &str, vals: &Values) -> IoResult<()> {
    let data_set = match vals {
        Values::F64(_) => group.new_dataset::<f64>(),
        Values::U64(_) => group.new_dataset::<u64>(),
    };

    let data_set = data_set
        .shape(vals.dimensions().0)
        .create(dataset_name)
        .map_err(std::io::Error::other)?;

    match vals {
        Values::F64(v) => data_set.write(v).map_err(std::io::Error::other),
        Values::U64(v) => data_set.write(v).map_err(std::io::Error::other),
    }
}

fn attribute_center_to_hdf5(center: attribute::Center) -> &'static str {
    match center {
        attribute::Center::Node => "point_data",
        attribute::Center::Cell => "cell_data",
        attribute::Center::Edge => "edge_data",
        attribute::Center::Face => "face_data",
        attribute::Center::Grid => "grid_data",
        attribute::Center::Other => "other_data",
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn write_values_works() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.h5");

        let h5_file = H5File::create(&file_name).unwrap();
        let group = h5_file.create_group("test_group").unwrap();

        let vec_f64 = vec![1., 2., 3., 4., 5., 6.];
        let vec_u64 = vec![10_u64, 20, 30, 40, 50, 60];

        write_values(&group, "test_f64", &vec_f64.clone().into()).unwrap();
        write_values(&group, "test_u64", &vec_u64.clone().into()).unwrap();

        // Verify the file exists
        assert!(file_name.exists());

        // Read back the data to verify
        let h5_file_read = H5File::open(&file_name).unwrap();
        let data_f64: Vec<f64> = h5_file_read
            .group("test_group")
            .unwrap()
            .dataset("test_f64")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();
        let data_u64: Vec<u64> = h5_file_read
            .group("test_group")
            .unwrap()
            .dataset("test_u64")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();

        assert_approx_eq!(&[f64], &vec_f64, &data_f64);
        assert_eq!(&vec_u64, &data_u64);
    }

    #[test]
    fn single_files_hdf5_writer_write_data_init_fin() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = SingleFileHdf5Writer::new(file_name).unwrap();

        assert!(writer.write_time.is_none());

        let res_fin = writer.write_data_finalize();
        assert_eq!(
            res_fin.unwrap_err().to_string(),
            "Writing data was not initialized"
        );

        let res_write = writer.write_data(
            "test_data",
            attribute::Center::Node,
            &Values::F64(vec![1.0, 2.0]),
        );
        assert_eq!(
            res_write.unwrap_err().to_string(),
            "Writing data was not initialized"
        );

        writer.write_data_initialize("0.0").unwrap();
        assert!(writer.write_time.is_some());

        let res_init = writer.write_data_initialize("0.0");
        assert_eq!(
            res_init.unwrap_err().to_string(),
            "Writing data was already initialized"
        );

        writer.write_data_finalize().unwrap();
    }

    #[test]
    fn mutliple_files_hdf5_writer_write_data_init_fin() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = MultipleFilesHdf5Writer::new(&file_name).unwrap();
        assert!(writer.h5_data_file.is_none());

        let res_fin = writer.write_data_finalize();
        assert_eq!(
            res_fin.unwrap_err().to_string(),
            "Writing data was not initialized"
        );

        let res_write = writer.write_data(
            "test_data",
            attribute::Center::Node,
            &Values::F64(vec![1.0, 2.0]),
        );
        assert_eq!(
            res_write.unwrap_err().to_string(),
            "Writing data was not initialized"
        );

        let exp_file_name = file_name.with_extension("h5").join("data_t_0.123.h5");
        writer.write_data_initialize("0.123").unwrap();
        assert!(writer.h5_data_file.is_some());

        assert_eq!(
            writer.h5_data_file.as_ref().unwrap().filename(),
            exp_file_name.to_string_lossy()
        );
        assert!(exp_file_name.exists());

        let res_init = writer.write_data_initialize("0.0");
        assert_eq!(
            res_init.unwrap_err().to_string(),
            "Writing data was already initialized"
        );

        writer.write_data_finalize().unwrap();
        assert!(writer.h5_data_file.is_none());
    }

    #[test]
    fn single_file_hdf5_writer_new() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let writer = SingleFileHdf5Writer::new(&file_name).unwrap();
        let exp_file_name = file_name.with_extension("h5");
        assert!(exp_file_name.exists());
        assert_eq!(writer.h5_file.filename(), exp_file_name.to_string_lossy());
    }

    #[test]
    fn mutliple_files_hdf5_writer_new() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let writer = MultipleFilesHdf5Writer::new(&file_name).unwrap();
        let exp_dir_name = file_name.with_extension("h5");
        assert_eq!(writer.h5_files_dir, exp_dir_name);
        assert!(writer.h5_files_dir.exists());
        assert!(writer.h5_files_dir.is_dir());
        assert!(writer.h5_data_file.is_none());
    }

    #[test]
    fn mutliple_files_hdf5_writer_write_mesh() {
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

    #[test]
    fn mutliple_files_hdf5_writer_write_data() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = MultipleFilesHdf5Writer::new(file_name).unwrap();
        let write_time = "12.258";
        let data_file = writer.h5_files_dir.join(format!("data_t_{write_time}.h5"));
        assert!(!data_file.exists());

        writer.write_data_initialize(write_time).unwrap();
        assert!(data_file.exists());

        // write points data
        let data_points = vec![0.0, 1.0, 2.0];
        let data_path_points = writer
            .write_data(
                "dummy_point_data",
                attribute::Center::Node,
                &Values::F64(data_points.clone()),
            )
            .unwrap();

        // write cell data
        let data_cells = vec![-9.0, 1.0, 2.0, 55.87];
        let data_path_cells = writer
            .write_data(
                "some_cell_data",
                attribute::Center::Cell,
                &Values::F64(data_cells.clone()),
            )
            .unwrap();

        writer.write_data_finalize().unwrap();
        assert!(data_file.exists());

        assert_eq!(
            data_path_points,
            data_file.to_string_lossy().to_string() + ":point_data/dummy_point_data"
        );
        assert_eq!(
            data_path_cells,
            data_file.to_string_lossy().to_string() + ":cell_data/some_cell_data"
        );

        // read back the data to verify
        let h5_file = H5File::open(&data_file).unwrap();
        let points_data: Vec<f64> = h5_file
            .dataset("point_data/dummy_point_data")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();
        let cells_data: Vec<f64> = h5_file
            .dataset("cell_data/some_cell_data")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();

        assert_approx_eq!(&[f64], &data_points, &points_data);
        assert_approx_eq!(&[f64], &data_cells, &cells_data);
    }
}

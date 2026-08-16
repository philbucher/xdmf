//! Implementations of writers for HDF5 data storage (single and multiple files).

use std::path::{Path, PathBuf};

use hdf5::{File as H5File, Group as H5Group, H5Type};

use crate::{
    DataStorage, DataWriter, Error, Result, Values,
    error::io_ctx,
    xdmf_elements::{
        attribute,
        data_item::{DataContent, Format},
    },
};

// Attach an operation description to an `hdf5::Error`, mirroring `error::io_ctx` for the
// filesystem case.
fn hdf5_ctx(operation: &'static str) -> impl FnOnce(hdf5::Error) -> Error {
    move |source| Error::Hdf5 { operation, source }
}

const MESH: &str = "mesh";
const DATA: &str = "data";
const POINTS: &str = "points";
const CELLS: &str = "cells";

// zlib/deflate level used when `deflate_level` is `None`. Benchmarked 3, 6 (zlib's default)
// and 9 (max) on a 10M-element CFD case: 9 gains ~0.1% size over 6 for up to 5x slower writes
// on noisy data. 3 writes ~30% faster than 6 for only ~1.3% larger output on noisy data
// (typical for real velocity/pressure fields, which carry solver noise/roundoff), though it
// costs ~25% more size on unrealistically smooth/structured data. Speed wins for the common
// case, hence the default; override per-writer via `DataStorage::Hdf5{SingleFile,MultipleFiles}`.
pub(crate) const DEFAULT_DEFLATE_LEVEL: u8 = 3;

pub(crate) struct SingleFileHdf5Writer {
    h5_file: H5File,
    h5_file_name: PathBuf,
    write_time: Option<String>,
    deflate_level: u8,
}

/// TODO show file hierarchy, and how data is structured
impl SingleFileHdf5Writer {
    pub(crate) fn new(file_name: impl AsRef<Path>, deflate_level: u8) -> Result<Self> {
        let h5_file_name_full = file_name.as_ref().to_path_buf().with_extension("h5");

        if let Some(parent) = h5_file_name_full.parent() {
            crate::mpi_safe_create_dir_all(parent)?;
        }

        let h5_file_name = h5_file_name_full
            .file_name()
            .ok_or(Error::Internal("output path has no file name component"))?;

        let h5_file = H5File::create(&h5_file_name_full).map_err(hdf5_ctx("creating HDF5 file"))?;

        Ok(Self {
            h5_file,
            h5_file_name: h5_file_name.into(),
            write_time: None,
            deflate_level,
        })
    }
}

impl DataWriter for SingleFileHdf5Writer {
    fn format(&self) -> Format {
        Format::HDF
    }

    fn data_storage(&self) -> DataStorage {
        DataStorage::Hdf5SingleFile {
            deflate_level: Some(self.deflate_level),
        }
    }

    fn write_mesh(
        &mut self,
        points: &Values<'_>,
        cells: &[u64],
    ) -> Result<(DataContent, DataContent)> {
        if self.h5_file.link_exists(MESH) {
            return Err(Error::InvalidMesh {
                reason: "mesh was already written".to_string(),
            });
        }

        let mesh_group = self
            .h5_file
            .create_group(MESH)
            .map_err(hdf5_ctx("creating mesh group"))?;

        let (data_name_points, data_name_cells) =
            write_mesh(&mesh_group, points, cells, self.deflate_level)?;

        let h5_file_name = self.h5_file_name.to_string_lossy();
        Ok((
            full_path(&h5_file_name, &data_name_points).into(),
            full_path(&h5_file_name, &data_name_cells).into(),
        ))
    }

    fn write_data(
        &mut self,
        name: &str,
        center: attribute::Center,
        data: &Values<'_>,
    ) -> Result<DataContent> {
        let time = self
            .write_time
            .as_ref()
            .ok_or(Error::Internal("writing data was not initialized"))?;

        let group_name = &format!(
            "{}/{}",
            time_group_name(time),
            attribute::center_to_data_tag(center)
        );

        // Create the group if it does not exist
        if !self.h5_file.link_exists(group_name) {
            self.h5_file
                .create_group(group_name)
                .map_err(hdf5_ctx("creating data group"))?;
        }

        let data_path = write_values(
            &self
                .h5_file
                .group(group_name)
                .map_err(hdf5_ctx("opening data group"))?,
            name,
            data,
            self.deflate_level,
        )?;

        Ok(full_path(&self.h5_file_name.to_string_lossy(), &data_path).into())
    }

    fn write_data_initialize(&mut self, time: &str) -> Result<()> {
        if self.write_time.is_some() {
            return Err(Error::Internal("writing data was already initialized"));
        }

        self.write_time = Some(time.to_string());
        Ok(())
    }
    fn write_data_finalize(&mut self) -> Result<()> {
        if self.write_time.is_none() {
            return Err(Error::Internal("writing data was not initialized"));
        }

        self.write_time = None;
        Ok(())
    }

    fn write_data_discard(&mut self) -> Result<()> {
        let time = self
            .write_time
            .take()
            .ok_or(Error::Internal("writing data was not initialized"))?;

        // Unlinking removes the names, so nothing dangles and the time can be written again --
        // but HDF5 does not hand the freed blocks back, so the file keeps the space until it is
        // repacked (`h5repack`). Discarding is the exceptional path, so that is an acceptable
        // trade against rewriting the file.
        let time_group = time_group_name(&time);
        if self.h5_file.link_exists(&time_group) {
            self.h5_file
                .unlink(&time_group)
                .map_err(hdf5_ctx("removing discarded data group"))?;
        }

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // Flush the HDF5 file
        self.h5_file.flush().map_err(hdf5_ctx("flushing file"))
    }
}

/// TODO show file hierarchy, and how data is structured
pub(crate) struct MultipleFilesHdf5Writer {
    h5_files_dir: PathBuf,
    h5_data_file: Option<H5File>,
    deflate_level: u8,
}

impl MultipleFilesHdf5Writer {
    pub(crate) fn new(file_name: impl AsRef<Path>, deflate_level: u8) -> Result<Self> {
        let h5_files_dir = file_name.as_ref().to_path_buf().with_extension("h5");

        h5_files_dir
            .file_name()
            .ok_or(Error::Internal("output path has no file name component"))?;

        crate::mpi_safe_create_dir_all(&h5_files_dir)?;

        Ok(Self {
            h5_files_dir,
            h5_data_file: None,
            deflate_level,
        })
    }
}

impl DataWriter for MultipleFilesHdf5Writer {
    fn format(&self) -> Format {
        Format::HDF
    }

    fn data_storage(&self) -> DataStorage {
        DataStorage::Hdf5MultipleFiles {
            deflate_level: Some(self.deflate_level),
        }
    }

    fn write_mesh(
        &mut self,
        points: &Values<'_>,
        cells: &[u64],
    ) -> Result<(DataContent, DataContent)> {
        let file_name = self.h5_files_dir.join(format!("{MESH}.h5"));
        let h5_file = H5File::create(&file_name).map_err(hdf5_ctx("creating mesh file"))?;

        let (data_name_points, data_name_cells) =
            write_mesh(&h5_file, points, cells, self.deflate_level)?;

        let rel_file_name = parent_and_filename(&file_name).ok_or(Error::Internal(
            "could not resolve parent directory and file name for an HDF5 path",
        ))?;

        Ok((
            full_path(&rel_file_name, &data_name_points).into(),
            full_path(&rel_file_name, &data_name_cells).into(),
        ))
    }

    fn write_data(
        &mut self,
        name: &str,
        center: attribute::Center,
        data: &Values<'_>,
    ) -> Result<DataContent> {
        // also double check that the name does not already exist

        let data_file = self
            .h5_data_file
            .as_ref()
            .ok_or(Error::Internal("writing data was not initialized"))?;

        let group_name = attribute::center_to_data_tag(center);

        // Create the group if it does not exist
        if !data_file.link_exists(group_name) {
            data_file
                .create_group(group_name)
                .map_err(hdf5_ctx("creating data group"))?;
        }

        let data_path = write_values(
            &data_file
                .group(group_name)
                .map_err(hdf5_ctx("opening data group"))?,
            name,
            data,
            self.deflate_level,
        )?;

        let rel_file_name = parent_and_filename(data_file.filename()).ok_or(Error::Internal(
            "could not resolve parent directory and file name for an HDF5 path",
        ))?;

        Ok(full_path(&rel_file_name, &data_path).into())
    }

    fn write_data_initialize(&mut self, time: &str) -> Result<()> {
        if self.h5_data_file.is_some() {
            return Err(Error::Internal("writing data was already initialized"));
        }

        let file_name = self.h5_files_dir.join(format!("data_t_{time}.h5"));
        self.h5_data_file =
            Some(H5File::create(&file_name).map_err(hdf5_ctx("creating data file"))?);

        Ok(())
    }

    fn write_data_finalize(&mut self) -> Result<()> {
        if self.h5_data_file.is_none() {
            return Err(Error::Internal("writing data was not initialized"));
        }

        // TODO check if this flushes the file etc
        self.h5_data_file = None;

        Ok(())
    }

    fn write_data_discard(&mut self) -> Result<()> {
        let data_file = self
            .h5_data_file
            .take()
            .ok_or(Error::Internal("writing data was not initialized"))?;

        // Read the path back off the handle, then close it before removing -- on Windows an open
        // handle blocks the removal.
        let file_name = PathBuf::from(data_file.filename());
        drop(data_file);

        // The whole step lives in this one file, so discarding it is just removing the file --
        // no other step's data can be in there.
        std::fs::remove_file(&file_name).map_err(io_ctx("removing discarded data file", &file_name))
    }
}

// Points and cells go through `write_values` like any other data, so the dataset type follows the
// `Values` variant (f32 or f64 coordinates) and the filter pipeline is shared with everything else.
fn write_mesh(
    group: &H5Group,
    points: &Values<'_>,
    cells: &[u64],
    deflate_level: u8,
) -> Result<(String, String)> {
    let data_name_points = write_values(group, POINTS, points, deflate_level)?;
    let data_name_cells = write_values(group, CELLS, &Values::from(cells), deflate_level)?;

    Ok((data_name_points, data_name_cells))
}

fn write_values(
    group: &H5Group,
    dataset_name: &str,
    vals: &Values<'_>,
    deflate_level: u8,
) -> Result<String> {
    let shape = vals.dimensions(crate::DataAttribute::Scalar).0;

    match vals {
        Values::F64(v) => create_and_write(group, dataset_name, v, shape, deflate_level),
        Values::F32(v) => create_and_write(group, dataset_name, v, shape, deflate_level),
        Values::I64(v) => create_and_write(group, dataset_name, v, shape, deflate_level),
        Values::I32(v) => create_and_write(group, dataset_name, v, shape, deflate_level),
        Values::U64(v) => create_and_write(group, dataset_name, v, shape, deflate_level),
        Values::U32(v) => create_and_write(group, dataset_name, v, shape, deflate_level),
    }
}

fn create_and_write<T: H5Type>(
    group: &H5Group,
    dataset_name: &str,
    data: &[T],
    shape: Vec<usize>,
    deflate_level: u8,
) -> Result<String> {
    let data_set = group
        .new_dataset::<T>()
        .shape(shape)
        .shuffle()
        .deflate(deflate_level)
        .create(dataset_name)
        .map_err(hdf5_ctx("creating dataset"))?;

    data_set.write(data).map_err(hdf5_ctx("writing dataset"))?;

    Ok(data_set.name())
}

// Group holding everything written for one time step, in the single-file layout. Both writing a
// step's data and discarding it again have to name this group, so the layout is spelled out here
// only -- a rename must not leave the discard removing a group that no longer exists.
fn time_group_name(time: &str) -> String {
    format!("{DATA}/t_{time}")
}
// Built with an explicit `/` rather than `PathBuf::join`/`to_string_lossy`, so the path
// embedded in the XDMF file is valid on every OS regardless of which OS wrote it (e.g. no
// backslashes from a Windows `PathBuf` ending up in a file read back on Linux).
fn parent_and_filename(path: impl AsRef<Path>) -> Option<String> {
    let path = path.as_ref();
    let parent = path.parent()?.file_name()?.to_string_lossy();
    let file_name = path.file_name()?.to_string_lossy();
    Some(format!("{parent}/{file_name}"))
}

// Path that is written to the xdmf file, specifying where the data is stored in the h5 file
// it consists of the path to the h5 file and the location within the file, which are separated by a colon
// e.g. /path/to/file.h5:mesh/points
fn full_path(path: &str, data_name: &str) -> String {
    format!("{path}{}", data_name.replacen('/', ":", 1))
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn full_path_works() {
        let file_name = "some/random/path/test.h5";
        let data_name = "/test_group/test_data";

        assert_eq!(
            full_path(file_name, data_name),
            format!("{file_name}:test_group/test_data")
        );
    }

    #[test]
    fn parent_and_filename_works() {
        assert_eq!(
            parent_and_filename(Path::new("some/random/path/test.h5")).unwrap(),
            "path/test.h5"
        );
        assert!(
            !parent_and_filename(Path::new("some/random/path/test.h5"))
                .unwrap()
                .contains('\\')
        );

        assert!(parent_and_filename(Path::new("test.h5")).is_none(),);
    }

    #[test]
    fn write_mesh_works() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.h5");

        let h5_file = H5File::create(&file_name).unwrap();
        assert!(file_name.exists());

        let group = h5_file.create_group("test_group").unwrap();

        let points = vec![0.0, 1.0, 2.0];
        let cells = vec![0, 1, 2];

        let (data_name_points, data_name_cells) =
            write_mesh(&group, &points.as_slice().into(), &cells, 6).unwrap();
        assert_eq!(data_name_points, "/test_group/points");
        assert_eq!(data_name_cells, "/test_group/cells");

        // Read back the data to verify
        let h5_file_read = H5File::open(&file_name).unwrap();
        let points_read: Vec<f64> = h5_file_read
            .group("test_group")
            .unwrap()
            .dataset("points")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();
        let cells_read: Vec<u64> = h5_file_read
            .group("test_group")
            .unwrap()
            .dataset("cells")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();

        assert_approx_eq!(&[f64], &points, &points_read);
        assert_eq!(&cells, &cells_read);
    }

    #[test]
    fn write_mesh_with_f32_points() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.h5");

        let h5_file = H5File::create(&file_name).unwrap();
        let group = h5_file.create_group("test_group").unwrap();

        let points = vec![0.0_f32, 1.0, 2.0];
        let cells = vec![0_u64, 1, 2];

        write_mesh(&group, &points.as_slice().into(), &cells, 6).unwrap();

        let h5_file_read = H5File::open(&file_name).unwrap();
        let dataset = h5_file_read
            .group("test_group")
            .unwrap()
            .dataset("points")
            .unwrap();

        // the dataset itself is 4-byte floats, not f64 values that happen to have been narrowed
        assert_eq!(dataset.dtype().unwrap().size(), 4);

        let points_read: Vec<f32> = dataset.read().unwrap().to_vec();
        assert_approx_eq!(&[f32], &points, &points_read);
    }

    #[test]
    fn write_values_works() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.h5");

        let h5_file = H5File::create(&file_name).unwrap();
        assert!(file_name.exists());

        let group = h5_file.create_group("test_group").unwrap();

        let vec_f64 = vec![1., 2., 3., 4., 5., 6.];
        let vec_u64 = vec![10_u64, 20, 30, 40, 50, 60];

        let f64_path = write_values(&group, "test_f64", &vec_f64.clone().into(), 6).unwrap();
        let u64_path = write_values(&group, "test_u64", &vec_u64.clone().into(), 6).unwrap();

        assert_eq!(f64_path, "/test_group/test_f64");
        assert_eq!(u64_path, "/test_group/test_u64");

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
    fn write_values_f32() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.h5");

        let h5_file = H5File::create(&file_name).unwrap();
        let group = h5_file.create_group("test_group").unwrap();

        let vec_f32 = vec![1., 2., 3., 4., 5., 6.0_f32];
        let f32_path = write_values(&group, "test_f32", &vec_f32.clone().into(), 6).unwrap();
        assert_eq!(f32_path, "/test_group/test_f32");

        let h5_file_read = H5File::open(&file_name).unwrap();
        let dataset = h5_file_read
            .group("test_group")
            .unwrap()
            .dataset("test_f32")
            .unwrap();

        // stored as 4-byte floats, so a reader gets f32 back rather than widened f64
        assert_eq!(dataset.dtype().unwrap().size(), 4);

        let data_f32: Vec<f32> = dataset.read().unwrap().to_vec();
        assert_approx_eq!(&[f32], &vec_f32, &data_f32);
    }

    #[test]
    fn write_values_integer_types() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.h5");

        let h5_file = H5File::create(&file_name).unwrap();
        let group = h5_file.create_group("test_group").unwrap();

        let vec_u32 = vec![1_u32, 2, 3];
        let vec_i64 = vec![-1_i64, 0, 1];
        let vec_i32 = vec![-2_i32, 0, 2];

        write_values(&group, "test_u32", &vec_u32.clone().into(), 6).unwrap();
        write_values(&group, "test_i64", &vec_i64.clone().into(), 6).unwrap();
        write_values(&group, "test_i32", &vec_i32.clone().into(), 6).unwrap();

        // each type keeps its own width and signedness in the file, no widening to u64/i64
        let h5_file_read = H5File::open(&file_name).unwrap();
        let group_read = h5_file_read.group("test_group").unwrap();

        let dataset_u32 = group_read.dataset("test_u32").unwrap();
        assert_eq!(dataset_u32.dtype().unwrap().size(), 4);
        assert_eq!(dataset_u32.read::<u32, _>().unwrap().to_vec(), vec_u32);

        let dataset_i64 = group_read.dataset("test_i64").unwrap();
        assert_eq!(dataset_i64.dtype().unwrap().size(), 8);
        assert_eq!(dataset_i64.read::<i64, _>().unwrap().to_vec(), vec_i64);

        let dataset_i32 = group_read.dataset("test_i32").unwrap();
        assert_eq!(dataset_i32.dtype().unwrap().size(), 4);
        assert_eq!(dataset_i32.read::<i32, _>().unwrap().to_vec(), vec_i32);
    }

    #[test]
    fn single_file_hdf5_writer_write_data_f32() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = SingleFileHdf5Writer::new(&file_name, DEFAULT_DEFLATE_LEVEL).unwrap();
        let h5_file_name = file_name.with_extension("h5");
        let write_time = "12.258";

        writer.write_data_initialize(write_time).unwrap();

        let data_points = vec![0.0_f32, 1.0, 2.5];
        let data_path_points = writer
            .write_data(
                "dummy_point_data",
                attribute::Center::Node,
                &Values::F32(data_points.as_slice().into()),
            )
            .unwrap();

        writer.write_data_finalize().unwrap();

        assert_eq!(
            data_path_points,
            ("test.h5:data/t_12.258/point_data/dummy_point_data").into()
        );

        let h5_file = H5File::open(h5_file_name).unwrap();
        let dataset = h5_file
            .dataset("data/t_12.258/point_data/dummy_point_data")
            .unwrap();

        assert_eq!(dataset.dtype().unwrap().size(), 4);

        let points_data: Vec<f32> = dataset.read().unwrap().to_vec();
        assert_approx_eq!(&[f32], &data_points, &points_data);
    }

    #[test]
    fn multiple_files_hdf5_writer_write_data_f32() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = MultipleFilesHdf5Writer::new(&file_name, DEFAULT_DEFLATE_LEVEL).unwrap();
        let write_time = "12.258";

        writer.write_data_initialize(write_time).unwrap();

        let data_cells = vec![-9.0_f32, 1.0, 2.0, 55.875];
        writer
            .write_data(
                "some_cell_data",
                attribute::Center::Cell,
                &Values::F32(data_cells.as_slice().into()),
            )
            .unwrap();

        writer.write_data_finalize().unwrap();

        let h5_file =
            H5File::open(writer.h5_files_dir.join(format!("data_t_{write_time}.h5"))).unwrap();
        let dataset = h5_file.dataset("cell_data/some_cell_data").unwrap();

        assert_eq!(dataset.dtype().unwrap().size(), 4);

        let cells_data: Vec<f32> = dataset.read().unwrap().to_vec();
        assert_approx_eq!(&[f32], &data_cells, &cells_data);
    }

    #[test]
    fn single_files_hdf5_writer_write_data_init_fin() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = SingleFileHdf5Writer::new(file_name, DEFAULT_DEFLATE_LEVEL).unwrap();

        assert!(writer.write_time.is_none());

        let res_fin = writer.write_data_finalize();
        std::assert_matches!(
            res_fin.unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        let res_write = writer.write_data(
            "test_data",
            attribute::Center::Node,
            &Values::F64(vec![1.0, 2.0].into()),
        );
        std::assert_matches!(
            res_write.unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        writer.write_data_initialize("1250.9").unwrap();
        assert_eq!(writer.write_time.clone().unwrap(), "1250.9");

        let res_init = writer.write_data_initialize("0.0");
        std::assert_matches!(
            res_init.unwrap_err(),
            Error::Internal("writing data was already initialized")
        );

        writer.write_data_finalize().unwrap();
        assert!(writer.write_time.is_none());
    }

    #[test]
    fn single_files_hdf5_writer_write_data_discard() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = SingleFileHdf5Writer::new(&file_name, DEFAULT_DEFLATE_LEVEL).unwrap();

        std::assert_matches!(
            writer.write_data_discard().unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        writer.write_data_initialize("0.5").unwrap();
        writer
            .write_data(
                "discarded",
                attribute::Center::Node,
                &Values::F64(vec![1.0, 2.0].into()),
            )
            .unwrap();
        assert!(writer.h5_file.link_exists("data/t_0.5"));

        writer.write_data_discard().unwrap();

        // the whole time group is gone, so nothing dangles for a `<Grid>` that was never written
        assert!(!writer.h5_file.link_exists("data/t_0.5"));
        assert!(writer.write_time.is_none());

        // the time can be written again afterwards
        writer.write_data_initialize("0.5").unwrap();
        writer
            .write_data(
                "kept",
                attribute::Center::Node,
                &Values::F64(vec![3.0, 4.0].into()),
            )
            .unwrap();
        writer.write_data_finalize().unwrap();

        assert!(writer.h5_file.link_exists("data/t_0.5/point_data/kept"));
        assert!(
            !writer
                .h5_file
                .link_exists("data/t_0.5/point_data/discarded")
        );
    }

    #[test]
    fn multiple_files_hdf5_writer_write_data_discard() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");
        let mut writer = MultipleFilesHdf5Writer::new(&file_name, DEFAULT_DEFLATE_LEVEL).unwrap();

        std::assert_matches!(
            writer.write_data_discard().unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        let data_file = file_name.with_extension("h5").join("data_t_0.5.h5");

        writer.write_data_initialize("0.5").unwrap();
        writer
            .write_data(
                "discarded",
                attribute::Center::Node,
                &Values::F64(vec![1.0, 2.0].into()),
            )
            .unwrap();
        assert!(data_file.exists());

        writer.write_data_discard().unwrap();

        // the step's whole file is removed, since no other step's data can be in it
        assert!(!data_file.exists());
        assert!(writer.h5_data_file.is_none());

        // the time can be written again afterwards
        writer.write_data_initialize("0.5").unwrap();
        writer
            .write_data(
                "kept",
                attribute::Center::Node,
                &Values::F64(vec![3.0, 4.0].into()),
            )
            .unwrap();
        writer.write_data_finalize().unwrap();
        assert!(data_file.exists());
    }

    #[test]
    fn multiple_files_hdf5_writer_write_data_init_fin() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = MultipleFilesHdf5Writer::new(&file_name, DEFAULT_DEFLATE_LEVEL).unwrap();
        assert!(writer.h5_data_file.is_none());

        let res_fin = writer.write_data_finalize();
        std::assert_matches!(
            res_fin.unwrap_err(),
            Error::Internal("writing data was not initialized")
        );

        let res_write = writer.write_data(
            "test_data",
            attribute::Center::Node,
            &Values::F64(vec![1.0, 2.0].into()),
        );
        std::assert_matches!(
            res_write.unwrap_err(),
            Error::Internal("writing data was not initialized")
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
        std::assert_matches!(
            res_init.unwrap_err(),
            Error::Internal("writing data was already initialized")
        );

        writer.write_data_finalize().unwrap();
        assert!(writer.h5_data_file.is_none());
    }

    #[test]
    fn single_file_hdf5_writer_new() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let writer = SingleFileHdf5Writer::new(&file_name, DEFAULT_DEFLATE_LEVEL).unwrap();
        let exp_file_name = file_name.with_extension("h5");
        assert!(exp_file_name.exists());
        assert_eq!(writer.h5_file.filename(), exp_file_name.to_string_lossy());
        assert_eq!(writer.h5_file_name, exp_file_name.file_name().unwrap());
    }

    #[test]
    fn multiple_files_hdf5_writer_new() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let writer = MultipleFilesHdf5Writer::new(&file_name, DEFAULT_DEFLATE_LEVEL).unwrap();
        let exp_dir_name = file_name.with_extension("h5");
        assert_eq!(writer.h5_files_dir, exp_dir_name);
        assert!(writer.h5_files_dir.exists());
        assert!(writer.h5_files_dir.is_dir());
        assert!(writer.h5_data_file.is_none());
    }

    #[test]
    fn single_file_hdf5_writer_write_mesh() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = SingleFileHdf5Writer::new(&file_name, DEFAULT_DEFLATE_LEVEL).unwrap();
        let h5_file = file_name.with_extension("h5");

        let points = vec![0.0, 1.0, 2.0];
        let cells = vec![0, 1, 2];
        let (points_path, cells_path) = writer
            .write_mesh(&points.as_slice().into(), &cells)
            .unwrap();

        assert_eq!(points_path, ("test.h5:mesh/points").into());
        assert_eq!(cells_path, ("test.h5:mesh/cells").into());

        // Ensure the file is closed before reading.
        // Seems to work also without, but better to be explicit.
        drop(writer);

        // read back the data to verify
        let h5_file = H5File::open(h5_file).unwrap();
        let points_data: Vec<f64> = h5_file
            .dataset("mesh/points")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();
        let cells_data: Vec<u64> = h5_file
            .dataset("mesh/cells")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();

        assert_approx_eq!(&[f64], &points, &points_data);
        assert_eq!(&cells, &cells_data);
    }

    #[test]
    fn multiple_files_hdf5_writer_write_mesh() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = MultipleFilesHdf5Writer::new(file_name, DEFAULT_DEFLATE_LEVEL).unwrap();
        let mesh_file = writer.h5_files_dir.join("mesh.h5");
        assert!(!mesh_file.exists());

        let points = vec![0.0, 1.0, 2.0];
        let cells = vec![0, 1, 2];
        let (points_path, cells_path) = writer
            .write_mesh(&points.as_slice().into(), &cells)
            .unwrap();
        assert!(mesh_file.exists());

        assert_eq!(points_path, ("test.h5/mesh.h5:points").into());
        assert_eq!(cells_path, ("test.h5/mesh.h5:cells").into());

        // read back the data to verify
        let h5_file = H5File::open(&mesh_file).unwrap();
        let points_data: Vec<f64> = h5_file.dataset("points").unwrap().read().unwrap().to_vec();
        let cells_data: Vec<u64> = h5_file.dataset("cells").unwrap().read().unwrap().to_vec();

        assert_approx_eq!(&[f64], &points, &points_data);
        assert_eq!(&cells, &cells_data);
    }

    #[test]
    fn single_file_hdf5_writer_write_data() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = SingleFileHdf5Writer::new(&file_name, DEFAULT_DEFLATE_LEVEL).unwrap();
        let h5_file = file_name.with_extension("h5");
        let write_time = "12.258";

        writer.write_data_initialize(write_time).unwrap();

        // write points data
        let data_points = vec![0.0, 1.0, 2.0];
        let data_path_points = writer
            .write_data(
                "dummy_point_data",
                attribute::Center::Node,
                &Values::F64(data_points.as_slice().into()),
            )
            .unwrap();

        // write cell data
        let data_cells = vec![-9.0, 1.0, 2.0, 55.87];
        let data_path_cells = writer
            .write_data(
                "some_cell_data",
                attribute::Center::Cell,
                &Values::F64(data_cells.as_slice().into()),
            )
            .unwrap();

        writer.write_data_finalize().unwrap();

        assert_eq!(
            data_path_points,
            ("test.h5:data/t_12.258/point_data/dummy_point_data").into()
        );
        assert_eq!(
            data_path_cells,
            ("test.h5:data/t_12.258/cell_data/some_cell_data").into()
        );

        // read back the data to verify
        let h5_file = H5File::open(h5_file).unwrap();
        let points_data: Vec<f64> = h5_file
            .dataset("data/t_12.258/point_data/dummy_point_data")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();

        let cells_data: Vec<f64> = h5_file
            .dataset("data/t_12.258/cell_data/some_cell_data")
            .unwrap()
            .read()
            .unwrap()
            .to_vec();

        assert_approx_eq!(&[f64], &data_points, &points_data);
        assert_approx_eq!(&[f64], &data_cells, &cells_data);
    }

    #[test]
    fn multiple_files_hdf5_writer_write_data() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("sub/folder/test.xdmf");
        let mut writer = MultipleFilesHdf5Writer::new(file_name, DEFAULT_DEFLATE_LEVEL).unwrap();
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
                &Values::F64(data_points.as_slice().into()),
            )
            .unwrap();

        // write cell data
        let data_cells = vec![-9.0, 1.0, 2.0, 55.87];
        let data_path_cells = writer
            .write_data(
                "some_cell_data",
                attribute::Center::Cell,
                &Values::F64(data_cells.as_slice().into()),
            )
            .unwrap();

        writer.write_data_finalize().unwrap();
        assert!(data_file.exists());

        assert_eq!(
            data_path_points,
            ("test.h5/data_t_12.258.h5:point_data/dummy_point_data").into()
        );
        assert_eq!(
            data_path_cells,
            ("test.h5/data_t_12.258.h5:cell_data/some_cell_data").into()
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

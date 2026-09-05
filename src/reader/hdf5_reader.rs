//! Reading a `Format="HDF"` `DataItem`'s heavy data: `file.h5:group/dataset` -> [`Values`].
//!
//! An HDF5 dataset is self-describing, so the light data's `NumberType`/`Precision` is not needed
//! here. The whole module is gated on the `hdf5` feature and holds no `cfg` of its own.

use std::{
    path::{Path, PathBuf},
    sync::{Mutex, PoisonError},
};

use hdf5::{Dataset, File as H5File, H5Type};

use crate::{Error, Result, Values, reader::sealed::SealedValueType};

/// The HDF5 file a read last opened, kept open for the next read that names the same one.
///
/// One slot rather than a map, which would hold one open file descriptor per time step and run a
/// long run out of them. Reads come in file order, so a single slot rarely misses.
#[derive(Default)]
pub(super) struct FileCache {
    last: Mutex<Option<(PathBuf, H5File)>>,
}

impl FileCache {
    /// The file at `path`, opened only if it is not the one already held.
    ///
    /// The handle is cloned out rather than borrowed, so the lock is released before the read
    /// itself runs.
    fn open(&self, path: &Path) -> Result<H5File> {
        // a previous read panicking says nothing about the handle it left behind
        let mut cached = self.last.lock().unwrap_or_else(PoisonError::into_inner);

        if let Some((cached_path, file)) = cached.as_ref()
            && cached_path == path
        {
            return Ok(file.clone());
        }

        let file = H5File::open(path).map_err(|source| Error::Hdf5 {
            operation: "opening HDF5 file",
            source,
        })?;
        *cached = Some((path.to_path_buf(), file.clone()));

        Ok(file)
    }
}

/// The whole dataset at `dataset_path` inside `file_path`.
pub(super) fn read(
    file_path: &Path,
    dataset_path: &str,
    files: &FileCache,
) -> Result<Values<'static>> {
    let dataset = open_dataset(file_path, dataset_path, files)?;

    values_of(&dataset)
}

/// The same read, straight into `into` where the dataset's own element type is already `T`,
/// reporting whether it was. `false` leaves `into` untouched, for the caller to convert.
///
/// This is the only path that skips an intermediate array, so a caller looping with the same
/// buffer allocates once rather than once per call.
pub(super) fn read_exact_into<T: SealedValueType>(
    file_path: &Path,
    dataset_path: &str,
    files: &FileCache,
    into: &mut Vec<T>,
) -> Result<bool> {
    let dataset = open_dataset(file_path, dataset_path, files)?;

    if !dtype_of(&dataset)?.is::<T>() {
        return Ok(false);
    }

    into.clear();
    into.resize(dataset.size(), T::default());
    dataset
        .read_into_raw(into.as_mut_slice())
        .map_err(|source| Error::Hdf5 {
            operation: "reading HDF5 dataset",
            source,
        })?;

    Ok(true)
}

fn open_dataset(file_path: &Path, dataset_path: &str, files: &FileCache) -> Result<Dataset> {
    files
        .open(file_path)?
        .dataset(dataset_path)
        .map_err(|source| Error::Hdf5 {
            operation: "opening HDF5 dataset",
            source,
        })
}

fn dtype_of(dataset: &Dataset) -> Result<hdf5::Datatype> {
    dataset.dtype().map_err(|source| Error::Hdf5 {
        operation: "reading HDF5 dataset type",
        source,
    })
}

/// The whole dataset as the [`Values`] variant its own element type names.
fn values_of(dataset: &Dataset) -> Result<Values<'static>> {
    let dtype = dtype_of(dataset)?;
    let ctx = "reading HDF5 dataset";

    if dtype.is::<f64>() {
        return Ok(Values::from(read_raw::<f64>(dataset, ctx)?));
    }
    if dtype.is::<f32>() {
        return Ok(Values::from(read_raw::<f32>(dataset, ctx)?));
    }
    if dtype.is::<i64>() {
        return Ok(Values::from(read_raw::<i64>(dataset, ctx)?));
    }
    if dtype.is::<i32>() {
        return Ok(Values::from(read_raw::<i32>(dataset, ctx)?));
    }
    if dtype.is::<u64>() {
        return Ok(Values::from(read_raw::<u64>(dataset, ctx)?));
    }
    if dtype.is::<u32>() {
        return Ok(Values::from(read_raw::<u32>(dataset, ctx)?));
    }

    Err(Error::Unsupported {
        reason: format!("HDF5 dataset has an unsupported element type: {dtype:?}"),
    })
}

fn read_raw<T: H5Type>(dataset: &Dataset, operation: &'static str) -> Result<Vec<T>> {
    dataset
        .read_raw::<T>()
        .map_err(|source| Error::Hdf5 { operation, source })
}

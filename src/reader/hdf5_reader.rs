//! Reading a `Format="HDF"` `DataItem`'s heavy data: `file.h5:group/dataset` -> [`Values`].
//!
//! An HDF5 dataset is self-describing, so this does not need `NumberType`/`Precision` from the
//! light data at all -- the dataset's own `H5Type` says which of the six [`Values`] variants to
//! read it as. Splitting the light data's `file:path` text is `selection.rs`'s job, so nothing
//! here touches a `DataItem`: this module is entirely gated on the `hdf5` feature (see
//! `reader.rs`) and holds no `cfg` of its own.

use std::{
    path::{Path, PathBuf},
    sync::{Mutex, PoisonError},
};

use hdf5::{Dataset, File as H5File, H5Type};

use crate::{Error, Result, Values, reader::sealed::SealedValueType};

/// The HDF5 file a read last opened, kept open for the next read that names the same one.
///
/// One slot rather than a map of every file seen: `Hdf5SingleFile` holds every array of a whole
/// time series in one file, so a single slot never misses there, and `Hdf5MultipleFiles` reads a
/// step's fields out of that step's one file before moving on to the next. A map would instead
/// hold one open file descriptor per time step, which a long run would run the process out of.
///
/// A `Mutex` rather than a `RefCell` because every read method takes `&self`, and
/// [`TimeSeriesReader`](crate::TimeSeriesReader) stays `Sync` this way. It is never contended in
/// practice -- the reader is used from one thread at a time -- and the work it guards is a
/// pointer comparison plus a handle refcount.
#[derive(Default)]
pub(super) struct FileCache {
    last: Mutex<Option<(PathBuf, H5File)>>,
}

impl FileCache {
    /// The file at `path`, opened only if it is not the one already held.
    ///
    /// The handle is cloned out rather than borrowed, so the lock is released before the read
    /// itself runs; cloning an `H5File` is a refcount bump on the same open file.
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

/// The same read, straight into `into` where the dataset's own element type is already `T` --
/// reporting whether it was.
///
/// `false` leaves `into` untouched: the dataset holds another type, and only the caller knows how
/// that one converts. Widening is what a field wants, a check against the index type is what a
/// connectivity wants, and a rejection is what a coordinate wants; each says so in its own words,
/// which is why that decision is not made here.
///
/// The matching case is the common one -- a mesh or a field read back at the width it was written
/// -- and it is the only one that can skip the intermediate array: `into` is resized to the
/// dataset and HDF5 fills it in place, so a caller looping with the same buffer allocates once
/// rather than once per call.
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

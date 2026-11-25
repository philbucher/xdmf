//! Reading a `Format="HDF"` `DataItem`'s heavy data: `file.h5:group/dataset` -> [`Values`].
//!
//! An HDF5 dataset is self-describing, so the light data's `NumberType`/`Precision` is not needed:
//! the dataset's own `H5Type` picks the [`Values`] variant. Splitting `file:path` is
//! `selection.rs`'s job, so nothing here touches a `DataItem`; the whole module is gated on the
//! `hdf5` feature and holds no `cfg` of its own.

use std::{
    path::{Path, PathBuf},
    sync::{Mutex, PoisonError},
};

use hdf5::{Dataset, File as H5File, H5Type, IndexType, IterationOrder, LinkType, dataset::Layout};

use crate::{Error, Result, Values, reader::sealed::SealedValueType};

/// The HDF5 file a read last opened, kept open for the next read that names the same one.
///
/// One slot rather than a map, which would hold one open file descriptor per time step and run a
/// long run out of them. Reads come in file order, so a single slot rarely misses.
///
/// A `Mutex` rather than a `RefCell` because the read methods take `&self`, which keeps
/// [`TimeSeriesReader`](crate::TimeSeriesReader) `Sync`. It guards a pointer comparison.
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

/// The same read, straight into `into` where the dataset's own element type is already `T`,
/// reporting whether it was. `false` leaves `into` untouched: only the caller knows how another
/// type converts -- a field widens it, a connectivity checks its values, a coordinate rejects it.
///
/// The matching case is the common one and the only one that skips the intermediate array, so a
/// caller looping with the same buffer allocates once rather than once per call.
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
    let file = files.open(file_path)?;
    reject_indirect_links(&file, dataset_path)?;

    let dataset = file.dataset(dataset_path).map_err(|source| Error::Hdf5 {
        operation: "opening HDF5 dataset",
        source,
    })?;
    reject_storage_in_other_files(&dataset, dataset_path)?;

    Ok(dataset)
}

/// Rejects a dataset path passing through anything but hard links, before HDF5 follows it: an
/// external link opens the file it names, and a soft link can point at an external one. That file
/// is named by a path the file's author chose, so following it could open any file.
fn reject_indirect_links(file: &H5File, dataset_path: &str) -> Result<()> {
    let link_error = |source| Error::Hdf5 {
        operation: "reading HDF5 links",
        source,
    };

    let mut group = file.group("/").map_err(link_error)?;
    // HDF5 skips `.` as it does an empty name, so a link named `.` cannot exist to look up
    let mut names = dataset_path
        .split('/')
        .filter(|name| !name.is_empty() && *name != ".")
        .peekable();
    while let Some(name) = names.next() {
        let link_type = group
            .find_link(IndexType::Name, IterationOrder::Native, |link, info| {
                Ok((link == name).then_some(info.link_type))
            })
            .map_err(link_error)?;

        match link_type {
            Some(LinkType::Hard) => {}
            Some(LinkType::Soft | LinkType::External) => {
                return Err(unsupported(
                    dataset_path,
                    "is reached through a soft or external link",
                ));
            }
            // opening the dataset reports the missing name
            None => return Ok(()),
        }

        if names.peek().is_some() {
            group = group.group(name).map_err(link_error)?;
        }
    }

    Ok(())
}

/// Rejects a dataset whose values HDF5 would fetch from another file: external raw storage or a
/// virtual dataset. Each names that file by a path the file's author chose, so reading it could
/// expose any file the process can read.
fn reject_storage_in_other_files(dataset: &Dataset, dataset_path: &str) -> Result<()> {
    let dcpl = dataset.dcpl().map_err(|source| Error::Hdf5 {
        operation: "reading HDF5 dataset creation properties",
        source,
    })?;
    let plist_error = |source| Error::Hdf5 {
        operation: "reading HDF5 dataset storage layout",
        source,
    };

    // the `get_` variants, because `layout()` and `external()` turn an error into a default that
    // would let the dataset through
    if !matches!(
        dcpl.get_layout().map_err(plist_error)?,
        Layout::Compact | Layout::Contiguous | Layout::Chunked
    ) {
        return Err(unsupported(
            dataset_path,
            "is a virtual dataset drawing on other files",
        ));
    }
    if !dcpl.get_external().map_err(plist_error)?.is_empty() {
        return Err(unsupported(
            dataset_path,
            "stores its values in external raw files",
        ));
    }

    Ok(())
}

fn unsupported(dataset_path: &str, what: &str) -> Error {
    Error::Unsupported {
        reason: format!("HDF5 dataset '{dataset_path}' {what}, which this reader does not follow"),
    }
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

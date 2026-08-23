//! Reading a `Format="HDF"` `DataItem`'s heavy data: `file.h5:group/dataset` -> [`Values`](crate::Values).
//!
//! An HDF5 dataset is self-describing, so this does not need `NumberType`/`Precision` from the
//! light data at all -- the dataset's own `H5Type` says which of the six
//! [`Values`](crate::Values) variants to read it as.

use std::path::Path;

use crate::{
    Error, Result,
    xdmf_elements::data_item::{DataContent, DataItem, Format},
};

/// Parse the `file:internal/path` heavy-data path this crate's HDF5 writer produces, and read the
/// dataset it names.
pub(super) fn read(item: &DataItem, base_dir: &Path) -> Result<crate::Values<'static>> {
    if item.format != Some(Format::HDF) {
        return Err(Error::Unsupported {
            reason: format!(
                "Format {:?} is not supported by this reader, only \"HDF\" is",
                item.format
            ),
        });
    }

    let DataContent::Raw(raw) = &item.data else {
        return Err(Error::InvalidDocument {
            reason: "a Format=\"HDF\" DataItem has no path text".to_string(),
        });
    };

    let (file_part, dataset_path) = raw.split_once(':').ok_or_else(|| Error::InvalidDocument {
        reason: format!("'{raw}' is not a valid HDF5 heavy-data path, expected 'file:path'"),
    })?;

    read_hdf5(&base_dir.join(file_part), dataset_path)
}

#[cfg(feature = "hdf5")]
fn read_hdf5(file_path: &Path, dataset_path: &str) -> Result<crate::Values<'static>> {
    use hdf5::File as H5File;

    use crate::Values;

    let h5_file = H5File::open(file_path).map_err(|source| Error::Hdf5 {
        operation: "opening HDF5 file",
        source,
    })?;
    let dataset = h5_file
        .dataset(dataset_path)
        .map_err(|source| Error::Hdf5 {
            operation: "opening HDF5 dataset",
            source,
        })?;

    let dtype = dataset.dtype().map_err(|source| Error::Hdf5 {
        operation: "reading HDF5 dataset type",
        source,
    })?;

    let ctx = "reading HDF5 dataset";

    if dtype.is::<f64>() {
        return Ok(Values::from(read_raw::<f64>(&dataset, ctx)?));
    }
    if dtype.is::<f32>() {
        return Ok(Values::from(read_raw::<f32>(&dataset, ctx)?));
    }
    if dtype.is::<i64>() {
        return Ok(Values::from(read_raw::<i64>(&dataset, ctx)?));
    }
    if dtype.is::<i32>() {
        return Ok(Values::from(read_raw::<i32>(&dataset, ctx)?));
    }
    if dtype.is::<u64>() {
        return Ok(Values::from(read_raw::<u64>(&dataset, ctx)?));
    }
    if dtype.is::<u32>() {
        return Ok(Values::from(read_raw::<u32>(&dataset, ctx)?));
    }

    Err(Error::Unsupported {
        reason: format!("HDF5 dataset has an unsupported element type: {dtype:?}"),
    })
}

#[cfg(feature = "hdf5")]
fn read_raw<T: hdf5::H5Type>(dataset: &hdf5::Dataset, operation: &'static str) -> Result<Vec<T>> {
    dataset
        .read_raw::<T>()
        .map_err(|source| Error::Hdf5 { operation, source })
}

#[cfg(not(feature = "hdf5"))]
fn read_hdf5(_file_path: &Path, _dataset_path: &str) -> Result<crate::Values<'static>> {
    Err(Error::Unsupported {
        reason: "the document holds Format=\"HDF\" data, but this build was compiled without \
                 the 'hdf5' feature"
            .to_string(),
    })
}

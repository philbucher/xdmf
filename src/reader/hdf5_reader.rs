//! Reader for `Format="HDF"` heavy data: `path/to/file.h5:/group/dataset`, covering both
//! `Hdf5SingleFile` and `Hdf5MultipleFiles` output — the file path is in the `DataItem` itself,
//! so there is nothing storage-mode-specific left to branch on by the time it reaches here.

use std::path::Path;

use hdf5::File as H5File;

use super::data_reader::{SourceKind, check_len};
use crate::{Error, Result, Values, xdmf_elements::data_item::DataItem};

fn hdf5_ctx(operation: &'static str) -> impl FnOnce(hdf5::Error) -> Error {
    move |source| Error::Hdf5 { operation, source }
}

pub(crate) fn read(
    item: &DataItem,
    base_dir: &Path,
    kind: SourceKind,
    expected_len: usize,
    xdmf_path: &Path,
) -> Result<Values<'static>> {
    let text = item.text.as_deref().ok_or_else(|| Error::InvalidFile {
        path: xdmf_path.to_path_buf(),
        reason: "DataItem with Format=\"HDF\" has no file/dataset reference".to_string(),
    })?;
    // The writer only ever produces one ':' (it replaces the first '/' of the in-file dataset
    // path with ':' — see `hdf5_writer::full_path`), so a relative file path containing a colon
    // of its own is a known, accepted limitation, not something this splits correctly.
    let (file_part, dataset_part) = text.split_once(':').ok_or_else(|| Error::InvalidFile {
        path: xdmf_path.to_path_buf(),
        reason: format!(
            "DataItem Format=\"HDF\" content '{text}' is not of the form 'file.h5:/path/to/dataset'"
        ),
    })?;

    let h5_path = base_dir.join(file_part);
    let h5_file = H5File::open(&h5_path).map_err(hdf5_ctx("opening HDF5 file"))?;
    let dataset = h5_file
        .dataset(dataset_part)
        .map_err(hdf5_ctx("opening dataset"))?;

    let values: Values<'static> = match kind {
        SourceKind::F32 => dataset
            .read_raw::<f32>()
            .map_err(hdf5_ctx("reading dataset"))?
            .into(),
        SourceKind::F64 => dataset
            .read_raw::<f64>()
            .map_err(hdf5_ctx("reading dataset"))?
            .into(),
        SourceKind::U32 => {
            let v: Vec<u32> = dataset
                .read_raw::<u32>()
                .map_err(hdf5_ctx("reading dataset"))?;
            v.into_iter().map(u64::from).collect::<Vec<_>>().into()
        }
        SourceKind::U64 => dataset
            .read_raw::<u64>()
            .map_err(hdf5_ctx("reading dataset"))?
            .into(),
    };

    check_len(values.len(), expected_len, xdmf_path)?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdmf_elements::{data_item::NumberType, dimensions::Dimensions};

    fn write_dataset(path: &Path, group: &str, name: &str, data: &[f64]) {
        let file = H5File::create(path).unwrap();
        let g = file.create_group(group).unwrap();
        g.new_dataset::<f64>()
            .shape(data.len())
            .create(name)
            .unwrap()
            .write(data)
            .unwrap();
    }

    #[test]
    fn reads_dataset_from_single_file() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let h5_path = tmp_dir.path().join("out.h5");
        write_dataset(&h5_path, "mesh", "points", &[1.0, 2.0, 3.0]);

        let item = DataItem {
            dimensions: Some(Dimensions(vec![3])),
            text: Some("out.h5:mesh/points".to_string()),
            number_type: Some(NumberType::Float),
            precision: Some(8),
            ..Default::default()
        };
        let result = read(
            &item,
            tmp_dir.path(),
            SourceKind::F64,
            3,
            Path::new("x.xdmf2"),
        )
        .unwrap();
        match result {
            Values::F64(v) => assert_eq!(v.into_owned(), vec![1.0, 2.0, 3.0]),
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn rejects_malformed_reference() {
        let item = DataItem {
            dimensions: Some(Dimensions(vec![3])),
            text: Some("no_colon_here".to_string()),
            number_type: Some(NumberType::Float),
            precision: Some(8),
            ..Default::default()
        };
        let err = read(
            &item,
            Path::new("."),
            SourceKind::F64,
            3,
            Path::new("x.xdmf2"),
        )
        .unwrap_err();
        std::assert_matches!(err, Error::InvalidFile { reason, .. } if reason.contains("file.h5:/path/to/dataset"));
    }
}

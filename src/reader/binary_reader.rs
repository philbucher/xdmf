//! Reader for `Format="Binary"` heavy data: a relative path to a raw little/big/native-endian file.

use std::path::Path;

use super::data_reader::{SourceKind, check_len};
use crate::{
    Error, Result, Values,
    error::io_ctx,
    xdmf_elements::data_item::{DataItem, Endian},
};

pub(crate) fn read(
    item: &DataItem,
    base_dir: &Path,
    kind: SourceKind,
    expected_len: usize,
    xdmf_path: &Path,
) -> Result<Values<'static>> {
    let relative_path = item.text.as_deref().ok_or_else(|| Error::InvalidFile {
        path: xdmf_path.to_path_buf(),
        reason: "DataItem with Format=\"Binary\" has no file path".to_string(),
    })?;
    let path = base_dir.join(relative_path);
    let bytes = std::fs::read(&path).map_err(io_ctx("reading binary data file", &path))?;
    let endian = item.endian.unwrap_or_default();

    let elem_size = match kind {
        SourceKind::F32 | SourceKind::U32 => 4,
        SourceKind::F64 | SourceKind::U64 => 8,
    };
    if !bytes.len().is_multiple_of(elem_size) {
        return Err(Error::InvalidFile {
            path: xdmf_path.to_path_buf(),
            reason: format!(
                "binary data file '{}' has {} bytes, which is not a multiple of the element size {elem_size}",
                path.display(),
                bytes.len()
            ),
        });
    }

    let values: Values<'static> = match kind {
        SourceKind::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32_from_bytes(c, endian))
            .collect::<Vec<_>>()
            .into(),
        SourceKind::F64 => bytes
            .chunks_exact(8)
            .map(|c| f64_from_bytes(c, endian))
            .collect::<Vec<_>>()
            .into(),
        SourceKind::U32 => bytes
            .chunks_exact(4)
            .map(|c| u64::from(u32_from_bytes(c, endian)))
            .collect::<Vec<_>>()
            .into(),
        SourceKind::U64 => bytes
            .chunks_exact(8)
            .map(|c| u64_from_bytes(c, endian))
            .collect::<Vec<_>>()
            .into(),
    };

    check_len(values.len(), expected_len, xdmf_path)?;
    Ok(values)
}

fn f32_from_bytes(bytes: &[u8], endian: Endian) -> f32 {
    let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
    match endian {
        Endian::Little => f32::from_le_bytes(bytes),
        Endian::Big => f32::from_be_bytes(bytes),
        Endian::Native => f32::from_ne_bytes(bytes),
    }
}

fn f64_from_bytes(bytes: &[u8], endian: Endian) -> f64 {
    let bytes = [
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ];
    match endian {
        Endian::Little => f64::from_le_bytes(bytes),
        Endian::Big => f64::from_be_bytes(bytes),
        Endian::Native => f64::from_ne_bytes(bytes),
    }
}

fn u32_from_bytes(bytes: &[u8], endian: Endian) -> u32 {
    let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
    match endian {
        Endian::Little => u32::from_le_bytes(bytes),
        Endian::Big => u32::from_be_bytes(bytes),
        Endian::Native => u32::from_ne_bytes(bytes),
    }
}

fn u64_from_bytes(bytes: &[u8], endian: Endian) -> u64 {
    let bytes = [
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ];
    match endian {
        Endian::Little => u64::from_le_bytes(bytes),
        Endian::Big => u64::from_be_bytes(bytes),
        Endian::Native => u64::from_ne_bytes(bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdmf_elements::{data_item::NumberType, dimensions::Dimensions};

    #[test]
    fn reads_little_endian_f64() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_path = tmp_dir.path().join("points.bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.5_f64.to_le_bytes());
        bytes.extend_from_slice(&(-2.5_f64).to_le_bytes());
        std::fs::write(&file_path, &bytes).unwrap();

        let item = DataItem {
            dimensions: Some(Dimensions(vec![2])),
            text: Some("points.bin".to_string()),
            number_type: Some(NumberType::Float),
            precision: Some(8),
            endian: Some(Endian::Little),
            ..Default::default()
        };
        let result = read(
            &item,
            tmp_dir.path(),
            SourceKind::F64,
            2,
            Path::new("x.xdmf2"),
        )
        .unwrap();
        match result {
            Values::F64(v) => assert_eq!(v.into_owned(), vec![1.5, -2.5]),
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn reads_big_endian_u32_widened_to_u64() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_path = tmp_dir.path().join("cells.bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&7_u32.to_be_bytes());
        bytes.extend_from_slice(&8_u32.to_be_bytes());
        std::fs::write(&file_path, &bytes).unwrap();

        let item = DataItem {
            dimensions: Some(Dimensions(vec![2])),
            text: Some("cells.bin".to_string()),
            number_type: Some(NumberType::UInt),
            precision: Some(4),
            endian: Some(Endian::Big),
            ..Default::default()
        };
        let result = read(
            &item,
            tmp_dir.path(),
            SourceKind::U32,
            2,
            Path::new("x.xdmf2"),
        )
        .unwrap();
        match result {
            Values::U64(v) => assert_eq!(v.into_owned(), vec![7, 8]),
            _ => panic!("expected U64"),
        }
    }

    #[test]
    fn rejects_truncated_file() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_path = tmp_dir.path().join("short.bin");
        std::fs::write(&file_path, [0_u8; 5]).unwrap();

        let item = DataItem {
            dimensions: Some(Dimensions(vec![2])),
            text: Some("short.bin".to_string()),
            number_type: Some(NumberType::UInt),
            precision: Some(4),
            endian: Some(Endian::Little),
            ..Default::default()
        };
        let err = read(
            &item,
            tmp_dir.path(),
            SourceKind::U32,
            2,
            Path::new("x.xdmf2"),
        )
        .unwrap_err();
        std::assert_matches!(err, Error::InvalidFile { reason, .. } if reason.contains("not a multiple"));
    }
}

//! Reader for `Format="XML"` heavy data: inline text, or an `xi:include`d text file.

use std::{path::Path, str::FromStr};

use super::data_reader::{SourceKind, check_len, resolve_text};
use crate::{Error, Result, Values, xdmf_elements::data_item::DataItem};

pub(crate) fn read(
    item: &DataItem,
    base_dir: &Path,
    kind: SourceKind,
    expected_len: usize,
    xdmf_path: &Path,
) -> Result<Values<'static>> {
    let text = resolve_text(item, base_dir, xdmf_path)?;
    // Multi-line inline text (`"\n0 1 2\n3 4 5\n"`) round-trips as-is, so this must split on any
    // whitespace, not just single spaces or single lines.
    let tokens = text.split_whitespace();

    let values: Values<'static> = match kind {
        SourceKind::F32 => parse_all::<f32>(tokens, xdmf_path)?.into(),
        SourceKind::F64 => parse_all::<f64>(tokens, xdmf_path)?.into(),
        SourceKind::U32 => {
            let v: Vec<u32> = parse_all(tokens, xdmf_path)?;
            v.into_iter().map(u64::from).collect::<Vec<_>>().into()
        }
        SourceKind::U64 => parse_all::<u64>(tokens, xdmf_path)?.into(),
    };

    check_len(values.len(), expected_len, xdmf_path)?;
    Ok(values)
}

fn parse_all<'a, T: FromStr>(
    tokens: impl Iterator<Item = &'a str>,
    xdmf_path: &Path,
) -> Result<Vec<T>> {
    tokens
        .map(|token| {
            token
                .parse::<T>()
                .map_err(|_parse_error| Error::InvalidFile {
                    path: xdmf_path.to_path_buf(),
                    reason: format!("could not parse '{token}' as a number"),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdmf_elements::{data_item::XInclude, dimensions::Dimensions};

    fn item_with_text(
        text: &str,
        number_type: crate::xdmf_elements::data_item::NumberType,
        precision: u8,
    ) -> DataItem {
        DataItem {
            text: Some(text.to_string()),
            number_type: Some(number_type),
            precision: Some(precision),
            ..Default::default()
        }
    }

    #[test]
    fn reads_inline_f64() {
        use crate::xdmf_elements::data_item::NumberType;
        let item = DataItem {
            dimensions: Some(Dimensions(vec![3])),
            ..item_with_text("1.0 2.0 3.0", NumberType::Float, 8)
        };
        let result = read(
            &item,
            Path::new("."),
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
    fn reads_multiline_inline_text() {
        use crate::xdmf_elements::data_item::NumberType;
        let item = DataItem {
            dimensions: Some(Dimensions(vec![6])),
            ..item_with_text("\n0 1 2\n3 4 5\n", NumberType::UInt, 8)
        };
        let result = read(
            &item,
            Path::new("."),
            SourceKind::U64,
            6,
            Path::new("x.xdmf2"),
        )
        .unwrap();
        match result {
            Values::U64(v) => assert_eq!(v.into_owned(), vec![0, 1, 2, 3, 4, 5]),
            _ => panic!("expected U64"),
        }
    }

    #[test]
    fn widens_u32_precision_to_u64() {
        use crate::xdmf_elements::data_item::NumberType;
        let item = DataItem {
            dimensions: Some(Dimensions(vec![2])),
            ..item_with_text("1 2", NumberType::UInt, 4)
        };
        let result = read(
            &item,
            Path::new("."),
            SourceKind::U32,
            2,
            Path::new("x.xdmf2"),
        )
        .unwrap();
        match result {
            Values::U64(v) => assert_eq!(v.into_owned(), vec![1, 2]),
            _ => panic!("expected U64"),
        }
    }

    #[test]
    fn reads_xi_include() {
        use crate::xdmf_elements::data_item::NumberType;
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let data_file = tmp_dir.path().join("data.txt");
        std::fs::write(&data_file, "5.0 6.0\n").unwrap();

        let item = DataItem {
            dimensions: Some(Dimensions(vec![2])),
            include: Some(XInclude::new("data.txt", true)),
            number_type: Some(NumberType::Float),
            precision: Some(8),
            text: None,
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
            Values::F64(v) => assert_eq!(v.into_owned(), vec![5.0, 6.0]),
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn rejects_unparsable_token() {
        use crate::xdmf_elements::data_item::NumberType;
        let item = DataItem {
            dimensions: Some(Dimensions(vec![2])),
            ..item_with_text("1.0 not_a_number", NumberType::Float, 8)
        };
        let err = read(
            &item,
            Path::new("."),
            SourceKind::F64,
            2,
            Path::new("x.xdmf2"),
        )
        .unwrap_err();
        std::assert_matches!(err, Error::InvalidFile { reason, .. } if reason.contains("not_a_number"));
    }

    #[test]
    fn rejects_length_mismatch() {
        use crate::xdmf_elements::data_item::NumberType;
        let item = DataItem {
            dimensions: Some(Dimensions(vec![3])),
            ..item_with_text("1.0 2.0", NumberType::Float, 8)
        };
        let err = read(
            &item,
            Path::new("."),
            SourceKind::F64,
            3,
            Path::new("x.xdmf2"),
        )
        .unwrap_err();
        std::assert_matches!(err, Error::InvalidFile { reason, .. } if reason.contains("promised 3") && reason.contains("has 2"));
    }
}

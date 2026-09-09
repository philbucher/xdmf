//! Reading a `Format="XML"` `DataItem`'s heavy data: whitespace-separated numbers, either inline
//! in the document or in the text file an `<xi:include>` names.
//!
//! A text file does not record its element type, so unlike an HDF5 dataset it is read as whatever
//! the light data's `NumberType`/`Precision` declares. [`element_type`] maps that pair onto a Rust
//! type, and [`binary_reader`](super::binary_reader) shares it: a binary file records no more
//! about itself than a text one does.

use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use crate::{
    Error, Result, Values, error::io_ctx, reader::sealed::SealedValueType,
    xdmf_elements::data_item::NumberType,
};

/// Bytes read at a time out of an included file.
const CHUNK_BYTES: usize = 8 * 1024;

/// Where one ascii item's numbers are: in the document itself, or in a file beside it.
///
/// [`selection`](super::selection) decides which, since that is light-data parsing. This module
/// only reads.
pub(super) enum Source<'i> {
    Inline(&'i str),
    File(PathBuf),
}

impl Source<'_> {
    /// An upper bound on how many numbers this source can hold, from its size alone: every value
    /// but the last takes at least a character and a separator.
    ///
    /// [`parse_into`] sizes its buffer by this as well as by `Dimensions`, which is only a claim
    /// the document makes and is checked once the values are in. On its own it would let a broken
    /// document reserve an array the file could never fill.
    /// [`binary_reader`](super::binary_reader) takes its count from the file's length for the same
    /// reason.
    fn max_values(&self) -> Result<usize> {
        let bytes = match self {
            Self::Inline(text) => text.len(),
            Self::File(path) => usize::try_from(
                std::fs::metadata(path)
                    .map_err(io_ctx("reading ascii data file size", path))?
                    .len(),
            )
            .unwrap_or(usize::MAX),
        };

        Ok(bytes / 2 + 1)
    }
}

/// All the numbers of one ascii item, as the type the light data declares.
pub(super) fn read(
    source: &Source<'_>,
    number_type: NumberType,
    precision: u8,
    expected_len: Option<usize>,
) -> Result<Values<'static>> {
    let values = match element_type(number_type, precision)? {
        ElementType::F64 => Values::from(parse_all::<f64>(source, expected_len)?),
        ElementType::F32 => Values::from(parse_all::<f32>(source, expected_len)?),
        ElementType::I64 => Values::from(parse_all::<i64>(source, expected_len)?),
        ElementType::I32 => Values::from(parse_all::<i32>(source, expected_len)?),
        ElementType::U64 => Values::from(parse_all::<u64>(source, expected_len)?),
        ElementType::U32 => Values::from(parse_all::<u32>(source, expected_len)?),
    };

    check_length(values.len(), expected_len)?;

    Ok(values)
}

/// The same read, straight into `into` where the declared element type is already `T`, reporting
/// whether it was -- see [`hdf5_reader::read_exact_into`](super::hdf5_reader::read_exact_into),
/// whose contract this mirrors. `false` leaves `into` untouched.
pub(super) fn read_exact_into<T: SealedValueType>(
    source: &Source<'_>,
    number_type: NumberType,
    precision: u8,
    expected_len: Option<usize>,
    into: &mut Vec<T>,
) -> Result<bool> {
    if number_type != T::NUMBER_TYPE || precision != T::PRECISION {
        return Ok(false);
    }

    into.clear();
    parse_into(source, expected_len, into)?;
    check_length(into.len(), expected_len)?;

    Ok(true)
}

/// The element type a `NumberType`/`Precision` pair names. One place, so the ascii and binary
/// storages accept the same pairs and turn down the rest with the same message.
#[derive(Debug)]
pub(super) enum ElementType {
    F64,
    F32,
    I64,
    I32,
    U64,
    U32,
}

pub(super) fn element_type(number_type: NumberType, precision: u8) -> Result<ElementType> {
    match (number_type, precision) {
        (NumberType::Float, 8) => Ok(ElementType::F64),
        (NumberType::Float, 4) => Ok(ElementType::F32),
        (NumberType::Int, 8) => Ok(ElementType::I64),
        (NumberType::Int, 4) => Ok(ElementType::I32),
        (NumberType::UInt, 8) => Ok(ElementType::U64),
        (NumberType::UInt, 4) => Ok(ElementType::U32),
        _ => Err(Error::Unsupported {
            reason: format!(
                "a DataItem with NumberType=\"{number_type:?}\" Precision=\"{precision}\" is not \
                 supported, only Float/Int/UInt at 4 or 8 bytes are"
            ),
        }),
    }
}

/// Reject a heavy-data array holding a different number of values than the light data says it
/// does: a truncated file, or a document edited without it.
fn check_length(found: usize, expected: Option<usize>) -> Result<()> {
    match expected {
        Some(expected) if expected != found => Err(Error::InvalidDocument {
            reason: format!(
                "a DataItem's Dimensions say it holds {expected} values, but its heavy data has \
                 {found}"
            ),
        }),
        _ => Ok(()),
    }
}

fn parse_all<T: SealedValueType>(
    source: &Source<'_>,
    expected_len: Option<usize>,
) -> Result<Vec<T>> {
    let mut values = Vec::new();
    parse_into(source, expected_len, &mut values)?;

    Ok(values)
}

/// Parse every value of `source` into `into`, which the caller has cleared.
///
/// The buffer takes its size up front where the document states one, so a read costs the single
/// allocation the values need. Growing it by doubling instead would end up on as much again.
fn parse_into<T: SealedValueType>(
    source: &Source<'_>,
    expected_len: Option<usize>,
    into: &mut Vec<T>,
) -> Result<()> {
    if let Some(expected) = expected_len {
        into.reserve(expected.min(source.max_values()?));
    }

    match source {
        Source::Inline(text) => {
            for token in text.split_ascii_whitespace() {
                into.push(parse_token(token.as_bytes())?);
            }

            Ok(())
        }
        Source::File(path) => parse_file_into(path, into),
    }
}

/// Parse an included file's numbers, a fixed buffer at a time, carrying the token that straddles
/// two reads across the boundary.
///
/// Reading the file into one `String` first would hold a second array the size of the text, which
/// is wider than the numbers it holds. A read into the caller's buffer would then cost more than
/// the buffer it fills, which no other storage does.
///
/// Bytes rather than `str`, since a chunk boundary can fall inside a multi-byte character. Only a
/// whole token goes back to text, and one that is not valid UTF-8 fails to parse anyway.
fn parse_file_into<T: SealedValueType>(path: &Path, into: &mut Vec<T>) -> Result<()> {
    let mut reader =
        BufReader::new(File::open(path).map_err(io_ctx("opening ascii data file", path))?);
    let mut chunk = [0_u8; CHUNK_BYTES];
    let mut token: Vec<u8> = Vec::new();

    loop {
        let read = reader
            .read(&mut chunk)
            .map_err(io_ctx("reading ascii data file", path))?;
        if read == 0 {
            break;
        }

        let mut fields = chunk[..read].split(u8::is_ascii_whitespace);

        // the first field continues whatever the previous chunk ended part-way through
        if let Some(first) = fields.next() {
            token.extend_from_slice(first);
        }

        // a separator precedes every field after that, so the token held is complete
        for field in fields {
            push_token(&token, into)?;
            token.clear();
            token.extend_from_slice(field);
        }
    }

    push_token(&token, into)
}

/// Push one token, unless a run of whitespace left it empty.
fn push_token<T: SealedValueType>(token: &[u8], into: &mut Vec<T>) -> Result<()> {
    if token.is_empty() {
        return Ok(());
    }

    into.push(parse_token(token)?);

    Ok(())
}

/// `T::from_str` is the whole check. It rejects a fractional or negative token for an integer
/// type, and one past that type's range, so a value the file cannot hold is reported rather than
/// wrapped.
fn parse_token<T: SealedValueType>(token: &[u8]) -> Result<T> {
    str::from_utf8(token)
        .ok()
        .and_then(|token| token.parse::<T>().ok())
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!(
                "'{}' is not a valid {} value",
                String::from_utf8_lossy(token),
                std::any::type_name::<T>()
            ),
        })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use temp_dir::TempDir;

    use super::*;

    fn write_text(dir: &TempDir, name: &str, text: &str) -> PathBuf {
        let path = dir.path().join(name);
        File::create(&path)
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();

        path
    }

    #[test]
    fn reads_the_declared_element_type() {
        let source = Source::Inline("1.5e0 -2e0 3.25e0");
        let values = read(&source, NumberType::Float, 8, Some(3)).unwrap();
        std::assert_matches!(&values, Values::F64(v) if **v == [1.5, -2.0, 3.25]);

        let source = Source::Inline("1 2 3");
        let values = read(&source, NumberType::UInt, 4, None).unwrap();
        std::assert_matches!(&values, Values::U32(v) if **v == [1, 2, 3]);

        let source = Source::Inline("-1 2");
        let values = read(&source, NumberType::Int, 8, None).unwrap();
        std::assert_matches!(&values, Values::I64(v) if **v == [-1, 2]);
    }

    #[test]
    fn fills_a_buffer_only_for_its_own_type() {
        let source = Source::Inline("1e0 2e0");

        let mut into: Vec<f64> = Vec::new();
        assert!(read_exact_into(&source, NumberType::Float, 8, Some(2), &mut into).unwrap());
        assert_eq!(into, [1.0, 2.0]);

        // an f32 array is not an f64 one, so the caller is told to convert instead
        let mut into: Vec<f64> = vec![9.0];
        assert!(!read_exact_into(&source, NumberType::Float, 4, None, &mut into).unwrap());
        assert_eq!(into, [9.0], "the buffer must be left alone");
    }

    #[test]
    fn a_token_that_is_not_a_number_of_that_type_is_rejected() {
        // a negative value in an unsigned array, which `u32::from_str` is what rejects
        std::assert_matches!(
            read(&Source::Inline("0 -1"), NumberType::UInt, 4, None).unwrap_err(),
            Error::InvalidDocument { reason } if reason.contains("'-1' is not a valid u32 value")
        );

        std::assert_matches!(
            read(&Source::Inline("1 nonsense"), NumberType::Float, 8, None).unwrap_err(),
            Error::InvalidDocument { reason } if reason.contains("'nonsense' is not a valid f64")
        );

        // and one past the width the file declares, rather than wrapping into it
        std::assert_matches!(
            read(&Source::Inline("4294967296"), NumberType::UInt, 4, None).unwrap_err(),
            Error::InvalidDocument { reason } if reason.contains("'4294967296' is not a valid u32")
        );
    }

    #[test]
    fn a_length_disagreeing_with_the_dimensions_is_rejected() {
        std::assert_matches!(
            read(&Source::Inline("1 2"), NumberType::Int, 4, Some(3)).unwrap_err(),
            Error::InvalidDocument { reason }
                if reason.contains("Dimensions say it holds 3 values") && reason.contains("has 2")
        );

        let mut into: Vec<i32> = Vec::new();
        std::assert_matches!(
            read_exact_into(&Source::Inline("1 2"), NumberType::Int, 4, Some(3), &mut into)
                .unwrap_err(),
            Error::InvalidDocument { reason } if reason.contains("Dimensions say it holds 3 values")
        );
    }

    #[test]
    fn an_element_type_this_reader_has_no_variant_for_is_rejected() {
        for (number_type, precision) in [
            (NumberType::Char, 1),
            (NumberType::UChar, 1),
            (NumberType::Float, 2),
            (NumberType::Int, 3),
        ] {
            std::assert_matches!(
                element_type(number_type, precision).unwrap_err(),
                Error::Unsupported { reason } if reason.contains("is not supported"),
                "{number_type:?}/{precision}"
            );
        }
    }

    #[test]
    fn newlines_and_runs_of_spaces_separate_values_like_a_single_space() {
        // the writer ends every file it writes with a newline, and one array per line is a shape
        // a hand-written document may well come in
        let text = "  1 2\n3\t4\r\n 5  \n";
        let dir = TempDir::new().unwrap();

        for source in [
            Source::Inline(text),
            Source::File(write_text(&dir, "spaced.txt", text)),
        ] {
            let values = read(&source, NumberType::Int, 4, Some(5)).unwrap();
            std::assert_matches!(&values, Values::I32(v) if **v == [1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn an_empty_item_reads_as_no_values() {
        let dir = TempDir::new().unwrap();

        for source in [
            Source::Inline("   \n"),
            Source::File(write_text(&dir, "empty.txt", "   \n")),
            Source::File(write_text(&dir, "nothing.txt", "")),
        ] {
            let values = read(&source, NumberType::Float, 8, Some(0)).unwrap();
            assert_eq!(values.len(), 0);
        }
    }

    #[test]
    fn a_missing_included_file_is_reported_with_its_path() {
        let source = Source::File(PathBuf::from("no_such_dir/no_such_file.txt"));

        std::assert_matches!(
            read(&source, NumberType::Float, 8, None).unwrap_err(),
            Error::Io { operation, path, .. }
                if operation == "opening ascii data file"
                    && path.ends_with("no_such_file.txt")
        );
    }

    // The file is streamed rather than read whole, so a number can land across the boundary
    // between two reads -- the case a whole-file parse could never get wrong, and the one this
    // has to.
    #[test]
    fn a_value_split_across_two_reads_is_parsed_as_one() {
        // written so that a value straddles every boundary in turn: each is 21 bytes with its
        // separator, and 8192 is not a multiple of 21
        let count = 3 * CHUNK_BYTES / 21 + 5;
        let expected: Vec<i64> = (0..count as i64)
            .map(|i| 1_000_000_000_000_000 + i)
            .collect();
        let text = expected
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(" ");

        let dir = TempDir::new().unwrap();
        let source = Source::File(write_text(&dir, "long.txt", &text));

        let mut into: Vec<i64> = Vec::new();
        assert!(read_exact_into(&source, NumberType::Int, 8, Some(count), &mut into).unwrap());
        assert_eq!(into, expected);
    }
}

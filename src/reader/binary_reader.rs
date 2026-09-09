//! Reading a `Format="Binary"` `DataItem`'s heavy data: one file of raw, packed numbers.
//!
//! Like a text file and unlike an HDF5 dataset, a binary file records neither its element type nor
//! its byte order, so both come from the light data: the type through
//! [`super::ascii_reader::element_type`], shared so the two storages accept the same
//! `NumberType`/`Precision` pairs, and the order from `Endian`. XDMF defaults that to `Native`,
//! which can only mean the byte order of whichever machine wrote the file.

use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use super::ascii_reader::{ElementType, element_type};
use crate::{
    Error, Result, Values,
    error::io_ctx,
    reader::sealed::SealedValueType,
    xdmf_elements::data_item::{Endian, NumberType},
};

/// Bytes read at a time, a multiple of both element widths so no value straddles two reads.
///
/// A fixed scratch rather than the whole file, so that reading a large array holds nothing of the
/// array's own size besides the caller's buffer.
const CHUNK_BYTES: usize = 8 * 1024;

/// The whole file at `path`, as the type the light data declares.
pub(super) fn read(
    path: &Path,
    number_type: NumberType,
    precision: u8,
    endian: Endian,
    expected_len: Option<usize>,
) -> Result<Values<'static>> {
    let element_type = element_type(number_type, precision)?;
    let count = element_count(path, precision, expected_len)?;

    Ok(match element_type {
        ElementType::F64 => Values::from(read_all::<f64>(path, count, endian)?),
        ElementType::F32 => Values::from(read_all::<f32>(path, count, endian)?),
        ElementType::I64 => Values::from(read_all::<i64>(path, count, endian)?),
        ElementType::I32 => Values::from(read_all::<i32>(path, count, endian)?),
        ElementType::U64 => Values::from(read_all::<u64>(path, count, endian)?),
        ElementType::U32 => Values::from(read_all::<u32>(path, count, endian)?),
    })
}

/// The same read, straight into `into` where the declared element type is already `T`, reporting
/// whether it was -- see [`hdf5_reader::read_exact_into`](super::hdf5_reader::read_exact_into),
/// whose contract this mirrors. `false` leaves `into` untouched.
pub(super) fn read_exact_into<T: SealedValueType>(
    path: &Path,
    number_type: NumberType,
    precision: u8,
    endian: Endian,
    expected_len: Option<usize>,
    into: &mut Vec<T>,
) -> Result<bool> {
    if number_type != T::NUMBER_TYPE || precision != T::PRECISION {
        return Ok(false);
    }

    let count = element_count(path, precision, expected_len)?;

    into.clear();
    read_into(path, count, endian, into)?;

    Ok(true)
}

/// How many values the file holds, from its own length, and whether the light data agrees.
///
/// The count comes from the file rather than from `Dimensions`. A `Dimensions` naming more values
/// than the file has would otherwise show up as a short read, reported as an I/O error instead of
/// as the document disagreeing with its own heavy data.
fn element_count(path: &Path, precision: u8, expected_len: Option<usize>) -> Result<usize> {
    let width = usize::from(precision);
    let len = usize::try_from(
        std::fs::metadata(path)
            .map_err(io_ctx("reading binary data file size", path))?
            .len(),
    )
    .map_err(|_source| Error::InvalidDocument {
        reason: format!("binary data file '{}' is too large to read", path.display()),
    })?;

    if width == 0 || !len.is_multiple_of(width) {
        return Err(Error::InvalidDocument {
            reason: format!(
                "binary data file '{}' is {len} bytes, which is not a whole number of \
                 {width}-byte values",
                path.display()
            ),
        });
    }

    let count = len / width;

    match expected_len {
        Some(expected) if expected != count => Err(Error::InvalidDocument {
            reason: format!(
                "a DataItem's Dimensions say it holds {expected} values, but its heavy data file \
                 '{}' has {count}",
                path.display()
            ),
        }),
        _ => Ok(count),
    }
}

fn read_all<T: SealedValueType>(path: &Path, count: usize, endian: Endian) -> Result<Vec<T>> {
    let mut values = Vec::new();
    read_into(path, count, endian, &mut values)?;

    Ok(values)
}

/// Decode `count` values through a fixed scratch into `into`, which the caller has cleared.
fn read_into<T: SealedValueType>(
    path: &Path,
    count: usize,
    endian: Endian,
    into: &mut Vec<T>,
) -> Result<()> {
    let width = usize::from(T::PRECISION);
    let mut reader = BufReader::new(File::open(path).map_err(io_ctx("opening binary data", path))?);
    let mut chunk = [0_u8; CHUNK_BYTES];

    into.reserve(count);

    let mut remaining = count;
    while remaining > 0 {
        let elements = remaining.min(CHUNK_BYTES / width);
        let bytes = &mut chunk[..elements * width];

        reader
            .read_exact(bytes)
            .map_err(io_ctx("reading binary data", path))?;

        for value in bytes.chunks_exact(width) {
            // `chunks_exact` hands over `width` bytes, which is `T::PRECISION`
            into.push(T::from_bytes(value, endian).ok_or(Error::Internal(
                "a binary value was decoded from the wrong number of bytes",
            ))?);
        }

        remaining -= elements;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use temp_dir::TempDir;

    use super::*;

    fn write_bytes(dir: &TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        File::create(&path).unwrap().write_all(bytes).unwrap();

        path
    }

    #[test]
    fn reads_the_declared_element_type() {
        let dir = TempDir::new().unwrap();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.5_f64.to_le_bytes());
        bytes.extend_from_slice(&(-2.0_f64).to_le_bytes());
        let path = write_bytes(&dir, "f64.bin", &bytes);

        let values = read(&path, NumberType::Float, 8, Endian::Little, Some(2)).unwrap();
        std::assert_matches!(&values, Values::F64(v) if **v == [1.5, -2.0]);

        let path = write_bytes(&dir, "u32.bin", &[1, 0, 0, 0, 2, 0, 0, 0]);
        let values = read(&path, NumberType::UInt, 4, Endian::Little, None).unwrap();
        std::assert_matches!(&values, Values::U32(v) if **v == [1, 2]);
    }

    #[test]
    fn honours_the_declared_byte_order() {
        let dir = TempDir::new().unwrap();
        let path = write_bytes(&dir, "be.bin", &0x0102_0304_u32.to_be_bytes());

        let values = read(&path, NumberType::UInt, 4, Endian::Big, Some(1)).unwrap();
        std::assert_matches!(&values, Values::U32(v) if **v == [0x0102_0304]);

        // the same bytes read the other way round are a different number, which is why the
        // attribute is honoured rather than assumed
        let values = read(&path, NumberType::UInt, 4, Endian::Little, Some(1)).unwrap();
        std::assert_matches!(&values, Values::U32(v) if **v == [0x0403_0201]);
    }

    #[test]
    fn fills_a_buffer_only_for_its_own_type() {
        let dir = TempDir::new().unwrap();
        let path = write_bytes(&dir, "i32.bin", &[7, 0, 0, 0]);

        let mut into: Vec<i32> = Vec::new();
        assert!(
            read_exact_into(
                &path,
                NumberType::Int,
                4,
                Endian::Little,
                Some(1),
                &mut into
            )
            .unwrap()
        );
        assert_eq!(into, [7]);

        let mut into: Vec<i64> = vec![9];
        assert!(
            !read_exact_into(&path, NumberType::Int, 4, Endian::Little, None, &mut into).unwrap()
        );
        assert_eq!(into, [9], "the buffer must be left alone");
    }

    #[test]
    fn a_file_that_is_not_a_whole_number_of_values_is_rejected() {
        let dir = TempDir::new().unwrap();
        let path = write_bytes(&dir, "ragged.bin", &[1, 2, 3, 4, 5]);

        std::assert_matches!(
            read(&path, NumberType::UInt, 4, Endian::Little, None).unwrap_err(),
            Error::InvalidDocument { reason }
                if reason.contains("5 bytes") && reason.contains("4-byte values")
        );
    }

    #[test]
    fn a_file_shorter_than_the_dimensions_is_rejected_as_a_disagreement() {
        // reported against the light data rather than as a short read, so the message names both
        // counts and the reader can see which of the two is wrong
        let dir = TempDir::new().unwrap();
        let path = write_bytes(&dir, "short.bin", &[1, 0, 0, 0]);

        std::assert_matches!(
            read(&path, NumberType::UInt, 4, Endian::Little, Some(3)).unwrap_err(),
            Error::InvalidDocument { reason }
                if reason.contains("Dimensions say it holds 3 values") && reason.contains("has 1")
        );
    }

    #[test]
    fn a_missing_file_is_reported_with_its_path() {
        let path = Path::new("no_such_dir/no_such_file.bin");

        std::assert_matches!(
            read(path, NumberType::Float, 8, Endian::Little, None).unwrap_err(),
            Error::Io { operation, path, .. }
                if operation == "reading binary data file size"
                    && path.ends_with("no_such_file.bin")
        );
    }

    // The scratch buffer is what makes a large read allocate nothing but the caller's own array.
    // An array several chunks long is what catches a mis-sized or mis-advanced one.
    #[test]
    fn an_array_longer_than_the_scratch_buffer_reads_back_whole() {
        let count = 5 * CHUNK_BYTES / size_of::<f64>() + 3;
        let expected: Vec<f64> = (0..count).map(|i| i as f64 * 0.5).collect();

        let mut bytes = Vec::with_capacity(count * size_of::<f64>());
        for value in &expected {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let dir = TempDir::new().unwrap();
        let path = write_bytes(&dir, "long.bin", &bytes);

        let mut into: Vec<f64> = Vec::new();
        assert!(
            read_exact_into(
                &path,
                NumberType::Float,
                8,
                Endian::Little,
                Some(count),
                &mut into,
            )
            .unwrap()
        );

        float_cmp::assert_approx_eq!(&[f64], &into, &expected);
    }
}

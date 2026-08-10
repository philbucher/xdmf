//! Shared value-parsing, dispatch, and widening logic used by every format-specific backend
//! (`ascii_reader`/`binary_reader`/`hdf5_reader`).
//!
//! Deliberately free functions rather than a `dyn DataReader` trait: unlike the writer, where one
//! backend is chosen once per `TimeSeriesWriter` and held for its lifetime, every single
//! `DataItem` carries its own `Format` (mixed-format documents are legal XDMF), so the format is
//! already known at each individual call site — a trait object would buy nothing here.

use std::path::Path;

#[cfg(feature = "hdf5")]
use super::hdf5_reader;
use super::{ascii_reader, binary_reader};
use crate::{
    Error, Result, Values,
    error::io_ctx,
    values::ValuesMut,
    xdmf_elements::data_item::{DataItem, Format, NumberType},
};

/// The numeric shape a `DataItem`'s heavy data is actually stored as, derived from its
/// `NumberType`/`Precision`. Distinct from `ValueKind` (the reader's public field on `DataInfo`):
/// this additionally distinguishes 4-byte vs. 8-byte integers, since a backend needs that to know
/// how many bytes to read, even though both widen to the same `Values::U64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceKind {
    F32,
    F64,
    U32,
    U64,
}

impl SourceKind {
    pub(crate) fn from_data_item(item: &DataItem, xdmf_path: &Path) -> Result<Self> {
        let number_type = item.number_type.unwrap_or_default();
        let precision = item.precision.unwrap_or(4);
        match (number_type, precision) {
            (NumberType::Float, 4) => Ok(Self::F32),
            (NumberType::Float, 8) => Ok(Self::F64),
            (NumberType::UInt, 4) => Ok(Self::U32),
            (NumberType::UInt, 8) => Ok(Self::U64),
            (NumberType::Float | NumberType::UInt, other) => Err(Error::InvalidFile {
                path: xdmf_path.to_path_buf(),
                reason: format!(
                    "unsupported Precision=\"{other}\" for NumberType=\"{number_type:?}\""
                ),
            }),
            (NumberType::Int | NumberType::Char | NumberType::UChar, _) => {
                Err(Error::Unsupported {
                    reason: format!(
                        "DataItem NumberType=\"{number_type:?}\" is not supported, only \"Float\" and \"UInt\" are"
                    ),
                })
            }
        }
    }
}

/// Read the heavy data a `DataItem` describes, dispatching on its `Format`. `item` must already
/// have any `Reference="XML"` resolved (see `light_data::resolve_reference`) — this function does
/// not resolve references itself.
///
/// Returns a plain [`Values`] (owned, i.e. `Cow::Owned` under the hood) rather than a
/// reader-specific type: a backend widens 4-byte integers to `U64` (and never produces a
/// 4-byte-integer variant of its own), so the parsed data already has exactly `Values`' shape —
/// three cases, same element types — and reusing it avoids a duplicate enum.
pub(crate) fn read_data_item(
    item: &DataItem,
    base_dir: &Path,
    xdmf_path: &Path,
) -> Result<Values<'static>> {
    let kind = SourceKind::from_data_item(item, xdmf_path)?;
    let expected_len = item
        .dimensions
        .as_ref()
        .map(|d| d.0.iter().product())
        .ok_or_else(|| Error::InvalidFile {
            path: xdmf_path.to_path_buf(),
            reason: "DataItem has no Dimensions".to_string(),
        })?;

    match item.format.unwrap_or_default() {
        Format::XML => ascii_reader::read(item, base_dir, kind, expected_len, xdmf_path),
        Format::Binary => binary_reader::read(item, base_dir, kind, expected_len, xdmf_path),
        Format::HDF => {
            #[cfg(feature = "hdf5")]
            {
                hdf5_reader::read(item, base_dir, kind, expected_len, xdmf_path)
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(Error::InvalidConfiguration {
                    reason:
                        "DataItem Format=\"HDF\" requires the 'hdf5' feature, which is not enabled"
                            .to_string(),
                })
            }
        }
    }
}

/// Parse whitespace/byte-separated tokens is done per-backend; this just checks the result has
/// the length `Dimensions` promised, which every backend needs after parsing.
pub(crate) fn check_len(actual: usize, expected: usize, xdmf_path: &Path) -> Result<()> {
    if actual != expected {
        return Err(Error::InvalidFile {
            path: xdmf_path.to_path_buf(),
            reason: format!(
                "DataItem's Dimensions promised {expected} values, but its heavy data has {actual}"
            ),
        });
    }
    Ok(())
}

/// Read the raw text a `Format="XML"` `DataItem` holds, whether inline or via `xi:include`.
pub(crate) fn resolve_text(item: &DataItem, base_dir: &Path, xdmf_path: &Path) -> Result<String> {
    if let Some(include) = &item.include {
        let path = base_dir.join(include.file_path());
        std::fs::read_to_string(&path).map_err(io_ctx("reading included data file", &path))
    } else if let Some(text) = &item.text {
        Ok(text.clone())
    } else {
        Err(Error::InvalidFile {
            path: xdmf_path.to_path_buf(),
            reason: "DataItem with Format=\"XML\" has neither inline text nor an xi:include child"
                .to_string(),
        })
    }
}

/// Widen/assign parsed heavy data into the caller's target buffer, rejecting anything that would
/// lose precision or change kind. Used only for attribute reads (`read_point_data`/
/// `read_cell_data`): points/connectivity have a fixed target type and are handled directly by
/// `light_data`/`topology` instead, since there is no caller-chosen `T` to widen against there.
pub(crate) fn assign(source: Values<'_>, into: ValuesMut<'_>) -> Result<()> {
    match (source, into) {
        (Values::F64(v), ValuesMut::F64(dst)) => *dst = v.into_owned(),
        (Values::F32(v), ValuesMut::F64(dst)) => {
            *dst = v.into_owned().into_iter().map(f64::from).collect();
        }
        (Values::F32(v), ValuesMut::F32(dst)) => *dst = v.into_owned(),
        (Values::U64(v), ValuesMut::U64(dst)) => *dst = v.into_owned(),
        (Values::F64(_), ValuesMut::F32(_)) => {
            return Err(Error::InvalidData {
                reason: "cannot read a Precision=\"8\" float DataItem into a f32 buffer without losing precision".to_string(),
            });
        }
        (Values::F64(_) | Values::F32(_), ValuesMut::U64(_)) => {
            return Err(Error::InvalidData {
                reason: "requested integer data, but the DataItem holds floating-point data"
                    .to_string(),
            });
        }
        (Values::U64(_), ValuesMut::F64(_) | ValuesMut::F32(_)) => {
            return Err(Error::InvalidData {
                reason: "requested floating-point data, but the DataItem holds integer data"
                    .to_string(),
            });
        }
    }
    Ok(())
}

/// Parsed heavy data always holding `f64`, for points (which have no caller-chosen type — a
/// mismatch here can only be the file's fault, hence `InvalidFile` rather than `InvalidData`).
pub(crate) fn into_f64(source: Values<'_>, xdmf_path: &Path) -> Result<Vec<f64>> {
    match source {
        Values::F64(v) => Ok(v.into_owned()),
        Values::F32(v) => Ok(v.into_owned().into_iter().map(f64::from).collect()),
        Values::U64(_) => Err(Error::InvalidFile {
            path: xdmf_path.to_path_buf(),
            reason: "Geometry DataItem must hold floating-point data".to_string(),
        }),
    }
}

/// Parsed heavy data always holding `u64`, for connectivity (see `into_f64` for why this is
/// `InvalidFile` rather than `InvalidData`).
pub(crate) fn into_u64(source: Values<'_>, xdmf_path: &Path) -> Result<Vec<u64>> {
    match source {
        Values::U64(v) => Ok(v.into_owned()),
        Values::F64(_) | Values::F32(_) => Err(Error::InvalidFile {
            path: xdmf_path.to_path_buf(),
            reason: "Topology DataItem must hold integer data".to_string(),
        }),
    }
}

//! Evaluating `ItemType::HyperSlab`/`ItemType::Coordinates` selections, and the general
//! `DataItem` -> [`Values`] dispatcher every other reader module reads heavy data through.

use std::path::PathBuf;

use super::{hdf5_reader, light_data, light_data::Document};
use crate::{
    Error, Result, Values,
    reader::sealed::SealedValueType,
    xdmf_elements::{
        Domain,
        data_item::{DataContent, DataItem, Format},
    },
};

/// Which positions of a source array one submesh holds: its cells or points out of the mesh's, or
/// its share of a per-step field. Both selector shapes this crate's writer emits (`HyperSlab`'s
/// `<start> 1 <count>` and `Coordinates`' explicit index list) collapse to this.
#[derive(Debug, Clone)]
pub(super) enum Membership {
    Contiguous { start: usize, len: usize },
    Explicit(Vec<usize>),
}

impl Membership {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Contiguous { len, .. } => *len,
            Self::Explicit(indices) => indices.len(),
        }
    }

    /// The source position the entry at `local` sits at, or `None` beyond the membership's end.
    pub(super) fn get(&self, local: usize) -> Option<usize> {
        match self {
            Self::Contiguous { start, len } => (local < *len).then(|| start + local),
            Self::Explicit(indices) => indices.get(local).copied(),
        }
    }

    /// The source positions, in local order.
    pub(super) fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match self {
            Self::Contiguous { start, len } => Box::new(*start..*start + *len),
            Self::Explicit(indices) => Box::new(indices.iter().copied()),
        }
    }

    /// The values this membership picks out of a fully-read source array.
    ///
    /// The positions come from the file, so neither shape is trusted to stay inside the array it
    /// selects from.
    fn apply(&self, source: &Values<'_>) -> Result<Values<'static>> {
        let out_of_range = |position: usize| Error::InvalidDocument {
            reason: format!(
                "a selection names position {position} of an array of only {} values",
                source.len()
            ),
        };

        match self {
            Self::Contiguous { start, len } => {
                let end = start.checked_add(*len).ok_or(Error::Internal(
                    "a HyperSlab selector's span does not fit a usize",
                ))?;

                if end > source.len() {
                    return Err(out_of_range(end - 1));
                }

                Ok(slice_owned(source, *start, *len))
            }
            Self::Explicit(indices) => {
                if let Some(&position) = indices.iter().find(|&&index| index >= source.len()) {
                    return Err(out_of_range(position));
                }

                Ok(gather_owned(source, indices))
            }
        }
    }
}

/// Read one `DataItem`'s values, following a `Reference="XML"` indirection and evaluating a
/// `HyperSlab`/`Coordinates` selection, down to the `Format="HDF"` array that actually holds them.
pub(super) fn read_data_item(
    item: &DataItem,
    document: &Document,
    domain: &Domain,
) -> Result<Values<'static>> {
    if item.reference.is_some() {
        let target = light_data::resolve_reference(item, domain)?;
        return read_data_item(target, document, domain);
    }

    if item.item_type.is_some() {
        let (selector, source) = selection_parts(item)?;
        let membership = parse_selector(selector, document, domain)?;
        let source_values = read_data_item(source, document, domain)?;
        return membership.apply(&source_values);
    }

    read_heavy(item, document)
}

/// The same read, into a caller's buffer rather than into a fresh [`Values`].
///
/// Only the plain array a reference chain ends at can be filled in place, and only when the
/// dataset already holds `T`. Another element type, or a selection, goes through `convert`
/// instead.
pub(super) fn read_data_item_into<T, F>(
    item: &DataItem,
    document: &Document,
    domain: &Domain,
    into: &mut Vec<T>,
    convert: F,
) -> Result<()>
where
    T: SealedValueType,
    F: FnOnce(Values<'static>) -> Result<Vec<T>>,
{
    if item.reference.is_some() {
        let target = light_data::resolve_reference(item, domain)?;
        return read_data_item_into(target, document, domain, into, convert);
    }

    if item.item_type.is_none() && read_heavy_exact_into(item, document, into)? {
        return Ok(());
    }

    let values = read_data_item(item, document, domain)?;
    into.clear();
    into.extend(convert(values)?);

    Ok(())
}

/// One plain `DataItem`'s heavy data, whole.
fn read_heavy(item: &DataItem, document: &Document) -> Result<Values<'static>> {
    let (file_path, dataset_path) = heavy_data_path(item, document)?;

    hdf5_reader::read(&file_path, dataset_path, &document.files)
}

/// The same, into the caller's buffer. The `bool` reports whether the dataset already held `T`.
fn read_heavy_exact_into<T: SealedValueType>(
    item: &DataItem,
    document: &Document,
    into: &mut Vec<T>,
) -> Result<bool> {
    let (file_path, dataset_path) = heavy_data_path(item, document)?;

    hdf5_reader::read_exact_into(&file_path, dataset_path, &document.files, into)
}

/// Where one `DataItem`'s heavy data lives: the file, and the path inside it.
///
/// Light-data parsing rather than heavy-data reading, so it happens on this side of the boundary
/// and a `Format` this reader does not support is reported as such in either build.
fn heavy_data_path<'i>(item: &'i DataItem, document: &Document) -> Result<(PathBuf, &'i str)> {
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

    Ok((document.base_dir.join(file_part), dataset_path))
}

/// The `<selector, source>` pair a `HyperSlab`/`Coordinates` `DataItem` carries as its nested
/// items, in that order.
pub(super) fn selection_parts(item: &DataItem) -> Result<(&DataItem, &DataItem)> {
    let DataContent::Items(children) = &item.data else {
        return Err(Error::InvalidDocument {
            reason: "a selection DataItem has no nested items".to_string(),
        });
    };

    match children.as_slice() {
        [selector, source] => Ok((selector, source)),
        other => Err(Error::InvalidDocument {
            reason: format!(
                "a selection DataItem must have exactly 2 nested items, found {}",
                other.len()
            ),
        }),
    }
}

/// The membership a selector names: which positions of the source it picks. A `Geometry`'s
/// selector answers which mesh points a submesh holds without reading the whole-mesh source.
pub(super) fn parse_selector(
    selector: &DataItem,
    document: &Document,
    domain: &Domain,
) -> Result<Membership> {
    if selector.reference.is_some() {
        let target = light_data::resolve_reference(selector, domain)?;
        let indices = read_heavy(target, document)?;
        return Ok(Membership::Explicit(values_to_usize(&indices)?));
    }

    let DataContent::Raw(text) = &selector.data else {
        return Err(Error::InvalidDocument {
            reason: "a HyperSlab selector DataItem has no text content".to_string(),
        });
    };

    let numbers = text
        .split_whitespace()
        .map(|part| {
            part.parse::<i64>()
                .map_err(|_source| Error::InvalidDocument {
                    reason: format!("HyperSlab selector '{text}' is not three integers"),
                })
        })
        .collect::<Result<Vec<i64>>>()?;

    let [start, stride, count] = numbers.as_slice() else {
        return Err(Error::InvalidDocument {
            reason: format!("HyperSlab selector '{text}' must have exactly 3 numbers"),
        });
    };

    if *stride != 1 {
        return Err(Error::Unsupported {
            reason: format!("HyperSlab selector with stride {stride} != 1 is not supported"),
        });
    }

    let start = usize::try_from(*start).map_err(|_source| Error::InvalidDocument {
        reason: format!("HyperSlab selector '{text}' has a negative start"),
    })?;
    let len = usize::try_from(*count).map_err(|_source| Error::InvalidDocument {
        reason: format!("HyperSlab selector '{text}' has a negative count"),
    })?;

    Ok(Membership::Contiguous { start, len })
}

/// Convert an index array's values, small signed integers by construction, to source positions.
pub(super) fn values_to_usize(values: &Values<'_>) -> Result<Vec<usize>> {
    let to_usize = |value: i128| {
        usize::try_from(value).map_err(|_source| Error::InvalidDocument {
            reason: format!("index value {value} is negative or does not fit a usize"),
        })
    };

    match values {
        Values::F64(_) | Values::F32(_) => Err(Error::InvalidDocument {
            reason: "an index array holds floating-point values".to_string(),
        }),
        Values::I64(v) => v.iter().map(|&x| to_usize(i128::from(x))).collect(),
        Values::I32(v) => v.iter().map(|&x| to_usize(i128::from(x))).collect(),
        Values::U64(v) => v.iter().map(|&x| to_usize(i128::from(x))).collect(),
        Values::U32(v) => v.iter().map(|&x| to_usize(i128::from(x))).collect(),
    }
}

fn slice_owned(source: &Values<'_>, start: usize, len: usize) -> Values<'static> {
    let end = start + len;
    match source {
        Values::F64(v) => Values::from(v[start..end].to_vec()),
        Values::F32(v) => Values::from(v[start..end].to_vec()),
        Values::I64(v) => Values::from(v[start..end].to_vec()),
        Values::I32(v) => Values::from(v[start..end].to_vec()),
        Values::U64(v) => Values::from(v[start..end].to_vec()),
        Values::U32(v) => Values::from(v[start..end].to_vec()),
    }
}

fn gather_owned(source: &Values<'_>, indices: &[usize]) -> Values<'static> {
    match source {
        Values::F64(v) => Values::from(indices.iter().map(|&i| v[i]).collect::<Vec<_>>()),
        Values::F32(v) => Values::from(indices.iter().map(|&i| v[i]).collect::<Vec<_>>()),
        Values::I64(v) => Values::from(indices.iter().map(|&i| v[i]).collect::<Vec<_>>()),
        Values::I32(v) => Values::from(indices.iter().map(|&i| v[i]).collect::<Vec<_>>()),
        Values::U64(v) => Values::from(indices.iter().map(|&i| v[i]).collect::<Vec<_>>()),
        Values::U32(v) => Values::from(indices.iter().map(|&i| v[i]).collect::<Vec<_>>()),
    }
}

//! Evaluating `ItemType::HyperSlab`/`ItemType::Coordinates` selections, and the general
//! `DataItem` -> [`Values`] dispatcher every other reader module reads heavy data through.

use std::path::PathBuf;

use super::{ascii_reader, binary_reader, hdf5_reader, light_data, light_data::Document};
use crate::{
    Error, Result, Values,
    reader::sealed::SealedValueType,
    xdmf_elements::{
        Domain,
        data_item::{DataContent, DataItem, Endian, Format, NumberType},
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

    /// The source position the entry at `local` sits at, or `None` beyond the membership's end,
    /// which a document read out of a file can always turn out to be.
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
    /// selects from: a truncated or foreign document is reported rather than indexed past the end
    /// of.
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
/// dataset already holds `T`. Another element type, or a selection (which must evaluate its whole
/// source before it knows what to keep), goes through `convert` instead.
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
    match heavy_source(item, document)? {
        HeavySource::Hdf5 { file, dataset } => hdf5_reader::read(&file, dataset, &document.files),
        HeavySource::Ascii(source) => {
            let (number_type, precision) = declared_element_type(item)?;
            ascii_reader::read(&source, number_type, precision, declared_len(item)?)
        }
        HeavySource::Binary(path) => {
            let (number_type, precision) = declared_element_type(item)?;
            binary_reader::read(
                &path,
                number_type,
                precision,
                declared_endian(item),
                declared_len(item)?,
            )
        }
    }
}

/// The same, into the caller's buffer -- see [`hdf5_reader::read_exact_into`] for what the `bool`
/// reports.
fn read_heavy_exact_into<T: SealedValueType>(
    item: &DataItem,
    document: &Document,
    into: &mut Vec<T>,
) -> Result<bool> {
    match heavy_source(item, document)? {
        HeavySource::Hdf5 { file, dataset } => {
            hdf5_reader::read_exact_into(&file, dataset, &document.files, into)
        }
        HeavySource::Ascii(source) => {
            let (number_type, precision) = declared_element_type(item)?;
            ascii_reader::read_exact_into(
                &source,
                number_type,
                precision,
                declared_len(item)?,
                into,
            )
        }
        HeavySource::Binary(path) => {
            let (number_type, precision) = declared_element_type(item)?;
            binary_reader::read_exact_into(
                &path,
                number_type,
                precision,
                declared_endian(item),
                declared_len(item)?,
                into,
            )
        }
    }
}

/// Where one plain `DataItem`'s heavy data lives, per storage.
enum HeavySource<'i> {
    Hdf5 { file: PathBuf, dataset: &'i str },
    Ascii(ascii_reader::Source<'i>),
    Binary(PathBuf),
}

/// Which of those it is, and where.
///
/// Light-data parsing rather than heavy-data reading, so it happens on this side of the boundary.
/// That is what lets a build without the `hdf5` feature report `Format="HDF"` as unsupported
/// instead of as a missing file.
fn heavy_source<'i>(item: &'i DataItem, document: &Document) -> Result<HeavySource<'i>> {
    match item.format {
        Some(Format::HDF) => {
            let raw = raw_text(item, "HDF")?;
            let (file_part, dataset) =
                raw.split_once(':').ok_or_else(|| Error::InvalidDocument {
                    reason: format!(
                        "'{raw}' is not a valid HDF5 heavy-data path, expected 'file:path'"
                    ),
                })?;

            Ok(HeavySource::Hdf5 {
                file: document.base_dir.join(file_part),
                dataset,
            })
        }
        // the two ascii storages share a `Format` and differ in where the numbers sit: in the
        // document, or in a file it includes
        Some(Format::XML) => Ok(HeavySource::Ascii(match &item.data {
            DataContent::Include(include) => {
                ascii_reader::Source::File(document.base_dir.join(include.file_path()))
            }
            DataContent::Raw(text) => ascii_reader::Source::Inline(text),
            DataContent::Items(_) => {
                return Err(Error::InvalidDocument {
                    reason: "a Format=\"XML\" DataItem holds nested items rather than values"
                        .to_string(),
                });
            }
        })),
        Some(Format::Binary) => Ok(HeavySource::Binary(
            document.base_dir.join(raw_text(item, "Binary")?),
        )),
        None => Err(Error::InvalidDocument {
            reason: "a DataItem holding heavy data has no Format".to_string(),
        }),
    }
}

fn raw_text<'i>(item: &'i DataItem, format: &str) -> Result<&'i str> {
    let DataContent::Raw(raw) = &item.data else {
        return Err(Error::InvalidDocument {
            reason: format!("a Format=\"{format}\" DataItem has no path text"),
        });
    };

    Ok(raw.trim())
}

/// The element type an ascii or binary item declares. Neither file records one, so an item
/// stating neither half of the pair cannot be read at all.
fn declared_element_type(item: &DataItem) -> Result<(NumberType, u8)> {
    let number_type = item.number_type.ok_or_else(|| Error::InvalidDocument {
        reason: "a DataItem holding ascii or binary values has no NumberType".to_string(),
    })?;
    let precision = item.precision.ok_or_else(|| Error::InvalidDocument {
        reason: "a DataItem holding ascii or binary values has no Precision".to_string(),
    })?;

    Ok((number_type, precision))
}

/// XDMF defaults to the writing machine's byte order, which is all a file that does not say can
/// mean.
fn declared_endian(item: &DataItem) -> Endian {
    item.endian.unwrap_or(Endian::Native)
}

/// How many values the item says it holds, for the two storages whose files can disagree with it.
/// `None` for a foreign item stating no `Dimensions`, which then goes unchecked.
fn declared_len(item: &DataItem) -> Result<Option<usize>> {
    let Some(dimensions) = item.dimensions.as_ref().filter(|dims| !dims.0.is_empty()) else {
        return Ok(None);
    };

    let len = dimensions
        .0
        .iter()
        .try_fold(1_usize, |product, &dimension| {
            product.checked_mul(dimension)
        })
        .ok_or(Error::Internal(
            "a DataItem's Dimensions do not multiply into a usize",
        ))?;

    Ok(Some(len))
}

/// The `<selector, source>` pair a `HyperSlab`/`Coordinates` `DataItem` carries as its nested
/// items, in that order -- the shape `selection()` (`time_series_writer.rs`) writes.
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

/// The membership a selector names: which positions of the source it picks. Used to evaluate a
/// selection's values (see [`read_data_item`]) and, for a `Geometry`'s selector alone, to learn
/// which mesh points a submesh holds without reading the whole-mesh source at all.
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

/// Convert an index array's values (small signed integers by construction, see
/// `time_series_writer.rs`'s `index_values`) to source positions.
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

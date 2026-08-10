//! This module contains the core datastructure used to specify data storage in XDMF files.

use serde::{Deserialize, Serialize};

use super::dimensions::Dimensions;

/// Core datastructure to define how, where, and in which format data is stored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataItem {
    #[serde(rename = "@Name", skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    pub name: Option<String>,

    #[serde(rename = "@Dimensions", skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    pub dimensions: Option<Dimensions>,

    #[serde(rename = "@NumberType", skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    pub number_type: Option<NumberType>,

    #[serde(rename = "@Format", skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    pub format: Option<Format>,

    #[serde(rename = "@Precision", skip_serializing_if = "Option::is_none")]
    /// Precision of the data, in bits (e.g. 4 for f32, 8 for f64)
    pub precision: Option<u8>,

    #[serde(rename = "@Endian", skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    pub endian: Option<Endian>,

    // Deliberately two plain fields rather than `#[serde(flatten)]` over a `DataContent` enum:
    // that combination does not deserialize with quick-xml (only serializes). The split `rename`
    // on `include` is required: quick-xml's serializer needs the literal `xi:include` to emit the
    // namespace prefix, but its deserializer strips the prefix and reports the field as `include`.
    // A single shared name silently fails to deserialize.
    #[serde(rename = "$text", skip_serializing_if = "Option::is_none", default)]
    #[doc(hidden)]
    pub text: Option<String>,

    #[serde(
        rename(serialize = "xi:include", deserialize = "include"),
        skip_serializing_if = "Option::is_none",
        default
    )]
    #[doc(hidden)]
    pub include: Option<XInclude>,

    #[serde(rename = "@Reference", skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    pub reference: Option<String>,
}

impl Default for DataItem {
    fn default() -> Self {
        Self {
            name: None,
            dimensions: Some(Dimensions(vec![1])),
            number_type: Some(NumberType::default()),
            format: Some(Format::default()),
            precision: Some(4),
            endian: None,
            text: Some(String::new()),
            include: None,
            reference: None,
        }
    }
}

impl DataItem {
    /// Create a new data item that references another data item
    pub fn new_reference(source: &Self, source_path: &str) -> Self {
        Self {
            name: None,
            dimensions: None,
            number_type: None,
            format: None,
            precision: None,
            endian: None,
            text: Some(format!(
                "{}[@Name=\"{}\"]",
                source_path,
                source.name.clone().unwrap_or("MISSING".to_string())
            )),
            include: None,
            reference: Some("XML".to_string()),
        }
    }
}

/// Used to include data from an external file using `XInclude`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename = "xi:include")]
pub struct XInclude {
    #[serde(rename = "@href")]
    #[doc(hidden)]
    file_path: String,

    #[serde(rename = "@parse", skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    parse: Option<String>,
}

impl XInclude {
    /// Create a new `XInclude` instance
    pub fn new(file_path: impl ToString, include_as_text: bool) -> Self {
        Self {
            file_path: file_path.to_string(),
            parse: include_as_text.then(|| "text".to_string()), // xml is default
        }
    }

    /// The `href` path, relative to the `.xdmf` file.
    pub(crate) fn file_path(&self) -> &str {
        &self.file_path
    }
}

/// Specifies where (ascii) data is stored, either inline or in an external file.
///
/// This is an internal writer-side value, not the wire representation: [`DataItem`] itself
/// carries `text`/`include` as two plain fields, not a flattened enum (`#[serde(flatten)]` over
/// an enum does not deserialize with quick-xml), so backends return a `DataContent` and
/// [`DataContent::into_parts`] splits it into those two fields.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DataContent {
    /// Store the data as raw text
    Raw(String),

    /// Store the data in an external file and include it using [XInclude](https://www.w3.org/TR/xinclude/)
    Include(XInclude),
}

impl DataContent {
    /// Split into the `(text, include)` fields of a [`DataItem`].
    pub(crate) fn into_parts(self) -> (Option<String>, Option<XInclude>) {
        match self {
            Self::Raw(text) => (Some(text), None),
            Self::Include(include) => (None, Some(include)),
        }
    }
}

impl From<String> for DataContent {
    fn from(data: String) -> Self {
        Self::Raw(data)
    }
}

impl From<&str> for DataContent {
    fn from(data: &str) -> Self {
        Self::Raw(data.to_string())
    }
}

impl From<XInclude> for DataContent {
    fn from(include: XInclude) -> Self {
        Self::Include(include)
    }
}

/// Specifies the type of data stored, such as f64 or i32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum NumberType {
    #[default]
    #[doc(hidden)]
    Float,
    #[doc(hidden)]
    Int,
    #[doc(hidden)]
    UInt,
    #[doc(hidden)]
    Char,
    #[doc(hidden)]
    UChar,
}

/// The format in which the heavy data is stored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Format {
    #[default]
    #[doc(hidden)]
    XML,
    #[doc(hidden)]
    HDF,
    #[doc(hidden)]
    Binary,
}

impl Format {
    /// Specify the endianness of the dataitem. Little by default to ensure OS-agnostic reading and writing
    pub(crate) fn endian(self) -> Option<Endian> {
        matches!(self, Self::Binary).then_some(Endian::Little)
    }

    ///  Paraview's legacy Xdmf2 reader silently misreads 64-bit integers in binary format, thus using 32-bit
    pub(crate) fn uint_precision(self) -> u8 {
        if matches!(self, Self::Binary) { 4 } else { 8 }
    }
}

/// Byte order of externally stored binary data (only meaningful for [`Format::Binary`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Endian {
    #[doc(hidden)]
    Native,
    #[doc(hidden)]
    Big,
    #[default]
    #[doc(hidden)]
    Little,
}

#[cfg(test)]
mod tests {
    use quick_xml::{de::from_str, se::to_string};

    use super::*;

    #[derive(Serialize, Deserialize)]
    struct XmlRoot {
        #[serde(rename = "DataItem")]
        data_item: DataItem,
    }

    #[test]
    fn data_item_default() {
        let default_item = DataItem::default();
        assert!(default_item.name.is_none());
        assert_eq!(default_item.dimensions, Some(Dimensions(vec![1])));
        assert_eq!(default_item.number_type, Some(NumberType::Float));
        assert_eq!(default_item.format, Some(Format::XML));
        assert_eq!(default_item.precision, Some(4));
        assert!(default_item.endian.is_none());
        assert_eq!(default_item.text, Some(String::new()));
        assert!(default_item.include.is_none());
        assert!(default_item.reference.is_none());
    }

    #[test]
    fn number_type_default() {
        assert_eq!(NumberType::default(), NumberType::Float);
    }

    #[test]
    fn format_default() {
        assert_eq!(Format::default(), Format::XML);
    }

    #[test]
    fn format_endian() {
        assert_eq!(Format::XML.endian(), None);
        assert_eq!(Format::HDF.endian(), None);
        assert_eq!(Format::Binary.endian(), Some(Endian::Little));
    }

    #[test]
    fn format_uint_precision() {
        assert_eq!(Format::XML.uint_precision(), 8);
        assert_eq!(Format::HDF.uint_precision(), 8);
        assert_eq!(Format::Binary.uint_precision(), 4);
    }

    #[test]
    fn data_item_custom() {
        let custom_item = DataItem {
            name: Some("custom_data_item".to_string()),
            dimensions: Some(Dimensions(vec![2, 3])),
            number_type: Some(NumberType::Int),
            format: Some(Format::HDF),
            precision: Some(8),
            endian: None,
            text: Some("custom_data".to_string()),
            include: None,
            reference: None,
        };
        assert_eq!(custom_item.name, Some("custom_data_item".to_string()));
        assert_eq!(custom_item.dimensions, Some(Dimensions(vec![2, 3])));
        assert_eq!(custom_item.number_type, Some(NumberType::Int));
        assert_eq!(custom_item.format, Some(Format::HDF));
        assert_eq!(custom_item.precision, Some(8));
        assert_eq!(custom_item.text, Some("custom_data".to_string()));
        assert!(custom_item.reference.is_none());
    }

    #[test]
    fn data_item_reference() {
        let source_data_item = DataItem {
            name: Some("source_data_item".to_string()),
            ..Default::default()
        };

        let ref_item = DataItem::new_reference(&source_data_item, "/Xdmf/Domain/DataItem");

        assert!(ref_item.name.is_none());
        assert!(ref_item.dimensions.is_none());
        assert!(ref_item.number_type.is_none());
        assert!(ref_item.format.is_none());
        assert!(ref_item.precision.is_none());
        assert!(ref_item.endian.is_none());
        assert_eq!(
            ref_item.text,
            Some("/Xdmf/Domain/DataItem[@Name=\"source_data_item\"]".to_string())
        );
        assert!(ref_item.include.is_none());
        assert_eq!(ref_item.reference, Some("XML".to_string()));
    }

    #[test]
    fn data_item_serialize() {
        let data_item = DataItem {
            name: Some("custom_data_item".to_string()),
            dimensions: Some(Dimensions(vec![2, 3])),
            number_type: Some(NumberType::Int),
            format: Some(Format::HDF),
            precision: Some(8),
            endian: None,
            text: Some("custom_data".to_string()),
            include: None,
            reference: None,
        };

        let xml = to_string(&XmlRoot { data_item }).unwrap();
        pretty_assertions::assert_eq!(
            xml,
            "<XmlRoot>\
            <DataItem Name=\"custom_data_item\" Dimensions=\"2 3\" NumberType=\"Int\" Format=\"HDF\" Precision=\"8\">custom_data</DataItem>\
            </XmlRoot>"
        );

        let round_tripped: XmlRoot = from_str(&xml).unwrap();
        assert_eq!(
            round_tripped.data_item.text,
            Some("custom_data".to_string())
        );
        assert!(round_tripped.data_item.include.is_none());
        assert_eq!(
            round_tripped.data_item.name,
            Some("custom_data_item".to_string())
        );
        assert_eq!(
            round_tripped.data_item.dimensions,
            Some(Dimensions(vec![2, 3]))
        );
        assert_eq!(round_tripped.data_item.number_type, Some(NumberType::Int));
        assert_eq!(round_tripped.data_item.format, Some(Format::HDF));
        assert_eq!(round_tripped.data_item.precision, Some(8));
    }

    #[test]
    fn data_item_reference_serialize() {
        let source_data_item = DataItem {
            name: Some("source_data_item".to_string()),
            ..Default::default()
        };

        let ref_item = DataItem::new_reference(&source_data_item, "/Xdmf/Domain/DataItem");

        let xml = to_string(&XmlRoot {
            data_item: ref_item,
        })
        .unwrap();
        pretty_assertions::assert_eq!(
            xml,
            "<XmlRoot>\
            <DataItem Reference=\"XML\">/Xdmf/Domain/DataItem[@Name=\"source_data_item\"]</DataItem>\
            </XmlRoot>"
        );

        let round_tripped: XmlRoot = from_str(&xml).unwrap();
        assert_eq!(
            round_tripped.data_item.text,
            Some("/Xdmf/Domain/DataItem[@Name=\"source_data_item\"]".to_string())
        );
        assert!(round_tripped.data_item.include.is_none());
        assert_eq!(round_tripped.data_item.reference, Some("XML".to_string()));
    }

    #[test]
    fn data_item_include_serialize() {
        let custom_item = DataItem {
            name: Some("custom_data_item".to_string()),
            dimensions: Some(Dimensions(vec![2, 3])),
            number_type: Some(NumberType::Int),
            format: Some(Format::HDF),
            precision: Some(8),
            endian: None,
            text: None,
            include: Some(XInclude::new("coords.txt".to_string(), true)),
            reference: None,
        };
        assert_eq!(custom_item.name, Some("custom_data_item".to_string()));
        assert_eq!(custom_item.dimensions, Some(Dimensions(vec![2, 3])));
        assert_eq!(custom_item.number_type, Some(NumberType::Int));
        assert_eq!(custom_item.format, Some(Format::HDF));
        assert_eq!(custom_item.precision, Some(8));
        assert!(custom_item.text.is_none());
        assert_eq!(
            custom_item.include,
            Some(XInclude::new("coords.txt".to_string(), true))
        );
        assert!(custom_item.reference.is_none());

        let xml = to_string(&XmlRoot {
            data_item: custom_item,
        })
        .unwrap();
        pretty_assertions::assert_eq!(
            xml,
            "<XmlRoot>\
                <DataItem Name=\"custom_data_item\" Dimensions=\"2 3\" NumberType=\"Int\" Format=\"HDF\" Precision=\"8\">\
                    <xi:include href=\"coords.txt\" parse=\"text\"/>\
                </DataItem>\
            </XmlRoot>"
        );

        let round_tripped: XmlRoot = from_str(&xml).unwrap();
        assert!(round_tripped.data_item.text.is_none());
        assert_eq!(
            round_tripped.data_item.include,
            Some(XInclude::new("coords.txt".to_string(), true))
        );
    }

    #[test]
    fn data_item_multiline_text_round_trips() {
        let data_item = DataItem {
            text: Some("\n0 1 2\n3 4 5\n".to_string()),
            include: None,
            ..Default::default()
        };

        let xml = to_string(&XmlRoot { data_item }).unwrap();
        let round_tripped: XmlRoot = from_str(&xml).unwrap();
        assert_eq!(
            round_tripped.data_item.text,
            Some("\n0 1 2\n3 4 5\n".to_string())
        );
    }

    #[test]
    fn xinclude_serialize() {
        pretty_assertions::assert_eq!(
            to_string(&XInclude::new("coords.txt".to_string(), false)).unwrap(),
            "<xi:include href=\"coords.txt\"/>"
        );
        pretty_assertions::assert_eq!(
            to_string(&XInclude::new("coords.txt".to_string(), true)).unwrap(),
            "<xi:include href=\"coords.txt\" parse=\"text\"/>"
        );
    }
}

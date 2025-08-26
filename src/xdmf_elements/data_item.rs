//! This module contains the core datastructure used to specify data storage in XDMF files.

use serde::Serialize;

use super::dimensions::Dimensions;

/// Core datastructure to define how, where, and in which format data is stored.
#[derive(Clone, Debug, Serialize)]
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

    #[serde(flatten)]
    #[doc(hidden)]
    pub data: DataContent,

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
            data: String::new().into(),
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
            data: format!(
                "{}[@Name=\"{}\"]",
                source_path,
                source.name.clone().unwrap_or("MISSING".to_string())
            )
            .into(),
            reference: Some("XML".to_string()),
        }
    }
}

/// Used to include data from an external file using `XInclude`
#[derive(Clone, Debug, PartialEq, Serialize)]
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
}

/// Specifies where (ascii) data is stored, either inline or in an external file.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum DataContent {
    #[serde(rename = "$value")]
    /// Store the data as raw text
    Raw(String),

    #[serde(rename = "xi:include")]
    /// Store the data in an external file and include it using [XInclude](https://www.w3.org/TR/xinclude/)
    Include(XInclude),
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum Format {
    #[default]
    #[doc(hidden)]
    XML,
    #[doc(hidden)]
    HDF,
    #[doc(hidden)]
    Binary,
}

#[cfg(test)]
mod tests {
    use quick_xml::se::to_string;

    use super::*;

    #[derive(Serialize)]
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
        assert_eq!(default_item.data, String::new().into());
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
    fn data_item_custom() {
        let custom_item = DataItem {
            name: Some("custom_data_item".to_string()),
            dimensions: Some(Dimensions(vec![2, 3])),
            number_type: Some(NumberType::Int),
            format: Some(Format::HDF),
            precision: Some(8),
            data: "custom_data".to_string().into(),
            reference: None,
        };
        assert_eq!(custom_item.name, Some("custom_data_item".to_string()));
        assert_eq!(custom_item.dimensions, Some(Dimensions(vec![2, 3])));
        assert_eq!(custom_item.number_type, Some(NumberType::Int));
        assert_eq!(custom_item.format, Some(Format::HDF));
        assert_eq!(custom_item.precision, Some(8));
        assert_eq!(custom_item.data, "custom_data".into());
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
        assert_eq!(
            ref_item.data,
            "/Xdmf/Domain/DataItem[@Name=\"source_data_item\"]".into()
        );
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
            data: "custom_data".to_string().into(),
            reference: None,
        };

        pretty_assertions::assert_eq!(
            to_string(&XmlRoot { data_item }).unwrap(),
            "<XmlRoot>\
            <DataItem Name=\"custom_data_item\" Dimensions=\"2 3\" NumberType=\"Int\" Format=\"HDF\" Precision=\"8\">custom_data</DataItem>\
            </XmlRoot>"
        );
    }

    #[test]
    fn data_item_reference_serialize() {
        let source_data_item = DataItem {
            name: Some("source_data_item".to_string()),
            ..Default::default()
        };

        let ref_item = DataItem::new_reference(&source_data_item, "/Xdmf/Domain/DataItem");

        pretty_assertions::assert_eq!(
            to_string(&XmlRoot {
                data_item: ref_item
            })
            .unwrap(),
            "<XmlRoot>\
            <DataItem Reference=\"XML\">/Xdmf/Domain/DataItem[@Name=\"source_data_item\"]</DataItem>\
            </XmlRoot>"
        );
    }

    #[test]
    fn data_item_include_serialize() {
        let custom_item = DataItem {
            name: Some("custom_data_item".to_string()),
            dimensions: Some(Dimensions(vec![2, 3])),
            number_type: Some(NumberType::Int),
            format: Some(Format::HDF),
            precision: Some(8),
            data: XInclude::new("coords.txt".to_string(), true).into(),
            reference: None,
        };
        assert_eq!(custom_item.name, Some("custom_data_item".to_string()));
        assert_eq!(custom_item.dimensions, Some(Dimensions(vec![2, 3])));
        assert_eq!(custom_item.number_type, Some(NumberType::Int));
        assert_eq!(custom_item.format, Some(Format::HDF));
        assert_eq!(custom_item.precision, Some(8));
        assert_eq!(
            custom_item.data,
            XInclude::new("coords.txt".to_string(), true).into()
        );
        assert!(custom_item.reference.is_none());

        pretty_assertions::assert_eq!(
            to_string(&XmlRoot {
                data_item: custom_item
            })
            .unwrap(),
            "<XmlRoot>\
                <DataItem Name=\"custom_data_item\" Dimensions=\"2 3\" NumberType=\"Int\" Format=\"HDF\" Precision=\"8\">\
                    <xi:include href=\"coords.txt\" parse=\"text\"/>\
                </DataItem>\
            </XmlRoot>"
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

use serde::Serialize;

use super::dimensions::Dimensions;

#[derive(Clone, Debug, Serialize)]
pub struct DataItem {
    #[serde(rename = "@Name", skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(rename = "@Dimensions", skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<Dimensions>,

    #[serde(rename = "@NumberType", skip_serializing_if = "Option::is_none")]
    pub number_type: Option<NumberType>,

    #[serde(rename = "@Format", skip_serializing_if = "Option::is_none")]
    pub format: Option<Format>,

    #[serde(rename = "@Precision", skip_serializing_if = "Option::is_none")]
    pub precision: Option<u8>,

    #[serde(rename = "$value")]
    pub data: String,

    #[serde(rename = "@Reference", skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
}

// <xi:include href="coords.txt" parse="text"/>
#[derive(Clone, Debug, Serialize)]
pub struct XInclude {
    href: String,
    parse: Option<String>,
}

impl XInclude {
    pub fn new(href: String, parse_as_text: bool) -> Self {
        Self {
            href,
            parse: if parse_as_text {
                Some("text".to_string())
            } else {
                None
            },
        }
    }
}

impl Default for DataItem {
    fn default() -> Self {
        Self {
            name: None,
            dimensions: Some(Dimensions(vec![1])),
            number_type: Some(NumberType::default()),
            format: Some(Format::default()),
            precision: Some(4),
            data: String::new(),
            reference: None,
        }
    }
}

impl DataItem {
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
            ),
            reference: Some("XML".to_string()),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum NumberType {
    #[default]
    Float,
    Int,
    UInt,
    Char,
    UChar,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum Format {
    #[default]
    XML,
    HDF,
    Binary,
}

#[cfg(test)]
mod tests {
    use quick_xml::se::to_string;

    use super::*;

    #[test]
    fn data_item_default() {
        let default_item = DataItem::default();
        assert!(default_item.name.is_none());
        assert_eq!(default_item.dimensions, Some(Dimensions(vec![1])));
        assert_eq!(default_item.number_type, Some(NumberType::Float));
        assert_eq!(default_item.format, Some(Format::XML));
        assert_eq!(default_item.precision, Some(4));
        assert_eq!(default_item.data, String::new());
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
            data: "custom_data".to_string(),
            reference: None,
        };
        assert_eq!(custom_item.name, Some("custom_data_item".to_string()));
        assert_eq!(custom_item.dimensions, Some(Dimensions(vec![2, 3])));
        assert_eq!(custom_item.number_type, Some(NumberType::Int));
        assert_eq!(custom_item.format, Some(Format::HDF));
        assert_eq!(custom_item.precision, Some(8));
        assert_eq!(custom_item.data, "custom_data");
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
            "/Xdmf/Domain/DataItem[@Name=\"source_data_item\"]"
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
            data: "custom_data".to_string(),
            reference: None,
        };

        pretty_assertions::assert_eq!(
            to_string(&data_item).unwrap(),
            "<DataItem Name=\"custom_data_item\" Dimensions=\"2 3\" NumberType=\"Int\" Format=\"HDF\" Precision=\"8\">custom_data</DataItem>"
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
            to_string(&ref_item).unwrap(),
            "<DataItem Reference=\"XML\">/Xdmf/Domain/DataItem[@Name=\"source_data_item\"]</DataItem>"
        );
    }
}

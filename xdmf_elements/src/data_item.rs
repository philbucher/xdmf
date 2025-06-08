use serde::Serialize;

use crate::dimensions::Dimensions;

#[derive(Debug, Serialize)]
pub struct DataItem {
    #[serde(rename = "@Dimensions")]
    pub dimensions: Dimensions,

    #[serde(rename = "@NumberType")]
    pub number_type: NumberType,

    #[serde(rename = "@Format")]
    pub format: Format,

    #[serde(rename = "@Precision")]
    pub precision: u8,

    #[serde(rename = "$value")]
    pub data: String,
}

impl Default for DataItem {
    fn default() -> Self {
        DataItem {
            dimensions: Dimensions(vec![1]),
            number_type: NumberType::default(),
            format: Format::default(),
            precision: 4,
            data: String::new(),
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub enum NumberType {
    Float,
    Int,
    UInt,
    Char,
    UChar,
}

impl Default for NumberType {
    fn default() -> Self {
        NumberType::Float
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub enum Format {
    XML,
    HDF,
    Binary,
}

impl Default for Format {
    fn default() -> Self {
        Format::XML
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quick_xml::se::to_string;

    #[test]
    fn test_data_item_default() {
        let default_item = DataItem::default();
        assert_eq!(default_item.dimensions, Dimensions(vec![1]));
        assert_eq!(default_item.number_type, NumberType::Float);
        assert_eq!(default_item.format, Format::XML);
        assert_eq!(default_item.precision, 4);
        assert_eq!(default_item.data, String::new());
    }

    #[test]
    fn test_number_type_default() {
        assert_eq!(NumberType::default(), NumberType::Float);
    }

    #[test]
    fn test_format_default() {
        assert_eq!(Format::default(), Format::XML);
    }

    #[test]
    fn test_data_item_custom() {
        let custom_item = DataItem {
            dimensions: Dimensions(vec![2, 3]),
            number_type: NumberType::Int,
            format: Format::HDF,
            precision: 8,
            data: "custom_data".to_string(),
        };
        assert_eq!(custom_item.dimensions, Dimensions(vec![2, 3]));
        assert_eq!(custom_item.number_type, NumberType::Int);
        assert_eq!(custom_item.format, Format::HDF);
        assert_eq!(custom_item.precision, 8);
        assert_eq!(custom_item.data, "custom_data");
    }

    #[test]
    fn test_data_item_serialize() {
        let data_item = DataItem {
            dimensions: Dimensions(vec![2, 3]),
            number_type: NumberType::Int,
            format: Format::HDF,
            precision: 8,
            data: "custom_data".to_string(),
        };

        assert_eq!(
            to_string(&data_item).unwrap(),
            "<DataItem Dimensions=\"2 3\" NumberType=\"Int\" Format=\"HDF\" Precision=\"8\">custom_data</DataItem>"
        );
    }
}

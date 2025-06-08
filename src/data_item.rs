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

#[derive(Debug, Serialize)]
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

#[derive(Debug, Serialize)]
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

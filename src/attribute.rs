use serde::Serialize;

use crate::data_item::DataItem;

#[derive(Debug, Default, Serialize)]
pub struct Attribute {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@AttributeType")]
    pub attribute_type: AttributeType,

    #[serde(rename = "@Center")]
    pub center: Center,

    #[serde(rename = "DataItem")]
    pub data_item: DataItem,
}

#[derive(Debug, Serialize)]
pub enum AttributeType {
    Scalar,
    Vector,
    Tensor,
    Tensor6,
    Matrix,
}

impl Default for AttributeType {
    fn default() -> Self {
        AttributeType::Scalar
    }
}

#[derive(Debug, Serialize)]
pub enum Center {
    Node,
    Edge,
    Face,
    Cell,
    Grid,
    Other,
}

impl Default for Center {
    fn default() -> Self {
        Center::Node
    }
}

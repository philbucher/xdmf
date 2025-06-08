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

#[derive(Debug, PartialEq, Serialize)]
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

#[derive(Debug, PartialEq, Serialize)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use quick_xml::se::to_string;

    #[test]
    fn test_attribute_default() {
        let attribute = Attribute::default();
        assert_eq!(attribute.name, "");
        assert_eq!(attribute.attribute_type, AttributeType::Scalar);
        assert_eq!(attribute.center, Center::Node);
    }

    #[test]
    fn test_attribute_serialization() {
        let attribute = Attribute {
            name: String::from("Temperature"),
            attribute_type: AttributeType::Scalar,
            center: Center::Cell,
            data_item: DataItem::default(),
        };

        assert_eq!(
            to_string(&attribute).unwrap(),
            "<Attribute Name=\"Temperature\" AttributeType=\"Scalar\" Center=\"Cell\"><DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/></Attribute>"
        );
    }

    #[test]
    fn test_attribute_type_default() {
        let attribute_type = AttributeType::default();
        assert_eq!(attribute_type, AttributeType::Scalar);
    }

    #[test]
    fn test_center_default() {
        let center = Center::default();
        assert_eq!(center, Center::Node);
    }
}

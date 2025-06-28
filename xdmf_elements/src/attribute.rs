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
    pub data_item: Vec<DataItem>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum AttributeType {
    #[default]
    Scalar,
    Vector,
    Tensor,
    Tensor6,
    Matrix,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum Center {
    #[default]
    Node,
    Edge,
    Face,
    Cell,
    Grid,
    Other,
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
            data_item: vec![DataItem::default(), DataItem::default()],
        };

        assert_eq!(
            to_string(&attribute).unwrap(),
            "<Attribute Name=\"Temperature\" AttributeType=\"Scalar\" Center=\"Cell\">\
                <DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/>\
                <DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/>\
            </Attribute>"
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

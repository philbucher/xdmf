use serde::Serialize;

use super::data_item::DataItem;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Attribute {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@AttributeType")]
    pub attribute_type: AttributeType,

    #[serde(rename = "@Center")]
    pub center: Center,

    #[serde(rename = "DataItem")]
    pub data_items: Vec<DataItem>,
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
pub(crate) fn center_to_data_tag(center: Center) -> &'static str {
    match center {
        Center::Node => "point_data",
        Center::Cell => "cell_data",
        Center::Edge => "edge_data",
        Center::Face => "face_data",
        Center::Grid => "grid_data",
        Center::Other => "other_data",
    }
}

#[cfg(test)]
mod tests {
    use quick_xml::se::to_string;

    use super::*;

    #[test]
    fn attribute_default() {
        let attribute = Attribute::default();
        assert_eq!(attribute.name, "");
        assert_eq!(attribute.attribute_type, AttributeType::Scalar);
        assert_eq!(attribute.center, Center::Node);
    }

    #[test]
    fn attribute_serialization() {
        let attribute = Attribute {
            name: String::from("Temperature"),
            attribute_type: AttributeType::Scalar,
            center: Center::Cell,
            data_items: vec![DataItem::default(), DataItem::default()],
        };

        pretty_assertions::assert_eq!(
            to_string(&attribute).unwrap(),
            "<Attribute Name=\"Temperature\" AttributeType=\"Scalar\" Center=\"Cell\">\
                <DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/>\
                <DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/>\
            </Attribute>"
        );
    }

    #[test]
    fn attribute_type_default() {
        assert_eq!(AttributeType::default(), AttributeType::Scalar);
    }

    #[test]
    fn center_default() {
        assert_eq!(Center::default(), Center::Node);
    }
}

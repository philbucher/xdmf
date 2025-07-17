use serde::Serialize;

use super::data_item::DataItem;

#[derive(Clone, Debug, Serialize)]
pub struct Geometry {
    #[serde(rename = "@GeometryType")]
    pub geometry_type: GeometryType,

    #[serde(rename = "DataItem")]
    pub data_item: DataItem,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum GeometryType {
    #[default]
    XYZ,
    XY,
}

#[cfg(test)]
mod tests {
    use quick_xml::se::to_string;

    use super::*;

    #[test]
    fn geometry_type_default() {
        assert_eq!(GeometryType::default(), GeometryType::XYZ);
    }

    #[test]
    fn geometry_serialization() {
        let geometry = Geometry {
            geometry_type: GeometryType::XY,
            data_item: DataItem::default(),
        };

        pretty_assertions::assert_eq!(
            to_string(&geometry).unwrap(),
            "<Geometry GeometryType=\"XY\"><DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/></Geometry>"
        );
    }
}

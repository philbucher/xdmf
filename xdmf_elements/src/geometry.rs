use serde::Serialize;

use crate::data_item::DataItem;

#[derive(Debug, Serialize)]
pub struct Geometry {
    #[serde(rename = "@GeometryType")]
    pub geometry_type: GeometryType,

    #[serde(rename = "DataItem")]
    pub data_item: DataItem,
}

#[derive(Debug, PartialEq, Serialize)]
pub enum GeometryType {
    XYZ,
    XY,
}

impl Default for GeometryType {
    fn default() -> Self {
        GeometryType::XYZ
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use quick_xml::se::to_string;

    #[test]
    fn test_geometry_type_default() {
        assert_eq!(GeometryType::default(), GeometryType::XYZ);
    }

    #[test]
    fn test_geometry_serialization() {
        let geometry = Geometry {
            geometry_type: GeometryType::XY,
            data_item: DataItem::default(),
        };

        assert_eq!(
            to_string(&geometry).unwrap(),
            "<Geometry GeometryType=\"XY\"><DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/></Geometry>"
        );
    }
}

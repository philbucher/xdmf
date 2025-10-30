//! This module contains the Geometry element, which describes the XYZ values of the mesh points.

use serde::{Deserialize, Serialize};

use super::data_item::DataItem;

/// The Geometry element describes the XYZ values of the mesh points.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Geometry {
    #[serde(rename = "@GeometryType")]
    #[doc(hidden)]
    pub geometry_type: GeometryType,

    #[serde(rename = "DataItem")]
    #[doc(hidden)]
    pub data_item: DataItem,
}

/// Type of geometry, either 3D (XYZ) or 2D (XY).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum GeometryType {
    #[default]
    #[doc(hidden)]
    XYZ,
    #[doc(hidden)]
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

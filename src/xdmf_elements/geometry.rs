//! This module contains the Geometry element, which describes the XYZ values of the mesh points.

use serde::{Deserialize, Serialize};

use super::data_item::DataItem;

/// The Geometry element describes the XYZ values of the mesh points.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Geometry {
    #[serde(rename = "@GeometryType")]
    #[doc(hidden)]
    pub geometry_type: GeometryType,

    /// One item for `XYZ`, three -- X, then Y, then Z -- for `X_Y_Z`.
    #[serde(rename = "DataItem")]
    #[doc(hidden)]
    pub data_items: Vec<DataItem>,
}

/// Type of geometry: 3D or 2D, interleaved or one array per coordinate direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum GeometryType {
    #[default]
    #[doc(hidden)]
    XYZ,
    #[doc(hidden)]
    XY,

    /// One array per coordinate direction rather than one of interleaved tuples. What a submesh
    /// selecting its points out of the mesh's is written as, since all three of its selections
    /// then share the one index list naming the points it holds.
    #[serde(rename = "X_Y_Z")]
    #[doc(hidden)]
    XYZSeparate,
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
            data_items: vec![DataItem::default()],
        };

        pretty_assertions::assert_eq!(
            to_string(&geometry).unwrap(),
            "<Geometry GeometryType=\"XY\"><DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/></Geometry>"
        );
    }
}

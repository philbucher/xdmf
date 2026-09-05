//! The Geometry element, which describes the XYZ values of the mesh points.

use serde::{Deserialize, Serialize};

use super::data_item::DataItem;

/// The Geometry element describes the XYZ values of the mesh points.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Geometry {
    #[serde(rename = "@GeometryType")]
    #[doc(hidden)]
    pub geometry_type: GeometryType,

    /// One item for `XYZ`; three for `X_Y_Z`, in the order X, Y, Z.
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

    /// One array per coordinate direction rather than interleaved tuples. A mesh whose submeshes
    /// select their points out of it is written this way, so all three of a submesh's selections
    /// share the one index list naming its points.
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

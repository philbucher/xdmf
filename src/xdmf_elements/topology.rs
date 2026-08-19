//! This module contains the Topology element, which describes how points are connected to form elements.

use serde::{Deserialize, Serialize};

use super::{CellType, data_item::DataItem};

/// Described the topology of the mesh, i.e. how the points are connected to form elements.
/// Check the documentation [here](https://www.xdmf.org/index.php/XDMF_Model_and_Format.html#Topology).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Topology {
    #[serde(rename = "@TopologyType")]
    #[doc(hidden)]
    pub topology_type: TopologyType,

    // Required for `Polyvertex`/`Polyline` when the per-element node count isn't the type's
    // implicit default
    #[serde(rename = "@NodesPerElement", skip_serializing_if = "Option::is_none")]
    #[doc(hidden)]
    pub nodes_per_element: Option<u8>,

    #[serde(rename = "@NumberOfElements")]
    #[doc(hidden)]
    pub number_of_elements: String,

    #[serde(rename = "DataItem")]
    #[doc(hidden)]
    pub data_item: DataItem,
}

/// Type of topology of the mesh.
/// Either a uniform type for all elements, or mixed for different element types.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum TopologyType {
    #[doc(hidden)]
    Mixed,
    #[doc(hidden)]
    Polyvertex,
    #[doc(hidden)]
    Polyline,
    #[doc(hidden)]
    Triangle,
    #[doc(hidden)]
    Quadrilateral,
    #[doc(hidden)]
    Tetrahedron,
    #[doc(hidden)]
    Pyramid,
    #[doc(hidden)]
    Wedge,
    #[doc(hidden)]
    Hexahedron,
    #[serde(rename = "Edge_3")]
    #[doc(hidden)]
    Edge3,
    #[serde(rename = "Triangle_6")]
    #[doc(hidden)]
    Triangle6,
    #[serde(rename = "Quadrilateral_8")]
    #[doc(hidden)]
    Quadrilateral8,
    #[serde(rename = "Quadrilateral_9")]
    #[doc(hidden)]
    Quadrilateral9,
    #[serde(rename = "Tetrahedron_10")]
    #[doc(hidden)]
    Tetrahedron10,
    #[serde(rename = "Pyramid_13")]
    #[doc(hidden)]
    Pyramid13,
    #[serde(rename = "Wedge_15")]
    #[doc(hidden)]
    Wedge15,
    #[serde(rename = "Wedge_18")]
    #[doc(hidden)]
    Wedge18,
    #[serde(rename = "Hexahedron_20")]
    #[doc(hidden)]
    Hexahedron20,
    #[serde(rename = "Hexahedron_24")]
    #[doc(hidden)]
    Hexahedron24,
    #[serde(rename = "Hexahedron_27")]
    #[doc(hidden)]
    Hexahedron27,
}

impl From<CellType> for TopologyType {
    fn from(cell_type: CellType) -> Self {
        match cell_type {
            CellType::Vertex => Self::Polyvertex,
            CellType::Edge => Self::Polyline,
            CellType::Triangle => Self::Triangle,
            CellType::Quadrilateral => Self::Quadrilateral,
            CellType::Tetrahedron => Self::Tetrahedron,
            CellType::Pyramid => Self::Pyramid,
            CellType::Wedge => Self::Wedge,
            CellType::Hexahedron => Self::Hexahedron,
            CellType::Edge3 => Self::Edge3,
            CellType::Quadrilateral9 => Self::Quadrilateral9,
            CellType::Triangle6 => Self::Triangle6,
            CellType::Quadrilateral8 => Self::Quadrilateral8,
            CellType::Tetrahedron10 => Self::Tetrahedron10,
            CellType::Pyramid13 => Self::Pyramid13,
            CellType::Wedge15 => Self::Wedge15,
            CellType::Wedge18 => Self::Wedge18,
            CellType::Hexahedron20 => Self::Hexahedron20,
            CellType::Hexahedron24 => Self::Hexahedron24,
            CellType::Hexahedron27 => Self::Hexahedron27,
        }
    }
}

#[cfg(test)]
mod tests {
    use quick_xml::se::to_string;

    use super::*;

    #[test]
    fn topology_serialization() {
        let topology = Topology {
            topology_type: TopologyType::Triangle,
            nodes_per_element: None,
            number_of_elements: "3".to_string(),
            data_item: DataItem::default(),
        };

        pretty_assertions::assert_eq!(
            to_string(&topology).unwrap(),
            "<Topology TopologyType=\"Triangle\" NumberOfElements=\"3\"><DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/></Topology>"
        );
    }

    #[test]
    fn topology_serialization_with_nodes_per_element() {
        let topology = Topology {
            topology_type: TopologyType::Polyline,
            nodes_per_element: Some(2),
            number_of_elements: "3".to_string(),
            data_item: DataItem::default(),
        };

        pretty_assertions::assert_eq!(
            to_string(&topology).unwrap(),
            "<Topology TopologyType=\"Polyline\" NodesPerElement=\"2\" NumberOfElements=\"3\"><DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/></Topology>"
        );
    }

    #[test]
    fn topology_type_serialization_names() {
        // the higher-order variants' XML names (with underscores) don't match their Rust
        // identifiers, so they carry an explicit `#[serde(rename)]` -- verified here through the
        // `@TopologyType` attribute context (a bare `to_string` on the enum instead serializes it
        // as its own element, e.g. `<Edge_3/>`, not the attribute value alone)
        let topology_type_attr = |topology_type| {
            let topology = Topology {
                topology_type,
                nodes_per_element: None,
                number_of_elements: "1".to_string(),
                data_item: DataItem::default(),
            };
            to_string(&topology).unwrap()
        };

        assert!(topology_type_attr(TopologyType::Edge3).contains("TopologyType=\"Edge_3\""));
        assert!(
            topology_type_attr(TopologyType::Triangle6).contains("TopologyType=\"Triangle_6\"")
        );
        assert!(
            topology_type_attr(TopologyType::Quadrilateral8)
                .contains("TopologyType=\"Quadrilateral_8\"")
        );
        assert!(
            topology_type_attr(TopologyType::Quadrilateral9)
                .contains("TopologyType=\"Quadrilateral_9\"")
        );
        assert!(
            topology_type_attr(TopologyType::Tetrahedron10)
                .contains("TopologyType=\"Tetrahedron_10\"")
        );
        assert!(
            topology_type_attr(TopologyType::Pyramid13).contains("TopologyType=\"Pyramid_13\"")
        );
        assert!(topology_type_attr(TopologyType::Wedge15).contains("TopologyType=\"Wedge_15\""));
        assert!(topology_type_attr(TopologyType::Wedge18).contains("TopologyType=\"Wedge_18\""));
        assert!(
            topology_type_attr(TopologyType::Hexahedron20)
                .contains("TopologyType=\"Hexahedron_20\"")
        );
        assert!(
            topology_type_attr(TopologyType::Hexahedron24)
                .contains("TopologyType=\"Hexahedron_24\"")
        );
        assert!(
            topology_type_attr(TopologyType::Hexahedron27)
                .contains("TopologyType=\"Hexahedron_27\"")
        );
    }
}

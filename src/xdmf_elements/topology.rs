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
    #[doc(hidden)]
    #[serde(rename = "Edge_3")]
    Edge3,
    #[doc(hidden)]
    #[serde(rename = "Triangle_6")]
    Triangle6,
    #[doc(hidden)]
    #[serde(rename = "Quadrilateral_8")]
    Quadrilateral8,
    #[doc(hidden)]
    #[serde(rename = "Quadrilateral_9")]
    Quadrilateral9,
    #[doc(hidden)]
    #[serde(rename = "Tetrahedron_10")]
    Tetrahedron10,
    #[doc(hidden)]
    #[serde(rename = "Pyramid_13")]
    Pyramid13,
    #[doc(hidden)]
    #[serde(rename = "Wedge_15")]
    Wedge15,
    #[doc(hidden)]
    #[serde(rename = "Wedge_18")]
    Wedge18,
    #[doc(hidden)]
    #[serde(rename = "Hexahedron_20")]
    Hexahedron20,
    #[doc(hidden)]
    #[serde(rename = "Hexahedron_24")]
    Hexahedron24,
    #[doc(hidden)]
    #[serde(rename = "Hexahedron_27")]
    Hexahedron27,
}

impl TopologyType {
    /// The [`CellType`] every element has in a homogeneous topology of this type, or `None` for
    /// `Mixed` (each element carries its own type inline) and `Polyvertex` (ambiguous by itself —
    /// see `reader::topology`, which special-cases it as either a point cloud or `Vertex` cells).
    pub(crate) fn cell_type(self) -> Option<CellType> {
        match self {
            Self::Mixed | Self::Polyvertex => None,
            Self::Polyline => Some(CellType::Edge),
            Self::Triangle => Some(CellType::Triangle),
            Self::Quadrilateral => Some(CellType::Quadrilateral),
            Self::Tetrahedron => Some(CellType::Tetrahedron),
            Self::Pyramid => Some(CellType::Pyramid),
            Self::Wedge => Some(CellType::Wedge),
            Self::Hexahedron => Some(CellType::Hexahedron),
            Self::Edge3 => Some(CellType::Edge3),
            Self::Triangle6 => Some(CellType::Triangle6),
            Self::Quadrilateral8 => Some(CellType::Quadrilateral8),
            Self::Quadrilateral9 => Some(CellType::Quadrilateral9),
            Self::Tetrahedron10 => Some(CellType::Tetrahedron10),
            Self::Pyramid13 => Some(CellType::Pyramid13),
            Self::Wedge15 => Some(CellType::Wedge15),
            Self::Wedge18 => Some(CellType::Wedge18),
            Self::Hexahedron20 => Some(CellType::Hexahedron20),
            Self::Hexahedron24 => Some(CellType::Hexahedron24),
            Self::Hexahedron27 => Some(CellType::Hexahedron27),
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
            number_of_elements: "3".to_string(),
            data_item: DataItem::default(),
        };

        pretty_assertions::assert_eq!(
            to_string(&topology).unwrap(),
            "<Topology TopologyType=\"Triangle\" NumberOfElements=\"3\"><DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/></Topology>"
        );
    }

    #[test]
    fn cell_type_mapping_is_exhaustive_and_consistent() {
        // Mixed/Polyvertex are deliberately not 1:1 with a single CellType.
        assert_eq!(TopologyType::Mixed.cell_type(), None);
        assert_eq!(TopologyType::Polyvertex.cell_type(), None);

        // every other variant maps to the CellType its name says it should.
        let cases = [
            (TopologyType::Polyline, CellType::Edge),
            (TopologyType::Triangle, CellType::Triangle),
            (TopologyType::Quadrilateral, CellType::Quadrilateral),
            (TopologyType::Tetrahedron, CellType::Tetrahedron),
            (TopologyType::Pyramid, CellType::Pyramid),
            (TopologyType::Wedge, CellType::Wedge),
            (TopologyType::Hexahedron, CellType::Hexahedron),
            (TopologyType::Edge3, CellType::Edge3),
            (TopologyType::Triangle6, CellType::Triangle6),
            (TopologyType::Quadrilateral8, CellType::Quadrilateral8),
            (TopologyType::Quadrilateral9, CellType::Quadrilateral9),
            (TopologyType::Tetrahedron10, CellType::Tetrahedron10),
            (TopologyType::Pyramid13, CellType::Pyramid13),
            (TopologyType::Wedge15, CellType::Wedge15),
            (TopologyType::Wedge18, CellType::Wedge18),
            (TopologyType::Hexahedron20, CellType::Hexahedron20),
            (TopologyType::Hexahedron24, CellType::Hexahedron24),
            (TopologyType::Hexahedron27, CellType::Hexahedron27),
        ];
        for (topology_type, cell_type) in cases {
            assert_eq!(topology_type.cell_type(), Some(cell_type));
        }
    }

    #[test]
    fn low_order_names_serialize_without_underscores() {
        for (topology_type, name) in [
            (TopologyType::Polyvertex, "Polyvertex"),
            (TopologyType::Polyline, "Polyline"),
            (TopologyType::Triangle, "Triangle"),
            (TopologyType::Quadrilateral, "Quadrilateral"),
            (TopologyType::Tetrahedron, "Tetrahedron"),
            (TopologyType::Pyramid, "Pyramid"),
            (TopologyType::Wedge, "Wedge"),
            (TopologyType::Hexahedron, "Hexahedron"),
            (TopologyType::Mixed, "Mixed"),
        ] {
            let topology = Topology {
                topology_type,
                number_of_elements: "1".to_string(),
                data_item: DataItem::default(),
            };
            assert!(
                to_string(&topology)
                    .unwrap()
                    .contains(&format!("TopologyType=\"{name}\""))
            );
        }
    }

    #[test]
    fn higher_order_names_use_underscore_suffix() {
        for (topology_type, name) in [
            (TopologyType::Edge3, "Edge_3"),
            (TopologyType::Triangle6, "Triangle_6"),
            (TopologyType::Quadrilateral8, "Quadrilateral_8"),
            (TopologyType::Quadrilateral9, "Quadrilateral_9"),
            (TopologyType::Tetrahedron10, "Tetrahedron_10"),
            (TopologyType::Pyramid13, "Pyramid_13"),
            (TopologyType::Wedge15, "Wedge_15"),
            (TopologyType::Wedge18, "Wedge_18"),
            (TopologyType::Hexahedron20, "Hexahedron_20"),
            (TopologyType::Hexahedron24, "Hexahedron_24"),
            (TopologyType::Hexahedron27, "Hexahedron_27"),
        ] {
            let topology = Topology {
                topology_type,
                number_of_elements: "1".to_string(),
                data_item: DataItem::default(),
            };
            assert!(
                to_string(&topology)
                    .unwrap()
                    .contains(&format!("TopologyType=\"{name}\""))
            );
        }
    }
}

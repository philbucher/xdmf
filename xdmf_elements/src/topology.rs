use serde::Serialize;

use crate::data_item::DataItem;

#[derive(Debug, Serialize)]
pub struct Topology {
    #[serde(rename = "@TopologyType")]
    pub topology_type: TopologyType,

    #[serde(rename = "@NumberOfElements")]
    pub number_of_elements: String,

    #[serde(rename = "DataItem")]
    pub data_item: DataItem,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum TopologyType {
    Mixed,
    Triangle,
    Quadrilateral,
    //     4 - TRIANGLE
    // 5 - QUADRILATERAL
    // 6 - TETRAHEDRON
    // 7 - PYRAMID
    // 8 - WEDGE
    // 9 - HEXAHEDRON
    // 16 - POLYHEDRON
    // 34 - EDGE_3
    // 35 - QUADRILATERAL_9
    // 36 - TRIANGLE_6
    // 37 - QUADRILATERAL_8
    // 38 - TETRAHEDRON_10
    // 39 - PYRAMID_13
    // 40 - WEDGE_15
    // 41 - WEDGE_18
    // 48 - HEXAHEDRON_20
    // 49 - HEXAHEDRON_24
    // 50 - HEXAHEDRON_27
}

#[cfg(test)]
mod tests {
    use super::*;
    use quick_xml::se::to_string;

    #[test]
    fn test_topology_serialization() {
        let topology = Topology {
            topology_type: TopologyType::Triangle,
            number_of_elements: "3".to_string(),
            data_item: DataItem::default(),
        };

        assert_eq!(
            to_string(&topology).unwrap(),
            "<Topology TopologyType=\"Triangle\" NumberOfElements=\"3\"><DataItem Dimensions=\"1\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\"/></Topology>"
        );
    }
}

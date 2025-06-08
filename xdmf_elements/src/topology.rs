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

#[derive(Debug, Serialize)]
pub enum TopologyType {
    Mixed,
    Triangle,
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

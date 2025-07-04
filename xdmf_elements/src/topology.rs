use serde::Serialize;

use crate::data_item::DataItem;

#[derive(Clone, Debug, Serialize)]
pub struct Topology {
    #[serde(rename = "@TopologyType")]
    pub topology_type: TopologyType,

    #[serde(rename = "@NumberOfElements")]
    pub number_of_elements: String,

    #[serde(rename = "DataItem")]
    pub data_item: DataItem,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub enum TopologyType {
    Mixed,
    Triangle,
    Quadrilateral,
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

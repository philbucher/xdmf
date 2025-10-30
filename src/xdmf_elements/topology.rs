//! This module contains the Topology element, which describes how points are connected to form elements.

use serde::{Deserialize, Serialize};

use super::data_item::DataItem;

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
/// Note: currently only the mixed type is used. Using a uniform type limits applicability but reduces file size slightly.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum TopologyType {
    #[doc(hidden)]
    Mixed,
    #[doc(hidden)]
    Triangle,
    #[doc(hidden)]
    Quadrilateral,
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
}

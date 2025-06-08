use quick_xml::Writer;
use serde::Serialize;

pub mod attribute;
pub mod data_item;
pub mod dimensions;
pub mod geometry;
pub mod grid;
pub mod topology;

use data_item::{DataItem, NumberType};
use dimensions::Dimensions;
use geometry::{Geometry, GeometryType};
use grid::Grid;
use topology::{Topology, TopologyType};

pub const XDMF_TAG: &str = "Xdmf";

#[derive(Debug, Serialize)]
pub struct Xdmf {
    #[serde(rename = "@Version")]
    pub version: String,

    #[serde(rename = "Domain")]
    pub domains: Vec<Domain>,
}

impl Xdmf {
    pub fn new(domain: Domain) -> Self {
        Xdmf {
            version: "3.0".to_string(),
            domains: vec![domain],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Domain {
    #[serde(rename = "Grid")]
    pub grid: Grid,
}

fn main2222() -> Result<(), Box<dyn std::error::Error>> {
    let xdmf = Xdmf {
        version: "3.0".into(),
        domains: vec![Domain {
            grid: Grid::new_tree(
                "Grid_Tree",
                vec![
                    Grid::new_uniform(
                        "Grid_1",
                        Geometry {
                            geometry_type: GeometryType::XY,
                            data_item: DataItem {
                                dimensions: Dimensions(vec![3, 2]),
                                data: "0 0 0 1 1 1".into(),
                                number_type: NumberType::Float,
                                ..Default::default()
                            },
                        },
                        Topology {
                            topology_type: TopologyType::Triangle,
                            number_of_elements: "1".into(),
                            data_item: DataItem {
                                dimensions: Dimensions(vec![3]),
                                number_type: NumberType::Int,
                                data: "0 1 2".into(),
                                ..Default::default()
                            },
                        },
                    ),
                    Grid::new_uniform(
                        "Grid_2",
                        Geometry {
                            geometry_type: GeometryType::XY,
                            data_item: DataItem {
                                dimensions: Dimensions(vec![3, 2]),
                                data: "1 1 1 2 2 2".into(),
                                number_type: NumberType::Float,
                                ..Default::default()
                            },
                        },
                        Topology {
                            topology_type: TopologyType::Triangle,
                            number_of_elements: "1".into(),
                            data_item: DataItem {
                                dimensions: Dimensions(vec![3]),
                                number_type: NumberType::Int,
                                data: "0 1 2".into(),
                                ..Default::default()
                            },
                        },
                    ),
                    Grid::new_tree(
                        "Grid_Tree2",
                        vec![
                            Grid::new_uniform(
                                "Grid_21",
                                Geometry {
                                    geometry_type: GeometryType::XY,
                                    data_item: DataItem {
                                        dimensions: Dimensions(vec![3, 2]),
                                        data: "3 3 3 4 4 4".into(),
                                        number_type: NumberType::Float,
                                        ..Default::default()
                                    },
                                },
                                Topology {
                                    topology_type: TopologyType::Triangle,
                                    number_of_elements: "1".into(),
                                    data_item: DataItem {
                                        dimensions: Dimensions(vec![3]),
                                        number_type: NumberType::Int,
                                        data: "0 1 2".into(),
                                        ..Default::default()
                                    },
                                },
                            ),
                            Grid::new_uniform(
                                "Grid_22",
                                Geometry {
                                    geometry_type: GeometryType::XY,
                                    data_item: DataItem {
                                        dimensions: Dimensions(vec![3, 2]),
                                        data: "4 4 4 5 5 5".into(),
                                        number_type: NumberType::Float,
                                        ..Default::default()
                                    },
                                },
                                Topology {
                                    topology_type: TopologyType::Triangle,
                                    number_of_elements: "1".into(),
                                    data_item: DataItem {
                                        dimensions: Dimensions(vec![3]),
                                        number_type: NumberType::Int,
                                        data: "0 1 2".into(),
                                        ..Default::default()
                                    },
                                },
                            ),
                        ],
                    ),
                ],
            ),
        }],
    };

    // Create an in-memory buffer (stdout here, but could be a file or Vec)
    let out_file = std::fs::File::create("output.xdmf")?;

    // Create quick_xml writer with indentation (pretty print)
    let mut writer = Writer::new_with_indent(out_file, b' ', 4);

    writer.write_serializable("Xdmf", &xdmf)?;

    Ok(())
}

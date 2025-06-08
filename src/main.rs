use quick_xml::Writer;
use serde::Serialize;

pub mod attribute;
pub mod data_item;
pub mod dimensions;
pub mod geometry;
pub mod grid;
pub mod topology;

use data_item::{DataItem, Format, NumberType};
use dimensions::Dimensions;
use geometry::{Geometry, GeometryType};
use grid::Grid;
use topology::{Topology, TopologyType};

#[derive(Debug, Serialize)]
#[serde(rename = "Xdmf")]
struct Xdmf {
    #[serde(rename = "@Version")]
    version: String,

    #[serde(rename = "Domain")]
    domains: Vec<Domain>,
}

#[derive(Debug, Serialize)]
struct Domain {
    #[serde(rename = "Grid")]
    grid: Grid,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xdmf = Xdmf {
        version: "3.0".into(),
        domains: vec![Domain {
            grid: Grid::new_tree(
                "Grid_Tree",
                vec![
                    Grid::new_uniform(
                        "Grid_1",
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
                        Geometry {
                            geometry_type: GeometryType::XY,
                            data_item: DataItem {
                                dimensions: Dimensions(vec![3, 2]),
                                data: "0 0 0 1 1 1".into(),
                                number_type: NumberType::Float,
                                ..Default::default()
                            },
                        },
                    ),
                    Grid::new_uniform(
                        "Grid_2",
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
                        Geometry {
                            geometry_type: GeometryType::XY,
                            data_item: DataItem {
                                dimensions: Dimensions(vec![3, 2]),
                                data: "1 1 1 2 2 2".into(),
                                number_type: NumberType::Float,
                                ..Default::default()
                            },
                        },
                    ),
                    Grid::new_tree(
                        "Grid_Tree2",
                        vec![
                            Grid::new_uniform(
                                "Grid_21",
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
                                Geometry {
                                    geometry_type: GeometryType::XY,
                                    data_item: DataItem {
                                        dimensions: Dimensions(vec![3, 2]),
                                        data: "3 3 3 4 4 4".into(),
                                        number_type: NumberType::Float,
                                        ..Default::default()
                                    },
                                },
                            ),
                            Grid::new_uniform(
                                "Grid_22",
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
                                Geometry {
                                    geometry_type: GeometryType::XY,
                                    data_item: DataItem {
                                        dimensions: Dimensions(vec![3, 2]),
                                        data: "4 4 4 5 5 5".into(),
                                        number_type: NumberType::Float,
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

    // let mut xdmf_mesh = XdmfMesh::new(points, cells);

    // xdmf_mesh.add_subset(...);

    // xdmf_mesh.add_results("results", time)?;

    // xdmf_mesh.write_to_file("output.xdmf")?;

    Ok(())
}

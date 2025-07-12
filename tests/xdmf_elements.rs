use quick_xml::Writer;

use xdmf::xdmf_elements::attribute::{Attribute, AttributeType, Center};
use xdmf::xdmf_elements::data_item::{DataItem, NumberType};
use xdmf::xdmf_elements::dimensions::Dimensions;
use xdmf::xdmf_elements::geometry::{Geometry, GeometryType};
use xdmf::xdmf_elements::grid::{CollectionType, Grid, Time, Uniform};
use xdmf::xdmf_elements::topology::{Topology, TopologyType};
use xdmf::xdmf_elements::{Domain, XDMF_TAG, Xdmf};

#[test]
fn basic_grid() {
    let xdmf = Xdmf::new(Domain::new(Grid::new_uniform(
        "Grid_1",
        Geometry {
            geometry_type: GeometryType::XYZ,
            data_item: DataItem {
                dimensions: Some(Dimensions(vec![4, 3])),
                data: "0 0 0 0 1 0 1 1 0 1 0 0.5".into(),
                number_type: Some(NumberType::Float),
                ..Default::default()
            },
        },
        Topology {
            topology_type: TopologyType::Triangle,
            number_of_elements: "2".into(),
            data_item: DataItem {
                dimensions: Some(Dimensions(vec![6])),
                number_type: Some(NumberType::Int),
                data: "0 1 2 0 2 3".into(),
                ..Default::default()
            },
        },
    )));

    // Create an in-memory buffer to serialize to
    let mut buffer = Vec::new();

    // Create quick_xml writer with indentation (pretty print)
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    writer.write_serializable(XDMF_TAG, &xdmf).unwrap();

    let expected_xdmf = r#"
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="Grid_1" GridType="Uniform">
            <Geometry GeometryType="XYZ">
                <DataItem Dimensions="4 3" NumberType="Float" Format="XML" Precision="4">0 0 0 0 1 0 1 1 0 1 0 0.5</DataItem>
            </Geometry>
            <Topology TopologyType="Triangle" NumberOfElements="2">
                <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 0 2 3</DataItem>
            </Topology>
        </Grid>
    </Domain>
</Xdmf>"#;

    pretty_assertions::assert_eq!(String::from_utf8(buffer).unwrap(), expected_xdmf);

    // write to file
    // let mut file_writer =
    //     Writer::new_with_indent(std::fs::File::create("basic_grid.xdmf").unwrap(), b' ', 4);
    // file_writer.write_serializable(XDMF_TAG, &xdmf).unwrap();
}
#[test]
fn hierarchical_tree_grid() {
    let xdmf = Xdmf::new(Domain::new(Grid::new_tree(
        "hierarchical_tree_grid",
        Some(vec![
            Grid::new_tree(
                "grid_level_1",
                Some(vec![
                    Grid::new_uniform(
                        "sub_grid_1",
                        Geometry {
                            geometry_type: GeometryType::XYZ,
                            data_item: DataItem {
                                dimensions: Some(Dimensions(vec![5, 3])),
                                data: "0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0".into(),
                                number_type: Some(NumberType::Float),
                                ..Default::default()
                            },
                        },
                        Topology {
                            topology_type: TopologyType::Triangle,
                            number_of_elements: "2".into(),
                            data_item: DataItem {
                                dimensions: Some(Dimensions(vec![6])),
                                number_type: Some(NumberType::Int),
                                data: "0 1 2 2 3 4".into(),
                                ..Default::default()
                            },
                        },
                    ),
                    Grid::new_uniform(
                        "sub_grid_2",
                        Geometry {
                            geometry_type: GeometryType::XYZ,
                            data_item: DataItem {
                                dimensions: Some(Dimensions(vec![6, 3])),
                                data: "1 1.5 0 1 1 0 1 0 0 1.3 1.5 0 1.3 1 0 1.3 0 0".into(),
                                number_type: Some(NumberType::Float),
                                ..Default::default()
                            },
                        },
                        Topology {
                            topology_type: TopologyType::Quadrilateral,
                            number_of_elements: "2".into(),
                            data_item: DataItem {
                                dimensions: Some(Dimensions(vec![8])),
                                number_type: Some(NumberType::Int),
                                data: "0 1 4 3 1 2 5 4".into(),
                                ..Default::default()
                            },
                        },
                    ),
                ]),
            ),
            Grid::new_uniform(
                "Grid_1",
                Geometry {
                    geometry_type: GeometryType::XYZ,
                    data_item: DataItem {
                        dimensions: Some(Dimensions(vec![5, 3])),
                        data: "0 0 0 0 1 0 1 1 0 1 0 0 0.5 1.5 0.5".into(),
                        number_type: Some(NumberType::Float),
                        ..Default::default()
                    },
                },
                Topology {
                    topology_type: TopologyType::Mixed,
                    number_of_elements: "2".into(),
                    data_item: DataItem {
                        dimensions: Some(Dimensions(vec![9])),
                        number_type: Some(NumberType::Int),
                        data: "5 0 1 2 3 4 1 2 4".into(),
                        ..Default::default()
                    },
                },
            ),
        ]),
    )));

    // Create an in-memory buffer to serialize to
    let mut buffer = Vec::new();

    // Create quick_xml writer with indentation (pretty print)
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    writer.write_serializable(XDMF_TAG, &xdmf).unwrap();

    let expected_xdmf = r#"
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="hierarchical_tree_grid" GridType="Tree">
            <Grid Name="grid_level_1" GridType="Tree">
                <Grid Name="sub_grid_1" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                    </Geometry>
                    <Topology TopologyType="Triangle" NumberOfElements="2">
                        <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                    </Topology>
                </Grid>
                <Grid Name="sub_grid_2" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Dimensions="6 3" NumberType="Float" Format="XML" Precision="4">1 1.5 0 1 1 0 1 0 0 1.3 1.5 0 1.3 1 0 1.3 0 0</DataItem>
                    </Geometry>
                    <Topology TopologyType="Quadrilateral" NumberOfElements="2">
                        <DataItem Dimensions="8" NumberType="Int" Format="XML" Precision="4">0 1 4 3 1 2 5 4</DataItem>
                    </Topology>
                </Grid>
            </Grid>
            <Grid Name="Grid_1" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 0 0 0 1 0 1 1 0 1 0 0 0.5 1.5 0.5</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="2">
                    <DataItem Dimensions="9" NumberType="Int" Format="XML" Precision="4">5 0 1 2 3 4 1 2 4</DataItem>
                </Topology>
            </Grid>
        </Grid>
    </Domain>
</Xdmf>"#;

    pretty_assertions::assert_eq!(String::from_utf8(buffer).unwrap(), expected_xdmf);

    // write to file
    // let mut file_writer = Writer::new_with_indent(
    //     std::fs::File::create("hierarchical_tree_grid.xdmf").unwrap(),
    //     b' ',
    //     4,
    // );
    // file_writer.write_serializable(XDMF_TAG, &xdmf).unwrap();
}
#[test]
fn mixed_grid() {
    let xdmf = Xdmf::new(Domain::new(Grid::new_uniform(
        "Grid_1",
        Geometry {
            geometry_type: GeometryType::XYZ,
            data_item: DataItem {
                dimensions: Some(Dimensions(vec![5, 3])),
                data: "0 0 0 0 1 0 1 1 0 1 0 0 0.5 1.5 0.5".into(),
                number_type: Some(NumberType::Float),
                ..Default::default()
            },
        },
        Topology {
            topology_type: TopologyType::Mixed,
            number_of_elements: "2".into(),
            data_item: DataItem {
                dimensions: Some(Dimensions(vec![9])),
                number_type: Some(NumberType::Int),
                data: "5 0 1 2 3 4 1 2 4".into(),
                ..Default::default()
            },
        },
    )));

    // Create an in-memory buffer to serialize to
    let mut buffer = Vec::new();

    // Create quick_xml writer with indentation (pretty print)
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    writer.write_serializable(XDMF_TAG, &xdmf).unwrap();

    let expected_xdmf = r#"
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="Grid_1" GridType="Uniform">
            <Geometry GeometryType="XYZ">
                <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 0 0 0 1 0 1 1 0 1 0 0 0.5 1.5 0.5</DataItem>
            </Geometry>
            <Topology TopologyType="Mixed" NumberOfElements="2">
                <DataItem Dimensions="9" NumberType="Int" Format="XML" Precision="4">5 0 1 2 3 4 1 2 4</DataItem>
            </Topology>
        </Grid>
    </Domain>
</Xdmf>"#;

    pretty_assertions::assert_eq!(String::from_utf8(buffer).unwrap(), expected_xdmf);

    // write to file
    // let mut file_writer =
    //     Writer::new_with_indent(std::fs::File::create("mixed_grid.xdmf").unwrap(), b' ', 4);
    // file_writer.write_serializable(XDMF_TAG, &xdmf).unwrap();
}
#[test]
fn spatial_collection_grid() {
    let xdmf = Xdmf::new(Domain::new(Grid::new_collection(
        "spatial_collection_grid",
        CollectionType::Spatial,
        Some(vec![
            Grid::new_uniform(
                "sub_grid_1",
                Geometry {
                    geometry_type: GeometryType::XYZ,
                    data_item: DataItem {
                        dimensions: Some(Dimensions(vec![5, 3])),
                        data: "0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0".into(),
                        number_type: Some(NumberType::Float),
                        ..Default::default()
                    },
                },
                Topology {
                    topology_type: TopologyType::Triangle,
                    number_of_elements: "2".into(),
                    data_item: DataItem {
                        dimensions: Some(Dimensions(vec![6])),
                        number_type: Some(NumberType::Int),
                        data: "0 1 2 2 3 4".into(),
                        ..Default::default()
                    },
                },
            ),
            Grid::new_uniform(
                "sub_grid_2",
                Geometry {
                    geometry_type: GeometryType::XYZ,
                    data_item: DataItem {
                        dimensions: Some(Dimensions(vec![6, 3])),
                        data: "1 1.5 0 1 1 0 1 0 0 1.3 1.5 0 1.3 1 0 1.3 0 0".into(),
                        number_type: Some(NumberType::Float),
                        ..Default::default()
                    },
                },
                Topology {
                    topology_type: TopologyType::Quadrilateral,
                    number_of_elements: "2".into(),
                    data_item: DataItem {
                        dimensions: Some(Dimensions(vec![8])),
                        number_type: Some(NumberType::Int),
                        data: "0 1 4 3 1 2 5 4".into(),
                        ..Default::default()
                    },
                },
            ),
            Grid::new_uniform(
                "Grid_1",
                Geometry {
                    geometry_type: GeometryType::XYZ,
                    data_item: DataItem {
                        dimensions: Some(Dimensions(vec![5, 3])),
                        data: "0 0 0 0 1 0 1 1 0 1 0 0 0.5 1.5 0.5".into(),
                        number_type: Some(NumberType::Float),
                        ..Default::default()
                    },
                },
                Topology {
                    topology_type: TopologyType::Mixed,
                    number_of_elements: "2".into(),
                    data_item: DataItem {
                        dimensions: Some(Dimensions(vec![9])),
                        number_type: Some(NumberType::Int),
                        data: "5 0 1 2 3 4 1 2 4".into(),
                        ..Default::default()
                    },
                },
            ),
        ]),
    )));

    // Create an in-memory buffer to serialize to
    let mut buffer = Vec::new();

    // Create quick_xml writer with indentation (pretty print)
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    writer.write_serializable(XDMF_TAG, &xdmf).unwrap();

    let expected_xdmf = r#"
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="spatial_collection_grid" GridType="Collection" CollectionType="Spatial">
            <Grid Name="sub_grid_1" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="2">
                    <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                </Topology>
            </Grid>
            <Grid Name="sub_grid_2" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="6 3" NumberType="Float" Format="XML" Precision="4">1 1.5 0 1 1 0 1 0 0 1.3 1.5 0 1.3 1 0 1.3 0 0</DataItem>
                </Geometry>
                <Topology TopologyType="Quadrilateral" NumberOfElements="2">
                    <DataItem Dimensions="8" NumberType="Int" Format="XML" Precision="4">0 1 4 3 1 2 5 4</DataItem>
                </Topology>
            </Grid>
            <Grid Name="Grid_1" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 0 0 0 1 0 1 1 0 1 0 0 0.5 1.5 0.5</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="2">
                    <DataItem Dimensions="9" NumberType="Int" Format="XML" Precision="4">5 0 1 2 3 4 1 2 4</DataItem>
                </Topology>
            </Grid>
        </Grid>
    </Domain>
</Xdmf>"#;

    pretty_assertions::assert_eq!(String::from_utf8(buffer).unwrap(), expected_xdmf);

    // write to file
    // let mut file_writer = Writer::new_with_indent(
    //     std::fs::File::create("spatial_collection_grid.xdmf").unwrap(),
    //     b' ',
    //     4,
    // );
    // file_writer.write_serializable(XDMF_TAG, &xdmf).unwrap();
}
#[test]
fn temporal_collection_grid() {
    let data_items = vec![
        DataItem {
            name: Some("coords".into()),
            dimensions: Some(Dimensions(vec![5, 3])),
            data: "0 0 0 0 1 0 1 1 0 1 0 0 0.5 1.5 0.5".into(),
            number_type: Some(NumberType::Float),
            ..Default::default()
        },
        DataItem {
            name: Some("connectivity".into()),
            dimensions: Some(Dimensions(vec![9])),
            number_type: Some(NumberType::Int),
            data: "5 0 1 2 3 4 1 2 4".into(),
            ..Default::default()
        },
    ];

    let xdmf = Xdmf::new(Domain {
        grids: vec![Grid::new_collection(
            "temporal_collection_grid",
            CollectionType::Temporal,
            Some(vec![
                Grid::Uniform(Uniform {
                    name: "Grid_t1".into(),
                    geometry: Geometry {
                        geometry_type: GeometryType::XYZ,
                        data_item: DataItem::new_reference(
                            &data_items[0],
                            "/Xdmf/Domain/DataItem".to_string(),
                        ),
                    },
                    topology: Topology {
                        topology_type: TopologyType::Mixed,
                        number_of_elements: "2".into(),
                        data_item: DataItem::new_reference(
                            &data_items[1],
                            "/Xdmf/Domain/DataItem".to_string(),
                        ),
                    },
                    grid_type: xdmf::xdmf_elements::grid::GridType::Uniform,
                    time: Some(Time {
                        value: "1.0".into(),
                    }),
                    attributes: Some(vec![
                        Attribute {
                            name: String::from("Pressure"),
                            attribute_type: AttributeType::Scalar,
                            center: Center::Node,
                            data_items: vec![DataItem {
                                dimensions: Some(Dimensions(vec![5])),
                                data: "1 2 2 3 9".into(),
                                number_type: Some(NumberType::Float),
                                ..Default::default()
                            }],
                        },
                        Attribute {
                            name: String::from("Temperature"),
                            attribute_type: AttributeType::Scalar,
                            center: Center::Cell,
                            data_items: vec![DataItem {
                                dimensions: Some(Dimensions(vec![2])),
                                data: "1 2".into(),
                                number_type: Some(NumberType::Float),
                                ..Default::default()
                            }],
                        },
                    ]),
                }),
                Grid::Uniform(Uniform {
                    name: "Grid_t2".into(),
                    geometry: Geometry {
                        geometry_type: GeometryType::XYZ,
                        data_item: DataItem::new_reference(
                            &data_items[0],
                            "/Xdmf/Domain/DataItem".to_string(),
                        ),
                    },
                    topology: Topology {
                        topology_type: TopologyType::Mixed,
                        number_of_elements: "2".into(),
                        data_item: DataItem::new_reference(
                            &data_items[1],
                            "/Xdmf/Domain/DataItem".to_string(),
                        ),
                    },
                    grid_type: xdmf::xdmf_elements::grid::GridType::Uniform,
                    time: Some(Time {
                        value: "2.0".into(),
                    }),
                    attributes: Some(vec![
                        Attribute {
                            name: String::from("Pressure"),
                            attribute_type: AttributeType::Scalar,
                            center: Center::Node,
                            data_items: vec![DataItem {
                                dimensions: Some(Dimensions(vec![5])),
                                data: "1 2 3 4 7".into(),
                                number_type: Some(NumberType::Float),
                                ..Default::default()
                            }],
                        },
                        Attribute {
                            name: String::from("Temperature"),
                            attribute_type: AttributeType::Scalar,
                            center: Center::Cell,
                            data_items: vec![DataItem {
                                dimensions: Some(Dimensions(vec![2])),
                                data: "2 3".into(),
                                number_type: Some(NumberType::Float),
                                ..Default::default()
                            }],
                        },
                    ]),
                }),
                Grid::Uniform(Uniform {
                    name: "Grid_t3".into(),
                    geometry: Geometry {
                        geometry_type: GeometryType::XYZ,
                        data_item: DataItem::new_reference(
                            &data_items[0],
                            "/Xdmf/Domain/DataItem".to_string(),
                        ),
                    },
                    topology: Topology {
                        topology_type: TopologyType::Mixed,
                        number_of_elements: "2".into(),
                        data_item: DataItem::new_reference(
                            &data_items[1],
                            "/Xdmf/Domain/DataItem".to_string(),
                        ),
                    },
                    grid_type: xdmf::xdmf_elements::grid::GridType::Uniform,
                    time: Some(Time {
                        value: "3.0".into(),
                    }),
                    attributes: Some(vec![
                        Attribute {
                            name: String::from("Pressure"),
                            attribute_type: AttributeType::Scalar,
                            center: Center::Node,
                            data_items: vec![DataItem {
                                dimensions: Some(Dimensions(vec![5])),
                                data: "3 2 2 3 8".into(),
                                number_type: Some(NumberType::Float),
                                ..Default::default()
                            }],
                        },
                        Attribute {
                            name: String::from("Temperature"),
                            attribute_type: AttributeType::Scalar,
                            center: Center::Cell,
                            data_items: vec![DataItem {
                                dimensions: Some(Dimensions(vec![2])),
                                data: "3 4".into(),
                                number_type: Some(NumberType::Float),
                                ..Default::default()
                            }],
                        },
                    ]),
                }),
            ]),
        )],
        data_items,
    });

    // Create an in-memory buffer to serialize to
    let mut buffer = Vec::new();

    // Create quick_xml writer with indentation (pretty print)
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    writer.write_serializable(XDMF_TAG, &xdmf).unwrap();

    let expected_xdmf = r#"
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="temporal_collection_grid" GridType="Collection" CollectionType="Temporal">
            <Grid Name="Grid_t1" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="2">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="1.0"/>
                <Attribute Name="Pressure" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="5" NumberType="Float" Format="XML" Precision="4">1 2 2 3 9</DataItem>
                </Attribute>
                <Attribute Name="Temperature" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="4">1 2</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="Grid_t2" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="2">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="2.0"/>
                <Attribute Name="Pressure" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="5" NumberType="Float" Format="XML" Precision="4">1 2 3 4 7</DataItem>
                </Attribute>
                <Attribute Name="Temperature" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="4">2 3</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="Grid_t3" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="2">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="3.0"/>
                <Attribute Name="Pressure" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="5" NumberType="Float" Format="XML" Precision="4">3 2 2 3 8</DataItem>
                </Attribute>
                <Attribute Name="Temperature" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="4">3 4</DataItem>
                </Attribute>
            </Grid>
        </Grid>
        <DataItem Name="coords" Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 0 0 0 1 0 1 1 0 1 0 0 0.5 1.5 0.5</DataItem>
        <DataItem Name="connectivity" Dimensions="9" NumberType="Int" Format="XML" Precision="4">5 0 1 2 3 4 1 2 4</DataItem>
    </Domain>
</Xdmf>"#;

    pretty_assertions::assert_eq!(String::from_utf8(buffer).unwrap(), expected_xdmf);

    // write to file
    // let mut file_writer = Writer::new_with_indent(
    //     std::fs::File::create("temporal_collection_grid.xdmf").unwrap(),
    //     b' ',
    //     4,
    // );
    // file_writer.write_serializable(XDMF_TAG, &xdmf).unwrap();
}

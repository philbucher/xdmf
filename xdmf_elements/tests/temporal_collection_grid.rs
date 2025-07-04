use std::vec;

use quick_xml::Writer;

use xdmf_elements::attribute::{Attribute, AttributeType, Center};
use xdmf_elements::data_item::{DataItem, NumberType};
use xdmf_elements::dimensions::Dimensions;
use xdmf_elements::geometry::{Geometry, GeometryType};
use xdmf_elements::grid::{CollectionType, Grid, Time, Uniform};
use xdmf_elements::topology::{Topology, TopologyType};
use xdmf_elements::{Domain, XDMF_TAG, Xdmf};

// this can only be opened with the "XDMF Reader" in Paraview, not the "Xdmf3ReaderS" and "Xdmf3ReaderT" readers (reason unknown)

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
                    grid_type: xdmf_elements::grid::GridType::Uniform,
                    indices: None,
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
                    grid_type: xdmf_elements::grid::GridType::Uniform,
                    indices: None,
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
                    grid_type: xdmf_elements::grid::GridType::Uniform,
                    indices: None,
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

    assert_eq!(String::from_utf8(buffer).unwrap(), expected_xdmf);

    // write to file
    // let mut file_writer = Writer::new_with_indent(
    //     std::fs::File::create("temporal_collection_grid.xdmf").unwrap(),
    //     b' ',
    //     4,
    // );
    // file_writer.write_serializable(XDMF_TAG, &xdmf).unwrap();
}

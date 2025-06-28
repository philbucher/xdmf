use quick_xml::Writer;

use xdmf_elements::data_item::{DataItem, NumberType};
use xdmf_elements::dimensions::Dimensions;
use xdmf_elements::geometry::{Geometry, GeometryType};
use xdmf_elements::grid::Grid;
use xdmf_elements::topology::{Topology, TopologyType};
use xdmf_elements::{Domain, XDMF_TAG, Xdmf};

// this can only be opened with the "XDMF Reader" in Paraview, not the "Xdmf3ReaderS" and "Xdmf3ReaderT" readers (reason unknown)

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

    assert_eq!(String::from_utf8(buffer).unwrap(), expected_xdmf);

    // write to file
    // let mut file_writer = Writer::new_with_indent(
    //     std::fs::File::create("hierarchical_tree_grid.xdmf").unwrap(),
    //     b' ',
    //     4,
    // );
    // file_writer.write_serializable(XDMF_TAG, &xdmf).unwrap();
}

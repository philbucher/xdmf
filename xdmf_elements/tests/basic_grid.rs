use quick_xml::Writer;

use xdmf_elements::data_item::{DataItem, NumberType};
use xdmf_elements::dimensions::Dimensions;
use xdmf_elements::geometry::{Geometry, GeometryType};
use xdmf_elements::grid::Grid;
use xdmf_elements::topology::{Topology, TopologyType};
use xdmf_elements::{Domain, XDMF_TAG, Xdmf};

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

    assert_eq!(String::from_utf8(buffer).unwrap(), expected_xdmf);

    // write to file
    // let mut file_writer =
    //     Writer::new_with_indent(std::fs::File::create("basic_grid.xdmf").unwrap(), b' ', 4);
    // file_writer.write_serializable(XDMF_TAG, &xdmf).unwrap();
}

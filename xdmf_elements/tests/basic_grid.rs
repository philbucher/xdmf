use quick_xml::Writer;

use xdmf_elements::Domain;
use xdmf_elements::Xdmf;
use xdmf_elements::data_item::{DataItem, NumberType};
use xdmf_elements::dimensions::Dimensions;
use xdmf_elements::geometry::{Geometry, GeometryType};
use xdmf_elements::grid::Grid;
use xdmf_elements::topology::{Topology, TopologyType};

#[test]
fn basic_grid() {
    let xdmf = Xdmf::new(Domain {
        grid: Grid::new_uniform(
            "Grid_1",
            Topology {
                topology_type: TopologyType::Triangle,
                number_of_elements: "2".into(),
                data_item: DataItem {
                    dimensions: Dimensions(vec![6]),
                    number_type: NumberType::Int,
                    data: "0 1 2 0 2 3".into(),
                    ..Default::default()
                },
            },
            Geometry {
                geometry_type: GeometryType::XYZ,
                data_item: DataItem {
                    dimensions: Dimensions(vec![3, 2]),
                    data: "0 0 0 0 1 0 1 1 0 1 0 0.5".into(),
                    number_type: NumberType::Float,
                    ..Default::default()
                },
            },
        ),
    });
    // Create an in-memory buffer (stdout here, but could be a file or Vec)
    let out_file = std::fs::File::create("basic_grid.xdmf").unwrap();

    // Create quick_xml writer with indentation (pretty print)
    let mut writer = Writer::new_with_indent(out_file, b' ', 4);

    writer.write_serializable("Xdmf", &xdmf).unwrap();

    let expected_xdmf = r#"
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="Grid_1" GridType="Uniform">
            <Geometry GeometryType="XYZ">
                <DataItem Dimensions="3 2" NumberType="Float" Format="XML" Precision="4">0 0 0 0 1 0 1 1 0 1 0 0.5</DataItem>
            </Geometry>
            <Topology TopologyType="Triangle" NumberOfElements="2">
                <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 0 2 3</DataItem>
            </Topology>
        </Grid>
    </Domain>
</Xdmf>"#;

    let output = std::fs::read_to_string("basic_grid.xdmf").unwrap();
    assert_eq!(output, expected_xdmf);
}

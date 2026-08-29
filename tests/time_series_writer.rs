use temp_dir::TempDir;
use xdmf::TimeSeriesWriter;

fn with_version(expected: &str) -> String {
    expected.replace("$VERSION", env!("CARGO_PKG_VERSION"))
}

#[test]
fn write_xdmf() {
    let node_coords = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0,
        0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.5, -0.5, 0.2, -0.5, 0.5, 0.2, 1.5, -0.5,
        0.2, 2.5, 0.5, 0.2, 0.5, 1.5, 0.2, 0.5, 2.5, 0.2, 1.5, 2.5, 0.2, 2.5, 1.5, 0.2,
    ];

    let connectivity = [
        0_u64, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7, 0, 1, 9, 3, 0, 10, 1, 2, 11, 2, 5, 12,
        6, 3, 13, 6, 7, 14, 7, 8, 15, 5, 8, 16,
    ];

    let cell_types = [
        xdmf::CellType::Quadrilateral,
        xdmf::CellType::Quadrilateral,
        xdmf::CellType::Quadrilateral,
        xdmf::CellType::Quadrilateral,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
    ];

    let num_nodes = node_coords.len() / 3;
    let num_cells = cell_types.len();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    let mut xdmf_writer = xdmf_writer
        .write_mesh(&node_coords, &connectivity, &cell_types)
        .unwrap();

    for i in 0..3 {
        let point_data_scalar: Vec<f64> = (0..num_nodes).map(|j| j as f64 + i as f64).collect();
        let point_data_vec: Vec<f64> = (0..num_nodes * 3).map(|j| (j % 3) as f64).collect();
        let point_data_tensor: Vec<f64> = (0..num_nodes * 9).map(|j| (j % 9) as f64).collect();
        let point_data_tensor6: Vec<f64> = (0..num_nodes * 6).map(|j| (j % 6) as f64).collect();
        let point_data_generic: Vec<f64> = (0..num_nodes * 5).map(|j| (j % 5) as f64).collect();
        let point_data_matrix2x2: Vec<f64> = (0..num_nodes * 4).map(|j| (j % 4) as f64).collect();

        let cell_data: Vec<f64> = (0..num_cells)
            .map(|j| 1. * j as f64 + 1.5 * i as f64)
            .collect();

        // deliberately not in alphabetical order: attributes must come out in the order written
        xdmf_writer
            .write_time_step(&i.to_string(), |step| {
                step.point_data(
                    "point_data_scalar",
                    xdmf::DataAttribute::Scalar,
                    &point_data_scalar,
                )?;
                step.point_data(
                    "point_data_vector",
                    xdmf::DataAttribute::Vector,
                    &point_data_vec,
                )?;
                step.point_data(
                    "point_data_tensor",
                    xdmf::DataAttribute::Tensor,
                    &point_data_tensor,
                )?;
                step.point_data(
                    "point_data_tensor6",
                    xdmf::DataAttribute::Tensor6,
                    &point_data_tensor6,
                )?;
                step.point_data(
                    "point_data_matrix_2x2",
                    xdmf::DataAttribute::Matrix(2, 2),
                    &point_data_matrix2x2,
                )?;
                step.point_data(
                    "point_data_generic-5",
                    xdmf::DataAttribute::Generic(5),
                    &point_data_generic,
                )?;

                step.cell_data("cell_data", xdmf::DataAttribute::Scalar, &cell_data)
            })
            .unwrap();
    }

    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">
            <Grid Name="time_series-t0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="12">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="0"/>
                <Attribute Name="point_data_scalar" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 9e0 1e1 1.1e1 1.2e1 1.3e1 1.4e1 1.5e1 1.6e1</DataItem>
                </Attribute>
                <Attribute Name="point_data_vector" AttributeType="Vector" Center="Node">
                    <DataItem Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_tensor" AttributeType="Tensor" Center="Node">
                    <DataItem Dimensions="17 9" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_tensor6" AttributeType="Matrix" Center="Node">
                    <DataItem Dimensions="17 6 1" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_matrix_2x2" AttributeType="Matrix" Center="Node">
                    <DataItem Dimensions="17 4 1" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_generic-5" AttributeType="Matrix" Center="Node">
                    <DataItem Dimensions="17 5 1" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0</DataItem>
                </Attribute>
                <Attribute Name="cell_data" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="12" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 9e0 1e1 1.1e1</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t1" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="12">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="1"/>
                <Attribute Name="point_data_scalar" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 9e0 1e1 1.1e1 1.2e1 1.3e1 1.4e1 1.5e1 1.6e1 1.7e1</DataItem>
                </Attribute>
                <Attribute Name="point_data_vector" AttributeType="Vector" Center="Node">
                    <DataItem Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_tensor" AttributeType="Tensor" Center="Node">
                    <DataItem Dimensions="17 9" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_tensor6" AttributeType="Matrix" Center="Node">
                    <DataItem Dimensions="17 6 1" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_matrix_2x2" AttributeType="Matrix" Center="Node">
                    <DataItem Dimensions="17 4 1" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_generic-5" AttributeType="Matrix" Center="Node">
                    <DataItem Dimensions="17 5 1" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0</DataItem>
                </Attribute>
                <Attribute Name="cell_data" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="12" NumberType="Float" Format="XML" Precision="8">1.5e0 2.5e0 3.5e0 4.5e0 5.5e0 6.5e0 7.5e0 8.5e0 9.5e0 1.05e1 1.15e1 1.25e1</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t2" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="12">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="2"/>
                <Attribute Name="point_data_scalar" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">2e0 3e0 4e0 5e0 6e0 7e0 8e0 9e0 1e1 1.1e1 1.2e1 1.3e1 1.4e1 1.5e1 1.6e1 1.7e1 1.8e1</DataItem>
                </Attribute>
                <Attribute Name="point_data_vector" AttributeType="Vector" Center="Node">
                    <DataItem Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0 0e0 1e0 2e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_tensor" AttributeType="Tensor" Center="Node">
                    <DataItem Dimensions="17 9" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_tensor6" AttributeType="Matrix" Center="Node">
                    <DataItem Dimensions="17 6 1" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0 0e0 1e0 2e0 3e0 4e0 5e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_matrix_2x2" AttributeType="Matrix" Center="Node">
                    <DataItem Dimensions="17 4 1" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0 0e0 1e0 2e0 3e0</DataItem>
                </Attribute>
                <Attribute Name="point_data_generic-5" AttributeType="Matrix" Center="Node">
                    <DataItem Dimensions="17 5 1" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0 0e0 1e0 2e0 3e0 4e0</DataItem>
                </Attribute>
                <Attribute Name="cell_data" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="12" NumberType="Float" Format="XML" Precision="8">3e0 4e0 5e0 6e0 7e0 8e0 9e0 1e1 1.1e1 1.2e1 1.3e1 1.4e1</DataItem>
                </Attribute>
            </Grid>
        </Grid>
        <DataItem Name="coords" Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0 2e0 0e0 0e0 0e0 1e0 0e0 1e0 1e0 0e0 2e0 1e0 0e0 0e0 2e0 0e0 1e0 2e0 0e0 2e0 2e0 0e0 5e-1 -5e-1 2e-1 -5e-1 5e-1 2e-1 1.5e0 -5e-1 2e-1 2.5e0 5e-1 2e-1 5e-1 1.5e0 2e-1 5e-1 2.5e0 2e-1 1.5e0 2.5e0 2e-1 2.5e0 1.5e0 2e-1</DataItem>
        <DataItem Name="connectivity" Dimensions="52" NumberType="UInt" Format="XML" Precision="8">5 0 1 4 3 5 1 2 5 4 5 3 4 7 6 5 4 5 8 7 4 0 1 9 4 3 0 10 4 1 2 11 4 2 5 12 4 6 3 13 4 6 7 14 4 7 8 15 4 5 8 16</DataItem>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="$VERSION"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    //  std::fs::copy(xdmf_file, "time_series_writer.xdmf2").unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

/// `file_name` names the XDMF file of the series, both before the mesh goes in and after, and
/// replaces an XDMF extension the caller spelled out rather than doubling it.
#[test]
fn the_file_name_is_the_xdmf_file_that_is_written() {
    let tmp_dir = TempDir::new().unwrap();

    for (given, base) in [("plain", "plain"), ("spelled.xdmf2", "spelled")] {
        let writer =
            TimeSeriesWriter::new(tmp_dir.path().join(given), xdmf::DataStorage::AsciiInline)
                .unwrap();

        let xdmf_file = tmp_dir.path().join(format!("{base}.xdmf2"));
        assert_eq!(writer.file_name(), xdmf_file);

        let writer = writer
            .write_mesh(&[0.0, 0.0, 0.0], &[] as &[u32], &[])
            .unwrap();

        assert_eq!(writer.file_name(), xdmf_file);
        assert!(
            xdmf_file.exists(),
            "{} was not written",
            xdmf_file.display()
        );
    }
}

#[test]
fn write_xdmf_only_mesh() {
    let node_coords = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0,
        0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.5, -0.5, 0.2, -0.5, 0.5, 0.2, 1.5, -0.5,
        0.2, 2.5, 0.5, 0.2, 0.5, 1.5, 0.2, 0.5, 2.5, 0.2, 1.5, 2.5, 0.2, 2.5, 1.5, 0.2,
    ];

    let connectivity = [
        0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7, 0, 1, 9, 3, 0, 10, 1, 2, 11, 2, 5, 12, 6,
        3, 13, 6, 7, 14, 7, 8, 15, 5, 8, 16,
    ];

    let cell_types = [
        xdmf::CellType::Quadrilateral,
        xdmf::CellType::Quadrilateral,
        xdmf::CellType::Quadrilateral,
        xdmf::CellType::Quadrilateral,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
    ];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    xdmf_writer
        .write_mesh(&node_coords, &connectivity, &cell_types)
        .unwrap();

    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="mesh" GridType="Uniform">
            <Geometry GeometryType="XYZ">
                <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
            </Geometry>
            <Topology TopologyType="Mixed" NumberOfElements="12">
                <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
            </Topology>
        </Grid>
        <DataItem Name="coords" Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0 2e0 0e0 0e0 0e0 1e0 0e0 1e0 1e0 0e0 2e0 1e0 0e0 0e0 2e0 0e0 1e0 2e0 0e0 2e0 2e0 0e0 5e-1 -5e-1 2e-1 -5e-1 5e-1 2e-1 1.5e0 -5e-1 2e-1 2.5e0 5e-1 2e-1 5e-1 1.5e0 2e-1 5e-1 2.5e0 2e-1 1.5e0 2.5e0 2e-1 2.5e0 1.5e0 2e-1</DataItem>
        <DataItem Name="connectivity" Dimensions="52" NumberType="Int" Format="XML" Precision="4">5 0 1 4 3 5 1 2 5 4 5 3 4 7 6 5 4 5 8 7 4 0 1 9 4 3 0 10 4 1 2 11 4 2 5 12 4 6 3 13 4 6 7 14 4 7 8 15 4 5 8 16</DataItem>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="$VERSION"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "time_series_writer_only_mesh.xdmf").unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

#[test]
fn write_xdmf_only_point_mesh() {
    let node_coords = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0,
        0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.5, -0.5, 0.2, -0.5, 0.5, 0.2, 1.5, -0.5,
        0.2, 2.5, 0.5, 0.2, 0.5, 1.5, 0.2, 0.5, 2.5, 0.2, 1.5, 2.5, 0.2, 2.5, 1.5, 0.2,
    ];

    let connectivity: [u64; 0] = [];

    let cell_types = [];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    xdmf_writer
        .write_mesh(&node_coords, &connectivity, &cell_types)
        .unwrap();

    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="mesh" GridType="Uniform">
            <Geometry GeometryType="XYZ">
                <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
            </Geometry>
            <Topology TopologyType="Polyvertex" NodesPerElement="1" NumberOfElements="17">
                <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
            </Topology>
        </Grid>
        <DataItem Name="coords" Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0 2e0 0e0 0e0 0e0 1e0 0e0 1e0 1e0 0e0 2e0 1e0 0e0 0e0 2e0 0e0 1e0 2e0 0e0 2e0 2e0 0e0 5e-1 -5e-1 2e-1 -5e-1 5e-1 2e-1 1.5e0 -5e-1 2e-1 2.5e0 5e-1 2e-1 5e-1 1.5e0 2e-1 5e-1 2.5e0 2e-1 1.5e0 2.5e0 2e-1 2.5e0 1.5e0 2e-1</DataItem>
        <DataItem Name="connectivity" Dimensions="17" NumberType="UInt" Format="XML" Precision="8">0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16</DataItem>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="$VERSION"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "time_series_writer_only_point_mesh.xdmf2").unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

#[test]
fn write_xdmf_point_mesh() {
    let node_coords = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0,
        0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.5, -0.5, 0.2, -0.5, 0.5, 0.2, 1.5, -0.5,
        0.2, 2.5, 0.5, 0.2, 0.5, 1.5, 0.2, 0.5, 2.5, 0.2, 1.5, 2.5, 0.2, 2.5, 1.5, 0.2,
    ];

    let connectivity: [u64; 0] = [];

    let cell_types = [];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    let mut xdmf_writer = xdmf_writer
        .write_mesh(&node_coords, &connectivity, &cell_types)
        .unwrap();

    for i in 0..3 {
        let point_data_scalar: Vec<f64> = (0..17).map(|j| j as f64 + i as f64).collect();

        xdmf_writer
            .write_time_step(&i.to_string(), |step| {
                step.point_data(
                    "point_data_scalar",
                    xdmf::DataAttribute::Scalar,
                    &point_data_scalar,
                )
            })
            .unwrap();
    }

    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">
            <Grid Name="time_series-t0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Polyvertex" NodesPerElement="1" NumberOfElements="17">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="0"/>
                <Attribute Name="point_data_scalar" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">0e0 1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 9e0 1e1 1.1e1 1.2e1 1.3e1 1.4e1 1.5e1 1.6e1</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t1" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Polyvertex" NodesPerElement="1" NumberOfElements="17">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="1"/>
                <Attribute Name="point_data_scalar" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">1e0 2e0 3e0 4e0 5e0 6e0 7e0 8e0 9e0 1e1 1.1e1 1.2e1 1.3e1 1.4e1 1.5e1 1.6e1 1.7e1</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t2" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Polyvertex" NodesPerElement="1" NumberOfElements="17">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="2"/>
                <Attribute Name="point_data_scalar" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">2e0 3e0 4e0 5e0 6e0 7e0 8e0 9e0 1e1 1.1e1 1.2e1 1.3e1 1.4e1 1.5e1 1.6e1 1.7e1 1.8e1</DataItem>
                </Attribute>
            </Grid>
        </Grid>
        <DataItem Name="coords" Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0 2e0 0e0 0e0 0e0 1e0 0e0 1e0 1e0 0e0 2e0 1e0 0e0 0e0 2e0 0e0 1e0 2e0 0e0 2e0 2e0 0e0 5e-1 -5e-1 2e-1 -5e-1 5e-1 2e-1 1.5e0 -5e-1 2e-1 2.5e0 5e-1 2e-1 5e-1 1.5e0 2e-1 5e-1 2.5e0 2e-1 1.5e0 2.5e0 2e-1 2.5e0 1.5e0 2e-1</DataItem>
        <DataItem Name="connectivity" Dimensions="17" NumberType="UInt" Format="XML" Precision="8">0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16</DataItem>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="$VERSION"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "write_xdmf_point_mesh.xdmf2").unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

#[test]
fn write_xdmf_f32_points() {
    // 3 points forming a triangle, held as f32 by the caller
    let node_coords: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 2];
    let cell_types = [xdmf::CellType::Triangle];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    let mut xdmf_writer = xdmf_writer
        .write_mesh(&node_coords, &connectivity, &cell_types)
        .unwrap();

    // the coordinate type says nothing about the attribute types: both widths in the same step
    xdmf_writer
        .write_time_step("0", |step| {
            step.point_data(
                "temperature_f32",
                xdmf::DataAttribute::Scalar,
                &[10.5_f32, 11.5, 12.5],
            )?;
            step.point_data(
                "temperature_f64",
                xdmf::DataAttribute::Scalar,
                &[10.5_f64, 11.5, 12.5],
            )
        })
        .unwrap();

    // the f32 coordinates and the f32 attribute are declared as 4-byte floats and written with
    // f32's digit count, while the f64 attribute in the same step keeps 8 bytes
    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">
            <Grid Name="time_series-t0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="1">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="0"/>
                <Attribute Name="temperature_f32" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="3" NumberType="Float" Format="XML" Precision="4">1.05e1 1.15e1 1.25e1</DataItem>
                </Attribute>
                <Attribute Name="temperature_f64" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="3" NumberType="Float" Format="XML" Precision="8">1.05e1 1.15e1 1.25e1</DataItem>
                </Attribute>
            </Grid>
        </Grid>
        <DataItem Name="coords" Dimensions="3 3" NumberType="Float" Format="XML" Precision="4">0e0 0e0 0e0 1e0 0e0 0e0 0e0 1e0 0e0</DataItem>
        <DataItem Name="connectivity" Dimensions="3" NumberType="UInt" Format="XML" Precision="8">0 1 2</DataItem>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="$VERSION"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "write_xdmf_f32_points.xdmf2").unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

#[test]
fn write_xdmf_integer_data() {
    // 3 points forming a triangle
    let node_coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 2];
    let cell_types = [xdmf::CellType::Triangle];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    let mut xdmf_writer = xdmf_writer
        .write_mesh(&node_coords, &connectivity, &cell_types)
        .unwrap();

    // all four integer widths in the same step; signed data may be negative
    xdmf_writer
        .write_time_step("0", |step| {
            step.point_data("rank_u64", xdmf::DataAttribute::Scalar, &[1_u64, 2, 3])?;
            step.point_data("rank_u32", xdmf::DataAttribute::Scalar, &[1_u32, 2, 3])?;
            step.point_data("level_i64", xdmf::DataAttribute::Scalar, &[-1_i64, 0, 1])?;
            step.point_data("level_i32", xdmf::DataAttribute::Scalar, &[-1_i32, 0, 1])
        })
        .unwrap();

    // the signed types are declared as `Int` and the unsigned ones as `UInt`, with the precision
    // following the width the caller handed over
    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">
            <Grid Name="time_series-t0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="1">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="0"/>
                <Attribute Name="rank_u64" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="3" NumberType="UInt" Format="XML" Precision="8">1 2 3</DataItem>
                </Attribute>
                <Attribute Name="rank_u32" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="3" NumberType="UInt" Format="XML" Precision="4">1 2 3</DataItem>
                </Attribute>
                <Attribute Name="level_i64" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="8">-1 0 1</DataItem>
                </Attribute>
                <Attribute Name="level_i32" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">-1 0 1</DataItem>
                </Attribute>
            </Grid>
        </Grid>
        <DataItem Name="coords" Dimensions="3 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0 0e0 1e0 0e0</DataItem>
        <DataItem Name="connectivity" Dimensions="3" NumberType="UInt" Format="XML" Precision="8">0 1 2</DataItem>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="$VERSION"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

// Counterpart to the u64 cap below: this limit is the ascii readers' alone, so the same value that
// the ascii storages refuse must still go through the HDF5 ones, which read i64 at its full width.
#[test]
fn write_data_rejects_i64_beyond_the_double_mantissa_for_ascii_only() {
    const TWO_POW_53: i64 = 1 << 53;

    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    // u32, so that the Binary case below fails on the attribute rather than on the mesh
    let connectivity = [0_u32, 1, 2];
    let cell_types = [xdmf::CellType::Triangle];

    let write = |storage, values: Vec<i64>| {
        let tmp_dir = TempDir::new().unwrap();
        let xdmf_writer =
            TimeSeriesWriter::new(tmp_dir.path().join("test_output"), storage).unwrap();
        let mut xdmf_writer = xdmf_writer
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap();
        // the TempDir has to outlive the write, so the result is produced before it is dropped
        let res = xdmf_writer.write_time_step("0", |step| {
            step.point_data("level", xdmf::DataAttribute::Scalar, values)
        });
        drop(tmp_dir);
        res
    };

    for storage in [xdmf::DataStorage::Ascii, xdmf::DataStorage::AsciiInline] {
        std::assert_matches!(
            write(storage, vec![0, 0, TWO_POW_53 + 1]).unwrap_err(),
            xdmf::Error::IntegerOutOfRange { value, reason }
                if value == i128::from(TWO_POW_53) + 1 && reason.contains("Hdf5"),
            "{storage:?} must reject an i64 past the double mantissa"
        );

        // the boundary itself is exact, so it is still accepted
        write(storage, vec![TWO_POW_53, -TWO_POW_53, 0]).unwrap();
    }

    // the HDF5 storages take the same value, which is what the error points the caller at
    #[cfg(feature = "hdf5")]
    for storage in [
        xdmf::DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
        xdmf::DataStorage::Hdf5MultipleFiles {
            deflate_level: None,
        },
    ] {
        write(storage, vec![i64::MAX, i64::MIN, 0]).unwrap();
    }

    // Binary does not take i64 at any magnitude, so it refuses the type rather than the range
    std::assert_matches!(
        write(xdmf::DataStorage::Binary, vec![0, 0, 0]).unwrap_err(),
        xdmf::Error::InvalidData { reason } if reason.contains("cannot hold i64 data")
    );
}

#[test]
fn write_data_rejects_u64_above_u32_max_for_every_storage() {
    // ParaView decodes NumberType="UInt" data into a 32-bit array whatever Precision the light
    // data declares, so a larger u64 is read back truncated or clamped without any reader error.
    // Unlike the Binary backend's narrowing of i64, no storage avoids this, so all of them reject.
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 2];
    let cell_types = [xdmf::CellType::Triangle];

    // Binary is absent on purpose: it refuses u64 outright, whatever the value, which
    // `write_mesh_rejects_64_bit_connectivity_for_binary` and the binary_writer tests cover
    let storages = [
        xdmf::DataStorage::Ascii,
        xdmf::DataStorage::AsciiInline,
        #[cfg(feature = "hdf5")]
        xdmf::DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
        #[cfg(feature = "hdf5")]
        xdmf::DataStorage::Hdf5MultipleFiles {
            deflate_level: None,
        },
    ];

    for storage in storages {
        let tmp_dir = TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output");

        let xdmf_writer = TimeSeriesWriter::new(&xdmf_file_path, storage).unwrap();
        let mut xdmf_writer = xdmf_writer
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap();

        let res = xdmf_writer.write_time_step("0", |step| {
            step.cell_data(
                "region_id",
                xdmf::DataAttribute::Scalar,
                vec![u64::from(u32::MAX) + 1],
            )
        });
        std::assert_matches!(
            res.unwrap_err(),
            xdmf::Error::IntegerOutOfRange { value, reason }
                if value == i128::from(u32::MAX) + 1 && reason.contains("no DataStorage avoids this"),
            "{storage:?} must reject a u64 above u32::MAX"
        );

        // u32::MAX itself is still fine, and the rejected step left the writer usable
        xdmf_writer
            .write_time_step("1", |step| {
                step.cell_data(
                    "region_id",
                    xdmf::DataAttribute::Scalar,
                    vec![u64::from(u32::MAX)],
                )
            })
            .unwrap();
    }
}

// The connectivity is written as the type it is passed in, so the light data has to declare that
// type -- and, for `i64`, the width the chosen storage can actually hold.
#[test]
fn write_mesh_connectivity_index_types() {
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let cell_types = [xdmf::CellType::Triangle];

    let connectivity_line = |xdmf_file_path: &std::path::Path| {
        std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2"))
            .unwrap()
            .lines()
            // the reference to it comes first, the definition is the one with the data
            .find(|line| line.contains(r#"<DataItem Name="connectivity""#))
            .unwrap()
            .trim()
            .to_string()
    };

    // only the index type differs between the cases, and it cannot be a closure argument
    macro_rules! connectivity_line_for {
        ($storage:expr, $ty:ty) => {{
            let tmp_dir = TempDir::new().unwrap();
            let xdmf_file_path = tmp_dir.path().join("test_output");

            TimeSeriesWriter::new(&xdmf_file_path, $storage)
                .unwrap()
                .write_mesh(&coords, &[0 as $ty, 1, 2], &cell_types)
                .unwrap();

            connectivity_line(&xdmf_file_path)
        }};
    }

    assert_eq!(
        connectivity_line_for!(xdmf::DataStorage::AsciiInline, u32),
        r#"<DataItem Name="connectivity" Dimensions="3" NumberType="UInt" Format="XML" Precision="4">0 1 2</DataItem>"#
    );
    assert_eq!(
        connectivity_line_for!(xdmf::DataStorage::AsciiInline, u64),
        r#"<DataItem Name="connectivity" Dimensions="3" NumberType="UInt" Format="XML" Precision="8">0 1 2</DataItem>"#
    );
    assert_eq!(
        connectivity_line_for!(xdmf::DataStorage::AsciiInline, i32),
        r#"<DataItem Name="connectivity" Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 2</DataItem>"#
    );
    // i64 is the one type that lifts the u32::MAX limit on the mesh size, and it too is written
    // at its own width
    assert_eq!(
        connectivity_line_for!(xdmf::DataStorage::AsciiInline, i64),
        r#"<DataItem Name="connectivity" Dimensions="3" NumberType="Int" Format="XML" Precision="8">0 1 2</DataItem>"#
    );

    // ...while Binary takes the 32-bit types only, since ParaView misreads 64-bit integers there
    assert_eq!(
        connectivity_line_for!(xdmf::DataStorage::Binary, u32),
        r#"<DataItem Name="connectivity" Dimensions="3" NumberType="UInt" Format="Binary" Precision="4" Endian="Little">test_output.bin/cells.bin</DataItem>"#
    );
}

#[test]
fn write_mesh_rejects_64_bit_connectivity_for_binary() {
    // the connectivity is data like any other, so the Binary storage refuses it at 64 bits too --
    // rather than narrowing it and putting a type in the file that the caller did not pass
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let cell_types = [xdmf::CellType::Triangle];

    for expected in ["cannot hold u64 data", "cannot hold i64 data"] {
        let tmp_dir = TempDir::new().unwrap();
        let xdmf_writer = TimeSeriesWriter::new(
            tmp_dir.path().join("test_output"),
            xdmf::DataStorage::Binary,
        )
        .unwrap();

        // `TimeSeriesDataWriter` is not `Debug`, so the error is taken out of the `Result` first
        let error = if expected.starts_with("cannot hold u64") {
            xdmf_writer
                .write_mesh(&coords, &[0_u64, 1, 2], &cell_types)
                .err()
        } else {
            xdmf_writer
                .write_mesh(&coords, &[0_i64, 1, 2], &cell_types)
                .err()
        }
        .unwrap();

        std::assert_matches!(
            error,
            xdmf::Error::InvalidData { reason } if reason.contains(expected),
            "Binary must refuse 64-bit connectivity"
        );
    }
}

#[test]
fn write_mesh_rejects_a_negative_connectivity_index() {
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let cell_types = [xdmf::CellType::Triangle];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_writer = TimeSeriesWriter::new(
        tmp_dir.path().join("test_output"),
        xdmf::DataStorage::AsciiInline,
    )
    .unwrap();

    // `TimeSeriesDataWriter` is not `Debug`, so the error is taken out of the `Result` first
    let error = xdmf_writer
        .write_mesh(&coords, &[0_i64, -2, 1], &cell_types)
        .err()
        .unwrap();

    std::assert_matches!(
        error,
        xdmf::Error::InvalidMesh { reason } if reason == "connectivity index -2 is negative"
    );
}

// A component count or a total number of values that does not fit a `usize`: in release builds
// both multiplications used to wrap, and a total that wraps back onto the real array length was
// accepted -- writing `Dimensions="0 4611686018427387905 1"` for four values.
#[test]
fn write_data_rejects_an_attribute_whose_size_is_zero_or_does_not_fit() {
    let coords = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0_f64,
    ];
    let cell_types = [xdmf::CellType::Quadrilateral];

    let write = |attribute, data: &[f64]| {
        let tmp_dir = TempDir::new().unwrap();
        let mut data_writer = TimeSeriesWriter::new(
            tmp_dir.path().join("test_output"),
            xdmf::DataStorage::AsciiInline,
        )
        .unwrap()
        .write_mesh(&coords, &[0_u64, 1, 2, 3], &cell_types)
        .unwrap();

        data_writer
            .write_time_step("0.0", |step| step.point_data("x", attribute, data))
            .unwrap_err()
    };

    // 4 points * (2^62 + 1) wraps to 4, exactly the number of values passed. The component count
    // itself is fine here, so it is the total that is reported, not the shape
    std::assert_matches!(
        write(xdmf::DataAttribute::Generic(usize::MAX / 4 + 2), &[1.0, 2.0, 3.0, 4.0]),
        xdmf::Error::InvalidData { reason } if reason.contains("whose total does not fit a usize")
    );
    // the component count itself does not fit
    std::assert_matches!(
        write(xdmf::DataAttribute::Matrix(usize::MAX, 2), &[1.0, 2.0, 3.0, 4.0]),
        xdmf::Error::InvalidData { reason } if reason.contains("has no usable size")
    );
    // zero components: an expected size of 0 that only empty data matches, and that divided by
    // zero when the shape was written
    std::assert_matches!(
        write(xdmf::DataAttribute::Generic(0), &[]),
        xdmf::Error::InvalidData { reason } if reason.contains("has no usable size")
    );
}

// A data name is only ever light data: it reaches an XML attribute value and nothing else, which
// is what lets `point_data`/`cell_data` accept characters like `<` and `&`. That rests entirely on
// the serializer escaping them, so it is asserted on the written file rather than taken on trust.
#[test]
fn write_data_escapes_xml_special_characters_in_a_name() {
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0_f64];
    let name = r#"a<b> & "c" 'd'"#;

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let mut data_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline)
        .unwrap()
        .write_mesh(&coords, &[0_u64, 1, 2], &[xdmf::CellType::Triangle])
        .unwrap();

    data_writer
        .write_time_step("0.0", |step| {
            step.point_data(name, xdmf::DataAttribute::Scalar, &[1.0, 2.0, 3.0])
        })
        .unwrap();

    let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
    assert!(
        read_xdmf.contains(
            r#"<Attribute Name="a&lt;b&gt; &amp; &quot;c&quot; 'd'" AttributeType="Scalar" Center="Node">"#
        ),
        "the name was not escaped as expected:\n{read_xdmf}"
    );

    // and a reader gets the name back unchanged, so the file is well-formed and not merely
    // escaped somewhere
    let mut reader = quick_xml::Reader::from_str(&read_xdmf);
    let mut names = Vec::new();
    loop {
        match reader.read_event().unwrap() {
            quick_xml::events::Event::Eof => break,
            quick_xml::events::Event::Start(tag) if tag.name().as_ref() == "Attribute" => {
                let attribute = tag.try_get_attribute("Name").unwrap().unwrap();
                names.push(
                    attribute
                        .normalized_value(quick_xml::XmlVersion::Implicit1_0)
                        .unwrap()
                        .into_owned(),
                );
            }
            _ => {}
        }
    }
    pretty_assertions::assert_eq!(names, vec![name.to_string()]);
}

// A small mixed mesh used by the submesh tests: 4 points, an edge (cell 0) and two triangles
// (cells 1 and 2) sharing it.
fn submesh_test_mesh() -> ([f64; 12], [u32; 8], [xdmf::CellType; 3]) {
    let node_coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let connectivity = [0_u32, 1, 0, 2, 1, 1, 2, 3];
    let cell_types = [
        xdmf::CellType::Edge,
        xdmf::CellType::Triangle,
        xdmf::CellType::Triangle,
    ];

    (node_coords, connectivity, cell_types)
}

#[test]
fn write_xdmf_with_submeshes() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    // "edge" is one cell, "surface" an ascending run, "corner" a scattered pair that overlaps
    // both of them -- so every case the writer distinguishes is present
    let mut ts_writer = xdmf_writer
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [
                ("edge", &[0][..]),
                ("surface", &[1, 2][..]),
                ("corner", &[2, 0][..]),
            ],
        )
        .unwrap();

    // two steps, so the expectation below covers what every step after the first adds: another
    // grid inside each submesh's temporal collection, and another point-data `DataItem`
    for (time, offset) in [("0.5", 0.0), ("1.5", 100.0)] {
        let temperature = [1.0, 2.0, 3.0, 4.0].map(|value: f64| value + offset);
        let material = [10.0, 20.0, 30.0].map(|value: f64| value + offset);

        ts_writer
            .write_time_step(time, |step| {
                step.point_data("temperature", xdmf::DataAttribute::Scalar, &temperature)?;
                step.cell_data("material", xdmf::DataAttribute::Scalar, &material)
            })
            .unwrap();
    }

    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="mesh" GridType="Collection" CollectionType="Spatial">
            <Grid Name="edge" GridType="Collection" CollectionType="Temporal">
                <Grid Name="edge-t0.5" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Polyline" NodesPerElement="2" NumberOfElements="1">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_0"]</DataItem>
                    </Topology>
                    <Time Value="0.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="8">1e0 2e0</DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="1" NumberType="Float" Format="XML" Precision="8">1e1</DataItem>
                    </Attribute>
                </Grid>
                <Grid Name="edge-t1.5" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Polyline" NodesPerElement="2" NumberOfElements="1">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_0"]</DataItem>
                    </Topology>
                    <Time Value="1.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="8">1.01e2 1.02e2</DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="1" NumberType="Float" Format="XML" Precision="8">1.1e2</DataItem>
                    </Attribute>
                </Grid>
            </Grid>
            <Grid Name="surface" GridType="Collection" CollectionType="Temporal">
                <Grid Name="surface-t0.5" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Triangle" NumberOfElements="2">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_1"]</DataItem>
                    </Topology>
                    <Time Value="0.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem Dimensions="4" NumberType="Float" Format="XML" Precision="8">1e0 2e0 3e0 4e0</DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="8">2e1 3e1</DataItem>
                    </Attribute>
                </Grid>
                <Grid Name="surface-t1.5" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Triangle" NumberOfElements="2">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_1"]</DataItem>
                    </Topology>
                    <Time Value="1.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem Dimensions="4" NumberType="Float" Format="XML" Precision="8">1.01e2 1.02e2 1.03e2 1.04e2</DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="8">1.2e2 1.3e2</DataItem>
                    </Attribute>
                </Grid>
            </Grid>
            <Grid Name="corner" GridType="Collection" CollectionType="Temporal">
                <Grid Name="corner-t0.5" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_2"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Mixed" NumberOfElements="2">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_2"]</DataItem>
                    </Topology>
                    <Time Value="0.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem Dimensions="4" NumberType="Float" Format="XML" Precision="8">1e0 2e0 3e0 4e0</DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="8">3e1 1e1</DataItem>
                    </Attribute>
                </Grid>
                <Grid Name="corner-t1.5" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_2"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Mixed" NumberOfElements="2">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_2"]</DataItem>
                    </Topology>
                    <Time Value="1.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem Dimensions="4" NumberType="Float" Format="XML" Precision="8">1.01e2 1.02e2 1.03e2 1.04e2</DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="8">1.3e2 1.1e2</DataItem>
                    </Attribute>
                </Grid>
            </Grid>
        </Grid>
        <DataItem Name="coords_0" Dimensions="2 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0</DataItem>
        <DataItem Name="connectivity_0" Dimensions="2" NumberType="UInt" Format="XML" Precision="4">0 1</DataItem>
        <DataItem Name="coords_1" Dimensions="4 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0 0e0 1e0 0e0 1e0 1e0 0e0</DataItem>
        <DataItem Name="connectivity_1" Dimensions="6" NumberType="UInt" Format="XML" Precision="4">0 2 1 1 2 3</DataItem>
        <DataItem Name="coords_2" Dimensions="4 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0 0e0 1e0 0e0 1e0 1e0 0e0</DataItem>
        <DataItem Name="connectivity_2" Dimensions="8" NumberType="UInt" Format="XML" Precision="4">4 1 2 3 2 2 0 1</DataItem>
        <DataItem Name="submesh_cells_2" Dimensions="2" NumberType="Int" Format="XML" Precision="4">2 0</DataItem>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="$VERSION"/>
    <Information Name="submesh_cells" Value="0:1 1:2 submesh_cells_2"/>
    <Information Name="submesh_points" Value="0:2 0:4 0:4"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

#[test]
fn write_xdmf_with_submeshes_names_each_block_once() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let mut ts_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline)
        .unwrap()
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [
                ("edge", &[0][..]),
                ("surface", &[1, 2][..]),
                ("corner", &[2, 0][..]),
            ],
        )
        .unwrap();

    for time in ["0.0", "1.0", "2.0"] {
        ts_writer
            .write_time_step(time, |step| {
                step.cell_data("material", xdmf::DataAttribute::Scalar, &[10.0, 20.0, 30.0])
            })
            .unwrap();
    }

    let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

    // `ParaView` makes a grid name unique across the whole document, so a submesh named once per
    // time step comes back as `edge`, `edge[1]`, `edge[2]`, ... and the block loses whatever the
    // user set for it in the Multi-block Inspector as the animation runs. Each submesh therefore
    // gets one grid carrying its name -- a temporal collection of that block's per-step grids --
    // however many steps are written.
    for name in ["edge", "surface", "corner"] {
        assert_eq!(
            read_xdmf
                .matches(&format!(r#"<Grid Name="{name}" "#))
                .count(),
            1,
            "submesh '{name}' must be named by exactly one grid, whatever the number of steps"
        );
    }
}

#[test]
fn write_xdmf_with_submeshes_only_mesh() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    xdmf_writer
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [("all", &[0, 1, 2][..])],
        )
        .unwrap();

    // without any time step the spatial collection is written directly, with no temporal one
    // around it -- the same as `write_xdmf_only_mesh` for a plain mesh
    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="mesh" GridType="Collection" CollectionType="Spatial">
            <Grid Name="all" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="3">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_0"]</DataItem>
                </Topology>
            </Grid>
        </Grid>
        <DataItem Name="coords_0" Dimensions="4 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0 0e0 1e0 0e0 1e0 1e0 0e0</DataItem>
        <DataItem Name="connectivity_0" Dimensions="12" NumberType="UInt" Format="XML" Precision="4">2 2 0 1 4 0 2 1 4 1 2 3</DataItem>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="$VERSION"/>
    <Information Name="submesh_cells" Value="0:3"/>
    <Information Name="submesh_points" Value="0:4"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

#[test]
fn write_xdmf_with_submeshes_of_a_uniform_topology() {
    // every cell shares one `CellType`, so the connectivity is written with no per-cell type
    // code (see `prepare_cells`) -- this exercises `cell_offsets`/`extract_connectivity` against
    // that fixed-stride layout instead of the type-code-prefixed `Mixed` one the other submesh
    // tests use.
    let node_coords = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0,
    ];
    let connectivity = [0_u32, 1, 2, 3, 1, 4, 5, 2];
    let cell_types = [xdmf::CellType::Quadrilateral, xdmf::CellType::Quadrilateral];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    xdmf_writer
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [("first", &[0][..]), ("second", &[1][..])],
        )
        .unwrap();

    let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

    assert!(read_xdmf.contains(r#"<Topology TopologyType="Quadrilateral" NumberOfElements="1">"#));
    assert!(read_xdmf.contains(
        r#"<DataItem Name="connectivity_0" Dimensions="4" NumberType="UInt" Format="XML" Precision="4">0 1 2 3</DataItem>"#
    ));

    // the second cell is `1 4 5 2` in the mesh's numbering; its submesh carries the four points
    // it uses, ascending, so those become 0, 2, 3 and 1 in the submesh's own
    assert!(read_xdmf.contains(
        r#"<DataItem Name="coords_1" Dimensions="4 3" NumberType="Float" Format="XML" Precision="8">1e0 0e0 0e0 1e0 1e0 0e0 2e0 0e0 0e0 2e0 1e0 0e0</DataItem>"#
    ));
    assert!(read_xdmf.contains(
        r#"<DataItem Name="connectivity_1" Dimensions="4" NumberType="UInt" Format="XML" Precision="4">0 2 3 1</DataItem>"#
    ));
    // which mesh points those were, for reading the file back
    assert!(read_xdmf.contains(
        r#"<DataItem Name="submesh_points_1" Dimensions="4" NumberType="Int" Format="XML" Precision="4">1 2 4 5</DataItem>"#
    ));
    assert!(
        read_xdmf.contains(r#"<Information Name="submesh_points" Value="0:4 submesh_points_1"/>"#)
    );
}

#[cfg(feature = "hdf5")]
#[test]
fn write_xdmf_with_a_scattered_submesh_selects_its_points_out_of_the_mesh() {
    // The counterpart of the test above for a storage that can be selected out of: the mesh's
    // coordinates are written once, as one array per direction, and each submesh's `<Geometry>`
    // says which of them it holds -- a start and a count for the first submesh, whose points are
    // one run, and the very index list that records those points for a reader for the second,
    // whose are not. Nothing is written per submesh but its connectivity.
    let node_coords = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0,
    ];
    let connectivity = [0_u32, 1, 2, 3, 1, 4, 5, 2];
    let cell_types = [xdmf::CellType::Quadrilateral, xdmf::CellType::Quadrilateral];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    TimeSeriesWriter::new(
        &xdmf_file_path,
        xdmf::DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
    )
    .unwrap()
    .write_mesh_with_submeshes(
        &node_coords,
        &connectivity,
        &cell_types,
        [("first", &[0][..]), ("second", &[1][..])],
    )
    .unwrap();

    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="mesh" GridType="Collection" CollectionType="Spatial">
            <Grid Name="first" GridType="Uniform">
                <Geometry GeometryType="X_Y_Z">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_x"]</DataItem>
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_y"]</DataItem>
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_z"]</DataItem>
                </Geometry>
                <Topology TopologyType="Quadrilateral" NumberOfElements="1">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_0"]</DataItem>
                </Topology>
            </Grid>
            <Grid Name="second" GridType="Uniform">
                <Geometry GeometryType="X_Y_Z">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_x"]</DataItem>
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_y"]</DataItem>
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_z"]</DataItem>
                </Geometry>
                <Topology TopologyType="Quadrilateral" NumberOfElements="1">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_1"]</DataItem>
                </Topology>
            </Grid>
        </Grid>
        <DataItem Name="coords_0_x" ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
            <DataItem Dimensions="6" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/0</DataItem>
        </DataItem>
        <DataItem Name="coords_0_y" ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
            <DataItem Dimensions="6" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/1</DataItem>
        </DataItem>
        <DataItem Name="coords_0_z" ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
            <DataItem Dimensions="6" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/2</DataItem>
        </DataItem>
        <DataItem Name="connectivity_0" Dimensions="4" NumberType="UInt" Format="HDF" Precision="4">test_output.h5:mesh/cells/0</DataItem>
        <DataItem Name="coords_1_x" ItemType="Coordinates" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="submesh_points_1"]</DataItem>
            <DataItem Dimensions="6" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/0</DataItem>
        </DataItem>
        <DataItem Name="coords_1_y" ItemType="Coordinates" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="submesh_points_1"]</DataItem>
            <DataItem Dimensions="6" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/1</DataItem>
        </DataItem>
        <DataItem Name="coords_1_z" ItemType="Coordinates" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="submesh_points_1"]</DataItem>
            <DataItem Dimensions="6" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/2</DataItem>
        </DataItem>
        <DataItem Name="connectivity_1" Dimensions="4" NumberType="UInt" Format="HDF" Precision="4">test_output.h5:mesh/cells/1</DataItem>
        <DataItem Name="submesh_points_1" Dimensions="4" NumberType="Int" Format="HDF" Precision="4">test_output.h5:mesh/submesh_points/1</DataItem>
    </Domain>
    <Information Name="data_storage" Value="Hdf5SingleFile { deflate_level: Some(3) }"/>
    <Information Name="version" Value="$VERSION"/>
    <Information Name="submesh_cells" Value="0:1 1:1"/>
</Xdmf>"#;

    let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

#[test]
fn write_xdmf_with_submeshes_accepts_a_step_of_cell_data_only() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    let mut ts_writer = xdmf_writer
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [("edge", &[0][..]), ("surface", &[1, 2][..])],
        )
        .unwrap();

    // cell data lands per submesh rather than in the step's shared attributes, so a step made of
    // nothing but cell data must still count as non-empty
    ts_writer
        .write_time_step("0.0", |step| {
            step.cell_data("material", xdmf::DataAttribute::Scalar, &[10.0, 20.0, 30.0])
        })
        .unwrap();

    let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

    assert!(read_xdmf.contains(r#"<Time Value="0.0"/>"#));
    assert_eq!(read_xdmf.matches(r#"Name="material""#).count(), 2);
}

#[test]
fn write_xdmf_with_submeshes_rejects_an_empty_step() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    let mut ts_writer = xdmf_writer
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [("all", &[0, 1, 2][..])],
        )
        .unwrap();

    let error = ts_writer
        .write_time_step("0.0", |_step| Ok(()))
        .unwrap_err();

    std::assert_matches!(
        error,
        xdmf::Error::InvalidTimeStep { time, reason }
            if time == "0.0" && reason.contains("no data written")
    );
}

#[test]
fn write_xdmf_with_submeshes_for_every_storage() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let storages = [
        xdmf::DataStorage::Ascii,
        xdmf::DataStorage::AsciiInline,
        xdmf::DataStorage::Binary,
        #[cfg(feature = "hdf5")]
        xdmf::DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
        #[cfg(feature = "hdf5")]
        xdmf::DataStorage::Hdf5MultipleFiles {
            deflate_level: None,
        },
    ];

    for storage in storages {
        let tmp_dir = TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output");

        let xdmf_writer = TimeSeriesWriter::new(&xdmf_file_path, storage).unwrap();

        let mut ts_writer = xdmf_writer
            .write_mesh_with_submeshes(
                &node_coords,
                &connectivity,
                &cell_types,
                [
                    ("edge", &[0][..]),
                    ("surface", &[1, 2][..]),
                    ("corner", &[2, 0][..]),
                ],
            )
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        for time in ["0.0", "1.0"] {
            ts_writer
                .write_time_step(time, |step| {
                    step.point_data(
                        "temperature",
                        xdmf::DataAttribute::Scalar,
                        &[1.0, 2.0, 3.0, 4.0],
                    )?;
                    step.cell_data("material", xdmf::DataAttribute::Scalar, &[10.0, 20.0, 30.0])
                })
                .unwrap_or_else(|error| panic!("{storage:?}: failed to write step: {error}"));
        }

        let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

        // one uniform grid per submesh per time step, plus one temporal collection per submesh
        // and the spatial collection gathering those
        assert_eq!(
            read_xdmf.matches("<Grid ").count(),
            3 * 2 + 3 + 1,
            "{storage:?} wrote an unexpected number of grids"
        );
        // the cell field carries the same name in every submesh, which is what lets ParaView
        // treat it as one field across the multi-block dataset
        assert_eq!(
            read_xdmf.matches(r#"Name="material""#).count(),
            3 * 2,
            "{storage:?} wrote an unexpected number of cell attributes"
        );
        // a point field is cut per submesh just as a cell field is, since each submesh carries
        // only its own points
        assert_eq!(
            read_xdmf.matches(r#"Name="temperature""#).count(),
            3 * 2,
            "{storage:?} wrote an unexpected number of point attributes"
        );
        // One coordinate item per submesh where each carries a copy of the points its own cells
        // use -- three, one per direction, where the submeshes select their own out of the mesh's
        // coordinates instead, which only the HDF5 storages can (see the two tests below).
        let selects = matches!(
            storage,
            xdmf::DataStorage::Hdf5SingleFile { .. } | xdmf::DataStorage::Hdf5MultipleFiles { .. }
        );
        assert_eq!(
            read_xdmf.matches(r#"<DataItem Name="coords_"#).count(),
            if selects { 3 * 3 } else { 3 },
            "{storage:?} wrote an unexpected number of coordinate items"
        );
        // which cells each submesh holds, for a reader: the two contiguous ones as a start and a
        // length, the scattered one as the `DataItem` every storage writes its indices to
        assert!(
            read_xdmf
                .contains(r#"<Information Name="submesh_cells" Value="0:1 1:2 submesh_cells_2"/>"#),
            "{storage:?} did not record which cells the submeshes hold"
        );
        assert_eq!(
            read_xdmf
                .matches(r#"<DataItem Name="submesh_cells_2""#)
                .count(),
            1,
            "{storage:?} wrote an unexpected number of submesh cell arrays"
        );
        // and which points, the counterpart a submesh's connectivity is renumbered against --
        // recorded the same way, but only where the geometry does not already say it: a submesh
        // selecting its points out of the mesh's names them in its own `<Geometry>`
        assert_eq!(
            read_xdmf.contains(r#"<Information Name="submesh_points" Value="0:2 0:4 0:4"/>"#),
            !selects,
            "{storage:?} recorded which points the submeshes hold in the wrong place"
        );
        assert!(
            !read_xdmf.contains(r#"<DataItem Name="submesh_points"#),
            "{storage:?} wrote a point index array for a submesh that is one run"
        );
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn write_xdmf_with_submeshes_selects_hdf5_data_written_once() {
    // What a selection buys: each field reaches the heavy data once per step, whole, and every
    // submesh's `<Attribute>` says which part of it that submesh holds -- so a step costs the same
    // however many submeshes there are and however much they overlap. The document below shows all
    // three ways that share is named:
    //
    // - a submesh whose entities are one run takes a `HyperSlab` of the field, whose start and
    //   count go into the XML itself ("edge" and "middle", and every submesh's point data here);
    // - a scattered one selects through the very index list that records which cells it holds for
    //   a reader ("ends" and `submesh_cells_2`), which is why a scalar field needs no new array;
    // - a field of more than one value per entity needs one anyway, since each of those values has
    //   to be named -- written once, at the step that first carries such a field, and referenced
    //   by every step after (`selections_0`).
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let mut ts_writer = TimeSeriesWriter::new(
        &xdmf_file_path,
        xdmf::DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
    )
    .unwrap()
    // two runs and a scattered list, which between them cover every cell of the mesh
    .write_mesh_with_submeshes(
        &node_coords,
        &connectivity,
        &cell_types,
        [
            ("edge", &[0][..]),
            ("middle", &[1][..]),
            ("ends", &[0, 2][..]),
        ],
    )
    .unwrap();

    // two steps, so the expectation below also covers what a later step reuses: the selections,
    // which name positions in a field rather than values, and so do not change with the step
    for (time, offset) in [("0.5", 0.0), ("1.5", 100.0)] {
        let temperature = [1.0, 2.0, 3.0, 4.0].map(|value: f64| value + offset);
        let material = [10.0, 20.0, 30.0].map(|value: f64| value + offset);
        let velocity: Vec<f64> = (1..=9).map(|value| f64::from(value) + offset).collect();

        ts_writer
            .write_time_step(time, |step| {
                step.point_data("temperature", xdmf::DataAttribute::Scalar, &temperature)?;
                step.cell_data("material", xdmf::DataAttribute::Scalar, &material)?;
                step.cell_data("velocity", xdmf::DataAttribute::Vector, &velocity)
            })
            .unwrap();
    }

    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="mesh" GridType="Collection" CollectionType="Spatial">
            <Grid Name="edge" GridType="Collection" CollectionType="Temporal">
                <Grid Name="edge-t0.5" GridType="Uniform">
                    <Geometry GeometryType="X_Y_Z">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_x"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_y"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_z"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Polyline" NodesPerElement="2" NumberOfElements="1">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_0"]</DataItem>
                    </Topology>
                    <Time Value="0.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem ItemType="HyperSlab" Dimensions="2" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 2</DataItem>
                            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.5/0</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem ItemType="HyperSlab" Dimensions="1" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 1</DataItem>
                            <DataItem Dimensions="3" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.5/1</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
                        <DataItem ItemType="HyperSlab" Dimensions="1 3" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
                            <DataItem Dimensions="9" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.5/2</DataItem>
                        </DataItem>
                    </Attribute>
                </Grid>
                <Grid Name="edge-t1.5" GridType="Uniform">
                    <Geometry GeometryType="X_Y_Z">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_x"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_y"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_z"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Polyline" NodesPerElement="2" NumberOfElements="1">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_0"]</DataItem>
                    </Topology>
                    <Time Value="1.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem ItemType="HyperSlab" Dimensions="2" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 2</DataItem>
                            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_1.5/0</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem ItemType="HyperSlab" Dimensions="1" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 1</DataItem>
                            <DataItem Dimensions="3" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_1.5/1</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
                        <DataItem ItemType="HyperSlab" Dimensions="1 3" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
                            <DataItem Dimensions="9" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_1.5/2</DataItem>
                        </DataItem>
                    </Attribute>
                </Grid>
            </Grid>
            <Grid Name="middle" GridType="Collection" CollectionType="Temporal">
                <Grid Name="middle-t0.5" GridType="Uniform">
                    <Geometry GeometryType="X_Y_Z">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_x"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_y"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_z"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Triangle" NumberOfElements="1">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_1"]</DataItem>
                    </Topology>
                    <Time Value="0.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem ItemType="HyperSlab" Dimensions="3" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
                            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.5/0</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem ItemType="HyperSlab" Dimensions="1" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">1 1 1</DataItem>
                            <DataItem Dimensions="3" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.5/1</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
                        <DataItem ItemType="HyperSlab" Dimensions="1 3" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">3 1 3</DataItem>
                            <DataItem Dimensions="9" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.5/2</DataItem>
                        </DataItem>
                    </Attribute>
                </Grid>
                <Grid Name="middle-t1.5" GridType="Uniform">
                    <Geometry GeometryType="X_Y_Z">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_x"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_y"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_z"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Triangle" NumberOfElements="1">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_1"]</DataItem>
                    </Topology>
                    <Time Value="1.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem ItemType="HyperSlab" Dimensions="3" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
                            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_1.5/0</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem ItemType="HyperSlab" Dimensions="1" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">1 1 1</DataItem>
                            <DataItem Dimensions="3" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_1.5/1</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
                        <DataItem ItemType="HyperSlab" Dimensions="1 3" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">3 1 3</DataItem>
                            <DataItem Dimensions="9" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_1.5/2</DataItem>
                        </DataItem>
                    </Attribute>
                </Grid>
            </Grid>
            <Grid Name="ends" GridType="Collection" CollectionType="Temporal">
                <Grid Name="ends-t0.5" GridType="Uniform">
                    <Geometry GeometryType="X_Y_Z">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_2_x"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_2_y"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_2_z"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Mixed" NumberOfElements="2">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_2"]</DataItem>
                    </Topology>
                    <Time Value="0.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
                            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.5/0</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem ItemType="Coordinates" Dimensions="2" NumberType="Float" Precision="8">
                            <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="submesh_cells_2"]</DataItem>
                            <DataItem Dimensions="3" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.5/1</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
                        <DataItem ItemType="Coordinates" Dimensions="2 3" NumberType="Float" Precision="8">
                            <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="selections_0"]</DataItem>
                            <DataItem Dimensions="9" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.5/2</DataItem>
                        </DataItem>
                    </Attribute>
                </Grid>
                <Grid Name="ends-t1.5" GridType="Uniform">
                    <Geometry GeometryType="X_Y_Z">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_2_x"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_2_y"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_2_z"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Mixed" NumberOfElements="2">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_2"]</DataItem>
                    </Topology>
                    <Time Value="1.5"/>
                    <Attribute Name="temperature" AttributeType="Scalar" Center="Node">
                        <DataItem ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
                            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_1.5/0</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem ItemType="Coordinates" Dimensions="2" NumberType="Float" Precision="8">
                            <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="submesh_cells_2"]</DataItem>
                            <DataItem Dimensions="3" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_1.5/1</DataItem>
                        </DataItem>
                    </Attribute>
                    <Attribute Name="velocity" AttributeType="Vector" Center="Cell">
                        <DataItem ItemType="Coordinates" Dimensions="2 3" NumberType="Float" Precision="8">
                            <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="selections_0"]</DataItem>
                            <DataItem Dimensions="9" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_1.5/2</DataItem>
                        </DataItem>
                    </Attribute>
                </Grid>
            </Grid>
        </Grid>
        <DataItem Name="coords_0_x" ItemType="HyperSlab" Dimensions="2" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 2</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/0</DataItem>
        </DataItem>
        <DataItem Name="coords_0_y" ItemType="HyperSlab" Dimensions="2" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 2</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/1</DataItem>
        </DataItem>
        <DataItem Name="coords_0_z" ItemType="HyperSlab" Dimensions="2" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 2</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/2</DataItem>
        </DataItem>
        <DataItem Name="connectivity_0" Dimensions="2" NumberType="UInt" Format="HDF" Precision="4">test_output.h5:mesh/cells/0</DataItem>
        <DataItem Name="coords_1_x" ItemType="HyperSlab" Dimensions="3" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/0</DataItem>
        </DataItem>
        <DataItem Name="coords_1_y" ItemType="HyperSlab" Dimensions="3" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/1</DataItem>
        </DataItem>
        <DataItem Name="coords_1_z" ItemType="HyperSlab" Dimensions="3" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/2</DataItem>
        </DataItem>
        <DataItem Name="connectivity_1" Dimensions="3" NumberType="UInt" Format="HDF" Precision="4">test_output.h5:mesh/cells/1</DataItem>
        <DataItem Name="coords_2_x" ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/0</DataItem>
        </DataItem>
        <DataItem Name="coords_2_y" ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/1</DataItem>
        </DataItem>
        <DataItem Name="coords_2_z" ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/2</DataItem>
        </DataItem>
        <DataItem Name="connectivity_2" Dimensions="8" NumberType="UInt" Format="HDF" Precision="4">test_output.h5:mesh/cells/2</DataItem>
        <DataItem Name="submesh_cells_2" Dimensions="2" NumberType="Int" Format="HDF" Precision="4">test_output.h5:mesh/submesh_cells/2</DataItem>
        <DataItem Name="selections_0" Dimensions="6" NumberType="Int" Format="HDF" Precision="4">test_output.h5:mesh/selections/0</DataItem>
    </Domain>
    <Information Name="data_storage" Value="Hdf5SingleFile { deflate_level: Some(3) }"/>
    <Information Name="version" Value="$VERSION"/>
    <Information Name="submesh_cells" Value="0:1 1:1 submesh_cells_2"/>
</Xdmf>"#;

    let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

#[cfg(feature = "hdf5")]
#[test]
fn write_xdmf_with_an_unordered_submesh_writes_its_share_out() {
    // `ParaView` hands back the values a `Coordinates` selection names in the order the array
    // holds them, not in the order they were named -- so "reversed", which lists its two cells the
    // other way round, gets a copy of its share (`data/t_0.0/1`) while "run" selects out of the
    // field itself (`data/t_0.0/0`), exactly as a storage without selections would write both.
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let mut ts_writer = TimeSeriesWriter::new(
        &xdmf_file_path,
        xdmf::DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
    )
    .unwrap()
    .write_mesh_with_submeshes(
        &node_coords,
        &connectivity,
        &cell_types,
        [("run", &[0, 1][..]), ("reversed", &[2, 1][..])],
    )
    .unwrap();

    ts_writer
        .write_time_step("0.0", |step| {
            step.cell_data("material", xdmf::DataAttribute::Scalar, &[10.0, 20.0, 30.0])
        })
        .unwrap();

    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="mesh" GridType="Collection" CollectionType="Spatial">
            <Grid Name="run" GridType="Collection" CollectionType="Temporal">
                <Grid Name="run-t0.0" GridType="Uniform">
                    <Geometry GeometryType="X_Y_Z">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_x"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_y"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0_z"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Mixed" NumberOfElements="2">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_0"]</DataItem>
                    </Topology>
                    <Time Value="0.0"/>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem ItemType="HyperSlab" Dimensions="2" NumberType="Float" Precision="8">
                            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 2</DataItem>
                            <DataItem Dimensions="3" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.0/0</DataItem>
                        </DataItem>
                    </Attribute>
                </Grid>
            </Grid>
            <Grid Name="reversed" GridType="Collection" CollectionType="Temporal">
                <Grid Name="reversed-t0.0" GridType="Uniform">
                    <Geometry GeometryType="X_Y_Z">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_x"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_y"]</DataItem>
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1_z"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Triangle" NumberOfElements="2">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_1"]</DataItem>
                    </Topology>
                    <Time Value="0.0"/>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="2" NumberType="Float" Format="HDF" Precision="8">test_output.h5:data/t_0.0/1</DataItem>
                    </Attribute>
                </Grid>
            </Grid>
        </Grid>
        <DataItem Name="coords_0_x" ItemType="HyperSlab" Dimensions="3" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/0</DataItem>
        </DataItem>
        <DataItem Name="coords_0_y" ItemType="HyperSlab" Dimensions="3" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/1</DataItem>
        </DataItem>
        <DataItem Name="coords_0_z" ItemType="HyperSlab" Dimensions="3" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 3</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/2</DataItem>
        </DataItem>
        <DataItem Name="connectivity_0" Dimensions="8" NumberType="UInt" Format="HDF" Precision="4">test_output.h5:mesh/cells/0</DataItem>
        <DataItem Name="coords_1_x" ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/0</DataItem>
        </DataItem>
        <DataItem Name="coords_1_y" ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/1</DataItem>
        </DataItem>
        <DataItem Name="coords_1_z" ItemType="HyperSlab" Dimensions="4" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" NumberType="Int" Format="XML" Precision="4">0 1 4</DataItem>
            <DataItem Dimensions="4" NumberType="Float" Format="HDF" Precision="8">test_output.h5:mesh/points/2</DataItem>
        </DataItem>
        <DataItem Name="connectivity_1" Dimensions="6" NumberType="UInt" Format="HDF" Precision="4">test_output.h5:mesh/cells/1</DataItem>
        <DataItem Name="submesh_cells_1" Dimensions="2" NumberType="Int" Format="HDF" Precision="4">test_output.h5:mesh/submesh_cells/1</DataItem>
    </Domain>
    <Information Name="data_storage" Value="Hdf5SingleFile { deflate_level: Some(3) }"/>
    <Information Name="version" Value="$VERSION"/>
    <Information Name="submesh_cells" Value="0:2 submesh_cells_1"/>
</Xdmf>"#;

    let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);
}

#[test]
fn write_xdmf_with_submeshes_never_selects_from_a_storage_that_misreads_it() {
    // `ParaView` ignores a selection for the ascii and binary storages and reads the source array
    // from its start instead, silently -- so those keep a copy per submesh however the submeshes
    // are shaped, as the document below shows: one data file per (field, submesh), no `ItemType`
    // anywhere. Guarding this is what keeps a selection from being written where it would show
    // numbers the file does not contain.
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let write_step = |storage, xdmf_file_path: &std::path::Path| {
        let mut ts_writer = TimeSeriesWriter::new(xdmf_file_path, storage)
            .unwrap()
            .write_mesh_with_submeshes(
                &node_coords,
                &connectivity,
                &cell_types,
                [
                    ("edge", &[0][..]),
                    ("middle", &[1][..]),
                    ("ends", &[0, 2][..]),
                ],
            )
            .unwrap();

        ts_writer
            .write_time_step("0.0", |step| {
                step.cell_data("material", xdmf::DataAttribute::Scalar, &[10.0, 20.0, 30.0])
            })
            .unwrap();

        std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap()
    };

    let tmp_dir = TempDir::new().unwrap();
    let read_xdmf = write_step(
        xdmf::DataStorage::Ascii,
        &tmp_dir.path().join("test_output"),
    );

    let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="mesh" GridType="Collection" CollectionType="Spatial">
            <Grid Name="edge" GridType="Collection" CollectionType="Temporal">
                <Grid Name="edge-t0.0" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_0"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Polyline" NodesPerElement="2" NumberOfElements="1">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_0"]</DataItem>
                    </Topology>
                    <Time Value="0.0"/>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="1" NumberType="Float" Format="XML" Precision="8">
                            <xi:include href="test_output.txt/data_t_0.0_0.txt" parse="text"/>
                        </DataItem>
                    </Attribute>
                </Grid>
            </Grid>
            <Grid Name="middle" GridType="Collection" CollectionType="Temporal">
                <Grid Name="middle-t0.0" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_1"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Triangle" NumberOfElements="1">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_1"]</DataItem>
                    </Topology>
                    <Time Value="0.0"/>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="1" NumberType="Float" Format="XML" Precision="8">
                            <xi:include href="test_output.txt/data_t_0.0_1.txt" parse="text"/>
                        </DataItem>
                    </Attribute>
                </Grid>
            </Grid>
            <Grid Name="ends" GridType="Collection" CollectionType="Temporal">
                <Grid Name="ends-t0.0" GridType="Uniform">
                    <Geometry GeometryType="XYZ">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords_2"]</DataItem>
                    </Geometry>
                    <Topology TopologyType="Mixed" NumberOfElements="2">
                        <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity_2"]</DataItem>
                    </Topology>
                    <Time Value="0.0"/>
                    <Attribute Name="material" AttributeType="Scalar" Center="Cell">
                        <DataItem Dimensions="2" NumberType="Float" Format="XML" Precision="8">
                            <xi:include href="test_output.txt/data_t_0.0_2.txt" parse="text"/>
                        </DataItem>
                    </Attribute>
                </Grid>
            </Grid>
        </Grid>
        <DataItem Name="coords_0" Dimensions="2 3" NumberType="Float" Format="XML" Precision="8">
            <xi:include href="test_output.txt/points_0.txt" parse="text"/>
        </DataItem>
        <DataItem Name="connectivity_0" Dimensions="2" NumberType="UInt" Format="XML" Precision="4">
            <xi:include href="test_output.txt/cells_0.txt" parse="text"/>
        </DataItem>
        <DataItem Name="coords_1" Dimensions="3 3" NumberType="Float" Format="XML" Precision="8">
            <xi:include href="test_output.txt/points_1.txt" parse="text"/>
        </DataItem>
        <DataItem Name="connectivity_1" Dimensions="3" NumberType="UInt" Format="XML" Precision="4">
            <xi:include href="test_output.txt/cells_1.txt" parse="text"/>
        </DataItem>
        <DataItem Name="coords_2" Dimensions="4 3" NumberType="Float" Format="XML" Precision="8">
            <xi:include href="test_output.txt/points_2.txt" parse="text"/>
        </DataItem>
        <DataItem Name="connectivity_2" Dimensions="8" NumberType="UInt" Format="XML" Precision="4">
            <xi:include href="test_output.txt/cells_2.txt" parse="text"/>
        </DataItem>
        <DataItem Name="submesh_cells_2" Dimensions="2" NumberType="Int" Format="XML" Precision="4">
            <xi:include href="test_output.txt/submesh_cells_2.txt" parse="text"/>
        </DataItem>
    </Domain>
    <Information Name="data_storage" Value="Ascii"/>
    <Information Name="version" Value="$VERSION"/>
    <Information Name="submesh_cells" Value="0:1 1:1 submesh_cells_2"/>
    <Information Name="submesh_points" Value="0:2 0:3 0:4"/>
</Xdmf>"#;

    pretty_assertions::assert_eq!(with_version(expected_xdmf), read_xdmf);

    // the same holds for the other two, whose light data differs only in where the values sit
    for storage in [xdmf::DataStorage::AsciiInline, xdmf::DataStorage::Binary] {
        let tmp_dir = TempDir::new().unwrap();
        let read_xdmf = write_step(storage, &tmp_dir.path().join("test_output"));

        assert!(
            !read_xdmf.contains("ItemType="),
            "{storage:?} wrote a selection its reader would misread: {read_xdmf}"
        );
    }
}

#[test]
fn write_xdmf_with_contiguous_submeshes_writes_no_cell_indices() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline)
        .unwrap()
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [("edge", &[0][..]), ("surface", &[1, 2][..])],
        )
        .unwrap();

    let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

    // a submesh that is one ascending run is a start and a length, so submeshes cost nothing to
    // read back in the case mesh generators produce -- no index array is written at all
    assert!(read_xdmf.contains(r#"<Information Name="submesh_cells" Value="0:1 1:2"/>"#));
    assert!(!read_xdmf.contains(r#"<DataItem Name="submesh_cells"#));
}

// A data name of the shape solvers actually hand over: spaces, parentheses, apostrophes, a dot and
// a comma. All are accepted (see `INVALID_DATA_NAME_CHARS` in the crate), and all of them end up in
// a file name for the ascii and binary storages and in a dataset name for the HDF5 ones.
const SOLVER_STYLE_NAME: &str = "Quantity('SOOT DENSITY'), U.component_0 [kg m-3]";

#[test]
fn write_xdmf_accepts_a_solver_style_data_name_for_every_storage() {
    let storages = [
        xdmf::DataStorage::Ascii,
        xdmf::DataStorage::AsciiInline,
        xdmf::DataStorage::Binary,
        #[cfg(feature = "hdf5")]
        xdmf::DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
        #[cfg(feature = "hdf5")]
        xdmf::DataStorage::Hdf5MultipleFiles {
            deflate_level: None,
        },
    ];

    for storage in storages {
        let tmp_dir = TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output");

        let mut ts_writer = TimeSeriesWriter::new(&xdmf_file_path, storage)
            .unwrap()
            .write_mesh(
                &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                &[0_u32, 1],
                &[xdmf::CellType::Edge],
            )
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        ts_writer
            .write_time_step("0.0", |step| {
                step.cell_data(SOLVER_STYLE_NAME, xdmf::DataAttribute::Scalar, &[1.5])
            })
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write step: {error}"));

        let xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

        // The name is what ParaView matches a field by, so it has to survive into the light data
        // exactly as given -- `quick-xml` quotes attributes with `"`, so the apostrophes in it need
        // no escaping and none is applied.
        assert!(
            xdmf.contains(&format!(r#"<Attribute Name="{SOLVER_STYLE_NAME}""#)),
            "{storage:?} did not write the name verbatim:\n{xdmf}"
        );

        // The name must not have reached the filesystem at all -- the heavy data is numbered, so
        // nothing the caller spelled has to be a legal path component.
        let extension = match storage {
            xdmf::DataStorage::Ascii => Some("txt"),
            xdmf::DataStorage::Binary => Some("bin"),
            _ => None,
        };
        if let Some(extension) = extension {
            let data_dir = xdmf_file_path.with_extension(extension);
            let expected = data_dir.join(format!("data_t_0.0_0.{extension}"));
            assert!(
                expected.exists(),
                "{storage:?} did not write {}",
                expected.display()
            );

            let written: Vec<String> = std::fs::read_dir(&data_dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            assert!(
                written.iter().all(|file| !file.contains("SOOT")),
                "{storage:?} put the caller's name in a file name: {written:?}"
            );
        }
    }
}

#[test]
fn write_xdmf_with_submeshes_gives_each_submesh_its_own_data_file() {
    // Every (field, submesh) pair is its own numbered array, so no two can land in one file however
    // the field and the submesh are spelled -- which a name-derived file name could not guarantee,
    // since `_` was legal in both (the field "a__b" of submesh "c" and the field "a" of submesh
    // "b__c" once collided).
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let mut ts_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::Ascii)
        .unwrap()
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [("c", &[0][..]), ("b__c", &[1, 2][..])],
        )
        .unwrap();

    ts_writer
        .write_time_step("0.0", |step| {
            step.cell_data("a__b", xdmf::DataAttribute::Scalar, &[10.0, 20.0, 30.0])?;
            step.cell_data("a", xdmf::DataAttribute::Scalar, &[40.0, 50.0, 60.0])
        })
        .unwrap();

    // four arrays, numbered in the order they were handed over: field "a__b" for each of the two
    // submeshes, then field "a" for each
    let data_dir = xdmf_file_path.with_extension("txt");
    let contents = |index: usize| {
        let file_name = format!("data_t_0.0_{index}.txt");
        std::fs::read_to_string(data_dir.join(&file_name))
            .unwrap_or_else(|error| panic!("{file_name} was not written: {error}"))
            .trim_end()
            .to_string()
    };

    assert_eq!(contents(0), "1e1"); // "a__b" of submesh "c"
    assert_eq!(contents(1), "2e1 3e1"); // "a__b" of submesh "b__c"
    assert_eq!(contents(2), "4e1"); // "a" of submesh "c"
    assert_eq!(contents(3), "5e1 6e1"); // "a" of submesh "b__c"

    // both fields keep the caller's name in every block, which is what ParaView matches them by
    let xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
    assert_eq!(xdmf.matches(r#"Name="a__b""#).count(), 2);
    assert_eq!(xdmf.matches(r#"Name="a" "#).count(), 2);
}

#[test]
fn debug_output_summarizes_the_writers_without_their_data() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let writer = TimeSeriesWriter::new(
        tmp_dir.path().join("test_output"),
        xdmf::DataStorage::AsciiInline,
    )
    .unwrap();

    let writer_debug = format!("{writer:?}");
    assert!(writer_debug.contains("AsciiInline"), "{writer_debug}");

    let mut ts_writer = writer
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [("edge", &[0][..]), ("surface", &[1, 2][..])],
        )
        .unwrap();

    ts_writer
        .write_time_step("0.5", |step| {
            step.point_data(
                "temperature",
                xdmf::DataAttribute::Scalar,
                &[1.0, 2.0, 3.0, 4.0],
            )?;

            // a step names the attributes it has taken, not their values
            let step_debug = format!("{step:?}");
            assert!(step_debug.contains(r#"time: "0.5""#), "{step_debug}");
            assert!(
                step_debug.contains(r#"point_data: ["temperature"]"#),
                "{step_debug}"
            );
            assert!(step_debug.contains("cell_data: []"), "{step_debug}");

            Ok::<(), xdmf::Error>(())
        })
        .unwrap();

    let debug = format!("{ts_writer:?}");
    assert!(debug.contains("AsciiInline"), "{debug}");
    assert!(debug.contains("num_points: 4"), "{debug}");
    assert!(debug.contains("num_cells: 3"), "{debug}");
    assert!(
        debug.contains(r#"submeshes: ["edge", "surface"]"#),
        "{debug}"
    );
    assert!(debug.contains(r#"written_times: ["0.5"]"#), "{debug}");

    // The point of the manual impls: the light data is summarized, not dumped. With `AsciiInline`
    // the `DataItem`s hold the values themselves, so a derived `Debug` would print the whole time
    // series -- "1e0" is how the first temperature above is written.
    assert!(debug.contains(".."), "{debug}");
    assert!(!debug.contains("1e0"), "{debug}");
}

#[test]
fn write_mesh_with_submeshes_rejects_a_cell_in_no_submesh() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    // `TimeSeriesDataWriter` is not `Debug`, so the error is taken out of the `Result` first
    let error = xdmf_writer
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [("edge", &[0][..])],
        )
        .err()
        .unwrap();

    std::assert_matches!(
        error,
        xdmf::Error::InvalidMesh { reason }
            if reason.contains("2 of 3 cells belong to no submesh: 1, 2")
    );
}

#[test]
fn write_mesh_with_submeshes_of_a_point_mesh() {
    let node_coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer =
        TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    // with no cell types the points themselves are the cells, so the submeshes index those, and
    // each submesh carries exactly the points it names
    xdmf_writer
        .write_mesh_with_submeshes(
            &node_coords,
            &[] as &[u32],
            &[],
            [("first", &[0, 1][..]), ("last", &[2][..])],
        )
        .unwrap();

    let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

    assert!(read_xdmf.contains(
        r#"<DataItem Name="coords_0" Dimensions="2 3" NumberType="Float" Format="XML" Precision="8">0e0 0e0 0e0 1e0 0e0 0e0</DataItem>"#
    ));
    assert!(read_xdmf.contains(
        r#"<DataItem Name="connectivity_0" Dimensions="2" NumberType="UInt" Format="XML" Precision="4">0 1</DataItem>"#
    ));
    assert!(read_xdmf.contains(
        r#"<DataItem Name="coords_1" Dimensions="1 3" NumberType="Float" Format="XML" Precision="8">2e0 0e0 0e0</DataItem>"#
    ));
    // the mesh's third point is the second submesh's first, so its cell indexes it as 0
    assert!(read_xdmf.contains(
        r#"<DataItem Name="connectivity_1" Dimensions="1" NumberType="UInt" Format="XML" Precision="4">0</DataItem>"#
    ));
}

#[test]
fn write_mesh_with_submeshes_writes_nothing_when_the_submeshes_are_rejected() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::Ascii).unwrap();

    // the submeshes are validated before the points are written, so a rejected list must not
    // leave heavy data behind
    let error = xdmf_writer
        .write_mesh_with_submeshes(
            &node_coords,
            &connectivity,
            &cell_types,
            [("not\tvalid", &[0, 1, 2][..])],
        )
        .err()
        .unwrap();

    std::assert_matches!(
        error,
        xdmf::Error::InvalidMesh { reason } if reason.contains("is not valid")
    );

    // nothing at all, rather than a named file: which arrays a mesh with submeshes writes, and
    // what they are called, is what the test below pins down
    assert_eq!(
        std::fs::read_dir(xdmf_file_path.with_extension("txt"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn write_xdmf_with_submeshes_names_the_heavy_data_by_array_and_submesh() {
    let (node_coords, connectivity, cell_types) = submesh_test_mesh();

    // one file per array for the two per-file storages, one dataset per array for HDF5 -- named
    // the same way in both: the array, then which submesh's copy of it
    let expected: [(xdmf::DataStorage, &str, &str); 3] = [
        (
            xdmf::DataStorage::Ascii,
            "txt",
            "test_output.txt/points_1.txt",
        ),
        (
            xdmf::DataStorage::Binary,
            "bin",
            "test_output.bin/cells_1.bin",
        ),
        cfg_select! {
            feature = "hdf5" => (
                xdmf::DataStorage::Hdf5SingleFile {
                    deflate_level: None,
                },
                "h5",
                "test_output.h5:mesh/points/1",
            ),
            _ => (
                xdmf::DataStorage::Ascii,
                "txt",
                "test_output.txt/cells_0.txt",
            ),
        },
    ];

    for (storage, extension, referenced) in expected {
        let tmp_dir = TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output");

        TimeSeriesWriter::new(&xdmf_file_path, storage)
            .unwrap()
            .write_mesh_with_submeshes(
                &node_coords,
                &connectivity,
                &cell_types,
                [("edge", &[0][..]), ("surface", &[1, 2][..])],
            )
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        let read_xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
        assert!(
            read_xdmf.contains(referenced),
            "{storage:?} did not reference {referenced}"
        );

        // the file the light data names is there, under exactly that name
        if let Some(file) = referenced
            .split('/')
            .next_back()
            .filter(|_| extension != "h5")
        {
            assert!(
                xdmf_file_path.with_extension(extension).join(file).exists(),
                "{storage:?} did not write {file}"
            );
        }
    }
}

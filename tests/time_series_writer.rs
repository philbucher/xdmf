use temp_dir::TempDir;
use xdmf::TimeSeriesWriter;

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
    <Information Name="version" Value="0.1.3"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    //  std::fs::copy(xdmf_file, "time_series_writer.xdmf2").unwrap();

    pretty_assertions::assert_eq!(expected_xdmf, read_xdmf);
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
    <Information Name="version" Value="0.1.3"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "time_series_writer_only_mesh.xdmf").unwrap();

    pretty_assertions::assert_eq!(expected_xdmf, read_xdmf);
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
    <Information Name="version" Value="0.1.3"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "time_series_writer_only_point_mesh.xdmf2").unwrap();

    pretty_assertions::assert_eq!(expected_xdmf, read_xdmf);
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
    <Information Name="version" Value="0.1.3"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "write_xdmf_point_mesh.xdmf2").unwrap();

    pretty_assertions::assert_eq!(expected_xdmf, read_xdmf);
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
    <Information Name="version" Value="0.1.3"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "write_xdmf_f32_points.xdmf2").unwrap();

    pretty_assertions::assert_eq!(expected_xdmf, read_xdmf);
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
    <Information Name="version" Value="0.1.3"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    pretty_assertions::assert_eq!(expected_xdmf, read_xdmf);
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

    // 4 points * (2^62 + 1) wraps to 4, exactly the number of values passed
    std::assert_matches!(
        write(xdmf::DataAttribute::Generic(usize::MAX / 4 + 2), &[1.0, 2.0, 3.0, 4.0]),
        xdmf::Error::InvalidData { reason } if reason.contains("has no usable size")
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
            quick_xml::events::Event::Start(tag) if tag.name().as_ref() == b"Attribute" => {
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

#[test]
fn debug_output_summarizes_the_writers_without_their_data() {
    let coords = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::AsciiInline).unwrap();

    let writer_debug = format!("{writer:?}");
    assert!(writer_debug.contains("AsciiInline"), "{writer_debug}");

    let mut ts_writer = writer
        .write_mesh(&coords, &[0_u64, 1, 2], &[xdmf::CellType::Triangle])
        .unwrap();

    ts_writer
        .write_time_step("0.5", |step| {
            step.point_data("temperature", xdmf::DataAttribute::Scalar, &[1.0, 2.0, 3.0])?;

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
    assert!(debug.contains("num_points: 3"), "{debug}");
    assert!(debug.contains("num_cells: 1"), "{debug}");
    assert!(debug.contains(r#"written_times: ["0.5"]"#), "{debug}");

    // The point of the manual impls: the light data is summarized, not dumped. With
    // `AsciiInline` the `DataItem`s hold the values themselves, so a derived `Debug` would print
    // the whole time series -- "1e0" is how the temperature value above is written.
    assert!(debug.contains(".."), "{debug}");
    assert!(!debug.contains("1e0"), "{debug}");
}

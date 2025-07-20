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

    let num_nodes = node_coords.len() / 3;
    let num_cells = cell_types.len();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer = TimeSeriesWriter::new_with_options(
        &xdmf_file_path,
        &TimeSeriesWriter::options().format(xdmf::Format::XML),
    )
    .unwrap();

    let mut xdmf_writer = xdmf_writer
        .write_mesh(&node_coords, (&connectivity, &cell_types))
        .unwrap();

    for i in 0..3 {
        let point_data_1: Vec<f64> = (0..num_nodes).map(|j| j as f64 + i as f64).collect();
        let point_data_2: Vec<f64> = (0..num_nodes)
            .map(|j| 1. * j as f64 + 2. + i as f64)
            .collect();
        let cell_data: Vec<f64> = (0..num_cells)
            .map(|j| 1. * j as f64 + 1.5 * i as f64)
            .collect();

        let point_data = vec![
            (
                "point_data1".to_string(),
                (xdmf::AttributeType::Scalar, point_data_1.into()),
            ),
            (
                "point_data2".to_string(),
                (xdmf::AttributeType::Scalar, point_data_2.into()),
            ),
        ]
        .into_iter()
        .collect();

        let cell_data = vec![(
            "cell_data".to_string(),
            (xdmf::AttributeType::Scalar, cell_data.into()),
        )]
        .into_iter()
        .collect();
        xdmf_writer
            .write_data(&i.to_string(), Some(&point_data), Some(&cell_data))
            .unwrap();
    }

    let expected_xdmf = r#"
<Xdmf Version="3.0">
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
                <Attribute Name="point_data1" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">0.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="point_data2" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1 1.8000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="cell_data" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="12" NumberType="Float" Format="XML" Precision="8">0.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1</DataItem>
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
                <Attribute Name="point_data1" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="point_data2" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1 1.8000000000000000e1 1.9000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="cell_data" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="12" NumberType="Float" Format="XML" Precision="8">1.5000000000000000e0 2.5000000000000000e0 3.5000000000000000e0 4.5000000000000000e0 5.5000000000000000e0 6.5000000000000000e0 7.5000000000000000e0 8.5000000000000000e0 9.5000000000000000e0 1.0500000000000000e1 1.1500000000000000e1 1.2500000000000000e1</DataItem>
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
                <Attribute Name="point_data1" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1 1.8000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="point_data2" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1 1.8000000000000000e1 1.9000000000000000e1 2.0000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="cell_data" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="12" NumberType="Float" Format="XML" Precision="8">3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1</DataItem>
                </Attribute>
            </Grid>
        </Grid>
        <DataItem Name="coords" Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 5.0000000000000000e-1 -5.0000000000000000e-1 2.0000000000000001e-1 -5.0000000000000000e-1 5.0000000000000000e-1 2.0000000000000001e-1 1.5000000000000000e0 -5.0000000000000000e-1 2.0000000000000001e-1 2.5000000000000000e0 5.0000000000000000e-1 2.0000000000000001e-1 5.0000000000000000e-1 1.5000000000000000e0 2.0000000000000001e-1 5.0000000000000000e-1 2.5000000000000000e0 2.0000000000000001e-1 1.5000000000000000e0 2.5000000000000000e0 2.0000000000000001e-1 2.5000000000000000e0 1.5000000000000000e0 2.0000000000000001e-1</DataItem>
        <DataItem Name="connectivity" Dimensions="52" NumberType="UInt" Format="XML" Precision="8">5 0 1 4 3 5 1 2 5 4 5 3 4 7 6 5 4 5 8 7 4 0 1 9 4 3 0 10 4 1 2 11 4 2 5 12 4 6 3 13 4 6 7 14 4 7 8 15 4 5 8 16</DataItem>
    </Domain>
    <MetaData Version="0.1.0" Format="Xml"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "time_series_writer.xdmf").unwrap();

    pretty_assertions::assert_eq!(read_xdmf, expected_xdmf);
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

    let xdmf_writer = TimeSeriesWriter::new_with_options(
        &xdmf_file_path,
        &TimeSeriesWriter::options().format(xdmf::Format::XML),
    )
    .unwrap();

    xdmf_writer
        .write_mesh(&node_coords, (&connectivity, &cell_types))
        .unwrap();

    let expected_xdmf = r#"
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="mesh" GridType="Uniform">
            <Geometry GeometryType="XYZ">
                <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
            </Geometry>
            <Topology TopologyType="Mixed" NumberOfElements="12">
                <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
            </Topology>
        </Grid>
        <DataItem Name="coords" Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 5.0000000000000000e-1 -5.0000000000000000e-1 2.0000000000000001e-1 -5.0000000000000000e-1 5.0000000000000000e-1 2.0000000000000001e-1 1.5000000000000000e0 -5.0000000000000000e-1 2.0000000000000001e-1 2.5000000000000000e0 5.0000000000000000e-1 2.0000000000000001e-1 5.0000000000000000e-1 1.5000000000000000e0 2.0000000000000001e-1 5.0000000000000000e-1 2.5000000000000000e0 2.0000000000000001e-1 1.5000000000000000e0 2.5000000000000000e0 2.0000000000000001e-1 2.5000000000000000e0 1.5000000000000000e0 2.0000000000000001e-1</DataItem>
        <DataItem Name="connectivity" Dimensions="52" NumberType="UInt" Format="XML" Precision="8">5 0 1 4 3 5 1 2 5 4 5 3 4 7 6 5 4 5 8 7 4 0 1 9 4 3 0 10 4 1 2 11 4 2 5 12 4 6 3 13 4 6 7 14 4 7 8 15 4 5 8 16</DataItem>
    </Domain>
    <MetaData Version="0.1.0" Format="Xml"/>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "time_series_writer_only_mesh.xdmf").unwrap();

    pretty_assertions::assert_eq!(read_xdmf, expected_xdmf);
}

#[cfg(feature = "unstable-submesh-api")]
#[test]
fn write_xdmf_with_submeshes() {
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
    ];

    let num_nodes = node_coords.len() / 3;
    let num_cells = cell_types.len();

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer = TimeSeriesWriter::new_with_options(
        &xdmf_file_path,
        &TimeSeriesWriter::options().format(xdmf::Format::XML),
    )
    .unwrap();

    let point_indx = vec![0, 1, 2];
    let cell_indx = vec![0];

    let mut xdmf_writer = xdmf_writer
        .write_mesh_and_submeshes(
            &node_coords,
            (&connectivity, &cell_types),
            &[(
                "submesh_1".to_string(),
                xdmf::SubMesh {
                    point_indices: point_indx,
                    cell_indices: cell_indx,
                },
            )]
            .into_iter()
            .collect(),
        )
        .unwrap();

    for i in 0..3 {
        let point_data_1: Vec<f64> = (0..num_nodes).map(|j| j as f64 + i as f64).collect();
        let point_data_2: Vec<f64> = (0..num_nodes)
            .map(|j| 1. * j as f64 + 2. + i as f64)
            .collect();
        let cell_data: Vec<f64> = (0..num_cells)
            .map(|j| 1. * j as f64 + 1.5 * i as f64)
            .collect();

        let point_data = vec![
            (
                "point_data1".to_string(),
                (xdmf::AttributeType::Scalar, point_data_1.into()),
            ),
            (
                "point_data2".to_string(),
                (xdmf::AttributeType::Scalar, point_data_2.into()),
            ),
        ]
        .into_iter()
        .collect();

        let cell_data = vec![(
            "cell_data".to_string(),
            (xdmf::AttributeType::Scalar, cell_data.into()),
        )]
        .into_iter()
        .collect();
        xdmf_writer
            .write_data(&i.to_string(), Some(&point_data), Some(&cell_data))
            .unwrap();
    }

    let expected_xdmf = r#"
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">
            <Grid Name="time_series-t0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="8">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="0"/>
                <Attribute Name="point_data1" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">0.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="point_data2" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1 1.8000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="cell_data" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="8" NumberType="Float" Format="XML" Precision="8">0.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t1" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="8">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="1"/>
                <Attribute Name="point_data1" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">1.0000000000000000e0 2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="point_data2" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1 1.8000000000000000e1 1.9000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="cell_data" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="8" NumberType="Float" Format="XML" Precision="8">1.5000000000000000e0 2.5000000000000000e0 3.5000000000000000e0 4.5000000000000000e0 5.5000000000000000e0 6.5000000000000000e0 7.5000000000000000e0 8.5000000000000000e0</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t2" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="coords"]</DataItem>
                </Geometry>
                <Topology TopologyType="Mixed" NumberOfElements="8">
                    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="connectivity"]</DataItem>
                </Topology>
                <Time Value="2"/>
                <Attribute Name="point_data1" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">2.0000000000000000e0 3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1 1.8000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="point_data2" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="17" NumberType="Float" Format="XML" Precision="8">4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1 1.1000000000000000e1 1.2000000000000000e1 1.3000000000000000e1 1.4000000000000000e1 1.5000000000000000e1 1.6000000000000000e1 1.7000000000000000e1 1.8000000000000000e1 1.9000000000000000e1 2.0000000000000000e1</DataItem>
                </Attribute>
                <Attribute Name="cell_data" AttributeType="Scalar" Center="Cell">
                    <DataItem Dimensions="8" NumberType="Float" Format="XML" Precision="8">3.0000000000000000e0 4.0000000000000000e0 5.0000000000000000e0 6.0000000000000000e0 7.0000000000000000e0 8.0000000000000000e0 9.0000000000000000e0 1.0000000000000000e1</DataItem>
                </Attribute>
            </Grid>
        </Grid>
        <DataItem Name="coords" Dimensions="17 3" NumberType="Float" Format="XML" Precision="8">0.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 1.0000000000000000e0 0.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 1.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 2.0000000000000000e0 2.0000000000000000e0 0.0000000000000000e0 5.0000000000000000e-1 -5.0000000000000000e-1 2.0000000000000001e-1 -5.0000000000000000e-1 5.0000000000000000e-1 2.0000000000000001e-1 1.5000000000000000e0 -5.0000000000000000e-1 2.0000000000000001e-1 2.5000000000000000e0 5.0000000000000000e-1 2.0000000000000001e-1 5.0000000000000000e-1 1.5000000000000000e0 2.0000000000000001e-1 5.0000000000000000e-1 2.5000000000000000e0 2.0000000000000001e-1 1.5000000000000000e0 2.5000000000000000e0 2.0000000000000001e-1 2.5000000000000000e0 1.5000000000000000e0 2.0000000000000001e-1</DataItem>
        <DataItem Name="connectivity" Dimensions="36" NumberType="UInt" Format="XML" Precision="8">5 0 1 4 3 5 1 2 5 4 5 3 4 7 6 5 4 5 8 7 4 0 1 9 4 3 0 10 4 1 2 11 4 2 5 12</DataItem>
    </Domain>
</Xdmf>"#;

    let xdmf_file = xdmf_file_path.with_extension("xdmf");
    let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

    // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
    // std::fs::copy(xdmf_file, "time_series_writer_with_submeshes.xdmf").unwrap();

    pretty_assertions::assert_eq!(read_xdmf, expected_xdmf);
}

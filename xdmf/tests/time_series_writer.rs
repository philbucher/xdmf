use xdmf::TimeSeriesWriter;

#[test]
fn test_write_xdmf() {
    let xdmf_writer = TimeSeriesWriter::new_with_options(
        "test_output",
        &TimeSeriesWriter::options().format(xdmf::Format::XML),
    );
    let xdmf_writer_submeshes = TimeSeriesWriter::new_with_options(
        "test_output-with-submeshes",
        &TimeSeriesWriter::options().format(xdmf::Format::XML),
    );

    let mut xdmf_writer = xdmf_writer
        .write_mesh(
            &[0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1.],
            (&[1, 2, 3], &[xdmf::CellType::Triangle]),
        )
        .unwrap();

    let point_indx = vec![0, 1, 2];
    let cell_indx = vec![0];

    let mut xdmf_writer_submeshes = xdmf_writer_submeshes
        .write_mesh_and_submeshes(
            &[0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1.],
            (&[1, 2, 3], &[xdmf::CellType::Triangle]),
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
        let cell_data = vec![1. * i as f64, 1.5 * i as f64];
        let point_data_1 = vec![1., 2., 3., 6. * i as f64];
        let point_data_2 = vec![1. * i as f64, 2., 5., 6.];

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
        xdmf_writer_submeshes
            .write_data(&i.to_string(), Some(&point_data), Some(&cell_data))
            .unwrap();
    }
}

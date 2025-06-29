use ndarray::prelude::*;
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

    let cells = vec![(
        xdmf::TopologyType::Triangle,
        ArrayView2::from_shape((2, 3), &[0, 1, 2, 0, 2, 3]).unwrap(),
    )];

    let mut xdmf_writer = xdmf_writer
        .write_mesh(
            &ArrayView2::from_shape((4, 3), &[0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1.])
                .unwrap(),
            &cells,
        )
        .unwrap();

    let mut xdmf_writer_submeshes = xdmf_writer_submeshes
        .write_mesh_and_submeshes(
            &ArrayView2::from_shape((4, 3), &[0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1.])
                .unwrap(),
            &cells,
            &[
                ("submesh1".to_string(), 0..2, 0..2),
                ("submesh2".to_string(), 1..3, 1..3),
            ],
        )
        .unwrap();

    for i in 0..3 {
        let cell_data = vec![1. * i as f64, 1.5 * i as f64];
        let point_data_1 = vec![1., 2., 3., 6. * i as f64];
        let point_data_2 = vec![1. * i as f64, 2., 5., 6.];

        let data = vec![
            xdmf::Data::new_point_data(
                "point_data1",
                xdmf::AttributeType::Scalar,
                xdmf::Values::View1Df64(ArrayView1::from(&point_data_1)),
            ),
            xdmf::Data::new_point_data(
                "point_data2",
                xdmf::AttributeType::Scalar,
                xdmf::Values::View1Df64(ArrayView1::from(&point_data_2)),
            ),
            xdmf::Data::new_cell_data(
                "cell_data",
                xdmf::AttributeType::Scalar,
                xdmf::Values::View1Df64(ArrayView1::from(&cell_data)),
            ),
        ];
        xdmf_writer.write_data(&i.to_string(), &data).unwrap();
        xdmf_writer_submeshes
            .write_data(&i.to_string(), &data)
            .unwrap();
    }
}

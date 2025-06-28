use std::collections::HashMap;

use ndarray::prelude::*;
use xdmf::TimeSeriesWriter;

#[test]
fn test_write_xdmf() {
    let xdmf_writer = TimeSeriesWriter::new_with_options(
        "test_output",
        &TimeSeriesWriter::options().format(xdmf::Format::XML),
    );

    let mut cells = HashMap::new();
    cells.insert(
        xdmf::TopologyType::Triangle,
        ArrayView2::from_shape((3, 1), &[0, 1, 2]).unwrap(),
    );

    let mut xdmf_writer = xdmf_writer
        .write_mesh(
            &ArrayView2::from_shape((4, 3), &[0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1.])
                .unwrap(),
            &cells,
        )
        .unwrap();

    let data = vec![1., 2., 3., 4., 5., 6.];
    let data2 = vec![1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.];

    // Create 1D view
    let _view1 = ArrayView1::from(&data[..3]);
    // Create 2D view (2x3)
    let view2 = ArrayView2::from_shape((2, 3), &data[..]).unwrap();
    // Create 3D view (1x2x3)
    let _view3 = ArrayView3::from_shape((1, 2, 3), &data[..]).unwrap();
    // Create dynamic view (2x3)
    let _view_dyn = ArrayViewD::from_shape(ndarray::IxDyn(&[2, 3]), &data[..]).unwrap();

    let data = vec![
        xdmf::Data::new_point_data(
            "point_data",
            xdmf::AttributeType::Scalar,
            xdmf::Values::View1Df64(ArrayView1::from(&data2)),
        ),
        xdmf::Data::new_point_data(
            "point_data2",
            xdmf::AttributeType::Scalar,
            xdmf::Values::View1Df64(ArrayView1::from(&data2)),
        ),
    ];

    xdmf_writer.write_data("1.0", &data).unwrap();
}

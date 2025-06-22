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
        .add_mesh(
            &ArrayView2::from_shape((2, 3), &[1., 2., 3., 4., 5., 6.]).unwrap(),
            &cells,
        )
        .unwrap();

    let data = vec![1., 2., 3., 4., 5., 6.];

    // Create 1D view
    let _view1 = ArrayView1::from(&data[..3]);
    // Create 2D view (2x3)
    let view2 = ArrayView2::from_shape((2, 3), &data[..]).unwrap();
    // Create 3D view (1x2x3)
    let _view3 = ArrayView3::from_shape((1, 2, 3), &data[..]).unwrap();
    // Create dynamic view (2x3)
    let _view_dyn = ArrayViewD::from_shape(ndarray::IxDyn(&[2, 3]), &data[..]).unwrap();

    let mut data_map = HashMap::new();

    // let v_data_2d = xdmf::Values::View2Df64(view2);
    data_map.insert("data_arr".to_string(), view2.into());

    xdmf_writer
        .add_data("time2", &data_map, &HashMap::new())
        .unwrap();

    // This should create a file named "test_output.xdmf"
    assert!(xdmf_writer.write().is_ok());
}

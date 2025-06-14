use std::collections::HashMap;

use xdmf::TimeSeriesWriter;

#[test]
fn test_write_xdmf() {
    let mut xdmf_writer = TimeSeriesWriter::new_with_options(
        "test_output",
        TimeSeriesWriter::options()
            .format(xdmf::Format::HDF)
            .multiple_files(false),
    );

    let mut xdmf_writer = xdmf_writer
        .add_mesh(&[0.0, 0.0, 0.0], HashMap::new())
        .unwrap();

    let f_data = vec![1.0, 2.0, 3.0];
    let mut data = xdmf::Value::Float64(&f_data);

    let mut data_map = HashMap::new();
    data_map.insert("data1".to_string(), data);

    xdmf_writer
        .add_data("time2", &data_map, &HashMap::new())
        .unwrap();
    // xdmf_writer.add_data("time2", false);

    // This should create a file named "test_output.xdmf"
    assert!(xdmf_writer.write().is_ok());
}

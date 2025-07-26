use std::path::Path;
use std::time::Instant;
use vtkio::model::*;
use xdmf::TimeSeriesWriter;

fn create_mesh(
    num_nodes_x: usize,
    num_nodes_y: usize,
    num_nodes_z: usize,
) -> (Vec<[f64; 3]>, Vec<Vec<usize>>) {
    let min_x = 0.0;
    let max_x = 1.0 * num_nodes_x as f64;
    let min_y = 0.0;
    let max_y = 1.0 * num_nodes_y as f64;
    let min_z = 0.0;
    let max_z = 1.0 * num_nodes_z as f64;

    let mut coords = Vec::with_capacity(num_nodes_x * num_nodes_y * num_nodes_z);
    for i in 0..num_nodes_x {
        for j in 0..num_nodes_y {
            for k in 0..num_nodes_z {
                let x = min_x + i as f64 * (max_x - min_x) / (num_nodes_x - 1) as f64;
                let y = min_y + j as f64 * (max_y - min_y) / (num_nodes_y - 1) as f64;
                let z = min_z + k as f64 * (max_z - min_z) / (num_nodes_z - 1) as f64;
                coords.push([x, y, z]);
            }
        }
    }

    let mut connectivities_hexa =
        Vec::with_capacity((num_nodes_x - 1) * (num_nodes_y - 1) * (num_nodes_z - 1));
    for i in 0..num_nodes_x - 1 {
        for j in 0..num_nodes_y - 1 {
            for k in 0..num_nodes_z - 1 {
                let node_index = |i: usize, j: usize, k: usize| {
                    i * num_nodes_y * num_nodes_z + j * num_nodes_z + k
                };
                connectivities_hexa.push(vec![
                    node_index(i, j, k),
                    node_index(i + 1, j, k),
                    node_index(i + 1, j + 1, k),
                    node_index(i, j + 1, k),
                    node_index(i, j, k + 1),
                    node_index(i + 1, j, k + 1),
                    node_index(i + 1, j + 1, k + 1),
                    node_index(i, j + 1, k + 1),
                ]);
            }
        }
    }

    (coords, connectivities_hexa)
}

#[test]
fn compare_xdmf_to_vtk_formats() {
    const NUM_STEPS: usize = 1000;

    let base_path = Path::new("tests/xdmf_vtk_comparison");
    if base_path.exists() {
        std::fs::remove_dir_all(base_path).unwrap();
    }
    std::fs::create_dir_all(base_path).unwrap();

    let start_time = Instant::now();
    let (coords, connectivity) = create_mesh(10, 10, 10);

    let start_time_xdmf_mesh_write = Instant::now();
    let xdmf_writer = TimeSeriesWriter::new_with_options(
        &xdmf_file_path,
        &TimeSeriesWriter::options().format(xdmf::Format::XML),
    )
    .unwrap();

    let mut xdmf_writer = xdmf_writer
        .write_mesh(&coords, (&connectivity, &cell_types))
        .unwrap();

    let time_xdmf_mesh_write_xml = start_time_xdmf_mesh_write.elapsed();
    let time_xdmf_mesh_write_h5_single = start_time_xdmf_mesh_write.elapsed();
    let time_xdmf_mesh_write_h5_multiple = start_time_xdmf_mesh_write.elapsed();

    // writing pressure (scalar) and velocity (vector) data

    for step in 0..NUM_STEPS {
        let time = step as f64 * 0.1;
        let data = vec![1.0; coords.len()]; // Example data

        xdmf_writer
            .write_data(&data, &xdmf::DataType::Scalar, "example_data", time)
            .unwrap();
    }
}

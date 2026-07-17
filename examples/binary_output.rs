//! Demonstrates `DataStorage::Binary`: writes a small animated mesh using uncompressed raw
//! binary data files (instead of ASCII text or HDF5). Open the resulting file in Paraview to
//! check that the `Format="Binary"` `DataItem`s are read correctly.
//!
//! Run with: `cargo run --example binary_output`
#![allow(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "example code: panicking on error and printing the result are both intentional here"
)]

use xdmf::{CellType, DataAttribute, DataStorage, TimeSeriesWriter};

fn main() {
    // 3x3 grid of points (9 points), 4 quads
    let mut points = Vec::new();
    for row in 0..3 {
        for col in 0..3 {
            points.extend_from_slice(&[col as f64, row as f64, 0.0]);
        }
    }

    let point_index = |row: usize, col: usize| (row * 3 + col) as u64;

    let mut connectivity = Vec::new();
    for row in 0..2 {
        for col in 0..2 {
            connectivity.extend_from_slice(&[
                point_index(row, col),
                point_index(row, col + 1),
                point_index(row + 1, col + 1),
                point_index(row + 1, col),
            ]);
        }
    }
    let cell_types = vec![CellType::Quadrilateral; 4];

    let xdmf_writer = TimeSeriesWriter::new("binary_output_example", DataStorage::Binary)
        .expect("failed to create XDMF writer");

    let mut ts_writer = xdmf_writer
        .write_mesh(&points, (&connectivity, &cell_types))
        .expect("failed to write mesh");

    let num_points = points.len() / 3;
    let num_cells = cell_types.len();

    for t in 0..3 {
        let time = t as f64;

        let height: xdmf::Values = (0..num_points)
            .map(|i| {
                let x = points[i * 3];
                let y = points[i * 3 + 1];
                (time + x * 0.5 + y * 0.3).sin()
            })
            .collect::<Vec<f64>>()
            .into();

        let cell_id: xdmf::Values = (0..num_cells).map(|i| i as u64).collect::<Vec<u64>>().into();

        let point_data = [("height", DataAttribute::Scalar, &height)];
        let cell_data = [("cell_id", DataAttribute::Scalar, &cell_id)];

        ts_writer
            .write_data(&t.to_string(), point_data, cell_data)
            .expect("failed to write time step data");
    }

    println!("wrote binary_output_example.xdmf2");
}

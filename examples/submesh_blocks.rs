//! Demonstrates `TimeSeriesWriter::write_mesh_with_blocks`: a single shared mesh (a 3x3 grid of
//! quads with a small triangular "fin" on one edge) split into several named, overlapping
//! submeshes, animated over 3 time steps. Open the resulting file in Paraview and use the
//! Multi-block Inspector (View -> Multi-block Inspector) to see the individual blocks.
//!
//! Run with: `cargo run --example submesh_blocks`
#![allow(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "example code: panicking on error and printing the result are both intentional here"
)]

use std::collections::BTreeSet;

use xdmf::{CellType, DataAttribute, DataStorage, TimeSeriesWriter};

fn point_index(row: usize, col: usize) -> u64 {
    (row * 4 + col) as u64
}

fn main() {
    // 4x4 grid of points (16 points) spanning a 3x3 grid of quads, plus 2 extra "apex" points
    // for a small triangular fin sticking up off the top edge (18 points total).
    let mut points = Vec::new();
    for row in 0..4 {
        for col in 0..4 {
            points.extend_from_slice(&[col as f64, row as f64, 0.0]);
        }
    }
    // apex points for the fin (index 16 and 17)
    points.extend_from_slice(&[0.5, 3.0, 1.5]);
    points.extend_from_slice(&[2.5, 3.0, 1.5]);

    // 9 quads (row-major, cell index = row * 3 + col) + 2 triangles for the fin (indices 9, 10)
    let mut connectivity = Vec::new();
    for row in 0..3 {
        for col in 0..3 {
            connectivity.extend_from_slice(&[
                point_index(row, col),
                point_index(row, col + 1),
                point_index(row + 1, col + 1),
                point_index(row + 1, col),
            ]);
        }
    }
    connectivity.extend_from_slice(&[point_index(3, 0), point_index(3, 1), 16]);
    connectivity.extend_from_slice(&[point_index(3, 2), point_index(3, 3), 17]);

    let mut cell_types = vec![CellType::Quadrilateral; 9];
    cell_types.extend_from_slice(&[CellType::Triangle, CellType::Triangle]);

    // Blocks deliberately overlap (e.g. "middle_row" shares a cell with both "left_column" and
    // "right_column", "top_row_with_fin" mixes cell types and shares a cell with "right_column")
    // to show that a cell can belong to more than one block. Every cell must belong to at least
    // one block, so "front_row" picks up cell 1, which none of the other blocks reference.
    let left_column: BTreeSet<usize> = [0, 3, 6].into();
    let right_column: BTreeSet<usize> = [2, 5, 8].into();
    let front_row: BTreeSet<usize> = [0, 1, 2].into();
    let middle_row: BTreeSet<usize> = [3, 4, 5].into();
    let fin: BTreeSet<usize> = [9, 10].into();
    let top_row_with_fin: BTreeSet<usize> = [6, 7, 8, 9, 10].into();

    let blocks = [
        ("left_column", &left_column),
        ("right_column", &right_column),
        ("front_row", &front_row),
        ("middle_row", &middle_row),
        ("fin", &fin),
        ("top_row_with_fin", &top_row_with_fin),
    ];

    let xdmf_writer = TimeSeriesWriter::new("dummy/submesh_blocks_example", DataStorage::Ascii)
        .expect("failed to create XDMF writer");

    let mut ts_writer = xdmf_writer
        .write_mesh_with_blocks(&points, (&connectivity, &cell_types), &blocks)
        .expect("failed to write mesh with blocks");

    let num_points = points.len() / 3;
    let num_cells = cell_types.len();

    // Buffers are allocated once and overwritten in place every time step (rather than
    // collected into a fresh `Vec` each iteration), since `write_data` only borrows its inputs.
    let mut height_wave: xdmf::Values = vec![0.0; num_points].into();
    let mut displacement: xdmf::Values = vec![0.0; num_points * 3].into();
    // `cell_id` never changes across time steps, so it is computed just once.
    let cell_id: xdmf::Values = (0..num_cells)
        .map(|i| i as f64)
        .collect::<Vec<f64>>()
        .into();
    let mut activity: xdmf::Values = vec![0.0; num_cells].into();

    for t in 0..3 {
        let time = t as f64;

        let height_wave_buf = height_wave
            .as_mut_slice::<f64>()
            .expect("height_wave holds f64 data");
        for (i, val) in height_wave_buf.iter_mut().enumerate() {
            let x = points[i * 3];
            let y = points[i * 3 + 1];
            *val = (time + x * 0.5 + y * 0.3).sin();
        }

        // x and y displacement stay 0.0, as set when the buffer was allocated.
        let displacement_buf = displacement
            .as_mut_slice::<f64>()
            .expect("displacement holds f64 data");
        for i in 0..num_points {
            let x = points[i * 3];
            displacement_buf[i * 3 + 2] = 0.3 * (time + x).sin();
        }

        let activity_buf = activity
            .as_mut_slice::<f64>()
            .expect("activity holds f64 data");
        for (i, val) in activity_buf.iter_mut().enumerate() {
            *val = i as f64 + time * 10.0;
        }

        let point_data = [
            ("height_wave", DataAttribute::Scalar, &height_wave),
            ("displacement", DataAttribute::Vector, &displacement),
        ];

        let cell_data = [
            ("cell_id", DataAttribute::Scalar, &cell_id),
            ("activity", DataAttribute::Scalar, &activity),
        ];

        ts_writer
            .write_data(&t.to_string(), point_data, cell_data)
            .expect("failed to write time step data");
    }

    println!("wrote submesh_blocks_example.xdmf2");
}

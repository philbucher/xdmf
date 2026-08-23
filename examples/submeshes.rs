//! Demonstrates [`TimeSeriesWriter::write_mesh_with_submeshes`]: one shared mesh -- a 3x3 grid of
//! quads with a small triangular "fin" on one edge -- split into several named, overlapping
//! submeshes and animated over 3 time steps.
//!
//! Open the resulting file in `ParaView` and use the Multi-block Inspector
//! (`View -> Multi-block Inspector`) to show, hide and colour the submeshes individually. Not the
//! `Grids` list in the Properties panel: that one holds a reader-side grid per submesh *and* time
//! step, so a selection made there falls apart as the animation runs.
//!
//! Run with `cargo run --example submeshes`, which writes `submeshes_example/mesh.xdmf2` (and
//! its heavy data next to it) into the current directory.

use xdmf::{CellType, DataAttribute, DataStorage, TimeSeriesWriter};

const COLUMNS: usize = 4;

fn point_index(row: usize, column: usize) -> u32 {
    (row * COLUMNS + column) as u32
}

fn main() -> xdmf::Result<()> {
    // 4x4 grid of points spanning a 3x3 grid of quads, plus two apex points (16 and 17) for a
    // small triangular fin sticking up off the top edge.
    let mut points = Vec::new();
    for row in 0..4 {
        for column in 0..COLUMNS {
            points.extend_from_slice(&[column as f64, row as f64, 0.0]);
        }
    }
    points.extend_from_slice(&[0.5, 3.0, 1.5]);
    points.extend_from_slice(&[2.5, 3.0, 1.5]);

    // 9 quads (cell index = row * 3 + column), then the fin's 2 triangles (cells 9 and 10)
    let mut connectivity = Vec::new();
    for row in 0..3 {
        for column in 0..3 {
            connectivity.extend_from_slice(&[
                point_index(row, column),
                point_index(row, column + 1),
                point_index(row + 1, column + 1),
                point_index(row + 1, column),
            ]);
        }
    }
    connectivity.extend_from_slice(&[point_index(3, 0), point_index(3, 1), 16]);
    connectivity.extend_from_slice(&[point_index(3, 2), point_index(3, 3), 17]);

    let mut cell_types = vec![CellType::Quadrilateral; 9];
    cell_types.extend_from_slice(&[CellType::Triangle, CellType::Triangle]);

    let num_points = points.len() / 3;
    let num_cells = cell_types.len();

    // The submeshes deliberately overlap: "top_row_with_fin" shares cells with both "right_column"
    // and "fin", and mixes cell types while doing so. Every cell has to be in at least one of
    // them, which is why "front_row" is here at all -- it picks up cell 1, which no other submesh
    // references.
    //
    // "left_column" and "right_column" are scattered (their cells are strided), the rest are
    // ascending runs, which the writer recognizes and stores as a range.
    let submeshes = [
        ("left_column", vec![0, 3, 6]),
        ("right_column", vec![2, 5, 8]),
        ("front_row", vec![0, 1, 2]),
        ("middle_row", vec![3, 4, 5]),
        ("fin", vec![9, 10]),
        ("top_row_with_fin", vec![6, 7, 8, 9, 10]),
    ];

    let xdmf_writer = TimeSeriesWriter::new("submeshes_example/mesh", DataStorage::Ascii)?;

    let mut ts_writer =
        xdmf_writer.write_mesh_with_submeshes(&points, &connectivity, &cell_types, submeshes)?;

    // Data is produced over the whole mesh, exactly as without submeshes -- the writer gives each
    // submesh its share. One buffer per field is allocated here and refilled in place every step,
    // since each attribute is written the moment it is passed.
    let mut height = vec![0.0; num_points];
    let mut displacement = vec![0.0; num_points * 3];
    let mut activity = vec![0.0; num_cells];
    // never changes across time steps, so it is filled just once
    let cell_id: Vec<f64> = (0..num_cells).map(|cell| cell as f64).collect();

    for step in 0..3 {
        let time = step as f64;

        for (index, value) in height.iter_mut().enumerate() {
            let (x, y) = (points[index * 3], points[index * 3 + 1]);
            *value = (time + x * 0.5 + y * 0.3).sin();
        }

        // x and y displacement stay at the 0.0 they were allocated with
        for index in 0..num_points {
            displacement[index * 3 + 2] = 0.3 * (time + points[index * 3]).sin();
        }

        for (index, value) in activity.iter_mut().enumerate() {
            *value = index as f64 + time * 10.0;
        }

        ts_writer.write_time_step(&time.to_string(), |step| {
            step.point_data("height", DataAttribute::Scalar, &height)?;
            step.point_data("displacement", DataAttribute::Vector, &displacement)?;
            step.cell_data("cell_id", DataAttribute::Scalar, &cell_id)?;
            step.cell_data("activity", DataAttribute::Scalar, &activity)
        })?;
    }

    Ok(())
}

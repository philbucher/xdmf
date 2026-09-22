//! Companion to `tests/paraview_smoke/write_with_paraview.py`: checks that this crate's own
//! [`TimeSeriesReader`] reads back the fixture that script writes with `ParaView`'s own
//! `vtkXdmfWriter`, correctly and without any of the leniency fixes that script's own doc comment
//! lists being needed again.
//!
//! Usage: `cargo run --example read_paraview_fixture -- <xdmf_file>`

use std::{
    env,
    io::{Error as IoError, ErrorKind::InvalidInput, Result as IoResult},
    path::Path,
};

use xdmf::{CellType, TimeSeriesReader};

const NUM_POINTS: usize = 5;
const NUM_CELLS: usize = 2;

const COORDS: [f64; NUM_POINTS * 3] = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.5, 0.0,
];
const CONNECTIVITY: [i64; 7] = [0, 1, 2, 3, 1, 4, 2];
const CELL_TYPES: [CellType; NUM_CELLS] = [CellType::Tetrahedron, CellType::Triangle];
const LEVEL_I32: [i32; NUM_CELLS] = [-2_000_000_000, 2_000_000_000];

fn main() -> IoResult<()> {
    let args: Vec<String> = env::args().collect();
    let [_, xdmf_file] = args.as_slice() else {
        return Err(IoError::new(
            InvalidInput,
            "usage: read_paraview_fixture <xdmf_file>",
        ));
    };
    let xdmf_file = Path::new(xdmf_file);

    let reader = TimeSeriesReader::new(xdmf_file)?;

    if reader.num_points() != NUM_POINTS {
        return Err(mismatch(xdmf_file, "num_points"));
    }
    if reader.num_cells() != NUM_CELLS {
        return Err(mismatch(xdmf_file, "num_cells"));
    }
    if reader.times() != ["0", "1"] {
        return Err(mismatch(xdmf_file, "times"));
    }

    let mut points: Vec<f64> = Vec::new();
    reader.read_points(&mut points)?;
    check_floats(xdmf_file, "points", &COORDS, &points)?;

    let mut connectivity: Vec<i64> = Vec::new();
    let mut cell_types = Vec::new();
    reader.read_topology(&mut connectivity, &mut cell_types)?;
    if connectivity != CONNECTIVITY || cell_types != CELL_TYPES {
        return Err(mismatch(xdmf_file, "topology"));
    }

    let mut temperature: Vec<f64> = Vec::new();
    let mut level_i32: Vec<i32> = Vec::new();
    for (step, scale) in [1.0, 2.0].into_iter().enumerate() {
        reader.read_point_data(step, "temperature", &mut temperature)?;
        let expected_temperature: Vec<f64> = [10.0, 11.0, 12.0, 13.0, 14.0]
            .into_iter()
            .map(|value| value * scale)
            .collect();
        check_floats(
            xdmf_file,
            "temperature",
            &expected_temperature,
            &temperature,
        )?;

        reader.read_cell_data(step, "level_i32", &mut level_i32)?;
        if level_i32 != LEVEL_I32 {
            return Err(mismatch(xdmf_file, "level_i32"));
        }
    }

    #[expect(
        clippy::print_stdout,
        reason = "CLI progress output expected from an example binary"
    )]
    {
        println!("OK: {} read back correctly", xdmf_file.display());
    }

    Ok(())
}

/// Compares two `f64` arrays through [`approx`], not `==` (`clippy::float_cmp` at the crate level)
/// and not a panicking assertion (`clippy::panic_in_result_fn` forbids one inside a function
/// returning [`IoResult`]).
fn check_floats(xdmf_file: &Path, field: &str, expected: &[f64], actual: &[f64]) -> IoResult<()> {
    let equal = expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(&e, &a)| approx::relative_eq!(e, a));

    if equal {
        Ok(())
    } else {
        Err(mismatch(xdmf_file, field))
    }
}

fn mismatch(xdmf_file: &Path, field: &str) -> IoError {
    IoError::new(
        InvalidInput,
        format!(
            "{}: reader read back '{field}' incorrectly",
            xdmf_file.display()
        ),
    )
}

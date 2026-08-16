//! Fixture generator for the `ParaView` compatibility smoke test (see `.github/workflows/paraview.yml`).
//! Writes a tiny XDMF time series with the requested `DataStorage` backend, plus an
//! `expected.json` recording the values written, so `tests/paraview_smoke/verify_with_pvpython.py`
//! can reopen the file in `ParaView` and check the two agree. The mesh mixes cell types
//! (Quadrilateral + Triangle) and the data fields mix `DataAttribute` variants (Scalar, Vector,
//! Tensor, Tensor6) so the verification script can also confirm `ParaView` reads back the correct
//! number of components per field, not just the right numeric values.
//!
//! Two fixtures are written per run, one with f64 coordinates and float attributes and one with
//! f32 ones, since the two produce different bytes *and* a different `Precision` in the light data.
//!
//! Usage: `cargo run --example paraview_smoke -- <output_dir> <storage>`
//! `<storage>` is any string accepted by `xdmf::DataStorage::from_str` (e.g. `Hdf5SingleFile`).

use std::{
    env,
    io::{Error as IoError, ErrorKind::InvalidInput, Result as IoResult},
    path::Path,
};

use serde::Serialize;
use xdmf::{CellType, DataAttribute, DataStorage, TimeSeriesWriter, Values};

const NUM_POINTS: usize = 5;
const NUM_CELLS: usize = 2;

// a quad and a triangle sharing an edge, to exercise a mixed-cell-type mesh
const COORDS: [f64; NUM_POINTS * 3] = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.5, 0.0,
];
const CONNECTIVITY: [u64; 7] = [0, 1, 2, 3, 1, 4, 2];
const CELL_TYPES: [CellType; NUM_CELLS] = [CellType::Quadrilateral, CellType::Triangle];
const REGION_ID: [u64; NUM_CELLS] = [100, 200];

#[derive(Serialize)]
struct ExpectedTimestep {
    time: f64,
    temperature: Vec<f64>,
    displacement: Vec<[f64; 3]>,
    velocity_gradient: Vec<[f64; 9]>,
    region_id: Vec<u64>,
    stress: Vec<[f64; 6]>,
}

#[derive(Serialize)]
struct ExpectedFixture {
    xdmf_file: String,
    points: Vec<Vec<f64>>,
    timesteps: Vec<ExpectedTimestep>,
}

#[derive(Serialize)]
struct Expected {
    fixtures: Vec<ExpectedFixture>,
}

/// Which width the coordinates and the float attributes of a fixture are written at.
#[derive(Clone, Copy, PartialEq)]
enum Precision {
    F64,
    F32,
}

impl Precision {
    /// Round every value to what it becomes as an `f32`, so the recorded expectations are the
    /// values `ParaView` must read back rather than the ones that went in.
    fn narrow(self, values: &mut [f64]) {
        if self == Self::F32 {
            for value in values {
                *value = f64::from(*value as f32);
            }
        }
    }
}

fn main() -> IoResult<()> {
    let args: Vec<String> = env::args().collect();
    let [_, output_dir, storage_arg] = args.as_slice() else {
        return Err(IoError::new(
            InvalidInput,
            "usage: paraview_smoke <output_dir> <storage>",
        ));
    };

    let storage: DataStorage = storage_arg
        .parse()
        .map_err(|e| IoError::new(InvalidInput, e))?;

    let output_dir = Path::new(output_dir);

    let expected = Expected {
        fixtures: vec![
            write_fixture(output_dir, storage_arg, storage, Precision::F64)?,
            write_fixture(output_dir, storage_arg, storage, Precision::F32)?,
        ],
    };

    let expected_json =
        serde_json::to_string(&expected).map_err(|e| IoError::new(InvalidInput, e.to_string()))?;
    std::fs::write(output_dir.join("expected.json"), expected_json)?;

    #[expect(
        clippy::print_stdout,
        reason = "CLI progress output expected from an example binary"
    )]
    {
        for fixture in &expected.fixtures {
            println!(
                "Wrote fixture to {}",
                output_dir.join(&fixture.xdmf_file).display()
            );
        }
    }

    Ok(())
}

fn write_fixture(
    output_dir: &Path,
    storage_arg: &str,
    storage: DataStorage,
    precision: Precision,
) -> IoResult<ExpectedFixture> {
    let suffix = match precision {
        Precision::F64 => "",
        Precision::F32 => "_f32",
    };
    let base_path = output_dir.join(format!("fixture_{}{suffix}", storage_arg.to_lowercase()));

    let mut coords = COORDS;
    precision.narrow(&mut coords);

    let xdmf_writer = TimeSeriesWriter::new(&base_path, storage)?;
    let mut xdmf_writer = match precision {
        Precision::F64 => xdmf_writer.write_mesh(&coords, &CONNECTIVITY, &CELL_TYPES)?,
        Precision::F32 => {
            let coords_f32: Vec<f32> = coords.iter().map(|&v| v as f32).collect();
            xdmf_writer.write_mesh(&coords_f32, &CONNECTIVITY, &CELL_TYPES)?
        }
    };

    let mut timesteps = Vec::new();
    for (step, scale) in [1.0, 2.0].into_iter().enumerate() {
        let mut temperature: Vec<f64> = [10.0, 11.0, 12.0, 13.0, 14.0]
            .into_iter()
            .map(|v| v * scale)
            .collect();

        let mut displacement: Vec<[f64; 3]> = (0..NUM_POINTS)
            .map(|i| [i as f64 * 0.1 * scale, i as f64 * 0.2 * scale, 0.0])
            .collect();

        let mut velocity_gradient: Vec<[f64; 9]> = (0..NUM_POINTS)
            .map(|i| std::array::from_fn(|j| (i * 9 + j) as f64 * scale))
            .collect();

        let mut stress: Vec<[f64; 6]> = (0..NUM_CELLS)
            .map(|i| {
                let base = (i + 1) as f64 * scale;
                [
                    base,
                    2.0 * base,
                    3.0 * base,
                    4.0 * base,
                    5.0 * base,
                    6.0 * base,
                ]
            })
            .collect();

        precision.narrow(&mut temperature);
        precision.narrow(displacement.as_flattened_mut());
        precision.narrow(velocity_gradient.as_flattened_mut());
        precision.narrow(stress.as_flattened_mut());

        timesteps.push(ExpectedTimestep {
            time: step as f64,
            temperature: temperature.clone(),
            displacement: displacement.clone(),
            velocity_gradient: velocity_gradient.clone(),
            region_id: REGION_ID.to_vec(),
            stress: stress.clone(),
        });

        xdmf_writer.write_time_step(&step.to_string(), |time_step| {
            // `as_flattened` reinterprets `&[[f64; N]]` as `&[f64]` without copying, so the
            // natural per-point/per-cell layout needs no intermediate `Vec`
            time_step.point_data(
                "temperature",
                DataAttribute::Scalar,
                at_precision(&temperature, precision),
            )?;
            time_step.point_data(
                "displacement",
                DataAttribute::Vector,
                at_precision(displacement.as_flattened(), precision),
            )?;
            time_step.point_data(
                "velocity_gradient",
                DataAttribute::Tensor,
                at_precision(velocity_gradient.as_flattened(), precision),
            )?;

            // integer data is unaffected by the fixture's float precision
            time_step.cell_data("region_id", DataAttribute::Scalar, &REGION_ID)?;
            time_step.cell_data(
                "stress",
                DataAttribute::Tensor6,
                at_precision(stress.as_flattened(), precision),
            )
        })?;
    }

    let xdmf_file = base_path
        .with_extension("xdmf2")
        .file_name()
        .ok_or_else(|| IoError::new(InvalidInput, "invalid output file name"))?
        .to_string_lossy()
        .into_owned();

    Ok(ExpectedFixture {
        xdmf_file,
        points: coords.chunks_exact(3).map(<[f64]>::to_vec).collect(),
        timesteps,
    })
}

/// The attribute values at the fixture's precision: the f64 values borrowed, or an f32 copy.
fn at_precision(values: &[f64], precision: Precision) -> Values<'_> {
    match precision {
        Precision::F64 => values.into(),
        Precision::F32 => values
            .iter()
            .map(|&v| v as f32)
            .collect::<Vec<f32>>()
            .into(),
    }
}

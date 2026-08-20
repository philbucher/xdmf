//! Fixture generator for the `ParaView` compatibility smoke test (see `.github/workflows/paraview.yml`).
//! Writes a tiny XDMF time series with the requested `DataStorage` backend, plus an
//! `expected.json` recording the values written, so `tests/paraview_smoke/verify_with_pvpython.py`
//! can reopen the file in `ParaView` and check the two agree. The mesh mixes cell types
//! (Quadrilateral + Triangle) and the data fields mix `DataAttribute` variants (Scalar, Vector,
//! Tensor, Tensor6) so the verification script can also confirm `ParaView` reads back the correct
//! number of components per field, not just the right numeric values.
//!
//! The cell data covers every integer element type the writer supports (`u64`, `i64`, `u32`,
//! `i32`) at the edges of its range, which is what pins down the 64-bit handling. `ParaView` reads
//! 64-bit integers correctly from the ascii and HDF5 storages but at the wrong stride from
//! `Format="Binary"`, so `DataStorage::Binary` refuses `i64`/`u64` outright and its fixtures carry
//! the 32-bit fields only.
//!
//! One fixture is written per (coordinate precision, connectivity index type) pair: f64 and f32
//! coordinates produce different bytes *and* a different `Precision` in the light data, and each
//! of the four index types (`u32`, `u64`, `i32`, `i64`) gives the connectivity a different
//! `NumberType`/`Precision` pair, which is what decides how large a mesh can be written. The
//! verification script checks the cells come back with the right type and point ids for each.
//!
//! Usage: `cargo run --example paraview_smoke -- <output_dir> <storage>`
//! `<storage>` is any string accepted by `xdmf::DataStorage::from_str` (e.g. `Hdf5SingleFile`).

use std::{
    env,
    io::{Error as IoError, ErrorKind::InvalidInput, Result as IoResult},
    path::Path,
};

use serde::Serialize;
use xdmf::{
    CellType, Coordinate, DataAttribute, DataStorage, TimeSeriesDataWriter, TimeSeriesWriter,
    Values,
};

const NUM_POINTS: usize = 5;
const NUM_CELLS: usize = 2;

// a quad and a triangle sharing an edge, to exercise a mixed-cell-type mesh
const COORDS: [f64; NUM_POINTS * 3] = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.5, 0.0,
];
const CONNECTIVITY: [u64; 7] = [0, 1, 2, 3, 1, 4, 2];
const CELL_TYPES: [CellType; NUM_CELLS] = [CellType::Quadrilateral, CellType::Triangle];

// What the connectivity above has to come back as, named the way VTK names the cell classes. The
// connectivity is the one array whose element type the caller picks, so a fixture that got its
// `NumberType`/`Precision` wrong shows up here as a mangled or missing cell rather than as a
// slightly-off number.
const EXPECTED_CELLS: [(&str, &[u64]); NUM_CELLS] =
    [("vtkQuad", &[0, 1, 2, 3]), ("vtkTriangle", &[1, 4, 2])];

const REGION_ID: [u64; NUM_CELLS] = [100, 200];
// the 32-bit fields cover both ends of their range, so a reader that gets the signedness wrong
// (u32 read as i32, or the sign bit of a negative i32 dropped) cannot pass
const LEVEL_I32: [i32; NUM_CELLS] = [i32::MIN, i32::MAX];
const FLAG_U32: [u32; NUM_CELLS] = [0, u32::MAX];
const LEVEL_I64: [i64; NUM_CELLS] = [i32::MIN as i64, i32::MAX as i64];
// deliberately out of 32-bit range, so the storages that carry 64-bit integers are checked on
// values that cannot survive unless they really are read back as 64-bit.
//
// +/-(2^53 - 1) rather than something wider, because that is the largest magnitude that reads back
// exactly from *every* storage: ParaView parses the ascii formats' integers through a double, so
// 2^53 + 1 comes back off by one there (HDF5 and Binary are exact). Signed because it has to be --
// ParaView's Xdmf2 reader builds a 32-bit array for `NumberType="UInt"` whatever `Precision` says,
// so the writer refuses a `u64` beyond `u32::MAX` outright, for every storage. See the README.
const LEVEL_I64_WIDE: [i64; NUM_CELLS] = [-9_007_199_254_740_991, 9_007_199_254_740_991];

/// One integer cell-data field, as `ParaView` must read it back. Every element type the writer
/// supports gets one, since the light data's `NumberType`/`Precision` pair -- and, for
/// `DataStorage::Binary`, the narrowing to 32 bits -- is what a reader has to agree with.
#[derive(Serialize)]
struct ExpectedIntegerField {
    name: String,
    values: Vec<i128>,
}

impl ExpectedIntegerField {
    fn new(name: &str, values: impl IntoIterator<Item = impl Into<i128>>) -> Self {
        Self {
            name: name.to_string(),
            values: values.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Serialize)]
struct ExpectedTimestep {
    time: f64,
    temperature: Vec<f64>,
    displacement: Vec<[f64; 3]>,
    velocity_gradient: Vec<[f64; 9]>,
    integers: Vec<ExpectedIntegerField>,
    stress: Vec<[f64; 6]>,
}

/// One cell as `ParaView` must read it back: the VTK class it becomes, and the point ids it spans.
#[derive(Serialize)]
struct ExpectedCell {
    r#type: String,
    points: Vec<u64>,
}

#[derive(Serialize)]
struct ExpectedFixture {
    xdmf_file: String,
    points: Vec<Vec<f64>>,
    cells: Vec<ExpectedCell>,
    timesteps: Vec<ExpectedTimestep>,
}

#[derive(Serialize)]
struct Expected {
    /// Which storage wrote these, so the verification script knows how many fixtures to expect --
    /// `Binary` carries fewer, having no 64-bit integer types.
    storage: String,
    fixtures: Vec<ExpectedFixture>,
}

/// Which width the coordinates and the float attributes of a fixture are written at.
///
/// The two methods below feed the two sides of the comparison the verification script makes, and
/// are both needed: [`narrow_expected`](Self::narrow_expected) fixes up what `expected.json` says
/// `ParaView` must read back, [`values`](Self::values) fixes up what is actually written. They stay
/// separate rather than becoming one call because the f64 case writes a *borrow* of the same buffer
/// the expectations are taken from, which rules out handing out a `&mut` and a `&` at once.
#[derive(Clone, Copy, PartialEq)]
enum Precision {
    F64,
    F32,
}

impl Precision {
    /// Rounds every value to what it becomes as an `f32`, so the recorded expectations are the
    /// values `ParaView` must read back rather than the ones that went in.
    ///
    /// Not a no-op on anything but round numbers: `0.1_f64 as f32` widens back to
    /// `0.10000000149011612`, and the script compares exactly.
    fn narrow_expected(self, values: &mut [f64]) {
        if self == Self::F32 {
            for value in values {
                *value = f64::from(*value as f32);
            }
        }
    }

    /// The values as they are written at this precision: the `f64` values borrowed, or an `f32`
    /// copy.
    ///
    /// Lossless in practice, since every caller narrows first.
    fn values(self, values: &[f64]) -> Values<'_> {
        match self {
            Self::F64 => values.into(),
            Self::F32 => values
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>()
                .into(),
        }
    }
}

/// Which integer type the connectivity of a fixture is written as.
///
/// The caller picks this, and it is what caps the mesh size: `UInt` (`u32`/`u64`) connectivity is
/// decoded at 32 bits by `ParaView` whatever precision is declared, while `Int` keeps its width.
#[derive(Clone, Copy)]
enum IndexType {
    U32,
    U64,
    I32,
    I64,
}

impl IndexType {
    /// The types a given storage can carry: `Binary` takes the 32-bit ones only, since `ParaView`
    /// reads 64-bit integers in `Format="Binary"` at the wrong stride and the writer refuses them.
    fn all_for(storage: DataStorage) -> &'static [Self] {
        if matches!(storage, DataStorage::Binary) {
            &[Self::U32, Self::I32]
        } else {
            &[Self::U32, Self::U64, Self::I32, Self::I64]
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }

    /// Writes the mesh with the connectivity converted to this type.
    ///
    /// A method rather than a value, since the index type is a type parameter of `write_mesh` and
    /// so cannot be passed through as data.
    fn write_mesh<C: Coordinate>(
        self,
        writer: TimeSeriesWriter,
        coords: &[C],
    ) -> IoResult<TimeSeriesDataWriter> {
        let mesh = match self {
            Self::U32 => writer.write_mesh(coords, &CONNECTIVITY.map(|i| i as u32), &CELL_TYPES),
            Self::U64 => writer.write_mesh(coords, &CONNECTIVITY, &CELL_TYPES),
            Self::I32 => writer.write_mesh(coords, &CONNECTIVITY.map(|i| i as i32), &CELL_TYPES),
            Self::I64 => writer.write_mesh(coords, &CONNECTIVITY.map(|i| i as i64), &CELL_TYPES),
        };

        Ok(mesh?)
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

    let mut fixtures = Vec::new();
    for precision in [Precision::F64, Precision::F32] {
        for &index_type in IndexType::all_for(storage) {
            fixtures.push(write_fixture(
                output_dir,
                storage_arg,
                storage,
                precision,
                index_type,
            )?);
        }
    }

    let expected = Expected {
        storage: storage_arg.to_lowercase(),
        fixtures,
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
    index_type: IndexType,
) -> IoResult<ExpectedFixture> {
    let suffix = match precision {
        Precision::F64 => "f64",
        Precision::F32 => "f32",
    };
    let base_path = output_dir.join(format!(
        "fixture_{}_{suffix}_{}",
        storage_arg.to_lowercase(),
        index_type.suffix()
    ));

    // `Binary` cannot carry 64-bit integers at all -- ParaView reads them at the wrong stride --
    // so it refuses them and its fixtures carry the 32-bit fields only
    let writes_64_bit_integers = !matches!(storage, DataStorage::Binary);

    let mut coords = COORDS;
    precision.narrow_expected(&mut coords);

    let xdmf_writer = TimeSeriesWriter::new(&base_path, storage)?;
    let mut xdmf_writer = match precision {
        Precision::F64 => index_type.write_mesh(xdmf_writer, &coords)?,
        Precision::F32 => {
            let coords_f32: Vec<f32> = coords.iter().map(|&v| v as f32).collect();
            index_type.write_mesh(xdmf_writer, &coords_f32)?
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

        precision.narrow_expected(&mut temperature);
        precision.narrow_expected(displacement.as_flattened_mut());
        precision.narrow_expected(velocity_gradient.as_flattened_mut());
        precision.narrow_expected(stress.as_flattened_mut());

        let mut integers = vec![
            ExpectedIntegerField::new("level_i32", LEVEL_I32),
            ExpectedIntegerField::new("flag_u32", FLAG_U32),
        ];
        if writes_64_bit_integers {
            integers.push(ExpectedIntegerField::new("region_id", REGION_ID));
            integers.push(ExpectedIntegerField::new("level_i64", LEVEL_I64));
            integers.push(ExpectedIntegerField::new("level_i64_wide", LEVEL_I64_WIDE));
        }

        timesteps.push(ExpectedTimestep {
            time: step as f64,
            temperature: temperature.clone(),
            displacement: displacement.clone(),
            velocity_gradient: velocity_gradient.clone(),
            integers,
            stress: stress.clone(),
        });

        xdmf_writer.write_time_step(&step.to_string(), |time_step| {
            // `as_flattened` reinterprets `&[[f64; N]]` as `&[f64]` without copying, so the
            // natural per-point/per-cell layout needs no intermediate `Vec`
            time_step.point_data(
                "temperature",
                DataAttribute::Scalar,
                precision.values(&temperature),
            )?;
            time_step.point_data(
                "displacement",
                DataAttribute::Vector,
                precision.values(displacement.as_flattened()),
            )?;
            time_step.point_data(
                "velocity_gradient",
                DataAttribute::Tensor,
                precision.values(velocity_gradient.as_flattened()),
            )?;

            // integer data is unaffected by the fixture's float precision
            time_step.cell_data("level_i32", DataAttribute::Scalar, &LEVEL_I32)?;
            time_step.cell_data("flag_u32", DataAttribute::Scalar, &FLAG_U32)?;
            if writes_64_bit_integers {
                time_step.cell_data("region_id", DataAttribute::Scalar, &REGION_ID)?;
                time_step.cell_data("level_i64", DataAttribute::Scalar, &LEVEL_I64)?;
                time_step.cell_data("level_i64_wide", DataAttribute::Scalar, &LEVEL_I64_WIDE)?;
            }

            time_step.cell_data(
                "stress",
                DataAttribute::Tensor6,
                precision.values(stress.as_flattened()),
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
        points: coords
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| c.to_vec())
            .collect(),
        cells: EXPECTED_CELLS
            .iter()
            .map(|(class_name, points)| ExpectedCell {
                r#type: (*class_name).to_string(),
                points: points.to_vec(),
            })
            .collect(),
        timesteps,
    })
}

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
//! One further fixture per storage covers `write_mesh_with_submeshes`, which is a different grid
//! structure rather than a different set of numbers: a `Spatial` collection of one `<Grid>` per
//! submesh, each holding the points its own cells use and its own share of every field.
//! `ParaView` reads that back as a multi-block dataset, so nothing about it is exercised by the
//! fixtures above. Its submeshes deliberately cover all four cases the
//! writer distinguishes -- a single cell, a contiguous run, an out-of-order (gathered) list, and
//! two blocks overlapping on the same cell -- and its cell data includes a `Vector` field, since a
//! per-cell component count is what the slicing and gathering multiply by.
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

// The submeshes of the submesh fixture, as (name, cell indices). Between them they cover every
// case the writer distinguishes: a single cell, a contiguous run of them, a list that is *not* an
// ascending run (so the writer gathers rather than borrows), and two blocks claiming the same cell.
const SUBMESHES: [(&str, &[usize]); 4] = [
    ("quad", &[0]),
    ("tri", &[1]),
    ("both", &[0, 1]),
    ("reversed", &[1, 0]),
];

// A per-cell `Vector` field for the submesh fixture: three components per cell, so a submesh's
// share is a strided slice rather than one value per cell, which is what the writer's `stride`
// arithmetic has to get right for ParaView to read back whole tuples.
const CELL_VELOCITY: [[f64; 3]; NUM_CELLS] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

// A data name of the shape solvers actually hand over, rather than one made of the alphanumerics
// the writer used to insist on. The heavy data is numbered, so this name reaches nothing but the
// `<Attribute Name=...>`: ParaView finding the field under this exact spelling is what proves
// quick-xml's escaping of it round-trips.
const SOLVER_STYLE_NAME: &str = "Quantity('SOOT DENSITY'), U.component_0 [kg m-3]";

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
    /// Carried in the fixture rather than restated in the verification script, so the two cannot
    /// disagree about the exact spelling that is under test.
    solver_style_name: String,
    solver_style_values: Vec<f64>,
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

/// One block of the submesh fixture as `ParaView` must read it back: the points its cells use, the
/// cells themselves in that block's own point numbering, and its share of every field.
///
/// A block carries only the points its cells touch, so both what it holds and how its cells index
/// it differ per block -- which is what these expectations pin down.
#[derive(Serialize)]
struct ExpectedBlock {
    name: String,
    points: Vec<Vec<f64>>,
    cells: Vec<ExpectedCell>,
    temperature: Vec<f64>,
    stress: Vec<[f64; 6]>,
    level_i32: Vec<i32>,
    cell_velocity: Vec<[f64; 3]>,
}

#[derive(Serialize)]
struct ExpectedSubmeshTimestep {
    time: f64,
    blocks: Vec<ExpectedBlock>,
}

#[derive(Serialize)]
struct ExpectedSubmeshFixture {
    xdmf_file: String,
    timesteps: Vec<ExpectedSubmeshTimestep>,
}

#[derive(Serialize)]
struct Expected {
    /// Which storage wrote these, so the verification script knows how many fixtures to expect --
    /// `Binary` carries fewer, having no 64-bit integer types.
    storage: String,
    fixtures: Vec<ExpectedFixture>,
    /// The multi-block fixture, kept apart from the ones above because it is read back as a
    /// composite dataset and so is checked by a traversal of its own.
    submesh_fixtures: Vec<ExpectedSubmeshFixture>,
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
        submesh_fixtures: vec![write_submesh_fixture(output_dir, storage_arg, storage)?],
    };

    let expected_json =
        serde_json::to_string(&expected).map_err(|e| IoError::new(InvalidInput, e.to_string()))?;
    std::fs::write(output_dir.join("expected.json"), expected_json)?;

    #[expect(
        clippy::print_stdout,
        reason = "CLI progress output expected from an example binary"
    )]
    {
        let fixture_files = expected
            .fixtures
            .iter()
            .map(|fixture| &fixture.xdmf_file)
            .chain(
                expected
                    .submesh_fixtures
                    .iter()
                    .map(|fixture| &fixture.xdmf_file),
            );
        for xdmf_file in fixture_files {
            println!("Wrote fixture to {}", output_dir.join(xdmf_file).display());
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

        let mut solver_style: Vec<f64> = (0..NUM_POINTS)
            .map(|point| (point as f64 + 0.5) * scale)
            .collect();
        precision.narrow_expected(&mut solver_style);

        timesteps.push(ExpectedTimestep {
            time: step as f64,
            temperature: temperature.clone(),
            solver_style_name: SOLVER_STYLE_NAME.to_string(),
            solver_style_values: solver_style.clone(),
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
            time_step.point_data(
                SOLVER_STYLE_NAME,
                DataAttribute::Scalar,
                precision.values(&solver_style),
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

/// Writes the multi-block fixture: the same mesh as above, split into the submeshes of
/// [`SUBMESHES`].
///
/// Only one is written per storage, at f64 precision with `u32` connectivity, because what is under
/// test here is the grid *structure* -- a spatial collection nested in the temporal one -- and not
/// the element types, which the fixtures above already cover for every storage across both
/// precisions and every index type. `u32` because it is the one index type every storage carries,
/// `Binary` included.
fn write_submesh_fixture(
    output_dir: &Path,
    storage_arg: &str,
    storage: DataStorage,
) -> IoResult<ExpectedSubmeshFixture> {
    let base_path = output_dir.join(format!("fixture_{}_submeshes", storage_arg.to_lowercase()));

    let xdmf_writer = TimeSeriesWriter::new(&base_path, storage)?;
    let mut xdmf_writer = xdmf_writer.write_mesh_with_submeshes(
        &COORDS,
        &CONNECTIVITY.map(|index| index as u32),
        &CELL_TYPES,
        SUBMESHES,
    )?;

    let mut timesteps = Vec::new();
    for (step, scale) in [1.0, 2.0].into_iter().enumerate() {
        let temperature: Vec<f64> = [10.0, 11.0, 12.0, 13.0, 14.0]
            .into_iter()
            .map(|value| value * scale)
            .collect();
        let cell_velocity: Vec<[f64; 3]> = CELL_VELOCITY
            .iter()
            .map(|tuple| tuple.map(|component| component * scale))
            .collect();
        // a point field of more than one value per point, which is what a submesh needs an index
        // array of its own to select -- and the only shape written as a rank-3 `Dimensions`
        let stress: Vec<[f64; 6]> = (0..NUM_POINTS)
            .map(|point| std::array::from_fn(|component| (point * 6 + component) as f64 * scale))
            .collect();

        // Each block's expectations are the whole-mesh arrays taken at that submesh's cell
        // indices, in the order the submesh names them -- which is what the writer is expected to
        // hand ParaView, and what the "reversed" submesh exists to pin down.
        let mut blocks = Vec::with_capacity(SUBMESHES.len());
        for (name, cells) in SUBMESHES {
            let points = submesh_points(cells);

            let mut expected_cells = Vec::with_capacity(cells.len());
            for &cell in cells {
                // renumbered into the block's own points, as the writer renumbers them
                let mut cell_points = Vec::with_capacity(EXPECTED_CELLS[cell].1.len());
                for point in EXPECTED_CELLS[cell].1 {
                    cell_points.push(local_index(&points, *point)?);
                }

                expected_cells.push(ExpectedCell {
                    r#type: EXPECTED_CELLS[cell].0.to_string(),
                    points: cell_points,
                });
            }

            blocks.push(ExpectedBlock {
                name: name.to_string(),
                points: points
                    .iter()
                    .map(|&point| COORDS.as_chunks::<3>().0[point as usize].to_vec())
                    .collect(),
                cells: expected_cells,
                temperature: points
                    .iter()
                    .map(|&point| temperature[point as usize])
                    .collect(),
                stress: points.iter().map(|&point| stress[point as usize]).collect(),
                level_i32: cells.iter().map(|&cell| LEVEL_I32[cell]).collect(),
                cell_velocity: cells.iter().map(|&cell| cell_velocity[cell]).collect(),
            });
        }

        timesteps.push(ExpectedSubmeshTimestep {
            time: step as f64,
            blocks,
        });

        xdmf_writer.write_time_step(&step.to_string(), |time_step| {
            // point and cell data are passed over the whole mesh, exactly as without submeshes --
            // the writer gives each block its share
            time_step.point_data("temperature", DataAttribute::Scalar, &temperature)?;
            time_step.point_data("stress", DataAttribute::Tensor6, stress.as_flattened())?;
            time_step.cell_data("level_i32", DataAttribute::Scalar, &LEVEL_I32)?;
            time_step.cell_data(
                "cell_velocity",
                DataAttribute::Vector,
                cell_velocity.as_flattened(),
            )
        })?;
    }

    let xdmf_file = base_path
        .with_extension("xdmf2")
        .file_name()
        .ok_or_else(|| IoError::new(InvalidInput, "invalid output file name"))?
        .to_string_lossy()
        .into_owned();

    Ok(ExpectedSubmeshFixture {
        xdmf_file,
        timesteps,
    })
}

/// The mesh points one submesh's cells use, ascending -- exactly the points the writer gives that
/// block, in the order it writes them.
fn submesh_points(cells: &[usize]) -> Vec<u64> {
    let mut points: Vec<u64> = cells
        .iter()
        .flat_map(|&cell| EXPECTED_CELLS[cell].1.iter().copied())
        .collect();

    points.sort_unstable();
    points.dedup();

    points
}

/// Where a mesh point sits in a block's own point list, which is how that block's cells index it.
fn local_index(points: &[u64], point: u64) -> IoResult<u64> {
    points
        .iter()
        .position(|&candidate| candidate == point)
        .map(|index| index as u64)
        .ok_or_else(|| {
            IoError::new(
                InvalidInput,
                format!("point {point} is not in the block that uses it"),
            )
        })
}

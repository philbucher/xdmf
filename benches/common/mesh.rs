//! CFD-like duct mesh generator for the M2 performance benchmarks (`plans/02_performance.md`
//! part A). A Rust port of `python/benchmarks/mesh_gen.py` on `origin/multiple-features`: a
//! structured hex duct (as an unstructured mesh) plus three quad boundary patches
//! (`inlet`/`outlet`/`sides`), all sharing one point array, with a parabolic-profile velocity
//! and a linear pressure-drop field. Scales to the 1e7-cell case used by
//! `examples/bench_cfd.rs`.

use xdmf::CellType;

const LX: f64 = 0.1; // duct cross-section 10cm x 10cm
const LY: f64 = 0.1;
const LZ: f64 = 1.0; // duct length 1m
const U_CENTERLINE: f64 = 1.0; // m/s
const P_INLET: f64 = 101_325.0; // Pa
const DP_DZ: f64 = 100.0; // Pa/m pressure drop along the duct

/// A structured hex duct with boundary patches, generated for benchmarking.
pub struct CfdCase {
    pub num_points: usize,
    pub num_hex: usize,
    pub num_inlet: usize,
    pub num_outlet: usize,
    pub num_sides: usize,
    pub points: Vec<f64>,
    pub connectivity: Vec<u64>,
    pub cell_types: Vec<CellType>,
    /// Block name paired with cell indices into `cell_types` / the connectivity array.
    #[expect(
        dead_code,
        reason = "not read by the M2 benches; generated now because M4 (`plans/04_submeshes.md`) \
                   needs the same case and the generator should not be forked in two"
    )]
    pub blocks: Vec<(&'static str, Vec<usize>)>,
    pub velocity: Vec<f64>,
    pub pressure: Vec<f64>,
}

impl CfdCase {
    pub fn num_cells(&self) -> usize {
        self.num_hex + self.num_inlet + self.num_outlet + self.num_sides
    }
}

fn node_index(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> u64 {
    (i * (ny + 1) * (nz + 1) + j * (nz + 1) + k) as u64
}

/// Builds an `nx * ny * nz` hexahedral duct. Node/cell counts follow `mesh_gen.py` exactly so
/// results are comparable to the earlier Python-based benchmark session.
pub fn build_case(nx: usize, ny: usize, nz: usize) -> CfdCase {
    let (nx1, ny1, nz1) = (nx + 1, ny + 1, nz + 1);
    let num_points = nx1 * ny1 * nz1;

    let mut points = Vec::with_capacity(num_points * 3);
    let mut velocity = Vec::with_capacity(num_points * 3);
    let mut pressure = Vec::with_capacity(num_points);

    for i in 0..nx1 {
        let x = LX * i as f64 / nx as f64;
        for j in 0..ny1 {
            let y = LY * j as f64 / ny as f64;
            let vz = U_CENTERLINE
                * (1.0 - ((x - LX / 2.0) / (LX / 2.0)).powi(2))
                * (1.0 - ((y - LY / 2.0) / (LY / 2.0)).powi(2));
            for k in 0..nz1 {
                let z = LZ * k as f64 / nz as f64;
                points.extend_from_slice(&[x, y, z]);
                velocity.extend_from_slice(&[0.0, 0.0, vz]);
                pressure.push(P_INLET - DP_DZ * z);
            }
        }
    }

    let num_hex = nx * ny * nz;
    let num_inlet = nx * ny;
    let num_outlet = nx * ny;
    let num_sides = 2 * (ny * nz + nx * nz);

    let mut connectivity =
        Vec::with_capacity(num_hex * 8 + (num_inlet + num_outlet + num_sides) * 4);

    // hexahedron volume cells, VTK node ordering (bottom face ccw, then top face ccw)
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                connectivity.extend_from_slice(&[
                    node_index(i, j, k, ny, nz),
                    node_index(i + 1, j, k, ny, nz),
                    node_index(i + 1, j + 1, k, ny, nz),
                    node_index(i, j + 1, k, ny, nz),
                    node_index(i, j, k + 1, ny, nz),
                    node_index(i + 1, j, k + 1, ny, nz),
                    node_index(i + 1, j + 1, k + 1, ny, nz),
                    node_index(i, j + 1, k + 1, ny, nz),
                ]);
            }
        }
    }

    // inlet (k=0) / outlet (k=nz) quad patches
    for i in 0..nx {
        for j in 0..ny {
            connectivity.extend_from_slice(&[
                node_index(i, j, 0, ny, nz),
                node_index(i + 1, j, 0, ny, nz),
                node_index(i + 1, j + 1, 0, ny, nz),
                node_index(i, j + 1, 0, ny, nz),
            ]);
        }
    }
    for i in 0..nx {
        for j in 0..ny {
            connectivity.extend_from_slice(&[
                node_index(i, j, nz, ny, nz),
                node_index(i + 1, j, nz, ny, nz),
                node_index(i + 1, j + 1, nz, ny, nz),
                node_index(i, j + 1, nz, ny, nz),
            ]);
        }
    }

    // side walls: i=0 / i=nx (vary j,k), then j=0 / j=ny (vary i,k)
    for i in [0, nx] {
        for j in 0..ny {
            for k in 0..nz {
                connectivity.extend_from_slice(&[
                    node_index(i, j, k, ny, nz),
                    node_index(i, j + 1, k, ny, nz),
                    node_index(i, j + 1, k + 1, ny, nz),
                    node_index(i, j, k + 1, ny, nz),
                ]);
            }
        }
    }
    for j in [0, ny] {
        for i in 0..nx {
            for k in 0..nz {
                connectivity.extend_from_slice(&[
                    node_index(i, j, k, ny, nz),
                    node_index(i + 1, j, k, ny, nz),
                    node_index(i + 1, j, k + 1, ny, nz),
                    node_index(i, j, k + 1, ny, nz),
                ]);
            }
        }
    }

    let mut cell_types = Vec::with_capacity(num_hex + num_inlet + num_outlet + num_sides);
    cell_types.extend(std::iter::repeat_n(CellType::Hexahedron, num_hex));
    cell_types.extend(std::iter::repeat_n(
        CellType::Quadrilateral,
        num_inlet + num_outlet + num_sides,
    ));

    let blocks = vec![
        ("domain", (0..num_hex).collect()),
        ("inlet", (num_hex..num_hex + num_inlet).collect()),
        (
            "outlet",
            (num_hex + num_inlet..num_hex + num_inlet + num_outlet).collect(),
        ),
        (
            "sides",
            (num_hex + num_inlet + num_outlet..num_hex + num_inlet + num_outlet + num_sides)
                .collect(),
        ),
    ];

    CfdCase {
        num_points,
        num_hex,
        num_inlet,
        num_outlet,
        num_sides,
        points,
        connectivity,
        cell_types,
        blocks,
        velocity,
        pressure,
    }
}

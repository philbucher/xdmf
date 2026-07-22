"""Generates a structured hex "CFD-like" mesh (rectangular duct) as an unstructured mesh:
a hexahedron volume domain plus 3 quad boundary patches (inlet, outlet, sides), all sharing
one point array. Fields: a parabolic-profile velocity (vector) and a linear pressure drop
(scalar), both as point data.

Node/cell layout is fully vectorized (numpy) so this scales to ~10M elements in seconds.
"""

from dataclasses import dataclass

import numpy as np

LX, LY, LZ = 0.1, 0.1, 1.0  # duct cross-section 10cm x 10cm, 1m long
U_CENTERLINE = 1.0  # m/s
P_INLET = 101325.0  # Pa
DP_DZ = 100.0  # Pa/m pressure drop along the duct


def _node_index(i, j, k, ny, nz):
    return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k


def _quad_sheet(idx0, idx1, idx2, idx3):
    """Stacks 4 per-quad node-index arrays (same shape) into a flat (n_quads*4,) uint64 array."""
    return np.stack([idx0, idx1, idx2, idx3], axis=-1).astype(np.uint64).reshape(-1)


@dataclass
class CfdCase:
    nx: int
    ny: int
    nz: int
    num_points: int
    num_hex: int
    num_inlet: int
    num_outlet: int
    num_sides: int
    points: np.ndarray  # flat float64, len = num_points * 3
    connectivity: np.ndarray  # flat uint64: hex cells, then inlet/outlet/sides quads
    cell_type_names: list  # one xdmf.CellType-name-ish string per cell (resolved by caller)
    blocks: list  # [(name, cell_indices_list), ...]
    velocity: np.ndarray  # flat float64, len = num_points * 3
    pressure: np.ndarray  # flat float64, len = num_points

    @property
    def num_cells(self):
        return self.num_hex + self.num_inlet + self.num_outlet + self.num_sides


def build_case(nx: int, ny: int, nz: int) -> CfdCase:
    ny1, nz1 = ny + 1, nz + 1

    # --- points -----------------------------------------------------------------
    xs = np.linspace(0.0, LX, nx + 1)
    ys = np.linspace(0.0, LY, ny + 1)
    zs = np.linspace(0.0, LZ, nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")  # each (nx+1, ny+1, nz+1)
    num_points = (nx + 1) * (ny + 1) * (nz + 1)
    points = np.stack([X, Y, Z], axis=-1).reshape(-1)

    # --- fields (point data) -----------------------------------------------------
    vz = (
        U_CENTERLINE
        * (1.0 - ((X - LX / 2) / (LX / 2)) ** 2)
        * (1.0 - ((Y - LY / 2) / (LY / 2)) ** 2)
    )
    velocity = np.stack([np.zeros_like(vz), np.zeros_like(vz), vz], axis=-1).reshape(-1)
    pressure = (P_INLET - DP_DZ * Z).reshape(-1)
    del X, Y, Z, vz

    # --- hexahedron volume cells --------------------------------------------------
    ii, jj, kk = np.meshgrid(
        np.arange(nx, dtype=np.uint64),
        np.arange(ny, dtype=np.uint64),
        np.arange(nz, dtype=np.uint64),
        indexing="ij",
    )
    ii, jj, kk = ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)

    def nidx(i, j, k):
        return _node_index(i, j, k, ny, nz)

    hex_conn = np.stack(
        [
            nidx(ii, jj, kk),
            nidx(ii + 1, jj, kk),
            nidx(ii + 1, jj + 1, kk),
            nidx(ii, jj + 1, kk),
            nidx(ii, jj, kk + 1),
            nidx(ii + 1, jj, kk + 1),
            nidx(ii + 1, jj + 1, kk + 1),
            nidx(ii, jj + 1, kk + 1),
        ],
        axis=-1,
    ).astype(np.uint64).reshape(-1)
    num_hex = nx * ny * nz
    del ii, jj, kk

    # --- boundary quad patches ----------------------------------------------------
    # inlet (k=0) / outlet (k=nz): vary i, j
    ig, jg = np.meshgrid(
        np.arange(nx, dtype=np.uint64), np.arange(ny, dtype=np.uint64), indexing="ij"
    )
    ig, jg = ig.reshape(-1), jg.reshape(-1)

    inlet_conn = _quad_sheet(
        nidx(ig, jg, 0), nidx(ig + 1, jg, 0), nidx(ig + 1, jg + 1, 0), nidx(ig, jg + 1, 0)
    )
    outlet_conn = _quad_sheet(
        nidx(ig, jg, nz), nidx(ig + 1, jg, nz), nidx(ig + 1, jg + 1, nz), nidx(ig, jg + 1, nz)
    )
    num_inlet = nx * ny
    num_outlet = nx * ny
    del ig, jg

    # sides: i=0, i=nx walls (vary j,k) + j=0, j=ny walls (vary i,k)
    jg2, kg2 = np.meshgrid(
        np.arange(ny, dtype=np.uint64), np.arange(nz, dtype=np.uint64), indexing="ij"
    )
    jg2, kg2 = jg2.reshape(-1), kg2.reshape(-1)
    wall_i0 = _quad_sheet(
        nidx(0, jg2, kg2), nidx(0, jg2 + 1, kg2), nidx(0, jg2 + 1, kg2 + 1), nidx(0, jg2, kg2 + 1)
    )
    wall_inx = _quad_sheet(
        nidx(nx, jg2, kg2),
        nidx(nx, jg2 + 1, kg2),
        nidx(nx, jg2 + 1, kg2 + 1),
        nidx(nx, jg2, kg2 + 1),
    )
    del jg2, kg2

    ig3, kg3 = np.meshgrid(
        np.arange(nx, dtype=np.uint64), np.arange(nz, dtype=np.uint64), indexing="ij"
    )
    ig3, kg3 = ig3.reshape(-1), kg3.reshape(-1)
    wall_j0 = _quad_sheet(
        nidx(ig3, 0, kg3), nidx(ig3 + 1, 0, kg3), nidx(ig3 + 1, 0, kg3 + 1), nidx(ig3, 0, kg3 + 1)
    )
    wall_jny = _quad_sheet(
        nidx(ig3, ny, kg3),
        nidx(ig3 + 1, ny, kg3),
        nidx(ig3 + 1, ny, kg3 + 1),
        nidx(ig3, ny, kg3 + 1),
    )
    del ig3, kg3

    sides_conn = np.concatenate([wall_i0, wall_inx, wall_j0, wall_jny])
    num_sides = 2 * (ny * nz + nx * nz)

    connectivity = np.concatenate([hex_conn, inlet_conn, outlet_conn, sides_conn])

    cell_type_names = (
        ["Hexahedron"] * num_hex + ["Quadrilateral"] * (num_inlet + num_outlet + num_sides)
    )

    domain_idx = np.arange(0, num_hex, dtype=np.int64)
    inlet_idx = np.arange(num_hex, num_hex + num_inlet, dtype=np.int64)
    outlet_idx = np.arange(num_hex + num_inlet, num_hex + num_inlet + num_outlet, dtype=np.int64)
    sides_idx = np.arange(
        num_hex + num_inlet + num_outlet,
        num_hex + num_inlet + num_outlet + num_sides,
        dtype=np.int64,
    )
    blocks = [
        ("domain", domain_idx.tolist()),
        ("inlet", inlet_idx.tolist()),
        ("outlet", outlet_idx.tolist()),
        ("sides", sides_idx.tolist()),
    ]

    return CfdCase(
        nx=nx,
        ny=ny,
        nz=nz,
        num_points=num_points,
        num_hex=num_hex,
        num_inlet=num_inlet,
        num_outlet=num_outlet,
        num_sides=num_sides,
        points=points,
        connectivity=connectivity,
        cell_type_names=cell_type_names,
        blocks=blocks,
        velocity=velocity,
        pressure=pressure,
    )


def extract_submesh(case: CfdCase, flat_start: int, flat_end: int, num_nodes_per_cell: int):
    """Extracts a local (points, connectivity, velocity, pressure) submesh for the cells
    occupying `case.connectivity[flat_start:flat_end]` (a contiguous run of same-size cells),
    remapping to a compact local point numbering. Used for pyvista's per-zone files.
    """
    local_conn_global = case.connectivity[flat_start:flat_end]

    unique_nodes, local_conn = np.unique(local_conn_global, return_inverse=True)
    local_conn = local_conn.astype(np.uint64).reshape(-1, num_nodes_per_cell)

    points_3d = case.points.reshape(-1, 3)
    velocity_3d = case.velocity.reshape(-1, 3)
    local_points = points_3d[unique_nodes]
    local_velocity = velocity_3d[unique_nodes]
    local_pressure = case.pressure[unique_nodes]

    return local_points, local_conn, local_velocity, local_pressure

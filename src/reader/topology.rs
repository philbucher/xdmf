//! Inverse of `time_series_writer::prepare_cells`: turns a `Topology`'s raw connectivity array
//! back into a flat `connectivity: Vec<u64>` plus one `CellType` per cell.

use std::path::Path;

use crate::{
    CellType, Error, Result, time_series_writer::poly_cell_points,
    xdmf_elements::topology::TopologyType,
};

/// `raw` is the `Topology` `DataItem`'s heavy data, already widened to `u64`. `num_points` and
/// `declared_num_elements` (`Topology::number_of_elements`, parsed) are used to recognize the
/// `Polyvertex` point-cloud special case and to validate the decoded cell count.
pub(crate) fn invert(
    topology_type: TopologyType,
    raw: &[u64],
    num_points: usize,
    declared_num_elements: usize,
    xdmf_path: &Path,
) -> Result<(Vec<u64>, Vec<CellType>)> {
    match topology_type {
        TopologyType::Mixed => invert_mixed(raw, declared_num_elements, xdmf_path),
        TopologyType::Polyvertex => {
            invert_polyvertex(raw, num_points, declared_num_elements, xdmf_path)
        }
        other => {
            let cell_type = other.cell_type().ok_or_else(|| Error::Unsupported {
                reason: format!("TopologyType \"{other:?}\" is not supported"),
            })?;
            invert_homogeneous(cell_type, raw, declared_num_elements, xdmf_path)
        }
    }
}

// `write_mesh` with no cells emits `Polyvertex` over the identity range `0..num_points` — see
// `time_series_writer::prepare_cells`. Recognizing exactly that pattern is what makes the
// point-cloud round trip return `cell_types = []` rather than `num_points` `Vertex` cells. Any
// other `Polyvertex` connectivity (a foreign file, or a permuted/partial one) is genuine
// one-point-per-cell `Vertex` topology.
fn invert_polyvertex(
    raw: &[u64],
    num_points: usize,
    declared_num_elements: usize,
    xdmf_path: &Path,
) -> Result<(Vec<u64>, Vec<CellType>)> {
    let is_point_cloud =
        raw.len() == num_points && raw.iter().enumerate().all(|(i, &v)| v as usize == i);
    if is_point_cloud {
        return Ok((Vec::new(), Vec::new()));
    }
    invert_homogeneous(CellType::Vertex, raw, declared_num_elements, xdmf_path)
}

fn invert_homogeneous(
    cell_type: CellType,
    raw: &[u64],
    declared_num_elements: usize,
    xdmf_path: &Path,
) -> Result<(Vec<u64>, Vec<CellType>)> {
    let stride = cell_type.num_points();
    if raw.len() != stride * declared_num_elements {
        return Err(Error::InvalidFile {
            path: xdmf_path.to_path_buf(),
            reason: format!(
                "Topology connectivity has {} values, expected {} ({declared_num_elements} elements of {stride} points each)",
                raw.len(),
                stride * declared_num_elements
            ),
        });
    }
    Ok((raw.to_vec(), vec![cell_type; declared_num_elements]))
}

fn invert_mixed(
    raw: &[u64],
    declared_num_elements: usize,
    xdmf_path: &Path,
) -> Result<(Vec<u64>, Vec<CellType>)> {
    let mut connectivity = Vec::new();
    let mut cell_types = Vec::new();
    let mut i = 0;

    while i < raw.len() {
        let code = raw[i];
        i += 1;
        let cell_type = CellType::from_code(code).ok_or_else(|| Error::Unsupported {
            reason: format!("unknown VTK cell type code {code} in Mixed topology connectivity"),
        })?;

        // Vertex/Edge carry a redundant point-count field the writer inserted (see
        // `poly_cell_points`); its value is always `cell_type.num_points()`, so it is skipped
        // rather than re-validated.
        if poly_cell_points(cell_type).is_some() {
            i += 1;
        }

        let n = cell_type.num_points();
        if i + n > raw.len() {
            return Err(Error::InvalidFile {
                path: xdmf_path.to_path_buf(),
                reason: "Mixed topology connectivity is truncated mid-cell".to_string(),
            });
        }
        connectivity.extend_from_slice(&raw[i..i + n]);
        cell_types.push(cell_type);
        i += n;
    }

    if cell_types.len() != declared_num_elements {
        return Err(Error::InvalidFile {
            path: xdmf_path.to_path_buf(),
            reason: format!(
                "Topology NumberOfElements={declared_num_elements} does not match the {} cells decoded from Mixed connectivity",
                cell_types.len()
            ),
        });
    }

    Ok((connectivity, cell_types))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series_writer::{IntConnectivity, prepare_cells};

    const ALL_CELL_TYPES: [CellType; 19] = [
        CellType::Vertex,
        CellType::Edge,
        CellType::Triangle,
        CellType::Quadrilateral,
        CellType::Tetrahedron,
        CellType::Pyramid,
        CellType::Wedge,
        CellType::Hexahedron,
        CellType::Edge3,
        CellType::Quadrilateral9,
        CellType::Triangle6,
        CellType::Quadrilateral8,
        CellType::Tetrahedron10,
        CellType::Pyramid13,
        CellType::Wedge15,
        CellType::Wedge18,
        CellType::Hexahedron20,
        CellType::Hexahedron24,
        CellType::Hexahedron27,
    ];

    #[test]
    fn mixed_round_trips_every_cell_type_individually() {
        for cell_type in ALL_CELL_TYPES {
            let n = cell_type.num_points();
            let connectivity: Vec<u64> = (0..n as u64).collect();
            let (topo_type, raw) =
                prepare_cells(IntConnectivity::U64(&connectivity), &[cell_type], n).unwrap();
            let raw = raw.as_slice::<u64>().unwrap();
            assert_eq!(topo_type, TopologyType::Mixed);

            let (decoded_connectivity, decoded_cell_types) =
                invert(topo_type, raw, n, 1, Path::new("x.xdmf2")).unwrap();
            assert_eq!(decoded_connectivity, connectivity, "{cell_type:?}");
            assert_eq!(decoded_cell_types, vec![cell_type], "{cell_type:?}");
        }
    }

    #[test]
    fn mixed_round_trips_several_cell_types_together() {
        let connectivity: Vec<u64> = vec![0, 1, 0, 2, 1, 0, 1, 2, 3];
        let cell_types = vec![CellType::Edge, CellType::Triangle, CellType::Tetrahedron];
        let num_points = 4;
        let (topo_type, raw) =
            prepare_cells(IntConnectivity::U64(&connectivity), &cell_types, num_points).unwrap();
        let raw = raw.as_slice::<u64>().unwrap();

        let (decoded_connectivity, decoded_cell_types) = invert(
            topo_type,
            raw,
            num_points,
            cell_types.len(),
            Path::new("x.xdmf2"),
        )
        .unwrap();
        assert_eq!(decoded_connectivity, connectivity);
        assert_eq!(decoded_cell_types, cell_types);
    }

    #[test]
    fn polyvertex_point_cloud_returns_empty_cell_types() {
        let num_points = 5;
        let (topo_type, raw) = prepare_cells(IntConnectivity::U64(&[]), &[], num_points).unwrap();
        let raw = raw.as_slice::<u64>().unwrap();
        assert_eq!(topo_type, TopologyType::Polyvertex);

        let (decoded_connectivity, decoded_cell_types) =
            invert(topo_type, raw, num_points, num_points, Path::new("x.xdmf2")).unwrap();
        assert!(decoded_connectivity.is_empty());
        assert!(decoded_cell_types.is_empty());
    }

    #[test]
    fn polyvertex_non_identity_is_genuine_vertex_cells() {
        let raw = vec![2_u64, 0, 1];
        let (connectivity, cell_types) =
            invert(TopologyType::Polyvertex, &raw, 3, 3, Path::new("x.xdmf2")).unwrap();
        assert_eq!(connectivity, raw);
        assert_eq!(cell_types, vec![CellType::Vertex; 3]);
    }

    #[test]
    fn homogeneous_triangle_topology() {
        let raw = vec![0_u64, 1, 2, 1, 2, 3];
        let (connectivity, cell_types) =
            invert(TopologyType::Triangle, &raw, 4, 2, Path::new("x.xdmf2")).unwrap();
        assert_eq!(connectivity, raw);
        assert_eq!(cell_types, vec![CellType::Triangle; 2]);
    }

    #[test]
    fn homogeneous_size_mismatch_is_invalid_file() {
        let raw = vec![0_u64, 1, 2, 1, 2];
        let err = invert(TopologyType::Triangle, &raw, 4, 2, Path::new("x.xdmf2")).unwrap_err();
        std::assert_matches!(err, Error::InvalidFile { reason, .. } if reason.contains("has 5 values"));
    }

    #[test]
    fn mixed_unknown_code_is_unsupported() {
        let raw = vec![999_u64, 0, 1];
        let err = invert(TopologyType::Mixed, &raw, 1, 1, Path::new("x.xdmf2")).unwrap_err();
        std::assert_matches!(err, Error::Unsupported { reason } if reason.contains("999"));
    }

    #[test]
    fn mixed_truncated_is_invalid_file() {
        let raw = vec![4_u64, 0, 1]; // Triangle needs 3 indices, only 2 given
        let err = invert(TopologyType::Mixed, &raw, 1, 1, Path::new("x.xdmf2")).unwrap_err();
        std::assert_matches!(err, Error::InvalidFile { reason, .. } if reason.contains("truncated"));
    }

    #[test]
    fn mixed_element_count_mismatch_is_invalid_file() {
        let connectivity: Vec<u64> = vec![0, 1, 2];
        let (topo_type, raw) = prepare_cells(
            IntConnectivity::U64(&connectivity),
            &[CellType::Triangle],
            3,
        )
        .unwrap();
        let raw = raw.as_slice::<u64>().unwrap();
        let err = invert(topo_type, raw, 3, 2, Path::new("x.xdmf2")).unwrap_err();
        std::assert_matches!(err, Error::InvalidFile { reason, .. } if reason.contains("NumberOfElements=2"));
    }
}

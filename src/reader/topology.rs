//! The inverse of `time_series_writer.rs`'s `prepare_cells`: a `Topology`'s raw connectivity
//! array -> per-cell [`CellType`]s and a plain (type-code-free) connectivity, local to whatever
//! submesh wrote it.

use crate::{
    CellType, ConnectivityIndex, Error, Result,
    xdmf_elements::topology::{Topology, TopologyType},
};

/// Decode a connectivity the caller has already read, in place, per `topology.topology_type`.
///
/// `connectivity` arrives holding the array as the file does and leaves holding the mesh's own:
/// for a uniform topology those are the same array and nothing is moved at all, and `Mixed` is
/// compacted where it stands by dropping each cell's type code (and a poly-cell's point count).
/// So neither shape needs a second array, which is what lets the whole read be one allocation --
/// the caller's buffer.
///
/// `cell_types` is filled to match and is cleared first.
pub(super) fn decode_in_place<I: ConnectivityIndex>(
    topology: &Topology,
    connectivity: &mut Vec<I>,
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    cell_types.clear();

    match topology.topology_type {
        TopologyType::Mixed => decode_mixed_in_place(connectivity, cell_types),
        uniform => decode_uniform_in_place(
            uniform,
            topology.nodes_per_element,
            connectivity,
            cell_types,
        ),
    }
}

/// Every cell shares one type, so the connectivity is already what the file holds -- only the
/// per-cell types have to be produced.
fn decode_uniform_in_place<I: ConnectivityIndex>(
    topology_type: TopologyType,
    nodes_per_element: Option<u8>,
    connectivity: &[I],
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    let cell_type = cell_type_of(topology_type, nodes_per_element)?;
    let stride = cell_type.num_points();

    if stride == 0 || !connectivity.len().is_multiple_of(stride) {
        return Err(Error::InvalidDocument {
            reason: format!(
                "a {topology_type:?} connectivity of {} values is not a multiple of {stride}",
                connectivity.len()
            ),
        });
    }

    cell_types.resize(connectivity.len() / stride, cell_type);

    Ok(())
}

/// `Mixed` prepends each cell's type code (and, for a poly-cell, its point count) to its points,
/// so the points are moved down over the entries dropped ahead of them.
///
/// The write position only ever trails the read position -- every cell drops at least the one
/// entry its type code takes -- so nothing is overwritten before it has been read.
fn decode_mixed_in_place<I: ConnectivityIndex>(
    connectivity: &mut Vec<I>,
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    let mut read = 0;
    let mut write = 0;

    while read < connectivity.len() {
        let entry = connectivity[read];
        let code = u8::try_from(entry.as_i128()).map_err(|_source| Error::InvalidDocument {
            reason: format!(
                "Mixed connectivity cell type code {} is out of range",
                entry.as_i128()
            ),
        })?;
        read += 1;

        let cell_type = CellType::from_code(code).ok_or_else(|| Error::InvalidDocument {
            reason: format!("Mixed connectivity has an unknown cell type code {code}"),
        })?;

        // Vertex/Edge poly-cells carry their point count next. For this crate's own files it is
        // redundant -- always the type's own fixed count -- but a foreign file may state another,
        // and taking it on trust would desynchronise the stream and surface as an "unknown cell
        // type code" further along. Rejected here instead, as `cell_type_of` rejects the
        // equivalent `NodesPerElement` for a uniform topology.
        if let Some(expected) = poly_cell_points(cell_type) {
            let stated = connectivity
                .get(read)
                .ok_or_else(|| Error::InvalidDocument {
                    reason: format!(
                        "Mixed connectivity ends after a {cell_type:?} cell's type code, before \
                         its point count"
                    ),
                })?
                .as_i128();

            if stated != i128::from(expected) {
                return Err(Error::Unsupported {
                    reason: format!(
                        "a Mixed connectivity's {cell_type:?} cell states {stated} points, only \
                         {expected} is supported"
                    ),
                });
            }

            read += 1;
        }

        let num_points = cell_type.num_points();
        let end = read.checked_add(num_points).ok_or(Error::Internal(
            "a Mixed connectivity's cell span does not fit a usize",
        ))?;

        if end > connectivity.len() {
            return Err(Error::InvalidDocument {
                reason: "Mixed connectivity ends in the middle of a cell".to_string(),
            });
        }

        connectivity.copy_within(read..end, write);
        write += num_points;
        read = end;
        cell_types.push(cell_type);
    }

    connectivity.truncate(write);

    Ok(())
}

/// The single `CellType` every cell of a uniform topology has, or `None` for `Mixed` -- whose
/// per-cell types only its connectivity itself states.
///
/// This is what lets `read_topology_with_submeshes` learn a submesh's cell types without reading
/// its heavy data at all, in the common case that the submesh is uniform.
pub(super) fn uniform_cell_type(topology: &Topology) -> Result<Option<CellType>> {
    match topology.topology_type {
        TopologyType::Mixed => Ok(None),
        uniform => cell_type_of(uniform, topology.nodes_per_element).map(Some),
    }
}

/// The one `CellType` a uniform `TopologyType` decodes to -- the inverse of
/// `topology.rs`'s `From<CellType> for TopologyType`.
fn cell_type_of(topology_type: TopologyType, nodes_per_element: Option<u8>) -> Result<CellType> {
    let cell_type = match topology_type {
        TopologyType::Mixed => {
            return Err(Error::Internal(
                "cell_type_of called with the Mixed topology type",
            ));
        }
        TopologyType::Polyvertex => CellType::Vertex,
        TopologyType::Polyline => CellType::Edge,
        TopologyType::Triangle => CellType::Triangle,
        TopologyType::Quadrilateral => CellType::Quadrilateral,
        TopologyType::Tetrahedron => CellType::Tetrahedron,
        TopologyType::Pyramid => CellType::Pyramid,
        TopologyType::Wedge => CellType::Wedge,
        TopologyType::Hexahedron => CellType::Hexahedron,
        TopologyType::Edge3 => CellType::Edge3,
        TopologyType::Triangle6 => CellType::Triangle6,
        TopologyType::Quadrilateral8 => CellType::Quadrilateral8,
        TopologyType::Quadrilateral9 => CellType::Quadrilateral9,
        TopologyType::Tetrahedron10 => CellType::Tetrahedron10,
        TopologyType::Pyramid13 => CellType::Pyramid13,
        TopologyType::Wedge15 => CellType::Wedge15,
        TopologyType::Wedge18 => CellType::Wedge18,
        TopologyType::Hexahedron20 => CellType::Hexahedron20,
        TopologyType::Hexahedron24 => CellType::Hexahedron24,
        TopologyType::Hexahedron27 => CellType::Hexahedron27,
    };

    // Polyvertex/Polyline are the only types whose per-element node count isn't the type's own
    // fixed one; this crate's own writer always states it for those two, but a foreign file might
    // rely on the XDMF-spec default (1 for Polyvertex, 2 for Polyline) instead.
    if matches!(
        topology_type,
        TopologyType::Polyvertex | TopologyType::Polyline
    ) && let Some(nodes_per_element) = nodes_per_element
        && usize::from(nodes_per_element) != cell_type.num_points()
    {
        return Err(Error::Unsupported {
            reason: format!(
                "{topology_type:?} with NodesPerElement={nodes_per_element} is not supported"
            ),
        });
    }

    Ok(cell_type)
}

/// Same table `prepare_mesh` (`time_series_writer.rs`) uses to decide whether a `Mixed`
/// connectivity's cell carries a point count.
fn poly_cell_points(cell_type: CellType) -> Option<u8> {
    match cell_type {
        CellType::Vertex => Some(1),
        CellType::Edge => Some(2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Values, values::sealed::SealedIndex, xdmf_elements::data_item::DataItem};

    /// A `Mixed` connectivity decoded in place, returned as the two buffers it fills.
    fn mixed<I: ConnectivityIndex>(values: &[I]) -> Result<(Vec<CellType>, Vec<I>)> {
        let topology = Topology {
            topology_type: TopologyType::Mixed,
            nodes_per_element: None,
            number_of_elements: "1".to_string(),
            data_item: DataItem::default(),
        };

        let mut connectivity = I::indices_from_values(I::as_values(values))?;
        let mut cell_types = Vec::new();
        decode_in_place(&topology, &mut connectivity, &mut cell_types)?;

        Ok((cell_types, connectivity))
    }

    #[test]
    fn a_poly_cells_point_count_is_consumed() {
        // a Vertex (code 1, 1 point) and an Edge (code 2, 2 points), each stating its own count
        let (cell_types, connectivity) = mixed(&[1_u64, 1, 7, 2, 2, 3, 4]).unwrap();

        assert_eq!(cell_types, [CellType::Vertex, CellType::Edge]);
        assert_eq!(connectivity, [7, 3, 4]);
    }

    /// Taking the stated count on trust would consume one value too few and read the rest of the
    /// stream shifted, which surfaces as a confusing "unknown cell type code" further along.
    #[test]
    fn a_poly_cell_stating_another_point_count_is_rejected() {
        std::assert_matches!(
            mixed(&[1_u64, 3, 7, 8, 9]).unwrap_err(),
            Error::Unsupported { reason }
                if reason.contains("Vertex cell states 3 points, only 1 is supported")
        );
    }

    #[test]
    fn a_poly_cell_without_its_point_count_is_rejected() {
        std::assert_matches!(
            mixed(&[1_u64]).unwrap_err(),
            Error::InvalidDocument { reason } if reason.contains("before its point count")
        );
    }

    /// The compaction moves each cell's points down over the entries dropped ahead of them, so a
    /// mesh of several cells is where a wrong write position would show.
    #[test]
    fn a_mixed_connectivity_is_compacted_in_place() {
        // three triangles (code 4, 3 points each), written as `prepare_cells` writes them
        let (cell_types, connectivity) = mixed(&[4_u32, 0, 1, 2, 4, 1, 2, 3, 4, 2, 3, 4]).unwrap();

        assert_eq!(cell_types, [CellType::Triangle; 3]);
        assert_eq!(connectivity, [0, 1, 2, 1, 2, 3, 2, 3, 4]);
    }

    /// The index type is checked against the *values*, whatever type the file itself holds -- a
    /// connectivity is positions, not a number format.
    #[test]
    fn an_index_beyond_the_requested_type_is_rejected() {
        std::assert_matches!(
            u32::indices_from_values(Values::from(&[u64::from(u32::MAX) + 1][..])).unwrap_err(),
            Error::IntegerOutOfRange { value, .. } if value == i128::from(u32::MAX) + 1
        );
    }

    #[test]
    fn a_negative_index_is_rejected() {
        // caught in the same-type arm too, which moves the array rather than converting it
        std::assert_matches!(
            i64::indices_from_values(Values::from(&[0_i64, -1][..])).unwrap_err(),
            Error::InvalidDocument { reason } if reason.contains("index -1 is negative")
        );
    }
}

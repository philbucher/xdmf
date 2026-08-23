//! The inverse of `time_series_writer.rs`'s `prepare_cells`: a `Topology`'s raw connectivity
//! array -> per-cell [`CellType`]s and a plain (type-code-free) connectivity, local to whatever
//! submesh wrote it.

use crate::{
    CellType, Error, Result, Values,
    xdmf_elements::topology::{Topology, TopologyType},
};

/// One submesh's decoded topology: its cells' types and its connectivity with any `Mixed` type
/// codes stripped out, indices still local to whatever point list the submesh was written against.
#[derive(Debug)]
pub(super) struct DecodedTopology {
    pub cell_types: Vec<CellType>,
    pub connectivity: Vec<u64>,
}

/// Decode a `Topology`'s raw connectivity values, per `topology.topology_type`.
pub(super) fn decode(topology: &Topology, raw: &Values<'_>) -> Result<DecodedTopology> {
    let values = widen_to_u64(raw)?;

    match topology.topology_type {
        TopologyType::Mixed => decode_mixed(&values),
        uniform => decode_uniform(uniform, topology.nodes_per_element, &values),
    }
}

fn decode_uniform(
    topology_type: TopologyType,
    nodes_per_element: Option<u8>,
    values: &[u64],
) -> Result<DecodedTopology> {
    let cell_type = cell_type_of(topology_type, nodes_per_element)?;
    let stride = cell_type.num_points();

    if stride == 0 || !values.len().is_multiple_of(stride) {
        return Err(Error::InvalidDocument {
            reason: format!(
                "a {topology_type:?} connectivity of {} values is not a multiple of {stride}",
                values.len()
            ),
        });
    }

    let num_cells = values.len() / stride;

    Ok(DecodedTopology {
        cell_types: vec![cell_type; num_cells],
        connectivity: values.to_vec(),
    })
}

fn decode_mixed(values: &[u64]) -> Result<DecodedTopology> {
    let mut cell_types = Vec::new();
    let mut connectivity = Vec::new();
    let mut position = 0;

    while position < values.len() {
        let code = u8::try_from(values[position]).map_err(|_source| Error::InvalidDocument {
            reason: format!(
                "Mixed connectivity cell type code {} is out of range",
                values[position]
            ),
        })?;
        position += 1;

        let cell_type = CellType::from_code(code).ok_or_else(|| Error::InvalidDocument {
            reason: format!("Mixed connectivity has an unknown cell type code {code}"),
        })?;

        // Vertex/Edge poly-cells carry their point count next. For this crate's own files it is
        // redundant -- always the type's own fixed count -- but a foreign file may state another,
        // and taking it on trust would desynchronise the stream and surface as an "unknown cell
        // type code" further along. Rejected here instead, as `cell_type_of` rejects the
        // equivalent `NodesPerElement` for a uniform topology.
        if let Some(expected) = poly_cell_points(cell_type) {
            let stated = *values.get(position).ok_or_else(|| Error::InvalidDocument {
                reason: format!(
                    "Mixed connectivity ends after a {cell_type:?} cell's type code, before its \
                     point count"
                ),
            })?;

            if stated != u64::from(expected) {
                return Err(Error::Unsupported {
                    reason: format!(
                        "a Mixed connectivity's {cell_type:?} cell states {stated} points, only \
                         {expected} is supported"
                    ),
                });
            }

            position += 1;
        }

        let num_points = cell_type.num_points();
        let end = position.checked_add(num_points).ok_or(Error::Internal(
            "a Mixed connectivity's cell span does not fit a usize",
        ))?;
        let cell_points = values
            .get(position..end)
            .ok_or_else(|| Error::InvalidDocument {
                reason: "Mixed connectivity ends in the middle of a cell".to_string(),
            })?;

        connectivity.extend_from_slice(cell_points);
        cell_types.push(cell_type);
        position = end;
    }

    Ok(DecodedTopology {
        cell_types,
        connectivity,
    })
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

/// Widen a raw connectivity array to `u64`, as every reader-facing connectivity is: the crate's
/// own writer never emits a negative index, and a foreign file's is rejected rather than wrapped.
fn widen_to_u64(values: &Values<'_>) -> Result<Vec<u64>> {
    let widen_signed = |value: i64| {
        u64::try_from(value).map_err(|_source| Error::InvalidDocument {
            reason: format!("connectivity index {value} is negative"),
        })
    };

    match values {
        Values::F64(_) | Values::F32(_) => Err(Error::InvalidDocument {
            reason: "a Topology's connectivity holds floating-point values".to_string(),
        }),
        Values::U64(v) => Ok(v.to_vec()),
        Values::U32(v) => Ok(v.iter().map(|&value| u64::from(value)).collect()),
        Values::I64(v) => v.iter().map(|&value| widen_signed(value)).collect(),
        Values::I32(v) => v
            .iter()
            .map(|&value| widen_signed(i64::from(value)))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdmf_elements::data_item::DataItem;

    fn mixed(values: &[u64]) -> Result<DecodedTopology> {
        let topology = Topology {
            topology_type: TopologyType::Mixed,
            nodes_per_element: None,
            number_of_elements: "1".to_string(),
            data_item: DataItem::default(),
        };

        decode(&topology, &Values::from(values))
    }

    #[test]
    fn a_poly_cells_point_count_is_consumed() {
        // a Vertex (code 1, 1 point) and an Edge (code 2, 2 points), each stating its own count
        let decoded = mixed(&[1, 1, 7, 2, 2, 3, 4]).unwrap();

        assert_eq!(decoded.cell_types, [CellType::Vertex, CellType::Edge]);
        assert_eq!(decoded.connectivity, [7, 3, 4]);
    }

    /// Taking the stated count on trust would consume one value too few and read the rest of the
    /// stream shifted, which surfaces as a confusing "unknown cell type code" further along.
    #[test]
    fn a_poly_cell_stating_another_point_count_is_rejected() {
        std::assert_matches!(
            mixed(&[1, 3, 7, 8, 9]).unwrap_err(),
            Error::Unsupported { reason }
                if reason.contains("Vertex cell states 3 points, only 1 is supported")
        );
    }

    #[test]
    fn a_poly_cell_without_its_point_count_is_rejected() {
        std::assert_matches!(
            mixed(&[1]).unwrap_err(),
            Error::InvalidDocument { reason } if reason.contains("before its point count")
        );
    }
}

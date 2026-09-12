//! Reconstructing a mesh, its topology, and per-step field data from submeshes: the inverse of
//! `writer/submesh.rs`'s splitting of a mesh's cells into named blocks.

use super::{
    grid_topology,
    light_data::{self, Analysis, Document},
    selection::{self, Membership},
    to_connectivity_index, topology,
};
use crate::{
    CellType, ConnectivityIndex, Coordinate, Error, Result, SUBMESH_CELLS, Values,
    xdmf_elements::{Domain, grid::Grid},
};

/// The mesh's own points, for a mesh with submeshes: read out of the source each submesh's
/// `<Geometry>` selects from.
///
/// One direction at a time, scattered into its stride of the interleaved output, so the three
/// whole-mesh arrays are never live together.
pub(super) fn read_points_with_submeshes<C: Coordinate>(
    document: &Document,
    domain: &Domain,
    first_grid: &Grid,
    points: &mut Vec<C>,
) -> Result<()> {
    let geometry = first_grid
        .geometry
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Geometry", first_grid.name),
        })?;

    if geometry.data_items.len() != 3 {
        return Err(Error::InvalidDocument {
            reason: format!(
                "a submesh's Geometry must have 3 DataItems (one per direction), found {}",
                geometry.data_items.len()
            ),
        });
    }

    let mut direction: Vec<C> = Vec::new();
    let mut num_points_total = 0;

    for (axis, item) in geometry.data_items.iter().enumerate() {
        let selection_item = light_data::resolve_reference(item, domain)?;
        let (_selector, source) = selection::selection_parts(selection_item)?;
        selection::read_data_item_into(
            source,
            document,
            domain,
            &mut direction,
            C::coordinates_from_values,
        )?;

        if axis == 0 {
            num_points_total = direction.len();
            points.clear();
            points.resize(num_points_total * 3, C::default());
        } else if direction.len() != num_points_total {
            return Err(Error::InvalidDocument {
                reason: "the mesh's per-direction coordinate arrays have different lengths"
                    .to_string(),
            });
        }

        for (point, &coordinate) in direction.iter().enumerate() {
            points[point * 3 + axis] = coordinate;
        }
    }

    Ok(())
}

/// Reconstruct the mesh's cell types and connectivity from its submeshes' own topology, scattered
/// into the mesh's indexing through each submesh's membership.
///
/// Two passes: the mesh's cell offsets need every cell's type before any cell's points can be
/// placed, but only one submesh's decoded topology is live at a time. A uniform submesh costs no
/// heavy-data read in the first pass; only `Mixed` is decoded twice.
pub(super) fn read_topology_with_submeshes<I: ConnectivityIndex>(
    document: &Document,
    analysis: &Analysis,
    points_membership: &[Membership],
    cells_membership: &[Membership],
    connectivity: &mut Vec<I>,
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    let domain = document.domain()?;
    let num_cells = mesh_num_cells_from_membership(cells_membership);

    if points_membership.len() != analysis.num_submeshes()
        || cells_membership.len() != analysis.num_submeshes()
    {
        return Err(Error::Internal(
            "read_topology_with_submeshes called with membership of the wrong length",
        ));
    }

    let mut scratch_connectivity: Vec<I> = Vec::new();
    let mut scratch_cell_types: Vec<CellType> = Vec::new();
    let mut covered = vec![false; num_cells];

    cell_types.clear();
    cell_types.resize(num_cells, CellType::Vertex);

    for (submesh, cells) in cells_membership.iter().enumerate() {
        let grid = analysis.mesh_grid(submesh, domain)?;

        if let Some(cell_type) = topology::uniform_cell_type(grid_topology(grid)?)? {
            for global_cell in cells.iter() {
                mark_cell(global_cell, cell_type, cell_types, &mut covered)?;
            }
        } else {
            decode_submesh_topology(
                grid,
                domain,
                document,
                cells,
                &mut scratch_connectivity,
                &mut scratch_cell_types,
            )?;

            for (&cell_type, global_cell) in scratch_cell_types.iter().zip(cells.iter()) {
                mark_cell(global_cell, cell_type, cell_types, &mut covered)?;
            }
        }
    }

    if covered.iter().any(|&is_covered| !is_covered) {
        return Err(Error::InvalidDocument {
            reason: "some mesh cells are not covered by any submesh".to_string(),
        });
    }

    let mut offsets = Vec::with_capacity(num_cells + 1);
    let mut offset = 0_usize;
    for cell_type in cell_types.iter() {
        offsets.push(offset);
        offset += cell_type.num_points();
    }
    offsets.push(offset);

    connectivity.clear();
    connectivity.resize(offset, I::default());

    for ((submesh, cells), points) in (0..analysis.num_submeshes())
        .zip(cells_membership)
        .zip(points_membership)
    {
        let grid = analysis.mesh_grid(submesh, domain)?;
        decode_submesh_topology(
            grid,
            domain,
            document,
            cells,
            &mut scratch_connectivity,
            &mut scratch_cell_types,
        )?;

        let mut local_offset = 0_usize;
        for (&cell_type, global_cell) in scratch_cell_types.iter().zip(cells.iter()) {
            // `offsets` was built from the cell types pass 1 recorded, where the last submesh to
            // claim an overlapped cell won. Two submeshes disagreeing about a shared cell's type
            // would make this cell's span here a different width than the slot reserved for it,
            // and write over its neighbour's -- so the disagreement is reported instead.
            if cell_type != cell_types[global_cell] {
                return Err(Error::InvalidDocument {
                    reason: format!(
                        "submeshes disagree about mesh cell {global_cell}: one holds it as \
                         {cell_type:?}, another as {:?}",
                        cell_types[global_cell]
                    ),
                });
            }

            let stride = cell_type.num_points();
            let global_start = offsets[global_cell];

            for component in 0..stride {
                // every value came through `SealedIndex::indices_from_values`, which rejects an
                // entry that is not a position, so this cannot be `None`
                let local_point = scratch_connectivity[local_offset + component]
                    .as_index()
                    .ok_or(Error::Internal(
                        "a decoded connectivity entry is not a position",
                    ))?;
                let global_point = points.get(local_point).ok_or_else(|| {
                    Error::InvalidDocument {
                        reason: format!(
                            "a submesh's connectivity references its point {local_point}, but its \
                             Geometry selects only {} points",
                            points.len()
                        ),
                    }
                })?;
                connectivity[global_start + component] = to_connectivity_index::<I>(global_point)?;
            }

            local_offset += stride;
        }
    }

    Ok(())
}

/// Record one mesh cell's type, and that some submesh covers it.
fn mark_cell(
    global_cell: usize,
    cell_type: CellType,
    cell_types: &mut [CellType],
    covered: &mut [bool],
) -> Result<()> {
    let num_cells = cell_types.len();
    let slot = covered
        .get_mut(global_cell)
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!(
                "'{SUBMESH_CELLS}' names cell {global_cell}, but the mesh only has {num_cells} \
                 cells"
            ),
        })?;

    *slot = true;
    cell_types[global_cell] = cell_type;

    Ok(())
}

/// Decode one submesh's topology into buffers the caller reuses across submeshes, rejecting a
/// `Topology` that holds a different number of cells than `submesh_cells` names for it.
fn decode_submesh_topology<I: ConnectivityIndex>(
    grid: &Grid,
    domain: &Domain,
    document: &Document,
    cells: &Membership,
    connectivity: &mut Vec<I>,
    cell_types: &mut Vec<CellType>,
) -> Result<()> {
    let topology = grid_topology(grid)?;

    selection::read_data_item_into(
        &topology.data_item,
        document,
        domain,
        connectivity,
        I::indices_from_values,
    )?;
    topology::decode_in_place(topology, connectivity, cell_types)?;

    if cell_types.len() != cells.len() {
        return Err(Error::InvalidDocument {
            reason: format!(
                "a submesh's Topology holds {} cells, but '{SUBMESH_CELLS}' names {} for it",
                cell_types.len(),
                cells.len()
            ),
        });
    }

    Ok(())
}
/// Which mesh points each submesh holds, read from its `<Geometry>` selector.
pub(super) fn submesh_points_membership(
    analysis: &Analysis,
    domain: &Domain,
    document: &Document,
) -> Result<Vec<Membership>> {
    (0..analysis.num_submeshes())
        .map(|submesh| {
            submesh_geometry_membership(analysis.mesh_grid(submesh, domain)?, domain, document)
        })
        .collect()
}

/// Reject a membership naming an entity the mesh does not have, once, when the document is opened.
///
/// Only points need it: the mesh's cell count is taken *from* the cell lists, while a submesh's
/// points and the mesh's point count are independent statements and can disagree. Everything later
/// treats these as positions -- [`scatter_field`] writes at them.
pub(super) fn check_membership_in_range(
    membership: &[Membership],
    num_entities: usize,
    entity: &str,
) -> Result<()> {
    for (submesh, entities) in membership.iter().enumerate() {
        if let Some(out_of_range) = entities.iter().find(|&index| index >= num_entities) {
            return Err(Error::InvalidDocument {
                reason: format!(
                    "submesh {submesh} holds {entity} {out_of_range}, but the mesh only has \
                     {num_entities} {entity}s"
                ),
            });
        }
    }

    Ok(())
}

fn submesh_geometry_membership(
    grid: &Grid,
    domain: &Domain,
    document: &Document,
) -> Result<Membership> {
    let geometry = grid
        .geometry
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Geometry", grid.name),
        })?;
    let first_item = geometry
        .data_items
        .first()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Geometry of Grid '{}' has no DataItem", grid.name),
        })?;
    let selection_item = light_data::resolve_reference(first_item, domain)?;
    let (selector, _source) = selection::selection_parts(selection_item)?;

    selection::parse_selector(selector, document, domain)
}

/// Parse `<Information Name="submesh_cells">`'s value into one [`Membership`] per submesh, in
/// submesh order. An entry is either `<start>:<len>` or the name of the `DataItem` holding its
/// indices.
pub(super) fn parse_submesh_cells(document: &Document, domain: &Domain) -> Result<Vec<Membership>> {
    let value = document
        .information(SUBMESH_CELLS)
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("the document has submeshes but no '{SUBMESH_CELLS}' Information"),
        })?;

    value
        .split_whitespace()
        .map(|entry| parse_submesh_cells_entry(entry, document, domain))
        .collect()
}

fn parse_submesh_cells_entry(
    entry: &str,
    document: &Document,
    domain: &Domain,
) -> Result<Membership> {
    if let Some((start, len)) = entry.split_once(':') {
        let start = start.parse().map_err(|_source| Error::InvalidDocument {
            reason: format!("'{SUBMESH_CELLS}' entry '{entry}' has an invalid start"),
        })?;
        let len = len.parse().map_err(|_source| Error::InvalidDocument {
            reason: format!("'{SUBMESH_CELLS}' entry '{entry}' has an invalid length"),
        })?;
        return Ok(Membership::Contiguous { start, len });
    }

    let item = light_data::find_by_name(domain, entry).ok_or_else(|| Error::InvalidDocument {
        reason: format!("'{SUBMESH_CELLS}' names a DataItem '{entry}' that does not exist"),
    })?;
    let values = selection::read_data_item(item, document, domain)?;

    Ok(Membership::Explicit(selection::values_to_usize(&values)?))
}

/// `num_points` for a mesh with submeshes: the `Dimensions` of the *source* array a submesh's
/// geometry selects out of, which is the mesh's own coordinates.
pub(super) fn mesh_num_points_with_submeshes(first_grid: &Grid, domain: &Domain) -> Result<usize> {
    let geometry = first_grid
        .geometry
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Grid '{}' has no Geometry", first_grid.name),
        })?;
    let item = geometry
        .data_items
        .first()
        .ok_or_else(|| Error::InvalidDocument {
            reason: format!("Geometry of Grid '{}' has no DataItem", first_grid.name),
        })?;

    let selection_item = light_data::resolve_reference(item, domain)?;
    let (_selector, source) = selection::selection_parts(selection_item)?;
    let dims = source
        .dimensions
        .as_ref()
        .ok_or_else(|| Error::InvalidDocument {
            reason: "the mesh's coordinate source DataItem has no Dimensions".to_string(),
        })?;

    dims.0
        .first()
        .copied()
        .ok_or_else(|| Error::InvalidDocument {
            reason: "the mesh's coordinate source DataItem has empty Dimensions".to_string(),
        })
}

/// `num_cells` for a mesh with submeshes: `1 + max` over every submesh's `submesh_cells` entry,
/// sound because the writer's `check_all_cells_covered` makes every cell belong to at least one
/// submesh.
pub(super) fn mesh_num_cells_from_membership(cells_membership: &[Membership]) -> usize {
    let max_cell = cells_membership.iter().flat_map(Membership::iter).max();

    max_cell.map_or(0, |max| max + 1)
}

/// Scatter every submesh's own share of a field into the mesh's own indexing through its
/// membership, the field-data counterpart of the connectivity scatter in
/// [`read_topology_with_submeshes`]. Works whether the submesh's `DataItem` was a selection or a
/// private copy, which this writer does not emit but a foreign file may.
pub(super) fn scatter_field(
    num_entities: usize,
    submesh_values: &[Values<'static>],
    membership: &[Membership],
) -> Result<Values<'static>> {
    if submesh_values.len() != membership.len() {
        return Err(Error::Internal(
            "scatter_field called with mismatched submeshes and membership",
        ));
    }

    let Some(first) = submesh_values.first() else {
        return Err(Error::InvalidDocument {
            reason: "a field with submeshes has no data at all".to_string(),
        });
    };

    let components = component_count(first, &membership[0])?;
    let total = num_entities.checked_mul(components).ok_or(Error::Internal(
        "scattered field length does not fit a usize",
    ))?;

    macro_rules! scatter_arm {
        ($variant:ident) => {{
            let mut entries: Vec<(&[_], &Membership)> = Vec::with_capacity(submesh_values.len());
            for (values, entities) in submesh_values.iter().zip(membership) {
                let Values::$variant(v) = values else {
                    return Err(Error::InvalidDocument {
                        reason: "a submesh's field data has a different type than another's"
                            .to_string(),
                    });
                };
                if v.len() != entities.len() * components {
                    return Err(Error::InvalidDocument {
                        reason: format!(
                            "a submesh's field data has {} values, expected {}",
                            v.len(),
                            entities.len() * components
                        ),
                    });
                }
                entries.push((v.as_ref(), entities));
            }
            Values::from(scatter_typed(total, &entries, components))
        }};
    }

    Ok(match first {
        Values::F64(_) => scatter_arm!(F64),
        Values::F32(_) => scatter_arm!(F32),
        Values::I64(_) => scatter_arm!(I64),
        Values::I32(_) => scatter_arm!(I32),
        Values::U64(_) => scatter_arm!(U64),
        Values::U32(_) => scatter_arm!(U32),
    })
}

fn component_count(values: &Values<'_>, membership: &Membership) -> Result<usize> {
    let len = values.len();
    let entities = membership.len();

    if entities == 0 || !len.is_multiple_of(entities) {
        return Err(Error::InvalidDocument {
            reason: format!(
                "a field's {len} values do not divide evenly over its submesh's {entities} entities"
            ),
        });
    }

    Ok(len / entities)
}

fn scatter_typed<T: Copy + Default>(
    total: usize,
    entries: &[(&[T], &Membership)],
    components: usize,
) -> Vec<T> {
    let mut buffer = vec![T::default(); total];

    for (values, membership) in entries {
        for (local, global_entity) in membership.iter().enumerate() {
            let src = local * components;
            let dst = global_entity * components;
            buffer[dst..dst + components].copy_from_slice(&values[src..src + components]);
        }
    }

    buffer
}

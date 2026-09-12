//! Preparing a mesh's cells for writing, and splitting them into named submeshes.

use std::{borrow::Cow, collections::HashSet, ops::Range};

use super::{DOMAIN_DATA_ITEMS, hyper_slab, is_valid_data_name, selection};
use crate::{
    CellType, ConnectivityIndex, Error, Result, SUBMESH_POINTS, Values,
    xdmf_elements::{
        data_item::DataItem,
        geometry::{Geometry, GeometryType},
        topology::TopologyType,
    },
};

/// A submesh's cell or point indices, as the narrowest integer type that holds all of them.
/// Narrowing is allowed here where it is nowhere else, since these indices are the crate's own.
/// Signed rather than unsigned, even though never negative, because `ParaView` decodes a
/// `NumberType="UInt"` array at 32 bits whatever `Precision` says.
pub(super) fn index_values(indices: &[usize]) -> Result<Values<'static>> {
    if let Some(indices) = indices
        .iter()
        .map(|&index| i32::try_from(index).ok())
        .collect::<Option<Vec<i32>>>()
    {
        return Ok(Values::from(indices));
    }

    let indices = indices
        .iter()
        .map(|&index| i64::try_from(index).ok())
        .collect::<Option<Vec<i64>>>()
        .ok_or(Error::Internal("an index does not fit into 64 bits"))?;

    Ok(Values::from(indices))
}

/// The parts of a written mesh that do not depend on how its cells are split into submeshes.
pub(super) struct PreparedMesh<'c, I: Clone> {
    pub num_points: usize,
    pub num_cells: usize,
    pub topology_type: TopologyType,
    /// Per-element node count, set only for the `Polyvertex`/`Polyline` topologies that carry one.
    pub nodes_per_element: Option<u8>,
    /// Connectivity of the whole mesh, with the cell types prepended for a `Mixed` mesh.
    ///
    /// Borrowed from the caller's array unless something had to be prepended (a `Mixed` mesh, or
    /// one of points only).
    pub cells: Cow<'c, [I]>,
}

/// One submesh's geometry, as a selection out of the mesh's coordinates: one item per direction,
/// selecting that submesh's points out of that direction's array. Named and `Domain`-level so
/// cloning the grid per time step repeats a short reference.
///
/// All three share the mesh's own `submesh_points` list as their selector for a scattered submesh.
/// The link is by name alone, and [`crate::TimeSeriesWriter::write_submesh_index_lists`] writes that item
/// only because of this reference, so change the two together.
pub(super) fn selected_coordinates(
    submesh: usize,
    coordinates: &[DataItem; 3],
    points: &IndexList,
    num_points: usize,
) -> Vec<DataItem> {
    coordinates
        .iter()
        .zip(["x", "y", "z"])
        .map(|(source, direction)| {
            let selector = match points {
                IndexList::Contiguous { start, len } => hyper_slab(*start, *len, 1),
                IndexList::Scattered(_) => submesh_index_reference(SUBMESH_POINTS, submesh),
            };

            let mut item = selection(selector, source, points.len(), &[num_points]);
            item.name = Some(format!("coords_{submesh}_{direction}"));

            item
        })
        .collect()
}

/// A grid's geometry over coordinates split by direction: the (short) references to the three
/// selections that cut its own points out of them.
pub(super) fn selected_geometry(coordinate_items: &[DataItem]) -> Geometry {
    Geometry {
        geometry_type: GeometryType::XYZSeparate,
        data_items: coordinate_items
            .iter()
            .map(|item| DataItem::new_reference(item, DOMAIN_DATA_ITEMS))
            .collect(),
    }
}

/// Name of the `DataItem` holding one submesh's cell or point index list.
pub(super) fn submesh_index_name(array: &str, submesh: usize) -> String {
    format!("{array}_{submesh}")
}

/// A reference to that list, for a submesh that has one -- which is every submesh whose entities
/// are not one run, and only those.
fn submesh_index_reference(array: &str, submesh: usize) -> DataItem {
    DataItem::new_reference(
        &DataItem {
            name: Some(submesh_index_name(array, submesh)),
            ..Default::default()
        },
        DOMAIN_DATA_ITEMS,
    )
}

/// A named subset of a mesh's cells as it comes out of validation, before the mesh has been walked
/// to find the points those cells use.
#[derive(Debug)]
pub(super) struct PreparedSubmesh {
    pub name: String,
    pub cells: IndexList,
}

/// A submesh as the data writer keeps it: its name, plus the cell and point lists its share of
/// every time step is cut with.
#[derive(Debug)]
pub(super) struct Submesh {
    pub name: String,
    pub cells: IndexList,
    pub points: IndexList,
}

/// Which of a submesh's two index lists cuts a field: a caller passes a field over the whole
/// mesh, so what a submesh's share is depends on which of them the field is indexed by.
pub(super) fn entities_of(submesh: &Submesh, point_data: bool) -> &IndexList {
    if point_data {
        &submesh.points
    } else {
        &submesh.cells
    }
}

/// Which index array a scattered submesh selects fields of one width with: a selector names the
/// position of every value it picks, so it depends on the entity's width as well as the submesh.
/// One array per (submesh, centering, component count) is written once and reused after.
#[derive(Debug, Eq, Hash, PartialEq)]
pub(super) struct SelectionKey {
    pub submesh: usize,
    /// point data is cut by the submesh's points, cell data by its cells
    pub point_data: bool,
    /// how many values one entity has, which is the `DataAttribute`'s component count
    pub components: usize,
}

/// An ascending list of a submesh's cells or points, as positions in the mesh.
///
/// Mesh generators usually produce element blocks, material zones and boundary patches grouped, so
/// a cell list is one ascending run. That case collapses to two numbers, after which every
/// per-step slice of a field is a borrow of the caller's array rather than a gather.
#[derive(Debug)]
pub(super) enum IndexList {
    Contiguous { start: usize, len: usize },
    Scattered(Vec<usize>),
}

impl IndexList {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Contiguous { len, .. } => *len,
            Self::Scattered(indices) => indices.len(),
        }
    }

    /// Whether the list is ascending, which a `Coordinates` selection needs: `ParaView` hands the
    /// values back in array order rather than the order they were named. A submesh listing its
    /// cells any other way gets a copy of its share instead.
    pub(super) fn is_ascending(&self) -> bool {
        match self {
            Self::Contiguous { .. } => true,
            Self::Scattered(indices) => indices.windows(2).all(|pair| pair[0] < pair[1]),
        }
    }

    /// The indices themselves, for a scattered list in the order it holds them.
    pub(super) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let (run, indices) = match self {
            Self::Contiguous { start, len } => (*start..*start + *len, [].as_slice()),
            Self::Scattered(indices) => (0..0, indices.as_slice()),
        };

        run.chain(indices.iter().copied())
    }
}

/// The cells of one submesh, as [`crate::TimeSeriesWriter::write_mesh_with_submeshes`] takes them.
///
/// Built with `.into()` from a slice, a `Vec`, an array, or a [`Range`], so a submesh of
/// consecutive cells can be given as `start..end` without building an index list at all.
#[derive(Clone, Debug)]
pub enum SubmeshCells<'a> {
    /// A block of consecutive cells, `start..end`.
    Range(Range<usize>),
    /// One index per cell, in the order the submesh holds them.
    Indices(Cow<'a, [usize]>),
}

impl SubmeshCells<'_> {
    /// The cheapest internal form that holds these cells, taking the caller's own allocation
    /// where it is already the right shape.
    fn into_index_list(self) -> IndexList {
        match self {
            Self::Range(range) => IndexList::Contiguous {
                start: range.start,
                len: range.end.saturating_sub(range.start),
            },
            Self::Indices(Cow::Borrowed(indices)) => collapse_indices(indices),
            Self::Indices(Cow::Owned(indices)) => {
                if is_contiguous(&indices) {
                    return IndexList::Contiguous {
                        start: indices.first().copied().unwrap_or(0),
                        len: indices.len(),
                    };
                }

                IndexList::Scattered(indices)
            }
        }
    }
}

impl From<Range<usize>> for SubmeshCells<'_> {
    fn from(range: Range<usize>) -> Self {
        Self::Range(range)
    }
}

impl<'a> From<&'a [usize]> for SubmeshCells<'a> {
    fn from(indices: &'a [usize]) -> Self {
        Self::Indices(Cow::Borrowed(indices))
    }
}

// The `&Vec<usize>` and `&[usize; N]` impls are not redundant with the `&[usize]` one, for the
// same reason `Values`' are not: an `impl Into<...>` argument is resolved by trait matching, which
// does not deref-coerce.
impl<'a> From<&'a Vec<usize>> for SubmeshCells<'a> {
    fn from(indices: &'a Vec<usize>) -> Self {
        Self::Indices(Cow::Borrowed(indices))
    }
}

impl<'a, const N: usize> From<&'a [usize; N]> for SubmeshCells<'a> {
    fn from(indices: &'a [usize; N]) -> Self {
        Self::Indices(Cow::Borrowed(indices))
    }
}

/// Moves the caller's own index list in, so a scattered submesh needs no copy of it.
impl From<Vec<usize>> for SubmeshCells<'_> {
    fn from(indices: Vec<usize>) -> Self {
        Self::Indices(Cow::Owned(indices))
    }
}

impl<const N: usize> From<[usize; N]> for SubmeshCells<'_> {
    fn from(indices: [usize; N]) -> Self {
        Self::Indices(Cow::Owned(indices.to_vec()))
    }
}

/// Validate the submeshes and collapse each one's index list to the cheapest form that holds it.
pub(super) fn prepare_submeshes<'c, N: AsRef<str>, B: Into<SubmeshCells<'c>>>(
    submeshes: impl IntoIterator<Item = (N, B)>,
    num_cells: usize,
) -> Result<Vec<PreparedSubmesh>> {
    let mut prepared: Vec<PreparedSubmesh> = Vec::new();
    let mut names = HashSet::new();

    // which cells any submesh has claimed, for the coverage check below
    let mut covered = CellBitSet::new(num_cells);
    // which cells the submesh being read has claimed, to tell a cell repeated within one submesh
    // apart from two submeshes overlapping on it; cleared per submesh, not per mesh
    let mut claimed_here = CellBitSet::new(num_cells);

    for (name, cells) in submeshes {
        let name = name.as_ref();
        let cells = cells.into().into_index_list();

        if !is_valid_data_name(name) {
            return Err(Error::InvalidMesh {
                reason: format!(
                    "submesh name '{name}' is not valid, must contain a non-whitespace character \
                     and must not contain control characters"
                ),
            });
        }

        // compared verbatim: the name reaches only the `<Grid>` element that carries it
        if !names.insert(name.to_string()) {
            return Err(Error::InvalidMesh {
                reason: format!("submesh name '{name}' is used more than once"),
            });
        }

        if cells.len() == 0 {
            return Err(Error::InvalidMesh {
                reason: format!("submesh '{name}' is empty, it must contain at least one cell"),
            });
        }

        match &cells {
            // A run needs neither the duplicate check (it has none by construction) nor a walk
            // over its own indices to bound it -- which is what lets a caller hand over a block of
            // a huge mesh as a range without ever materialising one index per cell.
            IndexList::Contiguous { start, len } => {
                let end = start.checked_add(*len).ok_or(Error::Internal(
                    "a submesh's cell range does not fit a usize",
                ))?;

                if end > num_cells {
                    return Err(Error::InvalidMesh {
                        reason: format!(
                            "submesh '{name}' references cell {}, but the mesh only has \
                             {num_cells} cells",
                            end - 1
                        ),
                    });
                }

                for index in *start..end {
                    covered.insert(index);
                }
            }
            IndexList::Scattered(indices) => {
                for &index in indices {
                    if index >= num_cells {
                        return Err(Error::InvalidMesh {
                            reason: format!(
                                "submesh '{name}' references cell {index}, but the mesh only has \
                                 {num_cells} cells"
                            ),
                        });
                    }

                    if claimed_here.contains(index) {
                        return Err(Error::InvalidMesh {
                            reason: format!(
                                "submesh '{name}' contains cell {index} more than once"
                            ),
                        });
                    }

                    claimed_here.insert(index);
                    covered.insert(index);
                }

                for &index in indices {
                    claimed_here.remove(index);
                }
            }
        }

        prepared.push(PreparedSubmesh {
            name: name.to_string(),
            cells,
        });
    }

    if prepared.is_empty() {
        return Err(Error::InvalidMesh {
            reason: "at least one submesh is required".to_string(),
        });
    }

    check_all_cells_covered(&covered)?;

    Ok(prepared)
}

/// One bit per cell, for `prepare_submeshes`'s two membership questions.
///
/// A `Vec<usize>` naming which submesh last claimed each cell would cost 8 bytes per cell (800 MB
/// on a 100M-cell mesh); two bit sets cost a quarter of a byte between them.
struct CellBitSet {
    words: Vec<u64>,
    len: usize,
}

impl CellBitSet {
    const BITS: usize = u64::BITS as usize;

    fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(Self::BITS)],
            len,
        }
    }

    // every index reaching these was already checked against the cell count, keeping the word
    // lookup in bounds
    fn contains(&self, index: usize) -> bool {
        self.words[index / Self::BITS] & (1 << (index % Self::BITS)) != 0
    }

    fn insert(&mut self, index: usize) {
        self.words[index / Self::BITS] |= 1 << (index % Self::BITS);
    }

    fn remove(&mut self, index: usize) {
        self.words[index / Self::BITS] &= !(1 << (index % Self::BITS));
    }

    /// The cells whose bit is unset, in ascending order.
    fn missing(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&index| !self.contains(index))
    }
}

/// Whether a list is one ascending run of consecutive indices, which needs no per-index storage.
/// An empty list counts; `prepare_submeshes` rejects it right after.
fn is_contiguous(cells: &[usize]) -> bool {
    cells.first().is_none_or(|&start| {
        cells
            .iter()
            .enumerate()
            .all(|(offset, &index)| index == start + offset)
    })
}

/// Recognize such a run, borrowing the indices only if it is not one.
fn collapse_indices(cells: &[usize]) -> IndexList {
    if is_contiguous(cells) {
        IndexList::Contiguous {
            start: cells.first().copied().unwrap_or(0),
            len: cells.len(),
        }
    } else {
        IndexList::Scattered(cells.to_vec())
    }
}

/// Reject a mesh with cells in no submesh: such a cell reaches none of the grids, so it would
/// vanish from the visualization rather than fail.
fn check_all_cells_covered(covered: &CellBitSet) -> Result<()> {
    const MAX_LISTED: usize = 10;

    let mut uncovered = covered.missing();

    // only a handful are collected; the rest are just counted, so reporting the mistake on a huge
    // mesh does not itself allocate an array as large as the mesh
    let listed_indices: Vec<usize> = uncovered.by_ref().take(MAX_LISTED).collect();

    if listed_indices.is_empty() {
        return Ok(());
    }

    let num_not_listed = uncovered.count();
    let num_uncovered = listed_indices.len() + num_not_listed;

    let listed = listed_indices
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let ellipsis = if num_not_listed > 0 {
        format!(", ... ({num_not_listed} more)")
    } else {
        String::new()
    };

    Err(Error::InvalidMesh {
        reason: format!(
            "{num_uncovered} of {} cells belong to no submesh: {listed}{ellipsis}. Every cell \
             must be in at least one submesh; leave the others out of the mesh instead",
            covered.len
        ),
    })
}

/// Where each cell's entries start in the prepared connectivity, with a final entry for its end.
///
/// `Mixed` connectivity prepends the cell type (and, for a poly-cell, the point count) to each
/// cell's points; a uniform topology stores only the points.
pub(super) fn cell_offsets(
    cell_types: &[CellType],
    topology_type: TopologyType,
    num_cells: usize,
) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(num_cells + 1);
    let mut offset = 0;

    for cell in 0..num_cells {
        offsets.push(offset);
        offset += cell_span(cell_types, topology_type, cell);
    }
    offsets.push(offset);

    offsets
}

/// How many entries one cell takes in a connectivity written as `topology_type`: its point ids,
/// behind whatever `Mixed` puts in front of them.
fn cell_span(cell_types: &[CellType], topology_type: TopologyType, cell: usize) -> usize {
    // the polyvertex fallback for a mesh of points only numbers each point one entry
    let Some(cell_type) = cell_types.get(cell) else {
        return 1;
    };

    leading_entries(cell_types, topology_type, cell) + cell_type.num_points()
}

/// A submesh's share of the prepared connectivity, in the order it lists its cells.
///
/// Owned even for one contiguous run, because the copy is renumbered into the submesh's own
/// points afterwards. A submesh written uniformly drops the type codes a `Mixed` mesh carries.
pub(super) fn extract_connectivity<I: ConnectivityIndex>(
    cells: &[I],
    offsets: &[usize],
    cell_types: &[CellType],
    mesh_topology: TopologyType,
    submesh_topology: TopologyType,
    submesh: &IndexList,
) -> Vec<I> {
    // the exact size is one pass over the cells away, cheaper than growing while gathering
    let size = submesh
        .iter()
        .map(|cell| cell_span(cell_types, submesh_topology, cell))
        .sum();

    let mut extracted = Vec::with_capacity(size);
    for cell in submesh.iter() {
        // what the mesh puts ahead of this cell's points, less what the submesh keeps of it (never
        // negative: a submesh is `Mixed` only where the mesh is, and then keeps all of it)
        let dropped = leading_entries(cell_types, mesh_topology, cell)
            - leading_entries(cell_types, submesh_topology, cell);

        extracted.extend_from_slice(&cells[offsets[cell] + dropped..offsets[cell + 1]]);
    }

    extracted
}

/// The topology one submesh's own cells are written as: the `CellType` all of them share, or the
/// mesh's own where they do not. Decided per submesh, since a `Mixed` mesh's blocks (e.g. a
/// hexahedra volume beside a quadrilateral boundary) are often individually uniform, saving one
/// index per cell. The second value is the per-element node count only `Polyvertex`/`Polyline`
/// carry.
pub(super) fn submesh_topology(
    cell_types: &[CellType],
    mesh_topology: TopologyType,
    mesh_nodes_per_element: Option<u8>,
    cells: &IndexList,
) -> (TopologyType, Option<u8>) {
    let mesh = (mesh_topology, mesh_nodes_per_element);

    // every cell of the mesh already shares one type, so every submesh's cells share that one too
    if mesh_topology != TopologyType::Mixed {
        return mesh;
    }

    let mut cells = cells.iter();
    // `prepare_submeshes` rejects an empty submesh, so the mesh's own is only a fallback here
    let Some(first) = cells.next().map(|cell| cell_types[cell]) else {
        return mesh;
    };

    if cells.any(|cell| cell_types[cell] != first) {
        return mesh;
    }

    (TopologyType::from(first), poly_cell_points(first))
}

/// How many entries of a cell's span come before its point ids: the cell type and, for a
/// poly-cell, its point count, which only `Mixed` connectivity carries.
fn leading_entries(cell_types: &[CellType], topology_type: TopologyType, cell: usize) -> usize {
    if topology_type != TopologyType::Mixed {
        return 0;
    }

    1 + usize::from(poly_cell_points(cell_types[cell]).is_some())
}

/// The mesh points one submesh's cells use, ascending.
///
/// The submesh's coordinates are cut out of the mesh's with this list, and its connectivity is
/// renumbered against it.
pub(super) fn submesh_points<I: ConnectivityIndex>(
    cells: &[I],
    offsets: &[usize],
    cell_types: &[CellType],
    topology_type: TopologyType,
    submesh: &IndexList,
) -> Result<IndexList> {
    let mut points = Vec::new();

    for cell in submesh.iter() {
        let start = offsets[cell] + leading_entries(cell_types, topology_type, cell);
        for entry in &cells[start..offsets[cell + 1]] {
            points.push(index_as_usize(*entry)?);
        }
    }

    // sorted rather than kept in the order the cells mention them, so that the numbering a
    // submesh's connectivity is remapped to follows the mesh's own, and a submesh cut out of one
    // region of the mesh collapses to a run
    points.sort_unstable();
    points.dedup();

    Ok(collapse_indices(&points))
}

/// Where each point of the mesh sits in the submesh currently being renumbered.
///
/// A lookup array rather than a binary search of the submesh's own point list, which cost 6-28% of
/// the whole mesh write when measured on a 4M-point mesh. Built only for a submesh whose points are
/// not one run, and sized by the largest point id that reaches it rather than by the mesh, so the
/// common case allocates nothing. Never cleared between submeshes: each one writes every entry it
/// goes on to read.
#[derive(Default)]
pub(super) struct LocalPoints {
    of_point: Vec<usize>,
}

impl LocalPoints {
    fn fill(&mut self, points: &[usize]) {
        // ascending, so the last is the largest -- nothing past it is ever looked up
        let needed = points.last().map_or(0, |last| last + 1);
        if self.of_point.len() < needed {
            self.of_point.resize(needed, 0);
        }

        for (local, &point) in points.iter().enumerate() {
            self.of_point[point] = local;
        }
    }
}

/// Renumber a submesh's connectivity from the mesh's point ids to its own, in place.
pub(super) fn renumber_connectivity<I: ConnectivityIndex>(
    cells: &mut [I],
    cell_types: &[CellType],
    submesh_topology: TopologyType,
    submesh: &IndexList,
    points: &IndexList,
    local_points: &mut LocalPoints,
) -> Result<()> {
    if let IndexList::Scattered(points) = points {
        local_points.fill(points);
    }

    let mut position = 0;

    for cell in submesh.iter() {
        // the extracted array's own layout, not the mesh's: it already dropped whatever the
        // submesh's topology does not carry
        let leading = leading_entries(cell_types, submesh_topology, cell);
        let span = cell_span(cell_types, submesh_topology, cell);

        for entry in &mut cells[position + leading..position + span] {
            let point = index_as_usize(*entry)?;
            let local = match points {
                IndexList::Contiguous { start, .. } => point - start,
                IndexList::Scattered(_) => local_points.of_point[point],
            };

            *entry = I::from_index(local).ok_or(Error::Internal(
                "a point index does not fit the connectivity type",
            ))?;
        }

        position += span;
    }

    Ok(())
}

/// One connectivity entry as a position into the points, which every index in a written mesh is:
/// `validate_points_and_cells` rejected a negative or out-of-range one before this can run.
fn index_as_usize<I: ConnectivityIndex>(index: I) -> Result<usize> {
    usize::try_from(index.as_i128())
        .ok()
        .ok_or(Error::Internal("a connectivity entry is not a point index"))
}

pub(super) fn validate_points_and_cells<I: ConnectivityIndex>(
    num_coordinates: usize,
    connectivity: &[I],
    cell_types: &[CellType],
) -> Result<()> {
    if num_coordinates == 0 {
        return Err(Error::InvalidMesh {
            reason: "at least one point is required".to_string(),
        });
    }

    if !num_coordinates.is_multiple_of(3) {
        return Err(Error::InvalidMesh {
            reason: format!(
                "points must have 3 dimensions, but {num_coordinates} is not a multiple of 3"
            ),
        });
    }

    // checked before anything is built, so a mesh too large for its index type is reported
    // without assembling its connectivity first; holds even with no connectivity passed at all,
    // since the polyvertex fallback below numbers the points itself
    let num_points = num_coordinates / 3;
    if num_points as i128 - 1 > I::MAX_INDEX {
        return Err(Error::InvalidMesh {
            reason: format!(
                "the mesh has {num_points} points, but its connectivity type can only index up \
                 to {}; a wider one is needed to write it",
                I::MAX_INDEX
            ),
        });
    }

    for index in connectivity {
        let index = index.as_i128();

        if index < 0 {
            return Err(Error::InvalidMesh {
                reason: format!("connectivity index {index} is negative"),
            });
        }
        if index >= num_points as i128 {
            return Err(Error::InvalidMesh {
                reason: format!(
                    "connectivity index {index} is out of bounds, the mesh only has \
                     {num_points} points"
                ),
            });
        }
    }

    let exp_num_points: usize = cell_types.iter().map(|ct| ct.num_points()).sum();
    if exp_num_points != connectivity.len() {
        return Err(Error::InvalidMesh {
            reason: format!(
                "size of connectivity ({}) does not match the number expected from the cell types ({exp_num_points})",
                connectivity.len()
            ),
        });
    }

    Ok(())
}

/// The point count a poly-cell (`Polyvertex`/`Polyline`) must additionally specify.
pub(super) fn poly_cell_points(cell_type: CellType) -> Option<u8> {
    match cell_type {
        CellType::Vertex => Some(1),
        CellType::Edge => Some(2),
        _ => None,
    }
}

/// Prepare cells/connectivity for writing. When every cell shares one `CellType`, that type is
/// written once as a uniform `TopologyType` instead of being prepended per cell; otherwise each
/// cell gets its type (and, for a poly-cell, its point count) prepended as `Mixed` requires.
pub(super) fn prepare_cells<'c, I: ConnectivityIndex>(
    connectivity: &'c [I],
    cell_types: &[CellType],
    num_points: usize,
) -> Result<(TopologyType, Cow<'c, [I]>)> {
    // every index fits by the time this runs: the point count was already checked against
    // `I::MAX_INDEX`
    let index_fits = || Error::Internal("a point index does not fit the connectivity type");

    if cell_types.is_empty() {
        // no cells: fall back to polyvertex on the points, which ParaView requires to visualize
        // points alone
        let indices = (0..num_points)
            .map(|index| I::from_index(index).ok_or_else(index_fits))
            .collect::<Result<Vec<_>>>()?;

        return Ok((TopologyType::Polyvertex, Cow::Owned(indices)));
    }

    if let [first, rest @ ..] = cell_types
        && rest.iter().all(|cell_type| cell_type == first)
    {
        // borrowed, not copied: a uniform topology stores the caller's indices as they are, so
        // the array that reaches the backend can be the caller's own however large the mesh is
        return Ok((TopologyType::from(*first), Cow::Borrowed(connectivity)));
    }

    let mut cells_with_types = Vec::with_capacity(connectivity.len() + cell_types.len());
    let mut index = 0_usize;

    for cell_type in cell_types {
        let num_points = cell_type.num_points();
        cells_with_types.push(I::from_u8(*cell_type as u8));

        if let Some(n_points_poly) = poly_cell_points(*cell_type) {
            cells_with_types.push(I::from_u8(n_points_poly));
        }

        cells_with_types.extend_from_slice(&connectivity[index..index + num_points]);

        index += num_points;
    }

    Ok((TopologyType::Mixed, Cow::Owned(cells_with_types)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{paraview, xdmf_elements::data_item::Format};

    #[test]
    fn test_poly_cell_points() {
        assert_eq!(poly_cell_points(CellType::Vertex), Some(1));
        assert_eq!(poly_cell_points(CellType::Edge), Some(2));
        assert_eq!(poly_cell_points(CellType::Triangle), None);
        assert_eq!(poly_cell_points(CellType::Quadrilateral), None);
        assert_eq!(poly_cell_points(CellType::Tetrahedron), None);
        assert_eq!(poly_cell_points(CellType::Pyramid), None);
        assert_eq!(poly_cell_points(CellType::Wedge), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron), None);
        assert_eq!(poly_cell_points(CellType::Edge3), None);
        assert_eq!(poly_cell_points(CellType::Quadrilateral9), None);
        assert_eq!(poly_cell_points(CellType::Triangle6), None);
        assert_eq!(poly_cell_points(CellType::Quadrilateral8), None);
        assert_eq!(poly_cell_points(CellType::Tetrahedron10), None);
        assert_eq!(poly_cell_points(CellType::Pyramid13), None);
        assert_eq!(poly_cell_points(CellType::Wedge15), None);
        assert_eq!(poly_cell_points(CellType::Wedge18), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron20), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron24), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron27), None);
    }

    /// `prepare_cells`, with the connectivity taken as a `Vec` so an expected value can be
    /// spelled `vec![..]` whether the real one borrows or owns its array.
    fn prepare_cells_vec<I: ConnectivityIndex>(
        connectivity: &[I],
        cell_types: &[CellType],
        num_points: usize,
    ) -> Result<(TopologyType, Vec<I>)> {
        let (topology_type, cells) = prepare_cells(connectivity, cell_types, num_points)?;

        Ok((topology_type, cells.into_owned()))
    }

    #[test]
    fn test_prepare_cells() {
        // mixed cell types can't be written as a uniform `TopologyType`, so the type is
        // prepended to every cell, as `Mixed` topology requires
        let (topo_type, cells_prep) = prepare_cells_vec(
            &[0_u64, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
            0,
        )
        .unwrap();

        assert_eq!(topo_type, TopologyType::Mixed);
        assert_eq!(
            cells_prep,
            vec![1, 1, 0, 2, 2, 1, 2, 4, 3, 4, 5, 5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn prepare_cells_by_celltype() {
        // when every cell shares the same type, no per-cell type code is written -- the type is
        // carried once as a uniform `TopologyType`, and the connectivity is written as-is
        assert_eq!(
            prepare_cells_vec(&[5_u64], &[CellType::Vertex], 0).unwrap(),
            (TopologyType::Polyvertex, vec![5])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6], &[CellType::Edge], 0).unwrap(),
            (TopologyType::Polyline, vec![5, 6])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7], &[CellType::Triangle], 0).unwrap(),
            (TopologyType::Triangle, vec![5, 6, 7])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8], &[CellType::Quadrilateral], 0).unwrap(),
            (TopologyType::Quadrilateral, vec![5, 6, 7, 8])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8], &[CellType::Tetrahedron], 0).unwrap(),
            (TopologyType::Tetrahedron, vec![5, 6, 7, 8])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8, 9], &[CellType::Pyramid], 0).unwrap(),
            (TopologyType::Pyramid, vec![5, 6, 7, 8, 9])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8, 9, 10], &[CellType::Wedge], 0).unwrap(),
            (TopologyType::Wedge, vec![5, 6, 7, 8, 9, 10])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8, 9, 10, 11, 12], &[CellType::Hexahedron], 0)
                .unwrap(),
            (TopologyType::Hexahedron, vec![5, 6, 7, 8, 9, 10, 11, 12])
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7], &[CellType::Edge3], 0).unwrap(),
            (TopologyType::Edge3, vec![5, 6, 7])
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12, 13],
                &[CellType::Quadrilateral9],
                0
            )
            .unwrap(),
            (
                TopologyType::Quadrilateral9,
                vec![5, 6, 7, 8, 9, 10, 11, 12, 13]
            )
        );

        assert_eq!(
            prepare_cells_vec(&[5_u64, 6, 7, 8, 9, 10], &[CellType::Triangle6], 0).unwrap(),
            (TopologyType::Triangle6, vec![5, 6, 7, 8, 9, 10])
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12],
                &[CellType::Quadrilateral8],
                0
            )
            .unwrap(),
            (
                TopologyType::Quadrilateral8,
                vec![5, 6, 7, 8, 9, 10, 11, 12]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                &[CellType::Tetrahedron10],
                0
            )
            .unwrap(),
            (
                TopologyType::Tetrahedron10,
                vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                &[CellType::Pyramid13],
                0
            )
            .unwrap(),
            (
                TopologyType::Pyramid13,
                vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                &[CellType::Wedge15],
                0
            )
            .unwrap(),
            (
                TopologyType::Wedge15,
                vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[
                    5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
                ],
                &[CellType::Wedge18],
                0
            )
            .unwrap(),
            (
                TopologyType::Wedge18,
                vec![
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
                ]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[
                    5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                ],
                &[CellType::Hexahedron20],
                0
            )
            .unwrap(),
            (
                TopologyType::Hexahedron20,
                vec![
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                ]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[
                    5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                    25, 26, 27, 28
                ],
                &[CellType::Hexahedron24],
                0
            )
            .unwrap(),
            (
                TopologyType::Hexahedron24,
                vec![
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28
                ]
            )
        );

        assert_eq!(
            prepare_cells_vec(
                &[
                    5_u64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                    25, 26, 27, 28, 29, 30, 31
                ],
                &[CellType::Hexahedron27],
                0
            )
            .unwrap(),
            (
                TopologyType::Hexahedron27,
                vec![
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31
                ]
            )
        );
    }

    #[test]
    fn prepare_cells_borrows_a_uniform_connectivity() {
        // the point of the `Cow`: a uniform topology writes the caller's indices as they are, so
        // no copy of them is made however large the mesh is
        let connectivity = [0_u64, 1, 2, 1, 2, 3];
        let (_topology_type, cells) =
            prepare_cells(&connectivity, &[CellType::Triangle; 2], 4).unwrap();

        std::assert_matches!(cells, Cow::Borrowed(borrowed) if borrowed.as_ptr() == connectivity.as_ptr());
    }

    #[test]
    fn prepare_cells_mixed_when_types_differ() {
        // more than one cell of the same repeated type still can't use a uniform `TopologyType`
        // once a different type is mixed in
        let (topo_type, cells_prep) = prepare_cells_vec(
            &[0_u64, 1, 2, 3, 4, 5, 6, 7],
            &[CellType::Triangle, CellType::Triangle, CellType::Edge],
            0,
        )
        .unwrap();

        assert_eq!(topo_type, TopologyType::Mixed);
        assert_eq!(cells_prep, vec![4, 0, 1, 2, 4, 3, 4, 5, 2, 2, 6, 7]);
    }

    #[test]
    fn test_prepare_cells_no_cells() {
        let (topo_type, cells_prep) = prepare_cells_vec(&[] as &[u64], &[], 5).unwrap();

        assert_eq!(topo_type, TopologyType::Polyvertex);
        assert_eq!(cells_prep, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_validate_points_and_cells() {
        // valid input, must not return an error
        validate_points_and_cells(
            33,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        )
        .unwrap();
    }

    // a mesh whose points cannot all be indexed by the connectivity type is rejected; exercised
    // through the validation helper, which only needs the coordinate count, since a mesh this
    // big cannot be allocated in a test
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn validate_points_and_cells_too_many_points() {
        // one point more than the type can index, since the last point is index `num_points - 1`
        let too_many_u32 = usize::try_from(u32::MAX).unwrap() + 2;
        let too_many_i32 = usize::try_from(i32::MAX).unwrap() + 2;

        std::assert_matches!(
            validate_points_and_cells(3 * too_many_u32, &[] as &[u32], &[]).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("can only index up to 4294967295")
        );
        std::assert_matches!(
            validate_points_and_cells(3 * too_many_i32, &[] as &[i32], &[]).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("can only index up to 2147483647")
        );

        // one point less is exactly what each type reaches, and is still addressable
        validate_points_and_cells(3 * (too_many_u32 - 1), &[] as &[u32], &[]).unwrap();
        validate_points_and_cells(3 * (too_many_i32 - 1), &[] as &[i32], &[]).unwrap();

        // the 64-bit types hold any index a mesh can have, so this helper lets both through --
        // the lower cap ParaView puts on `u64` is checked on the connectivity values instead
        validate_points_and_cells(3 * too_many_u32, &[] as &[u64], &[]).unwrap();
        validate_points_and_cells(3 * too_many_u32, &[] as &[i64], &[]).unwrap();
    }

    // the mesh that would trip this needs over 4 billion points, which cannot be built in a
    // test, so this checks the prepared connectivity directly instead
    #[test]
    fn connectivity_above_the_paraview_uint_cap_is_rejected() {
        let too_large = Values::from(vec![u64::from(u32::MAX) + 1]);

        std::assert_matches!(
            paraview::validate(&too_large, Format::XML).unwrap_err(),
            Error::IntegerOutOfRange { value, reason }
                if value == i128::from(u32::MAX) + 1 && reason.contains("no DataStorage avoids this")
        );

        // ...and the largest index it does allow is accepted
        paraview::validate(&Values::from(vec![u64::from(u32::MAX)]), Format::XML).unwrap();
    }

    #[test]
    fn validate_points_and_cells_negative_index() {
        std::assert_matches!(
            validate_points_and_cells(9, &[0_i32, -1, 2], &[CellType::Triangle]).unwrap_err(),
            Error::InvalidMesh { reason } if reason == "connectivity index -1 is negative"
        );
    }

    #[test]
    fn validate_points_and_cells_only_points() {
        // valid input, must not return an error
        validate_points_and_cells(33, &[] as &[u64], &[]).unwrap();
    }

    #[test]
    fn validate_points_and_cells_points_empty() {
        let res = validate_points_and_cells(
            0,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("at least one point")
        );
    }

    #[test]
    fn validate_points_and_cells_points_not_3d() {
        let res = validate_points_and_cells(
            22,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("22 is not a multiple of 3")
        );
    }

    #[test]
    fn validate_points_and_cells_conn_index_out_of_bounds() {
        let res = validate_points_and_cells(
            33,
            &[0, 1, 2, 3, 4, 5, 6, 70],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("connectivity index 70")
                    && reason.contains("only has 11 points")
        );
    }

    #[test]
    fn validate_points_and_cells_conn_mismatch() {
        let res = validate_points_and_cells(
            33,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("connectivity (8)") && reason.contains("cell types (10)")
        );
    }

    // A submesh list in the shape `prepare_submeshes` takes, with `&str` names and index slices.
    fn submeshes<'a>(entries: &'a [(&'a str, &'a [usize])]) -> Vec<(&'a str, &'a [usize])> {
        entries.to_vec()
    }

    #[test]
    fn prepare_submeshes_collapses_an_ascending_run() {
        let prepared = prepare_submeshes(submeshes(&[("all", &[0, 1, 2, 3])]), 4).unwrap();

        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].name, "all");
        std::assert_matches!(
            prepared[0].cells,
            IndexList::Contiguous { start: 0, len: 4 }
        );
    }

    #[test]
    fn prepare_submeshes_collapses_a_run_that_does_not_start_at_zero() {
        let prepared =
            prepare_submeshes(submeshes(&[("low", &[0, 1]), ("high", &[2, 3, 4])]), 5).unwrap();

        std::assert_matches!(
            prepared[1].cells,
            IndexList::Contiguous { start: 2, len: 3 }
        );
    }

    #[test]
    fn prepare_submeshes_keeps_a_scattered_list_in_the_given_order() {
        // descending, so it is a permutation of a run rather than one: the order the caller gave
        // is the order the submesh's cells (and its share of every cell field) are written in
        let prepared = prepare_submeshes(submeshes(&[("all", &[2, 0, 1])]), 3).unwrap();

        std::assert_matches!(
            &prepared[0].cells,
            IndexList::Scattered(indices) if indices == &[2, 0, 1]
        );
    }

    #[test]
    fn prepare_submeshes_takes_a_range_without_materialising_its_indices() {
        let prepared = prepare_submeshes([("lower", 0..2), ("upper", 2..3)], 3).unwrap();

        std::assert_matches!(
            prepared[0].cells,
            IndexList::Contiguous { start: 0, len: 2 }
        );
        std::assert_matches!(
            prepared[1].cells,
            IndexList::Contiguous { start: 2, len: 1 }
        );
    }

    #[test]
    fn prepare_submeshes_rejects_a_range_past_the_end_of_the_mesh() {
        let res = prepare_submeshes([("all", 0..4)], 3);

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("references cell 3")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_an_empty_range() {
        let res = prepare_submeshes([("all", 0..3), ("none", 1..1)], 3);

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("submesh 'none' is empty")
        );
    }

    #[test]
    fn prepare_submeshes_moves_an_owned_scattered_list_in() {
        let prepared = prepare_submeshes([("all", vec![2, 0, 1])], 3).unwrap();

        std::assert_matches!(
            &prepared[0].cells,
            IndexList::Scattered(indices) if indices == &[2, 0, 1]
        );
    }

    #[test]
    fn prepare_submeshes_allows_overlapping_submeshes() {
        let prepared = prepare_submeshes(
            submeshes(&[("left", &[0, 1]), ("right", &[1, 2]), ("all", &[0, 1, 2])]),
            3,
        )
        .unwrap();

        assert_eq!(prepared.len(), 3);
    }

    #[test]
    fn prepare_submeshes_rejects_no_submeshes() {
        let empty: Vec<(&str, &[usize])> = Vec::new();

        std::assert_matches!(
            prepare_submeshes(empty, 3).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("at least one submesh is required")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_an_invalid_name() {
        // a space is fine now -- the name only labels the block -- so it takes a control character,
        // which XML cannot represent at all
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("has space", &[0])]), 1),
            Ok(_)
        );
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("has\u{9}tab", &[0])]), 1).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("is not valid")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_a_duplicate_name() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0]), ("part", &[1])]), 2).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("submesh name 'part' is used more than once")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_an_empty_submesh() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("empty", &[])]), 1).unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("submesh 'empty' is empty")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_an_out_of_range_cell() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0, 5])]), 3).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("submesh 'part' references cell 5")
                    && reason.contains("only has 3 cells")
        );
    }

    #[test]
    fn prepare_submeshes_rejects_a_cell_repeated_within_one_submesh() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0, 1, 0])]), 2).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("submesh 'part' contains cell 0 more than once")
        );
    }

    #[test]
    fn prepare_submeshes_spans_the_bit_sets_words() {
        // every other test here fits within `CellBitSet`'s first word, hiding word-arithmetic
        // bugs; this one straddles three words instead
        let low: Vec<usize> = (0..70).collect();
        let high: Vec<usize> = (64..150).collect();

        // 150 cells covered by the two, which overlap on 64..70
        let prepared = prepare_submeshes(submeshes(&[("low", &low), ("high", &high)]), 150);
        assert_eq!(prepared.unwrap().len(), 2);

        // cell 149 is claimed twice by the same submesh, two words in
        let repeated: Vec<usize> = high.iter().copied().chain([149]).collect();
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("low", &low), ("high", &repeated)]), 150).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("submesh 'high' contains cell 149 more than once")
        );

        // and cell 130 is in none of them, which the coverage pass has to find past the boundary
        let gapped: Vec<usize> = high.iter().copied().filter(|index| *index != 130).collect();
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("low", &low), ("high", &gapped)]), 150).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("1 of 150 cells belong to no submesh: 130")
        );
    }

    // a `usize` only exceeds `i32::MAX` where it is wider than 32 bits
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn cell_indices_are_written_at_the_narrowest_type_that_holds_them() {
        let small = index_values(&[0, 7, 12]).unwrap();
        std::assert_matches!(&small, Values::I32(indices) if **indices == [0, 7, 12]);

        // one index past what an `i32` holds widens the whole array, since the type is the
        // `DataItem`'s and not each value's
        let large = index_values(&[1, usize::try_from(i32::MAX).unwrap() + 1]).unwrap();
        std::assert_matches!(
            &large,
            Values::I64(indices) if **indices == [1, i64::from(i32::MAX) + 1]
        );
    }

    #[test]
    fn prepare_submeshes_rejects_cells_in_no_submesh() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0, 2])]), 4).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("2 of 4 cells belong to no submesh: 1, 3")
        );
    }

    #[test]
    fn prepare_submeshes_truncates_a_long_list_of_uncovered_cells() {
        std::assert_matches!(
            prepare_submeshes(submeshes(&[("part", &[0])]), 20).unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("19 of 20 cells belong to no submesh")
                    && reason.contains("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... (9 more)")
        );
    }

    #[test]
    fn cell_offsets_of_mixed_cells() {
        // a triangle takes 1 + 3 entries, an edge 1 + 1 (its point count) + 2, a vertex 1 + 1 + 1
        let offsets = cell_offsets(
            &[CellType::Triangle, CellType::Edge, CellType::Vertex],
            TopologyType::Mixed,
            3,
        );

        assert_eq!(offsets, vec![0, 4, 8, 11]);
    }

    #[test]
    fn cell_offsets_of_uniform_cells() {
        // every cell shares one type, so no per-cell metadata is stored: each quad takes exactly
        // its 4 points
        let offsets = cell_offsets(
            &[CellType::Quadrilateral; 3],
            TopologyType::Quadrilateral,
            3,
        );

        assert_eq!(offsets, vec![0, 4, 8, 12]);
    }

    #[test]
    fn cell_offsets_of_a_point_mesh() {
        // no cell types means the polyvertex fallback, one index per point
        assert_eq!(
            cell_offsets(&[], TopologyType::Polyvertex, 3),
            vec![0, 1, 2, 3]
        );
    }

    // three quads as `prepare_cells` lays them out: the cell type code (5, XDMF's own code for
    // `Quadrilateral`, not VTK's 9) then its four points -- the code collides with a point index
    // here (points are 0..11), which is why offsets say where each cell starts
    const QUAD_CELLS: [u32; 15] = [5, 0, 1, 2, 3, 5, 4, 5, 6, 7, 5, 8, 9, 10, 11];
    const QUAD_OFFSETS: [usize; 4] = [0, 5, 10, 15];

    /// The three cell types `QUAD_CELLS` is laid out for.
    const QUAD_TYPES: [CellType; 3] = [CellType::Quadrilateral; 3];

    #[test]
    fn extract_connectivity_takes_a_contiguous_submesh() {
        let extracted = extract_connectivity(
            &QUAD_CELLS,
            &QUAD_OFFSETS,
            &QUAD_TYPES,
            TopologyType::Mixed,
            TopologyType::Mixed,
            &IndexList::Contiguous { start: 1, len: 2 },
        );

        assert_eq!(extracted, &[5, 4, 5, 6, 7, 5, 8, 9, 10, 11]);
    }

    #[test]
    fn extract_connectivity_gathers_a_scattered_submesh() {
        let extracted = extract_connectivity(
            &QUAD_CELLS,
            &QUAD_OFFSETS,
            &QUAD_TYPES,
            TopologyType::Mixed,
            TopologyType::Mixed,
            &IndexList::Scattered(vec![2, 0]),
        );

        // gathered in the order the submesh names its cells, not in ascending order
        assert_eq!(extracted, &[5, 8, 9, 10, 11, 5, 0, 1, 2, 3]);
    }

    /// A submesh whose cells share one type is written uniformly: the per-cell type code a
    /// `Mixed` mesh carries is dropped, since its `<Topology>` states the type once instead.
    #[test]
    fn extract_connectivity_drops_the_type_codes_of_a_uniform_submesh() {
        let contiguous = extract_connectivity(
            &QUAD_CELLS,
            &QUAD_OFFSETS,
            &QUAD_TYPES,
            TopologyType::Mixed,
            TopologyType::Quadrilateral,
            &IndexList::Contiguous { start: 1, len: 2 },
        );

        assert_eq!(contiguous, &[4, 5, 6, 7, 8, 9, 10, 11]);

        let scattered = extract_connectivity(
            &QUAD_CELLS,
            &QUAD_OFFSETS,
            &QUAD_TYPES,
            TopologyType::Mixed,
            TopologyType::Quadrilateral,
            &IndexList::Scattered(vec![2, 0]),
        );

        assert_eq!(scattered, &[8, 9, 10, 11, 0, 1, 2, 3]);
    }

    #[test]
    fn submesh_topology_is_the_type_its_own_cells_share() {
        let cell_types = [
            CellType::Hexahedron,
            CellType::Quadrilateral,
            CellType::Quadrilateral,
        ];

        // the block of quads is uniform even though the mesh it is cut out of is not
        assert_eq!(
            submesh_topology(
                &cell_types,
                TopologyType::Mixed,
                None,
                &IndexList::Contiguous { start: 1, len: 2 }
            ),
            (TopologyType::Quadrilateral, None)
        );

        // one that spans both types stays Mixed, as does one cell of each
        assert_eq!(
            submesh_topology(
                &cell_types,
                TopologyType::Mixed,
                None,
                &IndexList::Contiguous { start: 0, len: 3 }
            ),
            (TopologyType::Mixed, None)
        );
        assert_eq!(
            submesh_topology(
                &cell_types,
                TopologyType::Mixed,
                None,
                &IndexList::Scattered(vec![2, 0])
            ),
            (TopologyType::Mixed, None)
        );

        // a poly-cell block carries its node count, as the mesh's own topology would
        assert_eq!(
            submesh_topology(
                &[CellType::Hexahedron, CellType::Edge],
                TopologyType::Mixed,
                None,
                &IndexList::Contiguous { start: 1, len: 1 }
            ),
            (TopologyType::Polyline, Some(2))
        );

        // and a mesh that is uniform to begin with hands its own topology to every submesh
        assert_eq!(
            submesh_topology(
                &[CellType::Edge; 2],
                TopologyType::Polyline,
                Some(2),
                &IndexList::Contiguous { start: 0, len: 1 }
            ),
            (TopologyType::Polyline, Some(2))
        );
    }
}

//! Where each rank's share of the mesh sits in the whole, and collecting those shares on rank 0.
//!
//! Every function here that talks to the other ranks derives its verdict from data every rank
//! has gathered identically, so they all return the same `Ok` or the same `Err` without a further
//! [`agree`](super::sync::agree).

use mpi::{
    datatype::Equivalence,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Destination, Source},
};

use crate::{CellType, Error, Result, Values};

/// One rank's share of an array: where it starts in the whole, and how many entries it holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Share {
    offset: usize,
    len: usize,
}

impl Share {
    /// Where this share starts in the whole array.
    pub(super) const fn offset(self) -> usize {
        self.offset
    }

    /// How many entries this share holds.
    pub(super) const fn len(self) -> usize {
        self.len
    }

    /// The same share counted in values rather than entities, for a field of `components` values
    /// per entity. Cannot overflow: the whole array's value count was checked to fit first.
    const fn times(self, components: usize) -> Self {
        Self {
            offset: self.offset * components,
            len: self.len * components,
        }
    }
}

/// How a mesh is spread over the ranks, fixed when it is written and reused by every time step.
#[derive(Debug)]
pub(super) struct Layout {
    /// Every rank's share of the points, in rank order, placed at its first global id.
    points: Vec<Share>,
    num_points: usize,
    /// Every rank's share of the cells, in rank order, concatenated in that order.
    cells: Vec<Share>,
    num_cells: usize,
    rank: usize,
}

impl Layout {
    pub(super) fn points(&self) -> &[Share] {
        &self.points
    }

    pub(super) const fn num_points(&self) -> usize {
        self.num_points
    }

    pub(super) fn cells(&self) -> &[Share] {
        &self.cells
    }

    pub(super) const fn num_cells(&self) -> usize {
        self.num_cells
    }

    pub(super) fn local_points(&self) -> Share {
        self.points.get(self.rank).copied().unwrap_or_default()
    }

    pub(super) fn local_cells(&self) -> Share {
        self.cells.get(self.rank).copied().unwrap_or_default()
    }
}

/// Every rank's [`Share`] of the points, checked to tile `0..num_points` exactly.
///
/// Each rank's own `owned_global_ids` must be one ascending, contiguous block, which is what a
/// hyperslab write can place; every rank's block is gathered, so that together they can be
/// checked to leave no id owned by nobody and none owned twice. The blocks may come in any order
/// of ranks, as the ids say where each goes.
pub(super) fn point_shares(
    comm: &SimpleCommunicator,
    owned_global_ids: &[u64],
) -> Result<(Vec<Share>, usize)> {
    let contiguous = owned_global_ids
        .windows(2)
        .all(|pair| pair[0].checked_add(1) == Some(pair[1]));

    let local = [
        u64::from(contiguous),
        owned_global_ids.first().copied().unwrap_or(0),
        owned_global_ids.len() as u64,
    ];
    let all = all_gather(comm, &local);

    // every rank holds the same `all` from here on, so every rank reaches the same verdict

    if let Some(rank) = all.iter().position(|[contiguous, ..]| *contiguous == 0) {
        return Err(Error::InvalidMesh {
            reason: format!(
                "owned_global_ids must be one ascending, contiguous block on every rank, but is \
                 not on rank {rank}"
            ),
        });
    }

    let shares = all
        .iter()
        .map(|&[_, start, len]| {
            Ok(Share {
                offset: to_usize(start)?,
                len: to_usize(len)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut blocks: Vec<(usize, Share)> = shares
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, share)| share.len > 0)
        .collect();
    blocks.sort_unstable_by_key(|(_, share)| share.offset);

    let mut next = 0_usize;
    for (rank, share) in blocks {
        if share.offset != next {
            let problem = if share.offset < next {
                "overlaps the block before it, so some ids are owned by more than one rank"
            } else {
                "leaves a gap after the block before it, so some ids are owned by no rank"
            };
            return Err(Error::InvalidMesh {
                reason: format!(
                    "owned_global_ids across all ranks must cover 0..N exactly once, but rank \
                     {rank}'s block starting at id {} {problem}",
                    share.offset
                ),
            });
        }
        next = next.checked_add(share.len).ok_or_else(too_many)?;
    }

    if next == 0 {
        return Err(Error::InvalidMesh {
            reason: "the mesh has no points: no rank owns any".to_string(),
        });
    }

    Ok((shares, next))
}

/// The topology every rank's cells are written as, agreed across all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CellKind {
    /// Every cell on every rank is of this one type.
    Uniform(CellType),
    /// Cells of more than one type, written as a `Mixed` topology.
    Mixed,
}

/// Every rank's [`Share`] of the cells, concatenated in rank order, and what kind of topology all
/// of them together make. A rank may own no cells at all, but the mesh as a whole may not.
pub(super) fn cell_shares(
    comm: &SimpleCommunicator,
    cell_types: &[CellType],
) -> Result<(Vec<Share>, usize, CellKind)> {
    // `u8::MAX`/`0` for a rank without cells, so it agrees with whatever the others have
    let codes = cell_types.iter().map(|cell_type| *cell_type as u8);
    let local = [
        cell_types.len() as u64,
        u64::from(codes.clone().min().unwrap_or(u8::MAX)),
        u64::from(codes.max().unwrap_or(0)),
    ];
    let all = all_gather(comm, &local);

    let mut shares = Vec::with_capacity(all.len());
    let mut next = 0_usize;
    for &[len, ..] in &all {
        let len = to_usize(len)?;
        shares.push(Share { offset: next, len });
        next = next.checked_add(len).ok_or_else(too_many)?;
    }

    if next == 0 {
        return Err(Error::InvalidMesh {
            reason: "the mesh has no cells: no rank passed any".to_string(),
        });
    }

    let min_code = all.iter().map(|[_, min, _]| *min).min().unwrap_or(0);
    let max_code = all.iter().map(|[_, _, max]| *max).max().unwrap_or(0);

    let kind = match u8::try_from(min_code).ok().and_then(CellType::from_code) {
        Some(cell_type) if min_code == max_code => CellKind::Uniform(cell_type),
        _ => CellKind::Mixed,
    };

    Ok((shares, next, kind))
}

/// Every rank's share of an array whose length each rank only knows of its own, concatenated in
/// rank order, and the whole array's length.
pub(super) fn concatenated(
    comm: &SimpleCommunicator,
    local_len: usize,
) -> Result<(Vec<Share>, usize)> {
    let all = all_gather(comm, &[local_len as u64]);

    let mut shares = Vec::with_capacity(all.len());
    let mut next = 0_usize;
    for &[len] in &all {
        let len = to_usize(len)?;
        shares.push(Share { offset: next, len });
        next = next.checked_add(len).ok_or_else(too_many)?;
    }

    Ok((shares, next))
}

/// The layout of a mesh whose point and cell shares are known.
pub(super) fn layout(
    comm: &SimpleCommunicator,
    (points, num_points): (Vec<Share>, usize),
    (cells, num_cells): (Vec<Share>, usize),
) -> Layout {
    Layout {
        points,
        num_points,
        cells,
        num_cells,
        rank: usize::try_from(comm.rank()).unwrap_or(0),
    }
}

/// `shares` counted in values, for a field of `components` values per entity, with the whole
/// field's length -- or an error, on every rank alike, if that length does not fit a `usize` or
/// one rank's share is more than one MPI message can carry.
pub(super) fn value_shares(
    shares: &[Share],
    num_entities: usize,
    components: usize,
) -> Result<(Vec<Share>, usize)> {
    let total = num_entities.checked_mul(components).ok_or_else(too_many)?;

    let shares: Vec<Share> = shares.iter().map(|share| share.times(components)).collect();

    if shares.iter().any(|share| i32::try_from(share.len).is_err()) {
        return Err(Error::InvalidData {
            reason: format!(
                "one rank's share is more than the {} values a single MPI message carries, which \
                 gathering it to rank 0 needs; the Hdf5SingleFile storage writes every rank's \
                 share in place instead",
                i32::MAX
            ),
        });
    }

    Ok((shares, total))
}

/// Collect every rank's `local` share on rank 0, each at its own place in an array of `total`
/// values; `None` on every other rank. Every rank calls this together, with the same `shares`.
pub(super) fn gather(
    comm: &SimpleCommunicator,
    local: &Values<'_>,
    shares: &[Share],
    total: usize,
) -> Option<Values<'static>> {
    match local {
        Values::F64(v) => gather_slice(comm, v, shares, total).map(Values::from),
        Values::F32(v) => gather_slice(comm, v, shares, total).map(Values::from),
        Values::I64(v) => gather_slice(comm, v, shares, total).map(Values::from),
        Values::I32(v) => gather_slice(comm, v, shares, total).map(Values::from),
        Values::U64(v) => gather_slice(comm, v, shares, total).map(Values::from),
        Values::U32(v) => gather_slice(comm, v, shares, total).map(Values::from),
    }
}

/// [`gather`] for one element type.
///
/// Point-to-point rather than `MPI_Gatherv`, whose displacements are `int`s: a mesh of more than
/// `i32::MAX` values in all is gathered fine, as long as no single rank's share is that large.
/// The communicator is the writer's own duplicate, so nothing of the caller's can match these
/// messages.
pub(super) fn gather_slice<T: Equivalence + Copy + Default>(
    comm: &SimpleCommunicator,
    local: &[T],
    shares: &[Share],
    total: usize,
) -> Option<Vec<T>> {
    if comm.rank() != 0 {
        if !local.is_empty() {
            comm.process_at_rank(0).send(local);
        }
        return None;
    }

    let mut all = vec![T::default(); total];

    for (rank, share) in shares.iter().enumerate() {
        let Some(target) = all.get_mut(share.offset..share.offset + share.len) else {
            continue;
        };

        if rank == 0 {
            target.copy_from_slice(local.get(..share.len).unwrap_or_default());
        } else if !target.is_empty() {
            let source = i32::try_from(rank).unwrap_or(i32::MAX);
            comm.process_at_rank(source).receive_into(target);
        }
    }

    Some(all)
}

/// Every rank's `local` numbers, in rank order.
fn all_gather<const N: usize>(comm: &SimpleCommunicator, local: &[u64; N]) -> Vec<[u64; N]> {
    let size = usize::try_from(comm.size()).unwrap_or(1);
    let mut flat = vec![0_u64; size * N];
    comm.all_gather_into(&local[..], &mut flat[..]);

    flat.as_chunks::<N>().0.to_vec()
}

fn to_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_too_large| too_many())
}

fn too_many() -> Error {
    Error::InvalidMesh {
        reason: "the mesh is larger than a usize can count".to_string(),
    }
}

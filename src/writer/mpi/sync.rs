//! Keeping every rank on the same sequence of collective calls, whatever fails where.
//!
//! MPI matches collectives by the order they are called in, so a rank returning early on an error
//! of its own -- or a caller passing different arguments on different ranks -- leaves the others
//! waiting in a collective it never reaches, which hangs rather than fails. Every point at which
//! one rank's outcome can differ from another's therefore goes through [`agree`] (or
//! [`converge`], at the end of a time step), which reduces two things across every rank before
//! any of them acts on either:
//!
//! - whether any rank failed, so that all of them return an error together, and
//! - a hash of what every rank is doing there (the call, the time, the field's name and shape),
//!   so that ranks calling with different arguments, or in a different order, are told so
//!   instead of pairing up mismatched collectives.

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    path::PathBuf,
};

use mpi::{
    collective::SystemOperation,
    topology::SimpleCommunicator,
    traits::{Communicator, CommunicatorCollectives, Root},
};

use crate::{Error, Result};

/// One point in the sequence of collectives, hashed together with everything every rank must
/// agree on there.
///
/// [`DefaultHasher::new`] hashes the same input to the same value in every process running the
/// same build, which is all this needs -- the ranks of one job run one binary.
pub(super) fn phase(parts: impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    parts.hash(&mut hasher);
    hasher.finish()
}

/// What a rank reports into the reduction when it did not fail. Every real rank is smaller.
const NO_FAILURE: u64 = u64::MAX;

/// What every rank learns from one reduction -- the same on every rank, so whatever each does
/// next is the same collective.
enum Outcome {
    /// Every rank is at the same phase, and none of them failed.
    Agreed,
    /// Every rank is at the same phase, and at least this one failed: the lowest such rank.
    Failed(i32),
    /// The ranks are at different phases.
    OutOfStep,
}

/// The one collective [`agree`] and [`converge`] are built on: a single `MPI_Allreduce` of three
/// numbers, with `min` -- the first failed rank, and the phase both ways round, so its minimum and
/// maximum come out of one reduction.
fn reduce(comm: &SimpleCommunicator, phase: u64, failed: bool) -> Outcome {
    let rank = u64::try_from(comm.rank()).unwrap_or(NO_FAILURE - 1);
    let local = [if failed { rank } else { NO_FAILURE }, phase, !phase];
    let mut global = [0_u64; 3];
    comm.all_reduce_into(&local[..], &mut global[..], SystemOperation::min());

    let [first_failed, min_phase, not_max_phase] = global;

    if min_phase != !not_max_phase {
        return Outcome::OutOfStep;
    }

    if first_failed == NO_FAILURE {
        return Outcome::Agreed;
    }

    Outcome::Failed(i32::try_from(first_failed).unwrap_or(i32::MAX))
}

/// Make every rank leave this point together: with `local`'s value on every rank if no rank
/// failed and all are at the same `phase`, with an error on every rank otherwise.
///
/// A rank that failed returns its own error. The others return the first failed rank's, rebuilt
/// as the same variant with the rank named in it, or -- for ranks out of step -- a description
/// of that.
pub(super) fn agree<T>(comm: &SimpleCommunicator, phase: u64, local: Result<T>) -> Result<T> {
    match reduce(comm, phase, local.is_err()) {
        Outcome::Agreed => local,
        Outcome::OutOfStep => Err(local.err().unwrap_or_else(out_of_step)),
        Outcome::Failed(rank) => {
            let shared = share_error(comm, rank, local.as_ref().err());
            Err(local.err().unwrap_or(shared))
        }
    }
}

/// The end of a time step: wait until every rank has arrived here, then report the first rank
/// whose step failed, if any.
///
/// Unlike [`agree`], ranks that are out of step are waited for rather than failed: a rank whose
/// step ended early (its closure returned an error before writing every field) is here while the
/// others are still in a field's [`agree`]. Each reduction this rank takes part in fails one of
/// their fields as out of step, until their closures return too and every rank is here.
pub(super) fn converge(comm: &SimpleCommunicator, phase: u64, failed: bool) -> Option<i32> {
    loop {
        match reduce(comm, phase, failed) {
            Outcome::Agreed => return None,
            Outcome::Failed(rank) => return Some(rank),
            Outcome::OutOfStep => {}
        }
    }
}

/// Rank 0's `flag`, on every rank; every rank calls this together.
pub(super) fn shared_flag(comm: &SimpleCommunicator, flag: bool) -> bool {
    let mut shared = u8::from(flag);
    comm.process_at_rank(0).broadcast_into(&mut shared);
    shared != 0
}

/// The error every rank reports when the ranks are not making the same calls.
fn out_of_step() -> Error {
    Error::InvalidData {
        reason: "the ranks are out of step: every rank must make the same calls, in the same \
                 order, with the same time and the same field names, attribute types and \
                 number types"
            .to_string(),
    }
}

/// Send `failed_rank`'s error to every rank, which all call this together, and rebuild it on
/// each as the same variant.
///
/// `local` is this rank's own error, which `failed_rank` must have.
fn share_error(comm: &SimpleCommunicator, failed_rank: i32, local: Option<&Error>) -> Error {
    let root = comm.process_at_rank(failed_rank);

    let mut encoded = if comm.rank() == failed_rank {
        local.map(SharedError::encode).unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut len = u64::try_from(encoded.len()).unwrap_or(0);
    root.broadcast_into(&mut len);
    encoded.resize(usize::try_from(len).unwrap_or(0), 0);
    root.broadcast_into(&mut encoded[..]);

    SharedError::decode(&encoded).rebuild(failed_rank)
}

/// An [`Error`] reduced to what can cross a process boundary: the variant, and the variant's
/// strings and number.
#[derive(Debug, Default)]
struct SharedError {
    kind: u8,
    value: i128,
    first: String,
    second: String,
}

impl SharedError {
    fn encode(error: &Error) -> Vec<u8> {
        let (kind, value, first, second) = match error {
            Error::Io { path, source, .. } => {
                (0, 0, path.display().to_string(), source.to_string())
            }
            Error::Hdf5 { source, .. } => (1, 0, String::new(), source.to_string()),
            Error::InvalidFileName { path, reason } => {
                (2, 0, path.display().to_string(), reason.clone())
            }
            Error::InvalidConfiguration { reason } => (3, 0, String::new(), reason.clone()),
            Error::InvalidMesh { reason } => (4, 0, String::new(), reason.clone()),
            Error::InvalidTimeStep { time, reason } => (5, 0, time.clone(), reason.clone()),
            Error::InvalidData { reason } => (6, 0, String::new(), reason.clone()),
            Error::IntegerOutOfRange { value, reason } => {
                (7, *value, String::new(), reason.clone())
            }
            Error::Internal(message) => (8, 0, String::new(), (*message).to_string()),
            Error::InvalidDocument { reason } => (9, 0, String::new(), reason.clone()),
            Error::Unsupported { reason } => (10, 0, String::new(), reason.clone()),
            Error::NumberTypeMismatch { reason } => (11, 0, String::new(), reason.clone()),
        };

        let mut encoded = Vec::with_capacity(1 + 16 + 8 + first.len() + second.len());
        encoded.push(kind);
        encoded.extend_from_slice(&value.to_le_bytes());
        encoded.extend_from_slice(&(first.len() as u64).to_le_bytes());
        encoded.extend_from_slice(first.as_bytes());
        encoded.extend_from_slice(second.as_bytes());
        encoded
    }

    /// The inverse of [`encode`](Self::encode). Malformed input -- which only a bug here could
    /// produce -- decodes to an empty `Internal` error rather than failing a second time.
    fn decode(encoded: &[u8]) -> Self {
        let decoded = (|| {
            let (&kind, rest) = encoded.split_first()?;
            let (value, rest) = rest.split_first_chunk::<16>()?;
            let (first_len, rest) = rest.split_first_chunk::<8>()?;
            let first_len = usize::try_from(u64::from_le_bytes(*first_len)).ok()?;
            let (first, second) = rest.split_at_checked(first_len)?;

            Some(Self {
                kind,
                value: i128::from_le_bytes(*value),
                first: String::from_utf8_lossy(first).into_owned(),
                second: String::from_utf8_lossy(second).into_owned(),
            })
        })();

        decoded.unwrap_or(Self {
            kind: 8,
            ..Self::default()
        })
    }

    /// The error as the rank that did not fail reports it: the same variant, saying which rank
    /// it happened on.
    fn rebuild(self, rank: i32) -> Error {
        let Self {
            kind,
            value,
            first,
            second,
        } = self;
        let reason = format!("on rank {rank}: {second}");

        match kind {
            0 => Error::Io {
                operation: "an I/O operation on another rank",
                path: PathBuf::from(first),
                source: std::io::Error::other(reason),
            },
            1 => Error::Hdf5 {
                operation: "an HDF5 call on another rank",
                source: hdf5::Error::from(reason),
            },
            2 => Error::InvalidFileName {
                path: PathBuf::from(first),
                reason,
            },
            3 => Error::InvalidConfiguration { reason },
            4 => Error::InvalidMesh { reason },
            5 => Error::InvalidTimeStep {
                time: first,
                reason,
            },
            6 => Error::InvalidData { reason },
            7 => Error::IntegerOutOfRange { value, reason },
            9 => Error::InvalidDocument { reason },
            10 => Error::Unsupported { reason },
            11 => Error::NumberTypeMismatch { reason },
            _ => Error::Internal("an internal invariant was violated on another rank"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn round_trip(error: &Error) -> Error {
        SharedError::decode(&SharedError::encode(error)).rebuild(3)
    }

    #[test]
    fn a_shared_error_keeps_its_variant_and_names_the_rank() {
        std::assert_matches!(
            round_trip(&Error::InvalidTimeStep {
                time: "0.5".to_string(),
                reason: "already written".to_string(),
            }),
            Error::InvalidTimeStep { time, reason }
                if time == "0.5" && reason == "on rank 3: already written"
        );

        std::assert_matches!(
            round_trip(&Error::IntegerOutOfRange {
                value: -(1_i128 << 70),
                reason: "too large".to_string(),
            }),
            Error::IntegerOutOfRange { value, reason }
                if value == -(1_i128 << 70) && reason == "on rank 3: too large"
        );

        std::assert_matches!(
            round_trip(&Error::InvalidFileName {
                path: PathBuf::from("out/a?b.xdmf2"),
                reason: "bad".to_string(),
            }),
            Error::InvalidFileName { path, .. } if path == Path::new("out/a?b.xdmf2")
        );

        std::assert_matches!(
            round_trip(&Error::Io {
                operation: "writing",
                path: PathBuf::from("out.xdmf2"),
                source: std::io::Error::other("disk full"),
            }),
            Error::Io { path, source, .. }
                if path == Path::new("out.xdmf2") && source.to_string().contains("disk full")
        );
    }

    #[test]
    fn a_malformed_shared_error_decodes_to_internal() {
        std::assert_matches!(
            SharedError::decode(&[4, 1, 2]).rebuild(0),
            Error::Internal(message) if message.contains("another rank")
        );
    }
}

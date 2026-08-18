//! Everything this crate restricts only because of `ParaView`.
//!
//! None of the limits enforced here come from XDMF. Each one is a defect of `ParaView`'s legacy
//! Xdmf2 reader that would otherwise show a value the file does not contain -- silently, without
//! any reader error -- so the value is refused rather than written. A file that broke them would
//! be valid XDMF and would read back correctly elsewhere.
//!
//! [`validate`] is the single entry point, and it is called by [`TimeSeriesWriter`] and
//! [`TimeStep`], never by a [`DataWriter`] backend. It only ever *rejects*: the backends write
//! each element type at its natural width, so nothing here rewrites, narrows or casts the caller's
//! data on its way out, and what lands in the file is the type that was passed in. A path that
//! does not care about `ParaView` -- writing data only to read it back with this crate, restart
//! files and the like -- is therefore this one call away, with nothing else to undo.
//!
//! Each limit below was measured against `ParaView` 5.13 and 6.1; see `examples/paraview_smoke.rs`
//! and `tests/paraview_smoke/`.
//!
//! [`TimeSeriesWriter`]: crate::TimeSeriesWriter
//! [`TimeStep`]: crate::TimeStep
//! [`DataWriter`]: crate::DataWriter

use crate::{Error, Result, Values, xdmf_elements::data_item::Format};

/// Largest magnitude an `i64` may have in the ascii storage methods.
///
/// `ParaView` parses their integers through a `double`, whose mantissa holds 53 bits, so a value
/// past this comes back rounded -- and `i64::MAX` comes back as `i64::MIN`, sign flipped, which
/// looks like data rather than like a failure. The digits written to the file are exact either
/// way, so this is the reader's limit and not the writer's.
///
/// Values above this that happen to be even do survive the `double`. The check stays on the range
/// that is exact for *every* value rather than the one that is exact if you are lucky.
const MAX_EXACT_ASCII_INT: u64 = 1 << 53;

/// Reject data `ParaView` would read back as different numbers than it was given.
///
/// `format` selects the restrictions that apply on top of the one every format shares, and covers
/// them exactly: the ascii storages ([`Format::XML`]) are read through a `double`,
/// [`Format::Binary`] cannot carry 64-bit integers at all, and [`Format::HDF`] adds nothing.
pub(crate) fn validate(data: &Values<'_>, format: Format) -> Result<()> {
    validate_uint_range(data)?;

    match format {
        Format::XML => validate_ascii_int_range(data),
        Format::Binary => reject_64_bit_integers(data),
        Format::HDF => Ok(()),
    }
}

/// The restriction every format shares: `UInt` data is decoded into a 32-bit array whatever
/// `Precision` the light data declares, so a `u64` above `u32::MAX` comes back truncated (ascii)
/// or clamped (HDF5).
///
/// Values *within* that range are read back exactly at either width, so this caps the value and
/// says nothing about how wide it is stored -- `u64` is written as 8 bytes like any other type.
fn validate_uint_range(data: &Values<'_>) -> Result<()> {
    let Values::U64(values) = data else {
        return Ok(());
    };

    for &value in values.iter() {
        if value > u64::from(u32::MAX) {
            return Err(Error::IntegerOutOfRange {
                value: i128::from(value),
                reason: "u64 data must fit in 32 bits, since ParaView decodes UInt data into a \
                         32-bit array whatever precision is declared; no DataStorage avoids this, \
                         use i64 for integers beyond 32 bits"
                    .to_string(),
            });
        }
    }
    Ok(())
}

fn validate_ascii_int_range(data: &Values<'_>) -> Result<()> {
    let Values::I64(values) = data else {
        // u64 is already capped far below this by `validate_uint_range`, and no other element type
        // reaches 53 bits of mantissa in the first place
        return Ok(());
    };

    for &value in values.iter() {
        if value.unsigned_abs() > MAX_EXACT_ASCII_INT {
            return Err(Error::IntegerOutOfRange {
                value: i128::from(value),
                reason: format!(
                    "the ascii storage methods are read back through a double, so an i64 beyond \
                     +/-{MAX_EXACT_ASCII_INT} is shown rounded; the Hdf5SingleFile and \
                     Hdf5MultipleFiles storages keep the full width"
                ),
            });
        }
    }
    Ok(())
}

/// `Format="Binary"` is the one format whose 64-bit integers `ParaView` cannot read at all.
///
/// It reads the raw bytes at the wrong stride rather than misinterpreting a value, so the damage
/// does not depend on how large the numbers are: an `i64`/`u64` attribute comes back with every
/// second value replaced by zero, and 64-bit connectivity makes the reader give up outright
/// (`vtkXdmfReader: Failed to read data`). Both were reproduced on 5.13 and 6.1.
///
/// So this rejects the *type*, whatever it holds, rather than a range. Narrowing to 32 bits on the
/// way out would also load, but it would put a different type in the file than the caller passed,
/// silently -- the caller narrowing deliberately is the same file with none of the surprise.
fn reject_64_bit_integers(data: &Values<'_>) -> Result<()> {
    let element_type = match data {
        Values::I64(_) => "i64",
        Values::U64(_) => "u64",
        // spelled out rather than left to a wildcard, so a further element type has to decide
        // whether Binary can carry it instead of silently being let through
        Values::F64(_) | Values::F32(_) | Values::I32(_) | Values::U32(_) => return Ok(()),
    };

    Err(Error::InvalidData {
        reason: format!(
            "the Binary storage cannot hold {element_type} data, since ParaView reads 64-bit \
             integers in Format=\"Binary\" at the wrong stride and gets neither the values nor \
             the mesh back; pass the data as i32/u32, or use another DataStorage"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uint_range_is_capped_for_every_format() {
        for format in [Format::XML, Format::HDF] {
            std::assert_matches!(
                validate(&vec![u64::from(u32::MAX) + 1].into(), format).unwrap_err(),
                Error::IntegerOutOfRange { value, reason }
                    if value == i128::from(u32::MAX) + 1
                        && reason.contains("no DataStorage avoids this"),
                "{format:?} must reject a u64 above u32::MAX"
            );

            // the cap itself is still accepted, and is written at the full 8 bytes -- the cap is
            // on the value, not on the width
            validate(&vec![u64::from(u32::MAX)].into(), format).unwrap();
        }

        // Binary refuses the type before the range is ever reached
        std::assert_matches!(
            validate(&vec![0_u64].into(), Format::Binary).unwrap_err(),
            Error::InvalidData { reason } if reason.contains("cannot hold u64 data")
        );
    }

    #[test]
    fn ascii_int_range_boundary() {
        // 2^53 itself is the last integer a double holds exactly, so it is still accepted
        let max = i64::try_from(MAX_EXACT_ASCII_INT).unwrap();
        validate(&vec![max, -max, 0, 1].into(), Format::XML).unwrap();

        for out_of_range in [max + 1, -max - 1, i64::MAX, i64::MIN] {
            std::assert_matches!(
                validate(&vec![0_i64, out_of_range].into(), Format::XML).unwrap_err(),
                Error::IntegerOutOfRange { value, reason }
                    if value == i128::from(out_of_range)
                        && reason.contains("read back through a double"),
                "an i64 of {out_of_range} must be rejected for the ascii storages"
            );

            // ...which is a limit of the ascii storages only, and the error says so
            validate(&vec![0_i64, out_of_range].into(), Format::HDF).unwrap();
        }

        // every other element type is unaffected -- u64 is capped far below this already, and the
        // rest cannot reach 53 bits
        validate(&vec![u64::from(u32::MAX)].into(), Format::XML).unwrap();
        validate(&vec![i32::MIN, i32::MAX].into(), Format::XML).unwrap();
        validate(&vec![f64::MAX, f64::MIN].into(), Format::XML).unwrap();
    }

    #[test]
    fn binary_rejects_64_bit_integers_whatever_they_hold() {
        // rejected by type, so even values that would fit in 32 bits do not get through -- the
        // file would otherwise hold a narrower type than the caller passed
        for (data, expected) in [
            (Values::from(vec![0_i64, 1]), "cannot hold i64 data"),
            (Values::from(vec![0_u64, 1]), "cannot hold u64 data"),
        ] {
            std::assert_matches!(
                validate(&data, Format::Binary).unwrap_err(),
                Error::InvalidData { reason }
                    if reason.contains(expected) && reason.contains("use another DataStorage"),
                "Binary must reject {data:?}"
            );

            // ...and the storages the error points at do take it
            validate(&data, Format::HDF).unwrap();
        }

        // the 32-bit types are what Binary is asking for, and go through
        validate(&vec![i32::MIN, i32::MAX].into(), Format::Binary).unwrap();
        validate(&vec![0_u32, u32::MAX].into(), Format::Binary).unwrap();
        validate(&vec![1.5_f64].into(), Format::Binary).unwrap();
        validate(&vec![1.5_f32].into(), Format::Binary).unwrap();
    }
}

//! A wrapper type giving one interface over the different element types.

use std::borrow::Cow;

use crate::{
    DataAttribute,
    xdmf_elements::{data_item::NumberType, dimensions::Dimensions},
};

macro_rules! define_values {
    ($($variant:ident($ty:ty)),+ $(,)?) => {
        /// Wrapper around different types of data, used to provide a unified interface.
        ///
        /// Backed by [`Cow`] rather than an owned `Vec`, so a caller that already holds the data
        /// in a slice can wrap it without copying and hand the same buffer to every time step.
        #[derive(Debug)]
        pub enum Values<'a> {
            $(
                #[doc = concat!("`", stringify!($ty), "` values")]
                $variant(Cow<'a, [$ty]>),
            )+
        }

        $(
            /// Moves `vec` into the value. If the same buffer is reused across multiple
            /// `write_data` calls, borrow it instead (`buf.as_slice().into()`)
            impl From<Vec<$ty>> for Values<'_> {
                fn from(vec: Vec<$ty>) -> Self {
                    Self::$variant(Cow::Owned(vec))
                }
            }

            impl<'a> From<&'a [$ty]> for Values<'a> {
                fn from(slice: &'a [$ty]) -> Self {
                    Self::$variant(Cow::Borrowed(slice))
                }
            }

            // The `&Vec<T>` and `&[T; N]` impls are not redundant with the `&[T]` one: the
            // `impl Into<Values<'_>>` arguments of `TimeStep::point_data`/`cell_data` are resolved
            // by trait matching, which does not deref-coerce, so passing a `&vec` or a
            // `&[1.0, 2.0]` needs its own impl.
            impl<'a> From<&'a Vec<$ty>> for Values<'a> {
                fn from(vec: &'a Vec<$ty>) -> Self {
                    Self::$variant(Cow::Borrowed(vec))
                }
            }

            impl<'a, const N: usize> From<&'a [$ty; N]> for Values<'a> {
                fn from(array: &'a [$ty; N]) -> Self {
                    Self::$variant(Cow::Borrowed(array))
                }
            }
        )+
    };
}

define_values!(F64(f64), F32(f32), I64(i64), I32(i32), U64(u64), U32(u32));

impl Values<'_> {
    /// Width in bytes each element type is written at, which is simply its own width. A backend
    /// that cannot carry a type says so through [`crate::paraview`] rather than storing fewer
    /// bytes than the caller handed over.
    pub(crate) fn precision(&self) -> u8 {
        match self {
            Self::F64(_) | Self::I64(_) | Self::U64(_) => 8,
            Self::F32(_) | Self::I32(_) | Self::U32(_) => 4,
        }
    }

    pub(crate) fn number_type(&self) -> NumberType {
        match self {
            Self::F64(_) | Self::F32(_) => NumberType::Float,
            Self::I64(_) | Self::I32(_) => NumberType::Int,
            Self::U64(_) | Self::U32(_) => NumberType::UInt,
        }
    }

    /// The Rust type name of the values held, for the reader's type-mismatch messages.
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Self::F64(_) => "f64",
            Self::F32(_) => "f32",
            Self::I64(_) => "i64",
            Self::I32(_) => "i32",
            Self::U64(_) => "u64",
            Self::U32(_) => "u32",
        }
    }

    // Only the number of values matters here, never their type, so the length is taken once and
    // the match is on the attribute alone -- matching on both would be one arm per (attribute,
    // variant) pair for the same `Dimensions`.
    pub(crate) fn dimensions(&self, attribute: DataAttribute) -> Dimensions {
        let len = self.len();

        // zero here means a component count of zero or one that does not fit a `usize`;
        // `write_attribute` rejects both before any values get here, so the flat shape below is
        // what data that never reaches a file gets, rather than a division by zero
        let size = attribute.size().filter(|size| *size != 0);

        match (attribute, size) {
            (DataAttribute::Scalar, _) | (_, None) => Dimensions(vec![len]),
            // written as a rank-3 shape ("<count> <size> 1") rather than "<count> <size>": VTK's
            // XDMF2 reader (vtkXdmfHeavyData, since https://github.com/Kitware/VTK/commit/7199be5854,
            // shipped in VTK 9.6 / ParaView 6.1) computes an AttributeType="Matrix" attribute's
            // component count as the product of its *last two* Dimensions entries, so a 2D
            // "<count> <size>" shape gets misread as one giant tuple. Appending a trailing 1 keeps
            // that product equal to `size` while `count` is used for the tuple count.
            (
                DataAttribute::Tensor6 | DataAttribute::Matrix(_, _) | DataAttribute::Generic(_),
                Some(size),
            ) => Dimensions(vec![len / size, size, 1]),
            (_, Some(size)) => Dimensions(vec![len / size, size]),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::U32(v) => v.len(),
        }
    }

    /// Borrow `len` elements starting at `start`, without copying.
    pub(crate) fn slice(&self, start: usize, len: usize) -> Values<'_> {
        let end = start + len;

        match self {
            Self::F64(v) => Values::F64(Cow::Borrowed(&v[start..end])),
            Self::F32(v) => Values::F32(Cow::Borrowed(&v[start..end])),
            Self::I64(v) => Values::I64(Cow::Borrowed(&v[start..end])),
            Self::I32(v) => Values::I32(Cow::Borrowed(&v[start..end])),
            Self::U64(v) => Values::U64(Cow::Borrowed(&v[start..end])),
            Self::U32(v) => Values::U32(Cow::Borrowed(&v[start..end])),
        }
    }
}

/// Scratch space for gathering a scattered submesh's share of a cell field out of the global array.
#[derive(Debug, Default)]
pub(crate) struct GatherBuffers {
    f64: Vec<f64>,
    f32: Vec<f32>,
    i64: Vec<i64>,
    i32: Vec<i32>,
    u64: Vec<u64>,
    u32: Vec<u32>,
}

impl GatherBuffers {
    /// Collect the `stride`-sized tuples at `indices` into the buffer for this element type.
    ///
    /// The caller has already checked that `values` holds `stride` elements for every cell and
    /// that each index names one of them, so the indexing below cannot go out of bounds.
    pub(crate) fn gather<'b>(
        &'b mut self,
        values: &Values<'_>,
        stride: usize,
        indices: &[usize],
    ) -> Values<'b> {
        match values {
            Values::F64(v) => Values::F64(gather_into(&mut self.f64, v, stride, indices)),
            Values::F32(v) => Values::F32(gather_into(&mut self.f32, v, stride, indices)),
            Values::I64(v) => Values::I64(gather_into(&mut self.i64, v, stride, indices)),
            Values::I32(v) => Values::I32(gather_into(&mut self.i32, v, stride, indices)),
            Values::U64(v) => Values::U64(gather_into(&mut self.u64, v, stride, indices)),
            Values::U32(v) => Values::U32(gather_into(&mut self.u32, v, stride, indices)),
        }
    }

    /// Collect every `stride`-th value starting at `offset` into the buffer for this element type.
    ///
    /// One coordinate direction of a mesh's interleaved points, which is how they are written for
    /// a mesh whose submeshes select their own out of them.
    pub(crate) fn component<'b>(
        &'b mut self,
        values: &Values<'_>,
        stride: usize,
        offset: usize,
    ) -> Values<'b> {
        match values {
            Values::F64(v) => Values::F64(component_into(&mut self.f64, v, stride, offset)),
            Values::F32(v) => Values::F32(component_into(&mut self.f32, v, stride, offset)),
            Values::I64(v) => Values::I64(component_into(&mut self.i64, v, stride, offset)),
            Values::I32(v) => Values::I32(component_into(&mut self.i32, v, stride, offset)),
            Values::U64(v) => Values::U64(component_into(&mut self.u64, v, stride, offset)),
            Values::U32(v) => Values::U32(component_into(&mut self.u32, v, stride, offset)),
        }
    }
}

fn gather_into<'b, T: Copy>(
    buffer: &'b mut Vec<T>,
    values: &[T],
    stride: usize,
    indices: &[usize],
) -> Cow<'b, [T]> {
    buffer.clear();
    buffer.reserve(indices.len() * stride);

    for &index in indices {
        buffer.extend_from_slice(&values[index * stride..(index + 1) * stride]);
    }

    Cow::Borrowed(buffer)
}

fn component_into<'b, T: Copy>(
    buffer: &'b mut Vec<T>,
    values: &[T],
    stride: usize,
    offset: usize,
) -> Cow<'b, [T]> {
    buffer.clear();
    buffer.reserve(values.len() / stride);
    buffer.extend(values.iter().skip(offset).step_by(stride).copied());

    Cow::Borrowed(buffer)
}

// Sealed so that `Coordinate` and `ConnectivityIndex` name exactly the types the XDMF geometry and
// topology can hold, and stay closed to outside impls. The conversions live here rather than on the
// public traits, so they are callable inside the crate without becoming public API.
pub(crate) mod sealed {
    use std::borrow::Cow;

    use super::Values;
    use crate::{Error, Result, reader::sealed::SealedValueType};

    /// Conversion backing [`Coordinate`](super::Coordinate), not nameable outside the crate.
    /// [`SealedValueType`] is a supertrait so a coordinate array can be read straight into the
    /// caller's buffer.
    pub trait SealedCoordinate: SealedValueType {
        /// Borrow a slice of coordinates as [`Values`]
        fn as_values(points: &[Self]) -> Values<'_>;

        /// Take a read coordinate array as this type, widening `f32` to `f64` and rejecting the
        /// narrowing direction. Named apart from [`SealedValueType::from_values`] so a mismatch is
        /// reported as a coordinate.
        fn coordinates_from_values(values: Values<'_>) -> Result<Vec<Self>>;
    }

    fn not_floating_point(requested: &str, found: &Values<'_>) -> Error {
        Error::NumberTypeMismatch {
            reason: format!(
                "requested coordinates as {requested}, but the file holds {}",
                found.type_name()
            ),
        }
    }

    impl SealedCoordinate for f64 {
        fn as_values(points: &[Self]) -> Values<'_> {
            Values::F64(Cow::Borrowed(points))
        }

        fn coordinates_from_values(values: Values<'_>) -> Result<Vec<Self>> {
            match values {
                Values::F64(v) => Ok(v.into_owned()),
                Values::F32(v) => Ok(v.iter().map(|&value| Self::from(value)).collect()),
                other => Err(not_floating_point("f64", &other)),
            }
        }
    }

    impl SealedCoordinate for f32 {
        fn as_values(points: &[Self]) -> Values<'_> {
            Values::F32(Cow::Borrowed(points))
        }

        fn coordinates_from_values(values: Values<'_>) -> Result<Vec<Self>> {
            match values {
                Values::F32(v) => Ok(v.into_owned()),
                other => Err(not_floating_point("f32", &other)),
            }
        }
    }

    /// Conversion backing [`ConnectivityIndex`](super::ConnectivityIndex), not nameable outside
    /// the crate. [`SealedValueType`] is a supertrait for the reason it is on
    /// [`SealedCoordinate`].
    pub trait SealedIndex: SealedValueType {
        /// The largest index this type can hold. Deliberately the type's own limit: the lower cap
        /// `ParaView` puts on `UInt` connectivity restricts the *values*, so [`crate::paraview`]
        /// enforces it with the rest, where it can be skipped.
        const MAX_INDEX: i128;

        /// Borrow a slice of indices as [`Values`]
        fn as_values(cells: &[Self]) -> Values<'_>;

        /// A cell type code or poly-cell point count, small enough for every index type
        fn from_u8(value: u8) -> Self;

        /// The index as this type, `None` when the type cannot hold it
        fn from_index(index: usize) -> Option<Self>;

        /// Widened for bounds checking, so signed and unsigned indices compare the same way
        fn as_i128(self) -> i128;

        /// The index as a position, `None` when it is negative or beyond what a `usize` holds.
        fn as_index(self) -> Option<usize>;

        /// Take a read connectivity array as this type.
        ///
        /// Any integer array is acceptable, since a connectivity holds positions; the *values* are
        /// checked instead, against [`Self::MAX_INDEX`] and against being negative. Every value in
        /// the result is therefore a valid position, which lets a later [`Self::as_index`] be
        /// infallible.
        fn indices_from_values(values: Values<'_>) -> Result<Vec<Self>>;
    }

    /// One index array converted element by element, for the types that are not already the one
    /// asked for. Split out of the macro so the conversion is written once.
    fn convert_indices<I: SealedIndex>(values: &Values<'_>) -> Result<Vec<I>> {
        match values {
            Values::F64(_) | Values::F32(_) => Err(not_an_index_array()),
            Values::U64(v) => v.iter().map(|&value| index_of(i128::from(value))).collect(),
            Values::U32(v) => v.iter().map(|&value| index_of(i128::from(value))).collect(),
            Values::I64(v) => v.iter().map(|&value| index_of(i128::from(value))).collect(),
            Values::I32(v) => v.iter().map(|&value| index_of(i128::from(value))).collect(),
        }
    }

    fn not_an_index_array() -> Error {
        Error::InvalidDocument {
            reason: "a Topology's connectivity holds floating-point values".to_string(),
        }
    }

    /// One connectivity value as the requested index type, rejecting the two ways a file's own
    /// array can hold something that is not a position this reader can hand back.
    fn index_of<I: SealedIndex>(value: i128) -> Result<I> {
        if value < 0 {
            return Err(Error::InvalidDocument {
                reason: format!("connectivity index {value} is negative"),
            });
        }

        usize::try_from(value)
            .ok()
            .and_then(I::from_index)
            .ok_or_else(|| Error::IntegerOutOfRange {
                value,
                reason: format!(
                    "the connectivity index does not fit the requested index type, whose largest \
                     is {}",
                    I::MAX_INDEX
                ),
            })
    }

    macro_rules! impl_sealed_index {
        ($($variant:ident($ty:ty, $max_index:expr)),+ $(,)?) => {
            $(
                impl SealedIndex for $ty {
                    const MAX_INDEX: i128 = $max_index;

                    fn as_values(cells: &[Self]) -> Values<'_> {
                        Values::$variant(Cow::Borrowed(cells))
                    }

                    fn from_u8(value: u8) -> Self {
                        Self::from(value)
                    }

                    fn from_index(index: usize) -> Option<Self> {
                        Self::try_from(index).ok()
                    }

                    fn as_i128(self) -> i128 {
                        i128::from(self)
                    }

                    fn as_index(self) -> Option<usize> {
                        usize::try_from(self).ok()
                    }

                    fn indices_from_values(values: Values<'_>) -> Result<Vec<Self>> {
                        match values {
                            // already the type asked for, so it is moved rather than converted --
                            // but still walked once, since nothing has yet ruled out a negative
                            // index or (on a 32-bit target) one past `usize`
                            Values::$variant(v) => {
                                let v = v.into_owned();
                                for &value in &v {
                                    index_of::<Self>(value.as_i128())?;
                                }
                                Ok(v)
                            }
                            other => convert_indices(&other),
                        }
                    }
                }
            )+
        };
    }

    impl_sealed_index!(
        U32(u32, u32::MAX as i128),
        U64(u64, u64::MAX as i128),
        I32(i32, i32::MAX as i128),
        I64(i64, i64::MAX as i128),
    );
}

/// A type usable as a point coordinate: `f32` or `f64`. Also what
/// [`TimeSeriesReader::read_points`](crate::TimeSeriesReader::read_points) fills, so a mesh written
/// as `f32` reads back at that width.
pub trait Coordinate: sealed::SealedCoordinate {}

impl Coordinate for f64 {}

impl Coordinate for f32 {}

/// A type usable as a connectivity index: `u32`, `u64`, `i32` or `i64`.
///
/// The connectivity is written as the type it is passed in, so this choice sets the largest mesh
/// that can be written. It is also what
/// [`TimeSeriesReader::read_topology`](crate::TimeSeriesReader::read_topology) fills, and there it
/// caps what can be read back: an index the type cannot hold is
/// [`Error::IntegerOutOfRange`](crate::Error::IntegerOutOfRange).
pub trait ConnectivityIndex: sealed::SealedIndex {}

impl ConnectivityIndex for u32 {}

impl ConnectivityIndex for u64 {}

impl ConnectivityIndex for i32 {}

impl ConnectivityIndex for i64 {}

#[cfg(test)]
mod tests {
    use super::{
        sealed::{SealedCoordinate, SealedIndex},
        *,
    };

    #[test]
    fn vec_f64() {
        let vec_f64 = vec![1., 2., 3., 4., 5., 6.];

        let values = vec_f64.into();
        std::assert_matches!(values, Values::F64(_));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(), 8);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Vector),
            Dimensions(vec![2, 3])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Tensor6),
            Dimensions(vec![1, 6, 1])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Matrix(3, 2)),
            Dimensions(vec![1, 6, 1])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_f32() {
        let vec_f32 = vec![1., 2., 3., 4., 5., 6.0_f32];

        let values = vec_f32.into();
        std::assert_matches!(values, Values::F32(_));

        // same NumberType as f64, only the precision distinguishes the two in the light data
        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(values.precision(), 4);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Vector),
            Dimensions(vec![2, 3])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Tensor6),
            Dimensions(vec![1, 6, 1])
        );
        assert_eq!(
            values.dimensions(DataAttribute::Matrix(3, 2)),
            Dimensions(vec![1, 6, 1])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_u64() {
        let vec_u64 = vec![1_u64, 2, 3, 4, 5, 6];
        let values = vec_u64.into();
        std::assert_matches!(values, Values::U64(_));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.precision(), 8);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_u32() {
        let vec_u32 = vec![1_u32, 2, 3, 4, 5, 6];
        let values = vec_u32.into();
        std::assert_matches!(values, Values::U32(_));

        // same NumberType as u64, but half the width
        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.precision(), 4);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_i64() {
        let vec_i64 = vec![-1_i64, 2, 3, 4, 5, 6];
        let values = vec_i64.into();
        std::assert_matches!(values, Values::I64(_));

        assert_eq!(values.number_type(), NumberType::Int);
        assert_eq!(values.precision(), 8);
        assert_eq!(
            values.dimensions(DataAttribute::Vector),
            Dimensions(vec![2, 3])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn vec_i32() {
        let vec_i32 = vec![-1_i32, 2, 3, 4, 5, 6];
        let values = vec_i32.into();
        std::assert_matches!(values, Values::I32(_));

        assert_eq!(values.number_type(), NumberType::Int);
        assert_eq!(values.precision(), 4);
        assert_eq!(
            values.dimensions(DataAttribute::Scalar),
            Dimensions(vec![6])
        );
        assert_eq!(values.len(), 6);
    }

    #[test]
    fn borrowed_slices() {
        let vec_f64 = vec![1., 2., 3., 4., 5., 6.];
        let values = Values::from(vec_f64.as_slice());
        std::assert_matches!(values, Values::F64(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::Float);
        assert_eq!(
            values.dimensions(DataAttribute::Vector),
            Dimensions(vec![2, 3])
        );
        assert_eq!(values.len(), 6);

        let vec_f32 = vec![1., 2., 3., 4., 5., 6.0_f32];
        let values = Values::from(vec_f32.as_slice());
        std::assert_matches!(values, Values::F32(Cow::Borrowed(_)));

        assert_eq!(values.precision(), 4);
        assert_eq!(values.len(), 6);

        let vec_u64 = vec![1_u64, 2, 3];
        let values = Values::from(vec_u64.as_slice());
        std::assert_matches!(values, Values::U64(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.len(), 3);

        let vec_u32 = vec![1_u32, 2, 3];
        let values = Values::from(vec_u32.as_slice());
        std::assert_matches!(values, Values::U32(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::UInt);
        assert_eq!(values.len(), 3);

        let vec_i64 = vec![1_i64, 2, 3];
        let values = Values::from(vec_i64.as_slice());
        std::assert_matches!(values, Values::I64(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::Int);
        assert_eq!(values.len(), 3);

        let vec_i32 = vec![1_i32, 2, 3];
        let values = Values::from(vec_i32.as_slice());
        std::assert_matches!(values, Values::I32(Cow::Borrowed(_)));

        assert_eq!(values.number_type(), NumberType::Int);
        assert_eq!(values.len(), 3);
    }

    // the `&Vec<T>`/`&[T; N]` impls are generated for every element type, so a compile-time check
    // that they resolve is the point here
    #[test]
    fn borrowed_vecs_and_arrays() {
        let vec_i32 = vec![1_i32, 2, 3];
        std::assert_matches!(Values::from(&vec_i32), Values::I32(Cow::Borrowed(_)));
        std::assert_matches!(Values::from(&[1_u32, 2]), Values::U32(Cow::Borrowed(_)));
        std::assert_matches!(Values::from(&[1_i64, 2]), Values::I64(Cow::Borrowed(_)));
    }

    #[test]
    fn coordinates_borrow_as_values() {
        let points_f64 = [0.0_f64, 1.0, 2.0];
        std::assert_matches!(
            SealedCoordinate::as_values(&points_f64),
            Values::F64(Cow::Borrowed(_))
        );

        let points_f32 = [0.0_f32, 1.0, 2.0];
        std::assert_matches!(
            SealedCoordinate::as_values(&points_f32),
            Values::F32(Cow::Borrowed(_))
        );
    }

    #[test]
    fn connectivity_indices_borrow_as_values() {
        std::assert_matches!(
            SealedIndex::as_values(&[0_u32, 1]),
            Values::U32(Cow::Borrowed(_))
        );
        std::assert_matches!(
            SealedIndex::as_values(&[0_u64, 1]),
            Values::U64(Cow::Borrowed(_))
        );
        std::assert_matches!(
            SealedIndex::as_values(&[0_i32, 1]),
            Values::I32(Cow::Borrowed(_))
        );
        std::assert_matches!(
            SealedIndex::as_values(&[0_i64, 1]),
            Values::I64(Cow::Borrowed(_))
        );
    }

    #[test]
    fn slice_borrows_a_run_of_values() {
        let values: Values<'_> = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0].into();

        // two vector tuples' worth, starting at the second
        let sliced = values.slice(3, 3);

        std::assert_matches!(sliced, Values::F64(Cow::Borrowed(v)) if v == [4.0, 5.0, 6.0]);
    }

    #[test]
    fn gather_collects_scalars_in_the_given_order() {
        let values: Values<'_> = vec![10_i32, 11, 12, 13].into();
        let mut buffers = GatherBuffers::default();

        let gathered = buffers.gather(&values, 1, &[2, 0]);

        std::assert_matches!(gathered, Values::I32(v) if v.as_ref() == [12, 10]);
    }

    #[test]
    fn gather_keeps_strided_tuples_together() {
        // three points of xyz, of which the third and the first are wanted
        let values: Values<'_> = vec![0.0_f32, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2].into();
        let mut buffers = GatherBuffers::default();

        let gathered = buffers.gather(&values, 3, &[2, 0]);

        std::assert_matches!(
            gathered,
            Values::F32(v) if v.as_ref() == [2.0, 2.1, 2.2, 0.0, 0.1, 0.2]
        );
    }

    #[test]
    fn gather_reuses_the_buffer_of_its_element_type() {
        let values: Values<'_> = vec![1_u64, 2, 3, 4].into();
        let mut buffers = GatherBuffers::default();

        // a long gather first, so the buffer has grown before the short one reuses it
        let capacity = match buffers.gather(&values, 1, &[0, 1, 2, 3]) {
            Values::U64(v) => v.len(),
            other => panic!("expected U64, got {other:?}"),
        };
        assert_eq!(capacity, 4);

        // the second gather must not see anything left over from the first
        std::assert_matches!(
            buffers.gather(&values, 1, &[3]),
            Values::U64(v) if v.as_ref() == [4]
        );
    }

    #[test]
    fn connectivity_index_limits() {
        // each type's own limit -- the lower one ParaView puts on `u64` is checked on the values,
        // by `validate_paraview_uint_range`, so that it can be skipped with the rest
        assert_eq!(u32::MAX_INDEX, i128::from(u32::MAX));
        assert_eq!(u64::MAX_INDEX, i128::from(u64::MAX));
        assert_eq!(i32::MAX_INDEX, i128::from(i32::MAX));
        assert_eq!(i64::MAX_INDEX, i128::from(i64::MAX));

        assert_eq!(u32::from_index(7), Some(7_u32));
        assert_eq!(i32::from_index(usize::MAX), None);
        assert_eq!(i32::from_u8(9), 9_i32);
        assert_eq!((-1_i64).as_i128(), -1_i128);
    }
}

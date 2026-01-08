//! Typed arrays backed by byte buffers.
//!
//! This module provides [`Array`], a small wrapper around a contiguous byte buffer plus a shape.
//! The element interpretation is controlled by an [`ArrayDataType`].
//!
//! ## Buffer types
//! An `Array` is generic over its backing buffer `B`.
//! - The default is `Immutable<bytes::Bytes>` (owned, immutable).
//! - Use `Immutable<&[u8]>` for borrowed, zero-copy views.
//! - Use [`MutableRef`] for mutable borrowed byte slices.
//!
//! ## Typing model
//! `Array<T, B>` tracks two independent aspects at compile time:
//! - `T: ArrayDataType` describes how to validate and interpret the bytes (logical element type).
//! - `B` describes how the bytes are stored/borrowed (ownership and mutability).
//!
//! Most APIs that only need read access are generic over `B: AsRef<[u8]>`.
//! Mutable views (e.g. `as_mut_ndarray`) additionally require `B: AsMut<[u8]>`.
//!
//! `T::Native<'a>` is the element type produced when viewing bytes for a particular lifetime.
//! For fixed-width scalars it is typically the scalar itself (e.g. `i32`); for [`Utf8`] it is
//! `&'a str` (borrowing string slices from the underlying payload).
//!
//! ## Chapter: How the typing system works
//!
//! The design separates **what the bytes mean** from **how the bytes are stored**.
//!
//! ### 1) `T`: the logical element type
//! `T` implements [`ArrayDataType`] and defines:
//! - validation rules (`validate`) for a `(bytes, shape)` pair
//! - how to view bytes as an `ndarray` (`as_array`, optionally `as_array_mut`)
//! - metadata (`TYPE`, `ALIGNMENT`, `BYTE_WIDTH`)
//!
//! This makes it hard to mix up byte interpretations: `Array<i32>` and `Array<f32>` are distinct
//! Rust types.
//!
//! ### 2) `B`: the backing buffer type
//! `B` represents ownership/mutability:
//! - owned immutable: `Immutable<bytes::Bytes>` (default)
//! - borrowed immutable: `Immutable<&[u8]>`
//! - borrowed mutable: [`MutableRef`]
//!
//! The key trait bounds are simple:
//! - reading requires `B: AsRef<[u8]>`
//! - in-place mutation requires `B: AsMut<[u8]>`
//!
//! ### 3) Why `T::Native<'a>` has a lifetime
//! When you view an `Array` as an `ndarray`, you’re producing values that may borrow from the
//! underlying byte buffer.
//!
//! - For fixed-width scalars, `T::Native<'a>` is typically the scalar itself (e.g. `i32`).
//! - For variable-width encodings like [`Utf8`], `T::Native<'a>` is `&'a str`, borrowing string
//!   slices from the payload.
//!
//! ### 4) `PhantomData<T>`
//! `Array` doesn’t store any `T` values at runtime; `T` is purely compile-time information.
//! `PhantomData<T>` tells Rust that `Array` is parameterized by `T` for type checking and variance.
//!
//! ### Examples
//! Borrowed, zero-copy view from an `ndarray` (standard-layout only):
//! ```rust,ignore
//! use crate::variable::array::Array;
//!
//! let nd = ndarray::Array::from_shape_vec((2, 2), vec![1i32, 2, 3, 4]).unwrap().into_dyn();
//! let arr = Array::<i32>::try_from_ndarray(&nd).unwrap();
//! // arr is Array<i32, Immutable<&[u8]>> (borrows bytes from nd)
//! assert_eq!(arr.shape(), &[2, 2]);
//! ```
//!
//! Same logical type, but different buffer (owned):
//! ```rust,ignore
//! use bytes::Bytes;
//! use crate::variable::array::{Array, Immutable};
//!
//! let owned = Array::<i32>::try_new(Immutable(Bytes::from(vec![0u8; 16])), smallvec::smallvec![4])
//!     .unwrap_err();
//! // (this errors because the bytes aren't a valid i32 buffer; shown for illustration)
//! ```
//!
//! ## Validation and safety
//! Prefer [`Array::try_new`] to construct arrays with validation. [`Array::new`] panics on
//! validation failure.
//!
//! [`Array::new_unchecked`] skips validation and is `unsafe`: callers must uphold all invariants
//! required by the selected [`ArrayDataType`] (alignment, byte length, and for [`Utf8`] valid
//! UTF-8 and correct prefix/payload structure).
//!
//! ## ndarray interop
//! - [`Array::as_ndarray`] creates an `ndarray` view after validation.
//! - [`Array::try_from_ndarray`] creates a **borrowed, zero-copy** `Array` view over an
//!   `ndarray` backing buffer, but only for **standard-layout** (contiguous, row-major) arrays.
use smallvec::SmallVec;

use crate::{
    dtype::DataType,
    variable::array::{
        datatype::{ArrayDataType, Utf8},
        error::ArrayValidationError,
    },
};

pub mod blob;
pub mod datatype;
pub mod discovery;
pub mod error;
pub mod fill_value;
pub mod metadata;
pub mod reader;
pub mod util;
pub mod writer;

/// A mutable borrowed byte buffer wrapper.
///
/// This is primarily used as an `Array` backing buffer type (`Array<T, MutableRef<'_>>`).
pub struct MutableRef<'b>(&'b mut [u8]);
impl<'b> AsRef<[u8]> for MutableRef<'b> {
    /// Returns an immutable view of the underlying bytes.
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}
impl<'b> AsMut<[u8]> for MutableRef<'b> {
    /// Returns a mutable view of the underlying bytes.
    fn as_mut(&mut self) -> &mut [u8] {
        self.0
    }
}

/// An immutable byte buffer wrapper.
///
/// This is primarily used as an `Array` backing buffer type.
/// - `Immutable<bytes::Bytes>` is the default owned representation.
/// - `Immutable<&[u8]>` represents a borrowed, zero-copy view.
pub struct Immutable<T: AsRef<[u8]>>(T);
impl<T: AsRef<[u8]>> AsRef<[u8]> for Immutable<T> {
    /// Returns an immutable view of the underlying bytes.
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

/// A typed array backed by a contiguous byte buffer.
///
/// Construction validates the buffer against `T` and `shape` (unless using
/// [`Array::new_unchecked`]). After validation, [`Array::as_ndarray`] creates
/// an `ndarray` view.
pub struct Array<T: ArrayDataType, B = Immutable<bytes::Bytes>> {
    pub data: B,
    pub shape: SmallVec<[usize; 4]>,
    _type: std::marker::PhantomData<T>,
}

impl<'a, T: ArrayDataType + 'a> Array<T, Immutable<bytes::Bytes>>
where
    T::Native<'a>: Copy,
{
    /// Construct a borrowed, validated array from a typed slice and a shape.
    ///
    /// This is **zero-copy** and borrows the backing memory of `data`.
    ///
    /// # Arguments
    /// - `shape`: The logical array dimensions.
    /// - `data`: Elements in row-major order.
    ///
    /// # Errors
    /// Returns [`ArrayValidationError`] if:
    /// - `T` is variable-width (`T::BYTE_WIDTH = None`)
    /// - the number of elements implied by `shape` overflows
    /// - `data.len()` does not match the element count implied by `shape`
    pub fn try_from_shape_and_slice<'l>(
        shape: &'l [usize],
        data: &'l [T::Native<'a>],
    ) -> Result<Array<T, Immutable<&'l [u8]>>, ArrayValidationError> {
        match T::BYTE_WIDTH {
            Some(width) => {
                let expected_elems = crate::variable::array::util::num_elements(shape)?;
                if data.len() != expected_elems {
                    return Err(ArrayValidationError::WrongElementCount {
                        expected: expected_elems,
                        actual: data.len(),
                    });
                }

                let expected_len = expected_elems
                    .checked_mul(width)
                    .ok_or(ArrayValidationError::ShapeOverflow)?;
                let data_len = data
                    .len()
                    .checked_mul(width)
                    .ok_or(ArrayValidationError::ShapeOverflow)?;

                // Safety: the validated element count/width ensures this is the correct span.
                let data_bytes =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data_len) };
                if data_bytes.len() != expected_len {
                    return Err(ArrayValidationError::WrongByteLen {
                        expected: expected_len,
                        actual: data_bytes.len(),
                    });
                }

                Array::<T, Immutable<&'l [u8]>>::try_new(Immutable(data_bytes), shape.into())
            }
            None => Err(ArrayValidationError::UnsupportedVariableWidth),
        }
    }

    /// Alias for [`Self::try_from_shape_and_slice`].
    pub fn from_shape_and_slice<'l>(
        shape: &'l [usize],
        data: &'l [T::Native<'a>],
    ) -> Result<Array<T, Immutable<&'l [u8]>>, ArrayValidationError> {
        Self::try_from_shape_and_slice(shape, data)
    }
}

impl<T: ArrayDataType> Array<T, Immutable<bytes::Bytes>> {
    /// Create a **zero-copy** borrowed array view over an `ndarray`'s backing buffer.
    ///
    /// This only works for **standard-layout** ndarrays (contiguous, row-major). Non-standard
    /// layout views (e.g. transpose) are rejected.
    ///
    /// # Arguments
    /// - `array`: An `ndarray` with dynamic dimensionality (`IxDyn`).
    ///
    /// # Errors
    /// Returns [`ArrayValidationError`] if:
    /// - `T` is variable-width (`T::BYTE_WIDTH = None`)
    /// - `array` is not standard layout / not representable as a contiguous slice
    /// - the element count implied by `array.shape()` overflows
    /// - the slice length does not match the shape
    pub fn try_from_ndarray<'a, S>(
        array: &'a ndarray::ArrayBase<S, ndarray::IxDyn>,
    ) -> Result<Array<T, Immutable<&'a [u8]>>, ArrayValidationError>
    where
        S: ndarray::Data<Elem = T::Native<'a>>,
        T::Native<'a>: Copy,
    {
        let Some(width) = T::BYTE_WIDTH else {
            return Err(ArrayValidationError::UnsupportedVariableWidth);
        };

        if !array.is_standard_layout() {
            return Err(ArrayValidationError::NonContiguousNdarray);
        }

        let elems = array
            .as_slice_memory_order()
            .ok_or(ArrayValidationError::NonContiguousNdarray)?;

        let shape: SmallVec<[usize; 4]> = array.shape().iter().copied().collect();
        let expected_elems = crate::variable::array::util::num_elements(&shape)?;
        if elems.len() != expected_elems {
            return Err(ArrayValidationError::WrongElementCount {
                expected: expected_elems,
                actual: elems.len(),
            });
        }

        let total_len = expected_elems
            .checked_mul(width)
            .ok_or(ArrayValidationError::ShapeOverflow)?;
        let bytes_view =
            unsafe { std::slice::from_raw_parts(elems.as_ptr() as *const u8, total_len) };

        Array::<T, Immutable<&'a [u8]>>::try_new(Immutable(bytes_view), shape)
    }
}

impl Array<Utf8> {
    /// Construct a validated UTF-8 inline string array from a list of `&str`.
    ///
    /// The encoding is:
    /// - prefix: `n` native-endian `u32` lengths
    /// - payload: concatenated UTF-8 bytes
    ///
    /// # Arguments
    /// - `shape`: The logical array dimensions.
    /// - `data`: String elements in row-major order.
    ///
    /// # Errors
    /// Returns [`ArrayValidationError`] if:
    /// - `data.len()` does not match the element count implied by `shape`
    /// - length/payload computations overflow
    /// - any string is too large to encode in a `u32` length prefix
    pub fn try_from_shape_and_strs(
        shape: &[usize],
        data: &[&str],
    ) -> Result<Array<Utf8>, ArrayValidationError> {
        let n = crate::variable::array::util::num_elements(shape)?;
        if data.len() != n {
            return Err(ArrayValidationError::WrongElementCount {
                expected: n,
                actual: data.len(),
            });
        }

        let lengths_bytes = n
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(ArrayValidationError::ShapeOverflow)?;

        let mut payload_len = 0usize;
        for s in data {
            payload_len = payload_len
                .checked_add(s.len())
                .ok_or(ArrayValidationError::LengthsOverflow)?;
        }
        let total_len = lengths_bytes
            .checked_add(payload_len)
            .ok_or(ArrayValidationError::LengthsOverflow)?;

        let mut bytes = Vec::with_capacity(total_len);
        for s in data {
            let len_u32: u32 = s
                .len()
                .try_into()
                .map_err(|_| ArrayValidationError::StringLenTooLarge { len: s.len() })?;
            bytes.extend_from_slice(&len_u32.to_ne_bytes());
        }
        for s in data {
            bytes.extend_from_slice(s.as_bytes());
        }

        let array_data = bytes::Bytes::from(bytes);
        Array::try_new(Immutable(array_data), shape.into())
    }

    /// Alias for [`Self::try_from_shape_and_strs`].
    pub fn from_shape_and_strs(
        shape: &[usize],
        data: &[&str],
    ) -> Result<Array<Utf8>, ArrayValidationError> {
        Self::try_from_shape_and_strs(shape, data)
    }
}

impl<T: ArrayDataType, B: AsRef<[u8]>> Array<T, B> {
    /// Create a validated array.
    ///
    /// # Arguments
    /// - `data`: Backing bytes to interpret.
    /// - `shape`: Logical dimensions.
    ///
    /// # Panics
    /// Panics if [`ArrayDataType::validate`] fails for the given bytes/shape.
    pub fn new(data: B, shape: SmallVec<[usize; 4]>) -> Self {
        if let Err(e) = T::validate(data.as_ref(), &shape) {
            panic!("invalid array bytes/shape: {e}");
        }
        Self {
            data,
            shape,
            _type: std::marker::PhantomData,
        }
    }

    /// Deep clone the array into an owned immutable buffer.
    ///
    /// This always copies the underlying bytes into a new `bytes::Bytes` buffer.
    pub fn deep_clone(&self) -> Array<T> {
        let data_bytes = bytes::Bytes::copy_from_slice(self.data.as_ref());
        Array {
            data: Immutable(data_bytes),
            shape: self.shape.clone(),
            _type: std::marker::PhantomData,
        }
    }

    /// Create an array without validating bytes/shape.
    ///
    /// # Safety
    /// The caller must ensure that `data` matches `shape` and `T`'s expected layout.
    /// In particular, this must uphold all invariants relied on by [`ArrayDataType::as_array`],
    /// including alignment and length constraints, and (for [`Utf8`]) valid UTF-8.
    pub unsafe fn new_unchecked(data: B, shape: SmallVec<[usize; 4]>) -> Self {
        Self {
            data,
            shape,
            _type: std::marker::PhantomData,
        }
    }

    /// Create a validated array, returning a structured error instead of panicking.
    ///
    /// # Arguments
    /// - `data`: Backing bytes to interpret.
    /// - `shape`: Logical dimensions.
    ///
    /// # Errors
    /// Returns [`ArrayValidationError`] if the bytes/shape are invalid for `T`.
    pub fn try_new(data: B, shape: SmallVec<[usize; 4]>) -> Result<Self, ArrayValidationError> {
        T::validate(data.as_ref(), &shape)?;
        Ok(Self {
            data,
            shape,
            _type: std::marker::PhantomData,
        })
    }

    /// View the array as an `ndarray`.
    ///
    /// # Notes
    /// - For fixed-width types, this is typically **zero-copy**.
    /// - For [`Utf8`], this allocates a `Vec<&str>` holding references into the payload.
    pub fn as_ndarray<'a>(&'a self) -> ndarray::CowArray<'a, T::Native<'a>, ndarray::IxDyn> {
        T::as_array(self.data.as_ref(), &self.shape)
    }

    /// Return the logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the logical datatype of this array.
    pub const fn data_type(&self) -> DataType {
        T::TYPE
    }

    /// Return the required alignment (in bytes) for the backing buffer.
    pub const fn alignment(&self) -> usize {
        T::ALIGNMENT
    }
}

impl<T: ArrayDataType, B: AsMut<[u8]>> Array<T, B> {
    /// View the array as a mutable `ndarray`.
    ///
    /// # Returns
    /// - `Some(_)` when `T` supports mutable views.
    /// - `None` for types that don't expose a mutable view (e.g. variable-width encodings).
    pub fn as_mut_ndarray<'a>(
        &'a mut self,
    ) -> Option<ndarray::ArrayViewMut<'a, T::Native<'a>, ndarray::IxDyn>> {
        T::as_array_mut(self.data.as_mut(), &self.shape)
    }
}

#[cfg(test)]
mod tests {
    use crate::variable::array::{datatype::Utf8, error::ArrayValidationError};

    use super::*;
    use bytes::Bytes;

    fn imm(b: Bytes) -> Immutable<Bytes> {
        Immutable(b)
    }

    fn utf8_inline_bytes(strings: &[&str]) -> Vec<u8> {
        let mut out = Vec::new();
        for s in strings {
            let len: u32 = s.as_bytes().len().try_into().unwrap();
            out.extend_from_slice(&len.to_ne_bytes());
        }
        for s in strings {
            out.extend_from_slice(s.as_bytes());
        }
        out
    }

    #[test]
    fn numeric_try_new_validates_byte_len() {
        let data = Bytes::from(vec![0u8; 3]);
        let err = match Array::<i16>::try_new(imm(data), smallvec::smallvec![2]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            ArrayValidationError::WrongByteLen { expected, actual } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 3);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn numeric_try_new_validates_alignment() {
        let mut v = vec![0u8];
        v.extend_from_slice(&1i32.to_ne_bytes());
        let b = Bytes::from(v).slice(1..);

        let err = match Array::<i32>::try_new(imm(b), smallvec::smallvec![1]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            ArrayValidationError::Misaligned { alignment } => assert_eq!(alignment, 4),
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn numeric_good_array_validates_and_decodes() {
        let values = [1i32, -2i32, 3i32];
        let align = std::mem::align_of::<i32>();
        let size = std::mem::size_of::<i32>();

        // Make an aligned slice inside a larger buffer.
        let mut buf = vec![0u8; align + values.len() * size];
        let base = buf.as_ptr() as usize;
        let offset = (align - (base % align)) % align;
        for (i, v) in values.iter().copied().enumerate() {
            let start = offset + i * size;
            buf[start..start + size].copy_from_slice(&v.to_ne_bytes());
        }

        let b = Bytes::from(buf).slice(offset..offset + values.len() * size);
        let arr = Array::<i32>::try_new(imm(b), smallvec::smallvec![values.len()]).unwrap();
        let got: Vec<i32> = arr.as_ndarray().iter().copied().collect();
        assert_eq!(got, values);
    }

    #[test]
    fn bool_try_new_rejects_non_0_1() {
        let b = Bytes::from(vec![0u8, 2u8, 1u8]);
        let err = match Array::<bool>::try_new(imm(b), smallvec::smallvec![3]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(matches!(err, ArrayValidationError::InvalidBoolByte));
    }

    #[test]
    fn bool_try_new_rejects_wrong_len() {
        let b = Bytes::from(vec![0u8, 1u8]);
        let err = match Array::<bool>::try_new(imm(b), smallvec::smallvec![3]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            ArrayValidationError::WrongByteLen { expected, actual } => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn bool_good_array_validates_and_decodes() {
        let b = Bytes::from(vec![0u8, 1u8, 0u8, 1u8]);
        let arr = Array::<bool>::try_new(imm(b), smallvec::smallvec![4]).unwrap();
        let got: Vec<bool> = arr.as_ndarray().iter().copied().collect();
        assert_eq!(got, vec![false, true, false, true]);
    }

    #[test]
    fn utf8_try_new_and_decode() {
        let bytes = utf8_inline_bytes(&["hi", "", "world"]);
        let arr = Array::<Utf8>::try_new(imm(Bytes::from(bytes)), smallvec::smallvec![3]).unwrap();
        let a = arr.as_ndarray();
        let got: Vec<&str> = a.iter().copied().collect();
        assert_eq!(got, vec!["hi", "", "world"]);
    }

    #[test]
    fn utf8_validate_rejects_invalid_utf8() {
        // One string of length 1 containing invalid UTF-8 byte 0xFF.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_ne_bytes());
        bytes.push(0xFF);

        let err = match Array::<Utf8>::try_new(imm(Bytes::from(bytes)), smallvec::smallvec![1]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(matches!(err, ArrayValidationError::InvalidUtf8));
    }

    #[test]
    fn utf8_validate_rejects_payload_out_of_bounds() {
        // Declares len=10 but provides only 3 bytes.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&10u32.to_ne_bytes());
        bytes.extend_from_slice(b"abc");

        let err = match Array::<Utf8>::try_new(imm(Bytes::from(bytes)), smallvec::smallvec![1]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(matches!(err, ArrayValidationError::PayloadOutOfBounds));
    }

    #[test]
    fn utf8_validate_rejects_trailing_bytes() {
        let mut bytes = utf8_inline_bytes(&["a"]);
        bytes.push(0);

        let err = match Array::<Utf8>::try_new(imm(Bytes::from(bytes)), smallvec::smallvec![1]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            ArrayValidationError::WrongByteLen { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn new_unchecked_skips_validation() {
        // Wrong shape vs bytes for i32, but construction should not validate.
        let data = Bytes::from(vec![0u8; 1]);
        let _arr = unsafe { Array::<i32>::new_unchecked(imm(data), smallvec::smallvec![999]) };
    }

    #[test]
    #[should_panic(expected = "invalid array bytes/shape")]
    fn new_panics_on_invalid() {
        let data = Bytes::from(vec![0u8; 1]);
        let _ = Array::<i32>::new(imm(data), smallvec::smallvec![1]);
    }

    #[test]
    fn try_new_rejects_shape_overflow() {
        let data = Bytes::new();
        let err = match Array::<u8>::try_new(imm(data), smallvec::smallvec![usize::MAX, 2]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(matches!(err, ArrayValidationError::ShapeOverflow));
    }

    #[test]
    fn utf8_rejects_short_prefix() {
        // shape=[2] requires 2*u32=8 bytes prefix.
        let bytes = Bytes::from(vec![0u8; 4]);
        let err = match Array::<Utf8>::try_new(imm(bytes), smallvec::smallvec![2]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            ArrayValidationError::WrongPrefixLen { expected, actual } => {
                assert_eq!(expected, 8);
                assert_eq!(actual, 4);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn numeric_multidim_decodes_in_row_major() {
        let values = [1i32, 2i32, 3i32, 4i32];
        let align = std::mem::align_of::<i32>();
        let size = std::mem::size_of::<i32>();

        let mut buf = vec![0u8; align + values.len() * size];
        let base = buf.as_ptr() as usize;
        let offset = (align - (base % align)) % align;
        for (i, v) in values.iter().copied().enumerate() {
            let start = offset + i * size;
            buf[start..start + size].copy_from_slice(&v.to_ne_bytes());
        }

        let b = Bytes::from(buf).slice(offset..offset + values.len() * size);
        let arr = Array::<i32>::try_new(imm(b), smallvec::smallvec![2, 2]).unwrap();
        let a = arr.as_ndarray();
        assert_eq!(a[[0, 0]], 1);
        assert_eq!(a[[0, 1]], 2);
        assert_eq!(a[[1, 0]], 3);
        assert_eq!(a[[1, 1]], 4);
    }

    #[test]
    fn as_mut_ndarray_allows_in_place_edit() {
        let mut data = vec![1i32, 2, 3, 4];
        {
            let byte_len = data.len() * std::mem::size_of::<i32>();
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, byte_len) };

            let mut arr =
                Array::<i32, MutableRef>::new(MutableRef(bytes), smallvec::smallvec![2, 2]);
            let mut view = arr.as_mut_ndarray().expect("expected mutable ndarray view");
            view[[0, 1]] = 99;
        }
        assert_eq!(data, vec![1, 99, 3, 4]);
    }

    #[test]
    fn try_from_shape_and_strs_rejects_wrong_element_count() {
        let err = match Array::<Utf8>::try_from_shape_and_strs(&[2], &["only-one"]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        match err {
            ArrayValidationError::WrongElementCount { expected, actual } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn ndarray_to_array_roundtrip_fixed_width() {
        let a = ndarray::Array::from_shape_vec((2, 3), vec![1i32, 2, 3, 4, 5, 6])
            .unwrap()
            .into_dyn();

        let arr = Array::<i32>::try_from_ndarray(&a).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);

        let got: Vec<i32> = arr.as_ndarray().iter().copied().collect();
        assert_eq!(got, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn ndarray_to_array_handles_non_contiguous_view_by_copying() {
        let base = ndarray::Array::from_shape_vec((2, 3), vec![1i32, 2, 3, 4, 5, 6])
            .unwrap()
            .into_dyn();

        // Transpose creates a non-contiguous view.
        let t = base.view().reversed_axes();
        let err = match Array::<i32>::try_from_ndarray(&t) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(matches!(err, ArrayValidationError::NonContiguousNdarray));
    }
}

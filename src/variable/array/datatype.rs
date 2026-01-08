//! Array datatype definitions.
//!
//! This module defines [`ArrayDataType`], which describes how a typed [`crate::variable::array::Array`]
//! interprets and validates its backing byte buffer.
//!
//! ## Core responsibilities
//! - Define the logical [`crate::dtype::DataType`] for the array.
//! - Specify required byte alignment (`ALIGNMENT`).
//! - Specify fixed element width (`BYTE_WIDTH`) or implement variable-width validation.
//! - Provide `ndarray` views over validated bytes via [`ArrayDataType::as_array`] (and optionally
//!   [`ArrayDataType::as_array_mut`]).
//!
//! ## Validation
//! The default [`ArrayDataType::validate`] implementation checks:
//! - alignment of the byte buffer
//! - exact byte length for fixed-width types
//!
//! Variable-width types (e.g. [`Utf8`]) must override `validate` to enforce their encoding.
//!
//! ## ndarray views
//! Most fixed-width types can return **zero-copy** `ndarray` views by reinterpreting the byte
//! buffer as a slice of native elements.
//!
//! [`Utf8`] is variable-width and returns an `ndarray` backed by a `Vec<&str>` (string *references*
//! into the payload), so it allocates for the index vector even though it does not copy the
//! underlying string bytes.
//!
//!
//!
use crate::{
    dtype::DataType,
    variable::array::{error::ArrayValidationError, util::num_elements},
};

pub trait ArrayDataType: Default + Clone {
    /// The element type produced when viewing bytes for a particular lifetime.
    ///
    /// Most scalar types use `T` itself; [`Utf8`] uses `&'a str`.
    type Native<'a>: Clone
    where
        Self: 'a;

    /// The logical datatype for this array.
    const TYPE: DataType;

    /// Required alignment for the backing byte buffer.
    const ALIGNMENT: usize;

    /// Fixed byte width for fixed-width types.
    ///
    /// `None` indicates a variable-width encoding (e.g. [`Utf8`]).
    const BYTE_WIDTH: Option<usize>;

    /// Validate `bytes` for `shape`.
    ///
    /// The default implementation validates:
    /// - `bytes` alignment (`bytes.as_ptr() % ALIGNMENT == 0`)
    /// - exact byte length for fixed-width types (`product(shape) * BYTE_WIDTH`)
    ///
    /// Variable-width types should override this to validate their encoding.
    fn validate(bytes: &[u8], shape: &[usize]) -> Result<(), ArrayValidationError> {
        if Self::ALIGNMENT != 0 && (bytes.as_ptr() as usize) % Self::ALIGNMENT != 0 {
            return Err(ArrayValidationError::Misaligned {
                alignment: Self::ALIGNMENT,
            });
        }

        if let Some(width) = Self::BYTE_WIDTH {
            let n = num_elements(shape)?;
            let expected = n
                .checked_mul(width)
                .ok_or(ArrayValidationError::ShapeOverflow)?;
            if bytes.len() != expected {
                return Err(ArrayValidationError::WrongByteLen {
                    expected,
                    actual: bytes.len(),
                });
            }
        }

        Ok(())
    }

    /// Convert raw bytes into an `ndarray` view.
    ///
    /// This is typically **zero-copy** for fixed-width types.
    /// For [`Utf8`], this allocates a `Vec<&str>` holding string *references* into the payload.
    fn as_array<'a>(
        bytes: &'a [u8],
        shape: &'a [usize],
    ) -> ndarray::CowArray<'a, Self::Native<'a>, ndarray::IxDyn>;
    fn as_array_mut<'a>(
        _bytes: &'a mut [u8],
        _shape: &'a [usize],
    ) -> Option<ndarray::ArrayViewMut<'a, Self::Native<'a>, ndarray::IxDyn>> {
        None
    }
}

macro_rules! impl_array_datatype_numeric {
    ($ty:ty, $dtype:ident) => {
        impl ArrayDataType for $ty {
            type Native<'a> = $ty;
            const TYPE: DataType = DataType::$dtype;
            const ALIGNMENT: usize = std::mem::align_of::<$ty>();
            const BYTE_WIDTH: Option<usize> = Some(std::mem::size_of::<$ty>());

            fn as_array<'a>(
                bytes: &'a [u8],
                shape: &'a [usize],
            ) -> ndarray::CowArray<'a, Self::Native<'a>, ndarray::IxDyn> {
                // Safety: caller guarantees `bytes` is properly aligned for `$ty` and that
                // its length is a multiple of `size_of::<$ty>()`.
                let data: &'a [$ty] = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const $ty,
                        bytes.len() / std::mem::size_of::<$ty>(),
                    )
                };

                let shape = ndarray::IxDyn(shape);
                match ndarray::ArrayView::from_shape(shape, data) {
                    Ok(view) => view.into(),
                    Err(e) => panic!("invalid ndarray shape for validated array: {e}"),
                }
            }

            fn as_array_mut<'a>(
                bytes: &'a mut [u8],
                shape: &'a [usize],
            ) -> Option<ndarray::ArrayViewMut<'a, Self::Native<'a>, ndarray::IxDyn>> {
                let data: &'a mut [$ty] = unsafe {
                    std::slice::from_raw_parts_mut(
                        bytes.as_mut_ptr() as *mut $ty,
                        bytes.len() / std::mem::size_of::<$ty>(),
                    )
                };

                let shape = ndarray::IxDyn(shape);
                ndarray::ArrayViewMut::from_shape(shape, data).ok()
            }
        }
    };
}

impl ArrayDataType for bool {
    type Native<'a> = bool;
    const TYPE: DataType = DataType::Bool;
    const ALIGNMENT: usize = std::mem::align_of::<bool>();
    const BYTE_WIDTH: Option<usize> = Some(1);

    fn validate(bytes: &[u8], shape: &[usize]) -> Result<(), ArrayValidationError> {
        #[allow(clippy::modulo_one)] // for the case ALIGNMENT=1
        if Self::ALIGNMENT != 0 && (bytes.as_ptr() as usize) % Self::ALIGNMENT != 0 {
            return Err(ArrayValidationError::Misaligned {
                alignment: Self::ALIGNMENT,
            });
        }

        let n = num_elements(shape)?;
        if bytes.len() != n {
            return Err(ArrayValidationError::WrongByteLen {
                expected: n,
                actual: bytes.len(),
            });
        }

        if bytes.iter().any(|&b| b != 0 && b != 1) {
            return Err(ArrayValidationError::InvalidBoolByte);
        }

        Ok(())
    }

    fn as_array<'a>(
        bytes: &'a [u8],
        shape: &'a [usize],
    ) -> ndarray::CowArray<'a, Self::Native<'a>, ndarray::IxDyn> {
        // Safety: caller guarantees `bytes` is a valid bool buffer (0/1) and
        // properly aligned for `bool`.
        let bools: &'a [bool] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const bool, bytes.len()) };
        let shape = ndarray::IxDyn(shape);
        match ndarray::ArrayView::from_shape(shape, bools) {
            Ok(view) => view.into(),
            Err(e) => panic!("invalid ndarray shape for validated array: {e}"),
        }
    }
    fn as_array_mut<'a>(
        bytes: &'a mut [u8],
        shape: &'a [usize],
    ) -> Option<ndarray::ArrayViewMut<'a, Self::Native<'a>, ndarray::IxDyn>> {
        let bools: &'a mut [bool] =
            unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut bool, bytes.len()) };
        let shape = ndarray::IxDyn(shape);
        ndarray::ArrayViewMut::from_shape(shape, bools).ok()
    }
}

/// Marker type for UTF-8 inline string arrays.
#[repr(transparent)]
#[derive(Clone)]
pub struct Utf8(String);

impl Default for Utf8 {
    fn default() -> Self {
        Utf8(String::new())
    }
}

impl ArrayDataType for Utf8 {
    type Native<'a> = &'a str;
    const TYPE: DataType = DataType::Utf8;
    // The inline string representation starts with an array of u32 lengths.
    const ALIGNMENT: usize = std::mem::align_of::<u32>();
    const BYTE_WIDTH: Option<usize> = None;

    fn validate(bytes: &[u8], shape: &[usize]) -> Result<(), ArrayValidationError> {
        if Self::ALIGNMENT != 0 && (bytes.as_ptr() as usize) % Self::ALIGNMENT != 0 {
            return Err(ArrayValidationError::Misaligned {
                alignment: Self::ALIGNMENT,
            });
        }

        let n = num_elements(shape)?;
        let lengths_bytes = n
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(ArrayValidationError::ShapeOverflow)?;
        if bytes.len() < lengths_bytes {
            return Err(ArrayValidationError::WrongPrefixLen {
                expected: lengths_bytes,
                actual: bytes.len(),
            });
        }

        // Safety: the prefix is validated for size and alignment above.
        let lengths: &[u32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, n) };

        let payload = &bytes[lengths_bytes..];
        let mut offset = 0usize;
        for &len in lengths {
            let len = len as usize;
            let end = offset
                .checked_add(len)
                .ok_or(ArrayValidationError::LengthsOverflow)?;
            if end > payload.len() {
                return Err(ArrayValidationError::PayloadOutOfBounds);
            }
            if std::str::from_utf8(&payload[offset..end]).is_err() {
                return Err(ArrayValidationError::InvalidUtf8);
            }
            offset = end;
        }

        if lengths_bytes + offset != bytes.len() {
            return Err(ArrayValidationError::WrongByteLen {
                expected: lengths_bytes + offset,
                actual: bytes.len(),
            });
        }

        Ok(())
    }

    fn as_array<'a>(
        bytes: &'a [u8],
        shape: &'a [usize],
    ) -> ndarray::CowArray<'a, Self::Native<'a>, ndarray::IxDyn> {
        let n = match num_elements(shape) {
            Ok(n) => n,
            Err(e) => panic!("invalid shape for array view: {e}"),
        };
        let lengths_bytes = match n.checked_mul(std::mem::size_of::<u32>()) {
            Some(v) => v,
            None => panic!("invalid shape for array view: element count overflow"),
        };

        // Safety: `Array::new`/`try_new` validate alignment and the lengths prefix.
        let lengths: &'a [u32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, n) };
        let payload = &bytes[lengths_bytes..];

        let mut offset = 0usize;
        let mut out: Vec<&'a str> = Vec::with_capacity(n);
        for &len in lengths {
            let len = len as usize;
            let end = match offset.checked_add(len) {
                Some(v) => v,
                None => panic!("invalid utf8 lengths for array view"),
            };
            // Safety: `Array::new`/`try_new` validate each string span is valid UTF-8.
            let s = unsafe { std::str::from_utf8_unchecked(&payload[offset..end]) };
            out.push(s);
            offset = end;
        }

        match ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), out) {
            Ok(arr) => arr.into(),
            Err(e) => panic!("invalid ndarray shape for validated array: {e}"),
        }
    }
}

/// Marker type for timestamp arrays stored as i64 nanoseconds since UNIX epoch.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Timestamp(i64);

impl Default for Timestamp {
    fn default() -> Self {
        Timestamp(0)
    }
}

impl_array_datatype_numeric!(i8, I8);
impl_array_datatype_numeric!(i16, I16);
impl_array_datatype_numeric!(i32, I32);
impl_array_datatype_numeric!(i64, I64);
impl_array_datatype_numeric!(u8, U8);
impl_array_datatype_numeric!(u16, U16);
impl_array_datatype_numeric!(u32, U32);
impl_array_datatype_numeric!(u64, U64);
impl_array_datatype_numeric!(f32, F32);
impl_array_datatype_numeric!(f64, F64);
impl_array_datatype_numeric!(Timestamp, I64);

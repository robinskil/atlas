use smallvec::SmallVec;

use crate::{
    dtype::DataType,
    variable::array::{datatype::ArrayDataType, error::ArrayValidationError},
};

pub mod blob;
pub mod datatype;
pub mod error;
pub mod fill_value;
pub mod metadata;
pub mod reader;
pub mod util;
pub mod writer;

/// A typed array backed by a contiguous byte buffer.
///
/// Construction validates the buffer against `T` and `shape` (unless using
/// [`Array::new_unchecked`]). After validation, [`Array::as_ndarray`] creates
/// an `ndarray` view.
pub struct Array<T: ArrayDataType> {
    pub data: bytes::Bytes,
    pub shape: SmallVec<[usize; 4]>,
    _type: std::marker::PhantomData<T>,
}

impl<T: ArrayDataType> Array<T> {
    /// Create a validated array.
    ///
    /// Panics if `data` does not match `shape` and `T`'s expected layout.
    pub fn new(data: bytes::Bytes, shape: SmallVec<[usize; 4]>) -> Self {
        T::validate(&data, &shape).expect("invalid array bytes/shape");
        Self {
            data,
            shape,
            _type: std::marker::PhantomData,
        }
    }

    /// Create an array without validating bytes/shape.
    ///
    /// # Safety
    /// The caller must ensure that `data` matches `shape` and `T`'s expected layout.
    /// In particular, this must uphold all invariants relied on by [`ArrayDataType::as_array`],
    /// including alignment and length constraints, and (for [`Utf8`]) valid UTF-8.
    pub unsafe fn new_unchecked(data: bytes::Bytes, shape: SmallVec<[usize; 4]>) -> Self {
        Self {
            data,
            shape,
            _type: std::marker::PhantomData,
        }
    }

    /// Create a validated array, returning a structured error instead of panicking.
    pub fn try_new(
        data: bytes::Bytes,
        shape: SmallVec<[usize; 4]>,
    ) -> Result<Self, ArrayValidationError> {
        T::validate(&data, &shape)?;
        Ok(Self {
            data,
            shape,
            _type: std::marker::PhantomData,
        })
    }

    pub fn as_ndarray<'a>(&'a self) -> ndarray::CowArray<'a, T::Native<'a>, ndarray::IxDyn> {
        T::as_array(&self.data, &self.shape)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub const fn data_type(&self) -> DataType {
        T::TYPE
    }

    pub const fn alignment(&self) -> usize {
        T::ALIGNMENT
    }
}

#[cfg(test)]
mod tests {
    use crate::variable::array::{datatype::Utf8, error::ArrayValidationError};

    use super::*;
    use bytes::Bytes;

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
        let err = match Array::<i16>::try_new(data, smallvec::smallvec![2]) {
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

        let err = match Array::<i32>::try_new(b, smallvec::smallvec![1]) {
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
        let arr = Array::<i32>::try_new(b, smallvec::smallvec![values.len()]).unwrap();
        let got: Vec<i32> = arr.as_ndarray().iter().copied().collect();
        assert_eq!(got, values);
    }

    #[test]
    fn bool_try_new_rejects_non_0_1() {
        let b = Bytes::from(vec![0u8, 2u8, 1u8]);
        let err = match Array::<bool>::try_new(b, smallvec::smallvec![3]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(matches!(err, ArrayValidationError::InvalidBoolByte));
    }

    #[test]
    fn bool_try_new_rejects_wrong_len() {
        let b = Bytes::from(vec![0u8, 1u8]);
        let err = match Array::<bool>::try_new(b, smallvec::smallvec![3]) {
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
        let arr = Array::<bool>::try_new(b, smallvec::smallvec![4]).unwrap();
        let got: Vec<bool> = arr.as_ndarray().iter().copied().collect();
        assert_eq!(got, vec![false, true, false, true]);
    }

    #[test]
    fn utf8_try_new_and_decode() {
        let bytes = utf8_inline_bytes(&["hi", "", "world"]);
        let arr = Array::<Utf8>::try_new(Bytes::from(bytes), smallvec::smallvec![3]).unwrap();
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

        let err = match Array::<Utf8>::try_new(Bytes::from(bytes), smallvec::smallvec![1]) {
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

        let err = match Array::<Utf8>::try_new(Bytes::from(bytes), smallvec::smallvec![1]) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(matches!(err, ArrayValidationError::PayloadOutOfBounds));
    }

    #[test]
    fn utf8_validate_rejects_trailing_bytes() {
        let mut bytes = utf8_inline_bytes(&["a"]);
        bytes.push(0);

        let err = match Array::<Utf8>::try_new(Bytes::from(bytes), smallvec::smallvec![1]) {
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
        let _arr = unsafe { Array::<i32>::new_unchecked(data, smallvec::smallvec![999]) };
    }

    #[test]
    #[should_panic(expected = "invalid array bytes/shape")]
    fn new_panics_on_invalid() {
        let data = Bytes::from(vec![0u8; 1]);
        let _ = Array::<i32>::new(data, smallvec::smallvec![1]);
    }
}

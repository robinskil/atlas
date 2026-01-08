use crate::{
    dtype::DataType,
    variable::array::{error::ArrayValidationError, fill_value::FillValue, util::num_elements},
};

pub trait ArrayDataType {
    /// The element type produced when viewing bytes for a particular lifetime.
    ///
    /// Most scalar types use `T` itself; [`Utf8`] uses `&'a str`.
    type Native<'a>
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
                ndarray::ArrayView::from_shape(shape, data).unwrap().into()
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
        ndarray::ArrayView::from_shape(shape, bools).unwrap().into()
    }
}

/// Marker type for UTF-8 inline string arrays.
pub struct Utf8;

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
        let n: usize = shape.iter().product();
        let lengths_bytes = n * std::mem::size_of::<u32>();

        // Safety: `Array::new`/`try_new` validate alignment and the lengths prefix.
        let lengths: &'a [u32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, n) };
        let payload = &bytes[lengths_bytes..];

        let mut offset = 0usize;
        let mut out: Vec<&'a str> = Vec::with_capacity(n);
        for &len in lengths {
            let len = len as usize;
            let end = offset + len;
            // Safety: `Array::new`/`try_new` validate each string span is valid UTF-8.
            let s = unsafe { std::str::from_utf8_unchecked(&payload[offset..end]) };
            out.push(s);
            offset = end;
        }

        ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), out)
            .unwrap()
            .into()
    }
}

/// Marker type for timestamp arrays stored as i64 nanoseconds since UNIX epoch.
#[repr(transparent)]
pub struct Timestamp(i64);

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

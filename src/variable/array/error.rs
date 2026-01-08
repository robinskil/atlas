use thiserror::Error;

/// Errors returned by [`Array::try_new`] / [`ArrayDataType::validate`].
#[derive(Debug, Error)]
pub enum ArrayValidationError {
    /// `product(shape)` overflowed `usize`.
    #[error("shape element count overflow")]
    ShapeOverflow,
    /// The byte buffer pointer is not aligned to the required alignment.
    #[error("byte buffer is not aligned to {alignment} bytes")]
    Misaligned { alignment: usize },
    /// The byte buffer length doesn't match the expected size.
    #[error("wrong byte length: expected {expected}, got {actual}")]
    WrongByteLen { expected: usize, actual: usize },
    /// For variable-width types: the fixed prefix/header is missing.
    #[error("missing/short header: expected {expected} bytes, got {actual}")]
    WrongPrefixLen { expected: usize, actual: usize },
    /// `bool` arrays must be encoded as 0/1 bytes.
    #[error("invalid bool byte (expected 0 or 1)")]
    InvalidBoolByte,
    /// UTF-8 validation failed.
    #[error("invalid UTF-8")]
    InvalidUtf8,
    /// Summing lengths overflowed `usize`.
    #[error("string length sum overflow")]
    LengthsOverflow,
    /// A declared string length exceeded the available payload bytes.
    #[error("string payload out of bounds")]
    PayloadOutOfBounds,
}

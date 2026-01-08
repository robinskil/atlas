pub const MAGIC_NUMBER: &[u8; 4] = b"ATLS";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Directories for variable arrays:
///     variable_name/array/
///     variable_name/array_chunked/
///     variable_name/array_meta/
///
pub const VARIABLE_ARRAY_DIR: &str = "array/";
pub const VARIABLE_ARRAY_CHUNKED_DIR: &str = "array_chunked/";
pub const VARIABLE_ARRAY_META_DIR: &str = "array_meta/";

/// Directories for variable attributes:
///     variable_name/attributes/
///     variable_name/attributes_meta/
///
pub const VARIABLE_ATTRIBUTES_DIR: &str = "attributes/";
pub const VARIABLE_ATTRIBUTES_META_DIR: &str = "attributes_meta/";

pub const MAGIC_NUMBER_COMPRESSED: &[u8; 8] = b"ATLCZSTD";
pub const MAGIC_NUMBER_UNCOMPRESSED: &[u8; 8] = b"ATLU____";

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

/// Schema directories for the collection and individual datasets
/// collection.atlas
///     ├── schema/
///         ├── base_schema.json
///         ├── wal_<unix_timestamp>.json
///
pub const SCHEMA_DIR: &str = "schema/";
pub const SCHEMA_FILE_CACHED: &str = "collection_schema_cached.json";
pub const DATASETS_SCHEMA_DIR: &str = "datasets/";

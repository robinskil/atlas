//! Layout
//!
//! ```text
//! collection.atlas/
//!         ├── datasets/
//!
//!         ├── variables/
//!             ├── variable_0/
//!                 variable.json
//!                 array_meta/
//!                     array_meta_0_1000.json
//!                 array/
//!                     blob_0.bin
//!                 array_chunked/
//!                    1/2/3/blob.bin
//!                 attributes_meta/
//!                    attributes_meta_0_1000.json
//!                 attributes/
//!                    blob_0.bin
//! ```

pub mod array;
pub mod attributes;
pub mod blob_encoding;
pub mod consts;
pub mod data_blob;
pub mod metadata_blob;
pub mod reader;
pub mod util;
pub mod writer;

use crate::{dtype::DataType, variable::blob_encoding::EncodingType};

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct CollectionVariable {
    pub name: String,
    pub datatype: DataType,
    pub data_blob_encoding: EncodingType,
    pub metadata_blob_encoding: EncodingType,
}

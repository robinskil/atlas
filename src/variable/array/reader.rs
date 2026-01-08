//! Variable array reader.
//!
//! This module contains [`ArrayObjectReader`], an `object_store`-backed reader for variable arrays
//! and their metadata.
//!
//! ## Layout
//! This reader expects the same layout as [`crate::variable::array::writer::ArrayObjectWriter`]:
//! - `array_meta/*.jsonl` contains JSONL [`ArrayMetadata`] entries (one per line).
//! - `array/*` contains [`ArrayBlob`](crate::variable::array::blob::ArrayBlob) objects, named as
//!   `start_end` (both inclusive) describing allocation-index ranges.

use object_store::ObjectStore;

use crate::{
    dtype::DataType,
    variable::array::{
        Array, Immutable,
        blob::{ArrayBlob, ArrayBlobMetadata, Committed},
        datatype::ArrayDataType,
        discovery::{self, ArrayDiscoveryError},
        error::ArrayValidationError,
        metadata::ArrayMetadata,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum ArrayReadError {
    #[error(transparent)]
    ObjectStore(#[from] object_store::Error),

    #[error(transparent)]
    Blob(#[from] crate::variable::array::blob::ArrayBlobError),

    #[error(transparent)]
    Json(#[from] simd_json::Error),

    #[error(transparent)]
    InvalidArray(#[from] ArrayValidationError),

    #[error("dataset_index {dataset_index} out of bounds (len={len})")]
    DatasetIndexOutOfBounds { dataset_index: usize, len: usize },

    #[error("variable metadata not defined for dataset index {dataset_index}")]
    VariableMetadataMissing { dataset_index: usize },

    #[error("array has no allocation for dataset index {dataset_index}")]
    AllocationMissing { dataset_index: usize },

    #[error("data type mismatch when reading array: expected {expected:?}, got {actual:?}")]
    DataTypeMismatch {
        expected: DataType,
        actual: DataType,
    },

    #[error("no blob metadata found for allocation index {allocation_index}")]
    BlobMetadataNotFound { allocation_index: usize },

    #[error("allocation index {allocation_index} not present in blob")]
    AllocationIndexMissing { allocation_index: usize },
}

impl From<ArrayDiscoveryError> for ArrayReadError {
    fn from(value: ArrayDiscoveryError) -> Self {
        match value {
            ArrayDiscoveryError::ObjectStore(err) => Self::ObjectStore(err),
            ArrayDiscoveryError::Json(err) => Self::Json(err),
            ArrayDiscoveryError::DatasetIndexOutOfBounds { dataset_index, len } => {
                Self::DatasetIndexOutOfBounds { dataset_index, len }
            }
        }
    }
}

pub struct ArrayObjectReader<S: ObjectStore> {
    store: S,
    variable_dir: object_store::path::Path,
    array_metadatas: Vec<Option<Box<ArrayMetadata>>>,
    array_blob_metadatas: Vec<ArrayBlobMetadata>,
}

impl<S: ObjectStore> ArrayObjectReader<S> {
    /// Create a reader with empty in-memory metadata.
    ///
    /// This does **not** scan the store. Use [`Self::load_from_store`] or
    /// [`Self::load_from_store_infer`] to hydrate existing state.
    pub fn new(store: S, variable_dir: object_store::path::Path) -> Self {
        Self {
            store,
            variable_dir,
            array_metadatas: Vec::new(),
            array_blob_metadatas: Vec::new(),
        }
    }

    /// Create a reader pre-sized for `datasets_count` datasets.
    pub fn with_datasets_count(
        store: S,
        variable_dir: object_store::path::Path,
        datasets_count: usize,
    ) -> Self {
        Self {
            store,
            variable_dir,
            array_metadatas: vec![None; datasets_count],
            array_blob_metadatas: Vec::new(),
        }
    }

    /// Hydrate a reader from the store with an explicit dataset count.
    pub async fn load_from_store(
        store: S,
        variable_dir: object_store::path::Path,
        datasets_count: usize,
    ) -> Result<Self, ArrayReadError> {
        let array_metadatas =
            discovery::read_array_metadatas(&store, &variable_dir, datasets_count).await?;
        let array_blob_metadatas =
            discovery::read_array_blob_metadatas(&store, &variable_dir).await?;
        Ok(Self {
            store,
            variable_dir,
            array_metadatas,
            array_blob_metadatas,
        })
    }

    /// Hydrate a reader from the store by inferring the dataset count.
    ///
    /// This scans `array_meta/*.jsonl` for the max `dataset_index` and sets
    /// `datasets_count = max + 1`.
    pub async fn load_from_store_infer(
        store: S,
        variable_dir: object_store::path::Path,
    ) -> Result<Self, ArrayReadError> {
        let datasets_count =
            discovery::infer_datasets_count_from_store(&store, &variable_dir).await?;
        Self::load_from_store(store, variable_dir, datasets_count).await
    }

    /// Read the current array for `dataset_index` and validate it as `T`.
    pub async fn read_array<T: ArrayDataType>(
        &self,
        dataset_index: usize,
    ) -> Result<Array<T>, ArrayReadError> {
        if dataset_index >= self.array_metadatas.len() {
            return Err(ArrayReadError::DatasetIndexOutOfBounds {
                dataset_index,
                len: self.array_metadatas.len(),
            });
        }

        let metadata = self.array_metadatas[dataset_index]
            .as_ref()
            .ok_or(ArrayReadError::VariableMetadataMissing { dataset_index })?;

        if metadata.data_type != T::TYPE {
            return Err(ArrayReadError::DataTypeMismatch {
                expected: metadata.data_type,
                actual: T::TYPE,
            });
        }

        let Some(allocation_index) = metadata.allocation_index else {
            return Err(ArrayReadError::AllocationMissing { dataset_index });
        };

        let blob_metadata = discovery::find_array_blob_for_allocation_index(
            &self.array_blob_metadatas,
            allocation_index,
        )
        .ok_or(ArrayReadError::BlobMetadataNotFound { allocation_index })?;

        let blob = ArrayBlob::<Committed, bytes::Bytes>::try_from_existing(
            &self.store,
            &blob_metadata.path,
        )
        .await?;
        let blob_allocation_index = allocation_index - blob_metadata.range.start();
        let allocation_bytes = blob
            .get_allocation_bytes(blob_allocation_index)
            .ok_or(ArrayReadError::AllocationIndexMissing { allocation_index })?;

        Ok(Array::<T>::try_new(
            Immutable(allocation_bytes),
            metadata.shape.clone(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::array::{Utf8, writer::ArrayObjectWriter};
    use object_store::memory::InMemory;
    use std::sync::Arc;

    fn variable_dir() -> object_store::path::Path {
        object_store::path::Path::from("variables/variable_0")
    }

    #[tokio::test]
    async fn reader_reads_u8_roundtrip() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let variable_dir = variable_dir();

        let mut writer = ArrayObjectWriter::new(store.clone(), variable_dir.clone());
        let arr = Array::<u8>::try_from_shape_and_slice(&[4], &[1, 2, 3, 4]).unwrap();
        writer
            .define_variable_array::<u8, _, _>(0, &[4], &["x"], None, None, Some(arr))
            .await
            .unwrap();

        let reader = ArrayObjectReader::load_from_store_infer(store, variable_dir)
            .await
            .unwrap();
        let got = reader.read_array::<u8>(0).await.unwrap();
        let got_vals: Vec<u8> = got.as_ndarray().iter().copied().collect();
        assert_eq!(got_vals, vec![1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn reader_reads_utf8_roundtrip() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let variable_dir = variable_dir();

        let mut writer = ArrayObjectWriter::new(store.clone(), variable_dir.clone());
        let arr = Array::<Utf8>::try_from_shape_and_strs(&[2], &["a", "bb"]).unwrap();
        writer
            .define_variable_array::<Utf8, _, _>(0, &[2], &["x"], None, None, Some(arr))
            .await
            .unwrap();

        let reader = ArrayObjectReader::load_from_store_infer(store, variable_dir)
            .await
            .unwrap();
        let got = reader.read_array::<Utf8>(0).await.unwrap();
        let got_vals: Vec<&str> = got.as_ndarray().iter().copied().collect();
        assert_eq!(got_vals, vec!["a", "bb"]);
    }
}

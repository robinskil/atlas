//! Variable array writer.
//!
//! This module contains [`ArrayObjectWriter`], an `object_store`-backed writer for variable arrays
//! and their metadata.
//!
//! ## Layout
//! The writer stores variable-array state under a per-variable directory:
//! ```text
//! collection.atlas/
//!   ├── datasets/
//!   │   ...
//!   └── variables/
//!       └── variable_0/
//!           ├── variable.json
//!           ├── array_meta/
//!           │   ├── 12384358.jsonl
//!           │   └── 12384399.jsonl (timestamp-based filename)
//!           ├── array/
//!           │   ├── 0_250
//!           │   └── 251_251
//!           └── array_chunked/ (not yet implemented)
//! ```
//!
//! `array_meta/*.jsonl` are JSONL files (one [`ArrayMetadata`] per line). The `array/*` objects are
//! [`ArrayBlob`](crate::variable::array::blob::ArrayBlob) files containing one or more allocations.
//!
//! ## Construction
//! - [`ArrayObjectWriter::new`]: start with empty in-memory metadata.
//! - [`ArrayObjectWriter::with_datasets_count`]: start with a known dataset count.
//! - [`ArrayObjectWriter::load_from_store`]: hydrate metadata from the store using an explicit
//!   `datasets_count`.
//! - [`ArrayObjectWriter::load_from_store_infer`]: hydrate metadata from the store by scanning
//!   `array_meta/*.jsonl` for the maximum `dataset_index`.
//!
//! ## Notes
//! - Partial writes are supported for fixed-width types (non-UTF8) when updating an existing
//!   allocation.
//! - For non-UTF8 data types, writes may mutate existing allocations in-place.
//! - For [`DataType::Utf8`], updates always create a new allocation.
//!

use bytes::BytesMut;
use chrono::Utc;
use object_store::{ObjectStore, PutPayload};
use smallvec::SmallVec;

use crate::{
    consts,
    dtype::DataType,
    variable::array::{
        Array,
        blob::{ArrayBlob, ArrayBlobMetadata, Committed, Uncommitted},
        datatype::ArrayDataType,
        discovery::{self, ArrayDiscoveryError},
        fill_value::FillValue,
        metadata::ArrayMetadata,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum ArrayWriteError {
    #[error(transparent)]
    ObjectStore(#[from] object_store::Error),

    #[error(transparent)]
    Blob(#[from] crate::variable::array::blob::ArrayBlobError),

    #[error(transparent)]
    Json(#[from] simd_json::Error),

    #[error("dataset_index {dataset_index} out of bounds (len={len})")]
    DatasetIndexOutOfBounds { dataset_index: usize, len: usize },

    #[error("variable metadata not defined for dataset index {dataset_index}")]
    VariableMetadataMissing { dataset_index: usize },

    #[error("data type mismatch when writing array: expected {expected:?}, got {actual:?}")]
    DataTypeMismatch {
        expected: DataType,
        actual: DataType,
    },

    #[error("no blob metadata found for allocation index {allocation_index}")]
    BlobMetadataNotFound { allocation_index: usize },

    #[error("partial writes are not supported")]
    PartialWriteUnsupported,

    #[error("shape mismatch for in-place write")]
    ShapeMismatch,

    #[error("rank mismatch for in-place write: start has len {start_len}, expected rank {rank}")]
    StartRankMismatch { start_len: usize, rank: usize },

    #[error("rank mismatch for in-place write: data has rank {data_rank}, expected rank {rank}")]
    DataRankMismatch { data_rank: usize, rank: usize },

    #[error(
        "slice out of bounds in dimension {dim}: start={start}, len={len}, allocated={allocated}"
    )]
    SliceOutOfBounds {
        dim: usize,
        start: usize,
        len: usize,
        allocated: usize,
    },

    #[error("cannot get a mutable ndarray view for this array type")]
    MutableNdarrayViewUnavailable,

    #[error("shape mismatch for new allocation")]
    NewAllocationShapeMismatch,

    #[error("allocation byte length mismatch for in-place write")]
    AllocationLenMismatch,
}

impl From<ArrayDiscoveryError> for ArrayWriteError {
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

pub struct ArrayObjectWriter<S: ObjectStore> {
    store: S,
    variable_dir: object_store::path::Path,
    array_metadatas: Vec<Option<Box<ArrayMetadata>>>,
    array_blob_metadatas: Vec<ArrayBlobMetadata>,
}

pub use discovery::find_array_blob_for_allocation_index;

impl<S: ObjectStore> ArrayObjectWriter<S> {
    /// Create a writer with empty in-memory metadata.
    ///
    /// This does **not** scan the store. Use [`Self::load_from_store`] or
    /// [`Self::load_from_store_infer`] if you want to hydrate existing state.
    ///
    /// # Arguments
    /// - `store`: Object store to write to.
    /// - `variable_dir`: Root directory for the variable.
    pub fn new(store: S, variable_dir: object_store::path::Path) -> Self {
        Self {
            store,
            variable_dir,
            array_metadatas: Vec::new(),
            array_blob_metadatas: Vec::new(),
        }
    }

    /// Create a writer pre-sized for `datasets_count` datasets.
    ///
    /// # Arguments
    /// - `store`: Object store to write to.
    /// - `variable_dir`: Root directory for the variable.
    /// - `datasets_count`: Number of datasets.
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

    /// Load writer state from the store.
    ///
    /// This reads both `array_meta/*.jsonl` and `array/*` to populate in-memory metadata.
    ///
    /// # Arguments
    /// - `store`: Object store to read/write.
    /// - `variable_dir`: Root directory for the variable.
    /// - `datasets_count`: Number of datasets expected.
    ///
    /// # Errors
    /// Returns an error if any metadata object cannot be listed/read/parsed.
    pub async fn load_from_store(
        store: S,
        variable_dir: object_store::path::Path,
        datasets_count: usize,
    ) -> Result<Self, ArrayWriteError> {
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

    /// Load writer state from the store, inferring `datasets_count` by scanning metadata.
    ///
    /// This scans `array_meta/*.jsonl` for the maximum `dataset_index` and sets
    /// `datasets_count = max + 1`. If no metadata exists, the inferred count is `0`.
    ///
    /// # Errors
    /// Returns an error if any metadata object cannot be listed/read/parsed.
    pub async fn load_from_store_infer(
        store: S,
        variable_dir: object_store::path::Path,
    ) -> Result<Self, ArrayWriteError> {
        let datasets_count =
            discovery::infer_datasets_count_from_store(&store, &variable_dir).await?;
        Self::load_from_store(store, variable_dir, datasets_count).await
    }

    /// Define (or redefine) a variable array for a dataset index.
    ///
    /// This writes an [`ArrayMetadata`] entry to `array_meta/*.jsonl`, updates the in-memory
    /// metadata, and optionally writes the array payload.
    ///
    /// # Arguments
    /// - `dataset_index`: Dataset index to define.
    /// - `shape`: Logical array shape.
    /// - `dimensions`: Dimension names.
    /// - `fill_value`: Optional fill value; defaults to `None`.
    /// - `start`: Optional write start indices. If `None`, defaults to all zeros.
    /// - `array`: Optional initial array payload to write.
    ///
    /// # Errors
    /// Returns an error if writing metadata or the array payload fails.
    pub async fn define_variable_array<T: ArrayDataType, B: AsRef<[u8]>, D: AsRef<str>>(
        &mut self,
        dataset_index: usize,
        shape: &[usize],
        dimensions: &[D],
        fill_value: Option<FillValue>,
        start: Option<&[usize]>,
        array: Option<Array<T, B>>,
    ) -> Result<(), ArrayWriteError> {
        if dataset_index >= self.array_metadatas.len() {
            self.array_metadatas.resize_with(dataset_index + 1, || None);
        }

        // Create ArrayMetadata
        let metadata = ArrayMetadata {
            dataset_index,
            allocation_index: None,
            data_type: T::TYPE,
            shape: SmallVec::from_slice(shape),
            dimensions: dimensions.iter().map(|d| d.as_ref().into()).collect(),
            chunk_shape: None,
            fill_value: fill_value.unwrap_or(FillValue::None),
        };

        // Serialize and write metadata as a JSON line to an object in the store
        // File name is the current unix timestamp in nanoseconds
        self.write_metadata(&metadata).await?;
        self.array_metadatas[dataset_index] = Some(Box::new(metadata));

        // If an array is provided, write the array data to the store
        if let Some(array) = array {
            self.write_array::<T, B>(dataset_index, &array, start)
                .await?;
        }

        Ok(())
    }

    /// Compute the object-store path for a new metadata JSONL file.
    fn generate_metadata_object_path(&self, timestamp_nanos: i64) -> object_store::path::Path {
        let metadata_dir = self.variable_dir.child(consts::VARIABLE_ARRAY_META_DIR);
        metadata_dir.child(format!("{}.jsonl", timestamp_nanos))
    }

    /// Compute the blob path for a blob containing allocations `[start_dataset_index, end_dataset_index]`.
    fn generate_blob_object_path(
        &self,
        start_dataset_index: usize,
        end_dataset_index: usize,
    ) -> object_store::path::Path {
        let blob_dir = self.variable_dir.child(consts::VARIABLE_ARRAY_DIR);
        blob_dir.child(format!("{}_{}", start_dataset_index, end_dataset_index))
    }

    /// Generate the next allocation index based on the current known blob ranges.
    fn generate_allocation_index(&self) -> usize {
        let max_range = self
            .array_blob_metadatas
            .iter()
            .max_by(|l, r| r.range.end().cmp(l.range.end()));
        match max_range {
            Some(blob_metadata) => *blob_metadata.range.end() + 1,
            None => 0,
        }
    }

    /// Write a single [`ArrayMetadata`] as JSONL to the metadata directory.
    ///
    /// # Errors
    /// Returns an error if JSON serialization or the object store put fails.
    async fn write_metadata(&self, metadata: &ArrayMetadata) -> Result<(), ArrayWriteError> {
        let timestamp_nanos = Utc::now()
            .timestamp_nanos_opt()
            .unwrap_or_else(|| Utc::now().timestamp_micros() * 1_000);
        let metadata_path = self.generate_metadata_object_path(timestamp_nanos);
        let json_bytes: bytes::Bytes = simd_json::serde::to_vec(&metadata)?.into();
        self.store
            .put(&metadata_path, PutPayload::from_bytes(json_bytes))
            .await?;
        Ok(())
    }

    /// Update an existing allocation.
    ///
    /// - If `start` is all zeros and `data.shape() == allocated_array_shape`, this performs a
    ///   fast full overwrite by copying raw bytes.
    /// - Otherwise, this performs a partial update by slicing the destination ndarray and
    ///   assigning from the source ndarray.
    ///
    /// # Errors
    /// Returns an error if the slice is out of bounds, ranks mismatch, or allocation byte lengths
    /// differ for a full overwrite.
    async fn mutate_in_place<T: ArrayDataType, B: AsRef<[u8]>>(
        array_blob: &mut ArrayBlob<Committed, BytesMut>,
        blob_allocation_index: usize,
        allocated_array_shape: &[usize],
        data: &Array<T, B>,
        start: &[usize],
    ) -> Result<(), ArrayWriteError> {
        let rank = allocated_array_shape.len();
        if start.len() != rank {
            return Err(ArrayWriteError::StartRankMismatch {
                start_len: start.len(),
                rank,
            });
        }
        if data.shape().len() != rank {
            return Err(ArrayWriteError::DataRankMismatch {
                data_rank: data.shape().len(),
                rank,
            });
        }

        for dim in 0..rank {
            let len = data.shape()[dim];
            let end = start[dim].saturating_add(len);
            if end > allocated_array_shape[dim] {
                return Err(ArrayWriteError::SliceOutOfBounds {
                    dim,
                    start: start[dim],
                    len,
                    allocated: allocated_array_shape[dim],
                });
            }
        }

        let mut dst = array_blob
            .get_array_mut::<T>(blob_allocation_index, allocated_array_shape.into())
            .ok_or(ArrayWriteError::BlobMetadataNotFound {
                allocation_index: blob_allocation_index,
            })?;

        // Full overwrite so we can copy directly.
        if start.iter().all(|&v| v == 0) && data.shape() == allocated_array_shape {
            let src_bytes = data.data.as_ref();
            let dst_bytes = dst.data.as_mut();
            if src_bytes.len() != dst_bytes.len() {
                return Err(ArrayWriteError::AllocationLenMismatch);
            }

            dst_bytes.copy_from_slice(src_bytes);
        } else {
            let mut nd_array = dst
                .as_mut_ndarray()
                .ok_or(ArrayWriteError::MutableNdarrayViewUnavailable)?;
            // Slice out the region to write to
            let mut slice_args = vec![];
            for i in 0..rank {
                slice_args.push(ndarray::SliceInfoElem::Slice {
                    start: start[i] as isize,
                    end: Some((start[i] + data.shape()[i]) as isize),
                    step: 1,
                });
            }
            nd_array
                .slice_mut(slice_args.as_slice())
                .assign(&data.as_ndarray());
        }

        Ok(())
    }

    /// Write (or update) an array payload for an already-defined dataset index.
    ///
    /// For fixed-width types, if an allocation already exists, this attempts to mutate the
    /// existing allocation in-place.
    ///
    /// For [`DataType::Utf8`], updates always create a new allocation (payload size can change).
    ///
    /// # Arguments
    /// - `dataset_index`: Target dataset index.
    /// - `array`: Array payload to write.
    /// - `start`: Optional write start indices. If `None`, defaults to all zeros.
    ///
    /// # Errors
    /// Returns an error if the dataset is out of bounds, metadata is missing, data types mismatch,
    /// partial writes are requested, or any object store / blob operation fails.
    pub async fn write_array<T: ArrayDataType, B: AsRef<[u8]>>(
        &mut self,
        dataset_index: usize,
        array: &Array<T, B>,
        start: Option<&[usize]>,
    ) -> Result<(), ArrayWriteError> {
        if dataset_index >= self.array_metadatas.len() {
            return Err(ArrayWriteError::DatasetIndexOutOfBounds {
                dataset_index,
                len: self.array_metadatas.len(),
            });
        }

        // Read metadata up-front (immutably) to avoid borrow conflicts.
        let (data_type, shape, allocation_index) =
            match self.array_metadatas[dataset_index].as_ref() {
                Some(metadata) => (
                    metadata.data_type,
                    metadata.shape.clone(),
                    metadata.allocation_index,
                ),
                None => return Err(ArrayWriteError::VariableMetadataMissing { dataset_index }),
            };

        // Default start to all zeros.
        let start_owned;
        let start = match start {
            Some(s) => s,
            None => {
                start_owned = vec![0usize; shape.len()];
                &start_owned
            }
        };

        if data_type != T::TYPE {
            return Err(ArrayWriteError::DataTypeMismatch {
                expected: data_type,
                actual: T::TYPE,
            });
        }

        if let Some(allocation_index) = allocation_index
            && !matches!(T::TYPE, DataType::Utf8)
        {
            // Write directly to the existing allocation
            let blob_metadata =
                find_array_blob_for_allocation_index(&self.array_blob_metadatas, allocation_index);
            let blob_metadata =
                blob_metadata.ok_or(ArrayWriteError::BlobMetadataNotFound { allocation_index })?;

            let mut array_blob_bytes = ArrayBlob::<Committed, BytesMut>::try_from_existing_mut(
                &self.store,
                &blob_metadata.path,
            )
            .await?;

            // Mutate in place
            let blob_allocation_index = allocation_index - blob_metadata.range.start();
            Self::mutate_in_place(
                &mut array_blob_bytes,
                blob_allocation_index,
                &shape,
                array,
                start,
            )
            .await?;

            // Commit the mutated blob back to the store
            array_blob_bytes
                .into_immutable_blob()
                .commit::<S>(&self.store)
                .await?;
        } else {
            if start.iter().any(|&v| v != 0) {
                // Creating a new allocation requires a full write.
                return Err(ArrayWriteError::PartialWriteUnsupported);
            }
            if array.shape() != shape.as_slice() {
                return Err(ArrayWriteError::NewAllocationShapeMismatch);
            }

            // Create new blob allocation
            let new_blob_index = self.generate_allocation_index();
            let blob_path = self.generate_blob_object_path(new_blob_index, new_blob_index);

            // Generate new mutable blob
            let mut mutable_blob = ArrayBlob::<Uncommitted, BytesMut>::new_mut();
            // Write array data into the mutable blob
            mutable_blob.write_array(array);
            // Commit the blob to the object store
            mutable_blob
                .into_immutable_blob()
                .commit::<S>(&self.store, &blob_path)
                .await?;

            // We have now written a new blob for this array
            // Now update the metadata to point to this new blob
            let array_blob_metadata = ArrayBlobMetadata {
                range: new_blob_index..=new_blob_index,
                path: blob_path,
            };
            self.array_blob_metadatas.push(array_blob_metadata);

            // Update the variable metadata with the new allocation index
            {
                let array_metadata = self.array_metadatas[dataset_index]
                    .as_mut()
                    .ok_or(ArrayWriteError::VariableMetadataMissing { dataset_index })?;
                array_metadata.allocation_index = Some(new_blob_index);
            }

            // Write updated metadata back to the store as new entry
            if let Some(array_metadata) = self.array_metadatas[dataset_index].as_ref() {
                self.write_metadata(array_metadata).await?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::array::Utf8;
    use object_store::{PutPayload, memory::InMemory};
    use smallvec::smallvec;

    fn variable_dir() -> object_store::path::Path {
        object_store::path::Path::from("variables/variable_0")
    }

    #[tokio::test]
    async fn load_from_store_infer_empty_store_is_zero() {
        let store = InMemory::new();
        let writer = ArrayObjectWriter::load_from_store_infer(store, variable_dir())
            .await
            .unwrap();
        assert_eq!(writer.array_metadatas.len(), 0);
        assert_eq!(writer.array_blob_metadatas.len(), 0);
    }

    #[tokio::test]
    async fn load_from_store_infer_scans_jsonl_and_sizes_metadatas() {
        let store = InMemory::new();
        let variable_dir = variable_dir();

        // Write a JSONL metadata object with dataset_index 3.
        let metadata_dir = variable_dir.child(consts::VARIABLE_ARRAY_META_DIR);
        let object_path = metadata_dir.child("1.jsonl");
        let jsonl = b"{\"dataset_index\":3,\"allocation_index\":null,\"data_type\":\"U8\",\"shape\":[2],\"dimensions\":[\"x\"],\"chunk_shape\":null,\"fill_value\":\"None\"}\n";
        store
            .put(
                &object_path,
                PutPayload::from_bytes(bytes::Bytes::from_static(jsonl)),
            )
            .await
            .unwrap();

        let writer = ArrayObjectWriter::load_from_store_infer(store, variable_dir)
            .await
            .unwrap();

        assert_eq!(writer.array_metadatas.len(), 4);
        assert!(writer.array_metadatas[3].is_some());
    }

    #[tokio::test]
    async fn read_array_blob_metadatas_skips_invalid_filenames() {
        let store = InMemory::new();
        let variable_dir = variable_dir();
        let blob_dir = variable_dir.child(consts::VARIABLE_ARRAY_DIR);

        // Create two objects, one with a valid "start_end" name and one invalid.
        store
            .put(
                &blob_dir.child("0_1"),
                PutPayload::from_bytes(bytes::Bytes::from_static(b"dummy")),
            )
            .await
            .unwrap();
        store
            .put(
                &blob_dir.child("not_a_range"),
                PutPayload::from_bytes(bytes::Bytes::from_static(b"dummy")),
            )
            .await
            .unwrap();

        let metadatas = discovery::read_array_blob_metadatas(&store, &variable_dir)
            .await
            .unwrap();
        assert_eq!(metadatas.len(), 1);
        assert!(metadatas[0].range.contains(&0));
        assert!(metadatas[0].range.contains(&1));
    }

    #[tokio::test]
    async fn define_variable_array_writes_u8_and_creates_blob() {
        let store = InMemory::new();
        let variable_dir = variable_dir();
        let mut writer = ArrayObjectWriter::with_datasets_count(store, variable_dir, 1);

        let arr = Array::<u8>::try_from_shape_and_slice(&[4], &[1, 2, 3, 4]).unwrap();
        writer
            .define_variable_array::<u8, _, _>(0, &[4], &["x"], None, None, Some(arr))
            .await
            .unwrap();

        let meta = writer.array_metadatas[0].as_ref().unwrap();
        assert_eq!(meta.allocation_index, Some(0));
        assert_eq!(writer.array_blob_metadatas.len(), 1);

        let blob_meta = &writer.array_blob_metadatas[0];
        let loaded =
            ArrayBlob::<Committed, bytes::Bytes>::try_from_existing(&writer.store, &blob_meta.path)
                .await
                .unwrap();

        let stored = loaded.get_array::<u8>(0, smallvec![4usize]).unwrap();
        assert_eq!(stored.data.as_ref(), &[1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn write_array_updates_existing_allocation_in_place_for_u8() {
        let store = InMemory::new();
        let variable_dir = variable_dir();
        let mut writer = ArrayObjectWriter::with_datasets_count(store, variable_dir, 1);

        let arr1 = Array::<u8>::try_from_shape_and_slice(&[4], &[1, 2, 3, 4]).unwrap();
        writer
            .define_variable_array::<u8, _, _>(0, &[4], &["x"], None, None, Some(arr1))
            .await
            .unwrap();

        let arr2 = Array::<u8>::try_from_shape_and_slice(&[4], &[9, 8, 7, 6]).unwrap();
        writer.write_array::<u8, _>(0, &arr2, None).await.unwrap();

        // Allocation index stays the same and no new blob metadata is created.
        let meta = writer.array_metadatas[0].as_ref().unwrap();
        assert_eq!(meta.allocation_index, Some(0));
        assert_eq!(writer.array_blob_metadatas.len(), 1);

        let blob_meta = &writer.array_blob_metadatas[0];
        let loaded =
            ArrayBlob::<Committed, bytes::Bytes>::try_from_existing(&writer.store, &blob_meta.path)
                .await
                .unwrap();
        let stored = loaded.get_array::<u8>(0, smallvec![4usize]).unwrap();
        assert_eq!(stored.data.as_ref(), &[9, 8, 7, 6]);
    }

    #[tokio::test]
    async fn write_array_utf8_creates_new_allocation_on_update() {
        let store = InMemory::new();
        let variable_dir = variable_dir();
        let mut writer = ArrayObjectWriter::with_datasets_count(store, variable_dir, 1);

        let arr1 = Array::<Utf8>::try_from_shape_and_strs(&[2], &["a", "bb"]).unwrap();
        writer
            .define_variable_array::<Utf8, _, _>(0, &[2], &["x"], None, None, Some(arr1))
            .await
            .unwrap();

        let meta_before = writer.array_metadatas[0].as_ref().unwrap();
        assert_eq!(meta_before.allocation_index, Some(0));
        assert_eq!(writer.array_blob_metadatas.len(), 1);

        let arr2 = Array::<Utf8>::try_from_shape_and_strs(&[2], &["ccc", "d"]).unwrap();
        writer.write_array::<Utf8, _>(0, &arr2, None).await.unwrap();

        let meta_after = writer.array_metadatas[0].as_ref().unwrap();
        assert_eq!(meta_after.allocation_index, Some(1));
        assert_eq!(writer.array_blob_metadatas.len(), 2);

        let allocation_index = meta_after.allocation_index.unwrap();
        let blob_meta =
            find_array_blob_for_allocation_index(&writer.array_blob_metadatas, allocation_index)
                .unwrap();

        let loaded =
            ArrayBlob::<Committed, bytes::Bytes>::try_from_existing(&writer.store, &blob_meta.path)
                .await
                .unwrap();
        let stored = loaded.get_array::<Utf8>(0, smallvec![2usize]).unwrap();
        let got: Vec<&str> = stored.as_ndarray().iter().map(|s| *s).collect();
        assert_eq!(got, vec!["ccc", "d"]);
    }

    #[tokio::test]
    async fn write_array_partial_update_u8_1d_updates_subslice() {
        let store = InMemory::new();
        let variable_dir = variable_dir();
        let mut writer = ArrayObjectWriter::with_datasets_count(store, variable_dir, 1);

        let full = Array::<u8>::try_from_shape_and_slice(&[4], &[1, 2, 3, 4]).unwrap();
        writer
            .define_variable_array::<u8, _, _>(0, &[4], &["x"], None, None, Some(full))
            .await
            .unwrap();

        let patch = Array::<u8>::try_from_shape_and_slice(&[2], &[9, 9]).unwrap();
        writer
            .write_array::<u8, _>(0, &patch, Some(&[1]))
            .await
            .unwrap();

        let blob_meta = &writer.array_blob_metadatas[0];
        let loaded =
            ArrayBlob::<Committed, bytes::Bytes>::try_from_existing(&writer.store, &blob_meta.path)
                .await
                .unwrap();
        let stored = loaded.get_array::<u8>(0, smallvec![4usize]).unwrap();
        assert_eq!(stored.data.as_ref(), &[1, 9, 9, 4]);
    }

    #[tokio::test]
    async fn write_array_partial_update_u8_2d_updates_subslice() {
        let store = InMemory::new();
        let variable_dir = variable_dir();
        let mut writer = ArrayObjectWriter::with_datasets_count(store, variable_dir, 1);

        // 2x3 row-major: [[0,1,2],[3,4,5]]
        let full = Array::<u8>::try_from_shape_and_slice(&[2, 3], &[0, 1, 2, 3, 4, 5]).unwrap();
        writer
            .define_variable_array::<u8, _, _>(0, &[2, 3], &["y", "x"], None, None, Some(full))
            .await
            .unwrap();

        // Patch 1x2 at start (1,1): updates [4,5] -> [99,98]
        let patch = Array::<u8>::try_from_shape_and_slice(&[1, 2], &[99, 98]).unwrap();
        writer
            .write_array::<u8, _>(0, &patch, Some(&[1, 1]))
            .await
            .unwrap();

        let blob_meta = &writer.array_blob_metadatas[0];
        let loaded =
            ArrayBlob::<Committed, bytes::Bytes>::try_from_existing(&writer.store, &blob_meta.path)
                .await
                .unwrap();
        let stored = loaded
            .get_array::<u8>(0, smallvec![2usize, 3usize])
            .unwrap();
        assert_eq!(stored.data.as_ref(), &[0, 1, 2, 3, 99, 98]);
    }
}

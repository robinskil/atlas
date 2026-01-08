//! Shared object-store discovery helpers for variable arrays.
//!
//! Both the variable-array reader and writer need to discover:
//! - the latest [`ArrayMetadata`] per dataset index (from `array_meta/*.jsonl`)
//! - available blob objects (from `array/*`), named as `start_end`
//! - the dataset count (by scanning metadata for the max `dataset_index`)
//!
//! This module centralizes that logic to avoid duplication.

use futures::TryStreamExt;
use object_store::ObjectStore;

use crate::{
    consts, variable::array::blob::ArrayBlobMetadata, variable::array::metadata::ArrayMetadata,
};

#[derive(Debug, thiserror::Error)]
pub enum ArrayDiscoveryError {
    #[error(transparent)]
    ObjectStore(#[from] object_store::Error),

    #[error(transparent)]
    Json(#[from] simd_json::Error),

    #[error("dataset_index {dataset_index} out of bounds (len={len})")]
    DatasetIndexOutOfBounds { dataset_index: usize, len: usize },
}

/// Read the latest array metadata for each dataset index.
///
/// This scans `array_meta/*.jsonl` under `variable_dir` and returns a vector of length
/// `datasets_count` where each entry is the **last** [`ArrayMetadata`] encountered for that index.
pub async fn read_array_metadatas<S: ObjectStore>(
    store: &S,
    variable_dir: &object_store::path::Path,
    datasets_count: usize,
) -> Result<Vec<Option<Box<ArrayMetadata>>>, ArrayDiscoveryError> {
    let metadata_path = variable_dir.child(consts::VARIABLE_ARRAY_META_DIR);
    let mut objects = store.list(Some(&metadata_path));

    let mut array_metadatas = vec![None; datasets_count];

    while let Ok(Some(json_metadata_obj)) = objects.try_next().await {
        let get_result = store.get(&json_metadata_obj.location).await?;
        let bytes = get_result.bytes().await?;

        // File is expected to be JSONL (one JSON object per line).
        for line in bytes.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            let mut line_buf = line.to_vec();
            let metadata: ArrayMetadata = simd_json::from_slice(&mut line_buf)?;
            let index = metadata.dataset_index;
            if index >= datasets_count {
                return Err(ArrayDiscoveryError::DatasetIndexOutOfBounds {
                    dataset_index: index,
                    len: datasets_count,
                });
            }
            array_metadatas[index] = Some(Box::new(metadata));
        }
    }

    Ok(array_metadatas)
}

/// List blob objects under `array/` and interpret their filenames as allocation ranges.
///
/// Blob objects are expected to be named `start_end` (both inclusive), where `start` and `end`
/// are dataset allocation indices. Objects that do not match this pattern are skipped.
pub async fn read_array_blob_metadatas<S: ObjectStore>(
    store: &S,
    variable_dir: &object_store::path::Path,
) -> Result<Vec<ArrayBlobMetadata>, ArrayDiscoveryError> {
    let blob_metadata_path = variable_dir.child(consts::VARIABLE_ARRAY_DIR);
    let mut objects = store.list(Some(&blob_metadata_path));

    let mut array_blob_metadatas = Vec::new();

    while let Ok(Some(json_blob_metadata_obj)) = objects.try_next().await {
        // In the filename it is stored as: startDatasetIndex_endDatasetIndex
        let Some(file_name) = json_blob_metadata_obj.location.filename() else {
            continue;
        };
        let name_parts: Vec<&str> = file_name.split('_').collect();
        if name_parts.len() != 2 {
            continue; // Skip files that don't match the expected pattern
        }
        let Ok(start_dataset_index) = name_parts[0].parse::<usize>() else {
            continue;
        };
        let Ok(end_dataset_index) = name_parts[1].parse::<usize>() else {
            continue;
        };

        array_blob_metadatas.push(ArrayBlobMetadata {
            range: start_dataset_index..=end_dataset_index,
            path: json_blob_metadata_obj.location.clone(),
        });
    }

    Ok(array_blob_metadatas)
}

/// Find the blob metadata that contains `allocation_index`.
///
/// # Returns
/// Returns `Some(&ArrayBlobMetadata)` if any entry's `range` contains `allocation_index`.
pub fn find_array_blob_for_allocation_index(
    array_blob_metadatas: &[ArrayBlobMetadata],
    allocation_index: usize,
) -> Option<&ArrayBlobMetadata> {
    array_blob_metadatas
        .iter()
        .find(|&blob_metadata| blob_metadata.range.contains(&allocation_index))
        .map(|v| v as _)
}

/// Infer dataset count by scanning `array_meta/*.jsonl` for the maximum `dataset_index`.
///
/// Returns `max_dataset_index + 1`, or `0` if no metadata is present.
pub async fn infer_datasets_count_from_store<S: ObjectStore>(
    store: &S,
    variable_dir: &object_store::path::Path,
) -> Result<usize, ArrayDiscoveryError> {
    let metadata_path = variable_dir.child(consts::VARIABLE_ARRAY_META_DIR);
    let mut objects = store.list(Some(&metadata_path));

    let mut max_dataset_index: Option<usize> = None;
    while let Ok(Some(json_metadata_obj)) = objects.try_next().await {
        let get_result = store.get(&json_metadata_obj.location).await?;
        let bytes = get_result.bytes().await?;

        for line in bytes.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            let mut line_buf = line.to_vec();
            let metadata: ArrayMetadata = simd_json::from_slice(&mut line_buf)?;
            max_dataset_index = Some(
                max_dataset_index
                    .map(|cur| cur.max(metadata.dataset_index))
                    .unwrap_or(metadata.dataset_index),
            );
        }
    }

    Ok(max_dataset_index.map(|v| v + 1).unwrap_or(0))
}

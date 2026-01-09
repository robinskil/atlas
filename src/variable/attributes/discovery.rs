//! Shared object-store discovery helpers for variable attributes.
//!
//! Variable attributes are stored under `attributes/<attribute_name>/` as JSON objects whose keys
//! are dataset indices and whose values are attribute values.
//!
//! Files are named as `start_end.json` (both inclusive) to describe the dataset-index range that
//! the file may contain.

use futures::TryStreamExt;
use object_store::ObjectStore;

use crate::{consts, variable::attributes::AttributeValue};

#[derive(Debug, thiserror::Error)]
pub enum AttributesDiscoveryError {
    #[error(transparent)]
    ObjectStore(#[from] object_store::Error),

    #[error(transparent)]
    Json(#[from] simd_json::Error),

    #[error("invalid attribute object path: {path}")]
    InvalidAttributeObjectPath { path: String },

    #[error("invalid attribute range filename: {file_name}")]
    InvalidRangeFileName { file_name: String },

    #[error("invalid dataset index key: {key}")]
    InvalidDatasetIndexKey { key: String },

    #[error("dataset_index {dataset_index} out of bounds (len={len})")]
    DatasetIndexOutOfBounds { dataset_index: usize, len: usize },

    #[error("attribute JSON key {dataset_index} is outside file range {start}..={end}")]
    DatasetIndexOutsideFileRange {
        dataset_index: usize,
        start: usize,
        end: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeRangeObject {
    pub attribute_name: String,
    pub start: usize,
    pub end: usize,
    pub path: object_store::path::Path,
}

fn parse_range_file_name(file_name: &str) -> Result<(usize, usize), AttributesDiscoveryError> {
    let stem = file_name.strip_suffix(".json").unwrap_or(file_name);

    let (start_str, end_str) =
        stem.split_once('_')
            .ok_or_else(|| AttributesDiscoveryError::InvalidRangeFileName {
                file_name: file_name.to_string(),
            })?;

    let start =
        start_str
            .parse::<usize>()
            .map_err(|_| AttributesDiscoveryError::InvalidRangeFileName {
                file_name: file_name.to_string(),
            })?;
    let end =
        end_str
            .parse::<usize>()
            .map_err(|_| AttributesDiscoveryError::InvalidRangeFileName {
                file_name: file_name.to_string(),
            })?;

    Ok((start, end))
}

/// Read all values for a single attribute.
pub async fn read_attribute_values<S: ObjectStore>(
    store: &S,
    variable_dir: &object_store::path::Path,
    attribute_name: &str,
    datasets_count: usize,
) -> Result<Vec<Option<AttributeValue>>, AttributesDiscoveryError> {
    let attribute_dir = variable_dir
        .child(consts::VARIABLE_ATTRIBUTES_DIR)
        .child(attribute_name);
    let mut value_objects = store.list(Some(&attribute_dir));

    let mut out = vec![None; datasets_count];

    while let Ok(Some(object_meta)) = value_objects.try_next().await {
        let get_result = store.get(&object_meta.location).await?;
        let bytes = get_result.bytes().await?;

        let mut buf = bytes.to_vec();
        let values: Vec<(usize, AttributeValue)> = simd_json::from_slice(&mut buf)?;

        for (dataset_index, value) in values {
            if dataset_index >= datasets_count {
                return Err(AttributesDiscoveryError::DatasetIndexOutOfBounds {
                    dataset_index,
                    len: datasets_count,
                });
            }
            out[dataset_index] = Some(value);
        }
    }

    Ok(out)
}

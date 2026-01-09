//! Variable attributes writer.
//!
//! This module contains [`AttributeObjectWriter`], an `object_store`-backed writer for a *single*
//! variable attribute.
//!
//! ## Layout
//! The writer stores attribute state under a per-variable directory:
//! ```text
//! collection.atlas/
//!   ├── datasets/
//!   │   ...
//!   └── variables/
//!       └── variable_0/
//!           ├── variable.json
//!           └── attributes/
//!               └── units/
//!                   ├── 1234235435.json
//!                   └── 2154654654.json
//! ```
//!
//! Each `start_end.json` is a JSON object mapping dataset indices to values.

use object_store::{ObjectStore, PutPayload};

use crate::{
    consts,
    variable::attributes::{AttributeValue, discovery},
};

#[derive(Debug, thiserror::Error)]
pub enum AttributesWriteError {
    #[error(transparent)]
    ObjectStore(#[from] object_store::Error),

    #[error(transparent)]
    Json(#[from] simd_json::Error),

    #[error(transparent)]
    Discovery(#[from] discovery::AttributesDiscoveryError),

    #[error("dataset_index {dataset_index} out of bounds (len={len})")]
    DatasetIndexOutOfBounds { dataset_index: usize, len: usize },
}

pub struct AttributeObjectWriter<S: ObjectStore> {
    store: S,
    variable_dir: object_store::path::Path,
    attribute_name: String,
    attribute_values: Vec<Option<AttributeValue>>,
    datasets_count: usize,
}

impl<S: ObjectStore> AttributeObjectWriter<S> {
    /// Create a writer pre-sized for `datasets_count` datasets.
    pub async fn new(
        store: S,
        variable_dir: object_store::path::Path,
        attribute_name: impl Into<String>,
        datasets_count: usize,
    ) -> Self {
        let attribute_name = attribute_name.into();
        let attribute_values = discovery::read_attribute_values(
            &store,
            &variable_dir,
            &attribute_name,
            datasets_count,
        )
        .await
        .unwrap_or(vec![None; datasets_count]);

        Self {
            store,
            variable_dir,
            attribute_values,
            attribute_name,
            datasets_count,
        }
    }

    /// Return the attribute name this writer is bound to.
    pub fn attribute_name(&self) -> &str {
        &self.attribute_name
    }

    /// Set (or overwrite) a single attribute value.
    ///
    /// This writes a single-file range object
    /// `attributes/<attribute_name>/<unix_timestamp_nanos>.json`.
    pub async fn set_attribute(
        &mut self,
        dataset_index: usize,
        value: AttributeValue,
    ) -> Result<(), AttributesWriteError> {
        if dataset_index >= self.datasets_count {
            return Err(AttributesWriteError::DatasetIndexOutOfBounds {
                dataset_index,
                len: self.datasets_count,
            });
        }

        self.attribute_values[dataset_index] = Some(value.clone());
        // Discovery expects each object to be a JSON array of (dataset_index, value) pairs.
        let json_bytes = simd_json::to_vec(&vec![(dataset_index, value)]).unwrap();

        self.store
            .put(
                &self.attribute_object_path(&self.attribute_name),
                PutPayload::from_bytes(json_bytes.into()),
            )
            .await?;

        Ok(())
    }

    fn attribute_object_path(&self, attribute_name: &str) -> object_store::path::Path {
        let unix_timestamp_nanos = chrono::Utc::now().timestamp_nanos();
        let dir = self
            .variable_dir
            .child(consts::VARIABLE_ATTRIBUTES_DIR)
            .child(attribute_name);
        dir.child(format!("{}.json", unix_timestamp_nanos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use std::sync::Arc;

    fn variable_dir() -> object_store::path::Path {
        object_store::path::Path::from("variables/variable_0")
    }

    #[tokio::test]
    async fn writer_rejects_out_of_bounds_dataset_index() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let variable_dir = variable_dir();
        let datasets_count = 6;

        let mut writer =
            AttributeObjectWriter::new(store, variable_dir, "flag", datasets_count).await;

        writer
            .set_attribute(5, AttributeValue::Boolean(false))
            .await
            .unwrap();

        let err = writer
            .set_attribute(6, AttributeValue::Boolean(true))
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            AttributesWriteError::DatasetIndexOutOfBounds {
                dataset_index: 6,
                len: 6
            }
        ));
    }
}

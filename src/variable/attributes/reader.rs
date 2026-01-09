//! Variable attributes reader.
//!
//! This module contains [`AttributeObjectReader`], an `object_store`-backed reader for a *single*
//! variable attribute.
//!
//! ## Layout
//! This reader expects the same layout as [`crate::variable::attributes::writer::AttributeObjectWriter`]:
//! - `attributes/<attribute_name>/*` contains JSON objects mapping dataset indices to values.
//! - Files are named `start_end.json` (both inclusive).

use object_store::ObjectStore;

use crate::variable::attributes::{AttributeValue, discovery};

#[derive(Debug, thiserror::Error)]
pub enum AttributesReadError {
    #[error(transparent)]
    ObjectStore(#[from] object_store::Error),

    #[error(transparent)]
    Json(#[from] simd_json::Error),

    #[error(transparent)]
    Discovery(#[from] discovery::AttributesDiscoveryError),
}

pub struct AttributeObjectReader<S: ObjectStore> {
    store: S,
    variable_dir: object_store::path::Path,
    attribute_name: String,
    values: Vec<Option<AttributeValue>>,
}

impl<S: ObjectStore> AttributeObjectReader<S> {
    /// Create a reader pre-sized for `datasets_count` datasets.
    pub async fn new(
        store: S,
        variable_dir: object_store::path::Path,
        attribute_name: impl Into<String>,
        datasets_count: usize,
    ) -> Result<Self, AttributesReadError> {
        let attribute_name = attribute_name.into();
        let attribute_values = discovery::read_attribute_values(
            &store,
            &variable_dir,
            &attribute_name,
            datasets_count,
        )
        .await?;
        let _ = datasets_count;
        Ok(Self {
            store,
            variable_dir,
            attribute_name,
            values: attribute_values,
        })
    }

    /// Return the attribute name this reader is bound to.
    pub fn attribute_name(&self) -> &str {
        &self.attribute_name
    }

    /// Read an attribute value for `dataset_index`.
    ///
    /// Returns `Ok(None)` if the dataset index has no value.
    pub async fn read_attribute(&self, dataset_index: usize) -> Option<AttributeValue> {
        self.values.get(dataset_index).cloned().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::attributes::writer::AttributeObjectWriter;
    use object_store::memory::InMemory;
    use std::sync::Arc;

    fn variable_dir() -> object_store::path::Path {
        object_store::path::Path::from("variables/variable_0")
    }

    #[tokio::test]
    async fn reader_reads_attributes_roundtrip() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let variable_dir = variable_dir();
        let datasets_count = 2;

        let mut units_writer = AttributeObjectWriter::new(
            store.clone(),
            variable_dir.clone(),
            "units",
            datasets_count,
        )
        .await;
        units_writer
            .set_attribute(0, AttributeValue::String("m".to_string()))
            .await
            .unwrap();
        units_writer
            .set_attribute(1, AttributeValue::String("km".to_string()))
            .await
            .unwrap();

        let mut valid_writer = AttributeObjectWriter::new(
            store.clone(),
            variable_dir.clone(),
            "valid",
            datasets_count,
        )
        .await;
        valid_writer
            .set_attribute(0, AttributeValue::Boolean(true))
            .await
            .unwrap();

        let units_reader = AttributeObjectReader::new(
            store.clone(),
            variable_dir.clone(),
            "units",
            datasets_count,
        )
        .await
        .unwrap();
        let valid_reader = AttributeObjectReader::new(store, variable_dir, "valid", datasets_count)
            .await
            .unwrap();

        assert_eq!(
            units_reader.read_attribute(0).await,
            Some(AttributeValue::String("m".to_string()))
        );
        assert_eq!(
            units_reader.read_attribute(1).await,
            Some(AttributeValue::String("km".to_string()))
        );
        assert_eq!(
            valid_reader.read_attribute(0).await,
            Some(AttributeValue::Boolean(true))
        );
        assert_eq!(valid_reader.read_attribute(1).await, None);
    }
}

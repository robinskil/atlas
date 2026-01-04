use std::{path::Path, sync::Arc};

use crate::schema::Schema;

pub struct Collection {}

pub struct CollectionMut {
    file: Arc<std::fs::File>,
    schema: Schema,
}

impl CollectionMut {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        // Placeholder implementation
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .unwrap();
        Self {
            file: Arc::new(file),
            schema: Schema::new_empty(),
        }
    }

    // pub fn create_dataset(&mut self, dataset_name: &str) -> DatasetMut {}
}

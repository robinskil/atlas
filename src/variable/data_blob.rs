use core::panic;
use std::sync::Arc;

use bytes::Bytes;
use compact_str::CompactString;
use futures::StreamExt;
use object_store::{ObjectMeta, ObjectStore, PutPayload, local::LocalFileSystem};
use smallvec::SmallVec;

use crate::{
    dtype::DataType,
    variable::{
        array::{Array, datatype::ArrayDataType},
        util::find_range_index,
    },
};

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct ArrayMetadataBlob {
    pub start: usize,
    pub end: usize,
    pub variable_data_blobs: Vec<VariableDataBlob>,
}

#[derive(Clone, serde::Deserialize, serde::Serialize, Debug)]
pub struct VariableDataBlob {
    pub dataset_index: usize,
    pub dimensions: SmallVec<[CompactString; 4]>,
    pub shape: SmallVec<[usize; 4]>,
    pub chunk_shape: SmallVec<[usize; 4]>,
    pub chunk_array_index: SmallVec<[Option<u32>; 4]>,
    pub datatype: DataType,
}

pub(crate) struct ObjectVariableBlobReader {
    store: Arc<dyn ObjectStore>,

    array_blob_json_files: Vec<ObjectMeta>,
    array_blob_json_cache: moka::future::Cache<usize, Arc<ArrayMetadataBlob>>,
    array_blob_json_ranges: Vec<std::ops::RangeInclusive<usize>>,

    array_blob_files: Vec<ObjectMeta>,
    array_blob_cache: moka::future::Cache<usize, Bytes>,
    array_blob_ranges: Vec<std::ops::RangeInclusive<usize>>,
}

impl ObjectVariableBlobReader {
    const ARRAY_BLOB_DIR: &'static str = "data_blobs/";
    /// JSON blobs define the layout of the binary blobs
    /// naming convention: data_blob_0_1000.json
    const JSON_ARRAY_BLOB_DIR: &'static str = "json_blobs/";

    pub async fn new(store: Arc<dyn ObjectStore>, variable_dir: object_store::path::Path) -> Self {
        // List all the json files in the variable directory
        let listed_array_files =
            Self::collect_object_metas(store.clone(), variable_dir.child(Self::ARRAY_BLOB_DIR))
                .await;
        let listed_json_array_files = Self::collect_object_metas(
            store.clone(),
            variable_dir.child(Self::JSON_ARRAY_BLOB_DIR),
        )
        .await;

        Self {
            store,
            array_blob_json_files: listed_json_array_files,
            array_blob_json_cache: moka::future::Cache::builder().max_capacity(16).build(),
            array_blob_json_ranges: Vec::new(),
            array_blob_files: listed_array_files,
            array_blob_cache: moka::future::Cache::builder()
                .max_capacity(32 * 1024 * 1024)
                .build(),
            array_blob_ranges: Vec::new(),
        }
    }

    async fn fetch_data_blob_json(&self, blob_index: usize) -> Arc<ArrayMetadataBlob> {
        let object_meta = &self.array_blob_json_files[blob_index];
        let store = self.store.clone();

        let data_blob = self
            .array_blob_json_cache
            .try_get_with(blob_index, async move {
                let get_result = store.get(&object_meta.location).await;
                match get_result {
                    Ok(get_result) => {
                        let mut bytes = Vec::with_capacity(object_meta.size as usize);
                        let mut bytes_stream = get_result.into_stream();
                        while let Some(chunk) = bytes_stream.next().await {
                            let chunk = chunk.expect("Error reading data blob json");
                            bytes.extend_from_slice(&chunk);
                        }
                        let data_blob: ArrayMetadataBlob = simd_json::serde::from_slice(&mut bytes)
                            .expect("Error deserializing data blob json");
                        Ok::<_, String>(Arc::new(data_blob))
                    }
                    Err(e) => {
                        panic!("Error fetching data blob json: {}", e);
                    }
                }
            })
            .await
            .unwrap();

        data_blob
    }

    pub async fn read_variable_array_chunk<T: ArrayDataType>(
        &'_ self,
        dataset_index: usize,
        chunk_index: Option<Vec<usize>>,
    ) -> Option<Array<T>> {
        // Find which data blob contains the requested dataset_index
        let blob_index = find_range_index(&self.array_blob_json_ranges, dataset_index);
        let blob_index = blob_index.unwrap();

        let data_blob = self.fetch_data_blob_json(blob_index).await;
        let variable_info = &data_blob.variable_data_blobs[dataset_index - data_blob.start];

        if let Some(chunk_index) = chunk_index {
            todo!()
        } else {
            // Read the entire array span
            let array = self
                .read_array::<T>(variable_info.shape.clone(), variable_info.dataset_index)
                .await?;
            Some(array)
        }
    }

    async fn read_array<T: ArrayDataType>(
        &self,
        shape: SmallVec<[usize; 4]>,
        array_blob_index: usize,
    ) -> Option<Array<T>> {
        let blob_file_index = find_range_index(&self.array_blob_ranges, array_blob_index)?;
        let array_blob = self.read_array_blob(blob_file_index).await?;
        let index_in_blob = array_blob_index - self.array_blob_ranges[blob_file_index].start();
        let array = array_blob.get_array::<T>(index_in_blob, shape)?;

        Some(array)
    }

    async fn read_array_blob(&self, array_blob_file_index: usize) -> Option<ArrayBlob> {
        let object_meta = &self.array_blob_files[array_blob_file_index].clone();
        let store = self.store.clone();

        let blob_bytes = self
            .array_blob_cache
            .try_get_with(array_blob_file_index, async move {
                let get_result = store.get(&object_meta.location).await;
                match get_result {
                    Ok(get_result) => {
                        let mut bytes = Vec::with_capacity(object_meta.size as usize);
                        let mut bytes_stream = get_result.into_stream();
                        while let Some(chunk) = bytes_stream.next().await {
                            let chunk = chunk.expect("Error reading array blob");
                            bytes.extend_from_slice(&chunk);
                        }
                        Ok::<_, String>(Bytes::from(bytes))
                    }
                    Err(e) => {
                        panic!("Error fetching array blob: {}", e);
                    }
                }
            })
            .await;

        blob_bytes
            .ok()
            .map(|bytes| ArrayBlob::new_from_bytes(bytes))
    }

    pub async fn read_variable_array<T: ArrayDataType>(
        &'_ self,
        dataset_index: usize,
        slice: Option<std::ops::Range<usize>>,
    ) -> Array<T> {
        todo!()
    }

    async fn collect_object_metas(
        store: Arc<dyn ObjectStore>,
        prefix: object_store::path::Path,
    ) -> Vec<ObjectMeta> {
        let listed_files = store.list(Some(&prefix)).collect::<Vec<_>>().await;

        let mut object_metas = Vec::new();
        for res in listed_files {
            match res {
                Ok(meta) => object_metas.push(meta),
                Err(e) => {
                    panic!("Error listing files: {}", e);
                }
            }
        }
        object_metas
    }
}

pub const DATA_BLOB_DIR: &str = "array_blobs/";
pub const JSON_DATA_BLOB_DIR: &str = "json_array_blobs/";

pub struct ObjectVariableBlobWriter<S: ObjectStore> {
    store: S,
    variable_dir: object_store::path::Path,
}

impl<S: ObjectStore> ObjectVariableBlobWriter<S> {
    pub fn define_variable_json_path(
        variable_dir: &object_store::path::Path,
        dataset_index: usize,
    ) -> object_store::path::Path {
        variable_dir
            .child(JSON_DATA_BLOB_DIR)
            .child(format!("{}.json", dataset_index))
    }
}

impl ObjectVariableBlobWriter<LocalFileSystem> {
    pub fn new(store: LocalFileSystem, variable_dir: object_store::path::Path) -> Self {
        Self {
            store,
            variable_dir,
        }
    }

    pub async fn define_variable<T: ArrayDataType, D: AsRef<str>>(
        &mut self,
        dataset_index: usize,
        dimensions: &[D],
        shape: &[usize],
        chunk_shape: Option<&[usize]>,
    ) {
        assert!(dimensions.len() == shape.len());
        let chunk_shape = match chunk_shape {
            Some(cs) => cs.to_vec(),
            None => shape.to_vec(),
        };

        let json_object_path = Self::define_variable_json_path(&self.variable_dir, dataset_index);

        let json_object = VariableDataBlob {
            dataset_index,
            dimensions: dimensions
                .iter()
                .map(|d| d.as_ref().into())
                .collect::<SmallVec<[CompactString; 4]>>(),
            shape: shape.iter().copied().collect::<SmallVec<[usize; 4]>>(),
            chunk_shape: chunk_shape
                .iter()
                .copied()
                .collect::<SmallVec<[usize; 4]>>(),
            chunk_array_index: SmallVec::from_buf([None; 4]),
            datatype: T::TYPE,
        };

        let json = simd_json::serde::to_vec(&json_object).unwrap();

        // Write the JSON to the filesystem
        self.store
            .put(&json_object_path, PutPayload::from_bytes(json.into()))
            .await
            .unwrap();
    }

    pub async fn write_variable_array<A: Into<Array<T>>, T: ArrayDataType>(
        &mut self,
        dataset_index: usize,
        start: Option<&[usize]>,
        array: A,
    ) {
        let array = array.into();
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {
        let fs_store = LocalFileSystem::new_with_prefix("./test_data").unwrap();
        let json_path =
            object_store::path::Path::from("variable_0/json_blobs/data_blob_0_999.json");
        let fs_path = fs_store.path_to_filesystem(&json_path).unwrap();
        println!("Filesystem path: {:?}", fs_path);
    }
}

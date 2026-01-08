use std::{
    alloc::System,
    time::{Instant, SystemTime},
};

use chrono::Utc;
use futures::TryStreamExt;
use object_store::{ObjectStore, PutPayload};
use smallvec::SmallVec;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::io::StreamReader;

use crate::{
    consts,
    variable::array::{
        Array, datatype::ArrayDataType, fill_value::FillValue, metadata::ArrayMetadata,
    },
};

pub struct ArrayObjectWriter<S: ObjectStore> {
    store: S,
    variable_dir: object_store::path::Path,
    array_metadatas: Vec<Option<Box<ArrayMetadata>>>,
}

pub async fn read_array_metadatas<S: ObjectStore>(
    store: &S,
    variable_dir: &object_store::path::Path,
    datasets_count: usize,
) -> object_store::Result<Vec<Option<Box<ArrayMetadata>>>> {
    let metadata_path = variable_dir.child(consts::VARIABLE_ARRAY_META_DIR);
    let mut objects = store.list(Some(&metadata_path));

    let mut array_metadatas = Vec::with_capacity(datasets_count); // Preallocate 64KB for performance
    for _ in 0..datasets_count {
        array_metadatas.push(None);
    }

    while let Ok(Some(json_metadata_obj)) = objects.try_next().await {
        let get_result = store.get(&json_metadata_obj.location).await?;
        let bytes_stream = get_result.into_stream();
        let stream_reader = StreamReader::new(bytes_stream);
        let buf_reader = BufReader::new(stream_reader);
        let mut lines = buf_reader.lines();
        while let Some(line) = lines.next_line().await.unwrap() {
            let mut bytes_ref_mut = line.into_bytes();
            let metadata: ArrayMetadata = simd_json::from_slice(&mut bytes_ref_mut).unwrap();
            let index = metadata.dataset_index;
            array_metadatas[index] = Some(Box::new(metadata));
        }
    }

    Ok(array_metadatas)
}

impl<S: ObjectStore> ArrayObjectWriter<S> {
    pub fn new(store: S, variable_dir: object_store::path::Path) -> Self {
        todo!()
    }

    pub async fn define_variable_array<T: ArrayDataType, D: AsRef<str>>(
        &self,
        dataset_index: usize,
        shape: &[usize],
        dimensions: &[D],
        fill_value: Option<FillValue>,
        start: &[usize],
        array: Option<Array<T>>,
    ) {
        // Create ArrayMetadata
        let metadata = ArrayMetadata {
            dataset_index,
            data_type: T::TYPE,
            shape: SmallVec::from_slice(shape),
            dimensions: dimensions.iter().map(|d| d.as_ref().into()).collect(),
            chunk_shape: None,
            fill_value: fill_value.unwrap_or(FillValue::None),
        };

        // Serialize and write metadata as a JSON line to an object in the store
        // File name is the current unix timestamp in nanoseconds
        let timestamp_nanos = Utc::now().timestamp_nanos();
        let metadata_path = self.generate_metadata_object_path(timestamp_nanos);
        let json_bytes: bytes::Bytes = simd_json::serde::to_vec(&metadata).unwrap().into();
        self.store
            .put(&metadata_path, PutPayload::from_bytes(json_bytes))
            .await
            .unwrap();

        // If an array is provided, write the array data to the store
        if let Some(array) = array {
            Self::write_array::<T>(dataset_index, &array, start).await;
        }
    }

    fn generate_metadata_object_path(&self, timestamp_nanos: i64) -> object_store::path::Path {
        let metadata_dir = self.variable_dir.child(consts::VARIABLE_ARRAY_META_DIR);
        metadata_dir.child(format!("{}.jsonl", timestamp_nanos))
    }

    pub async fn write_array<T: ArrayDataType>(
        dataset_index: usize,
        array: &Array<T>,
        start: &[usize],
    ) {
        todo!()
    }
}

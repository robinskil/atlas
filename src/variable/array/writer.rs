use std::{
    alloc::System,
    time::{Instant, SystemTime},
};

use chrono::Utc;
use futures::TryStreamExt;
use ndarray::SliceInfoElem;
use object_store::{ObjectStore, PutPayload};
use smallvec::SmallVec;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::io::StreamReader;

use crate::{
    consts,
    dtype::DataType,
    variable::array::{
        Array,
        blob::{ArrayBlob, ArrayBlobMetadata},
        datatype::ArrayDataType,
        fill_value::FillValue,
        metadata::ArrayMetadata,
    },
};

pub struct ArrayObjectWriter<S: ObjectStore> {
    store: S,
    variable_dir: object_store::path::Path,
    array_metadatas: Vec<Option<Box<ArrayMetadata>>>,
    array_blob_metadatas: Vec<ArrayBlobMetadata>,
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

pub async fn read_array_blob_metadatas<S: ObjectStore>(
    store: &S,
    variable_dir: &object_store::path::Path,
) -> object_store::Result<Vec<ArrayBlobMetadata>> {
    let blob_metadata_path = variable_dir.child(consts::VARIABLE_ARRAY_DIR);
    let mut objects = store.list(Some(&blob_metadata_path));

    let mut array_blob_metadatas = Vec::new();

    while let Ok(Some(json_blob_metadata_obj)) = objects.try_next().await {
        // In the filename it is stored as: startDatasetIndex_endDatasetIndex
        let file_name = json_blob_metadata_obj.location.filename().unwrap();
        let name_parts: Vec<&str> = file_name.split('_').collect();
        if name_parts.len() != 2 {
            continue; // Skip files that don't match the expected pattern
        }
        let start_dataset_index: usize = name_parts[0].parse().unwrap();
        let end_dataset_index: usize = name_parts[1].parse().unwrap();

        let metadata = ArrayBlobMetadata {
            start_dataset_index,
            end_dataset_index,
            num_allocations: end_dataset_index - start_dataset_index,
            path: json_blob_metadata_obj.location.clone(),
        };
        array_blob_metadatas.push(metadata);
    }

    Ok(array_blob_metadatas)
}

pub fn find_array_blob_for_dataset_index(
    array_blob_metadatas: &[ArrayBlobMetadata],
    dataset_index: usize,
) -> Option<&ArrayBlobMetadata> {
    for blob_metadata in array_blob_metadatas {
        if dataset_index >= blob_metadata.start_dataset_index
            && dataset_index <= blob_metadata.end_dataset_index
        {
            return Some(blob_metadata);
        }
    }
    None
}

impl<S: ObjectStore> ArrayObjectWriter<S> {
    pub fn new(store: S, variable_dir: object_store::path::Path) -> Self {
        todo!()
    }

    pub async fn define_variable_array<T: ArrayDataType, D: AsRef<str>>(
        &mut self,
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
            allocation_index: None,
            data_type: T::TYPE,
            shape: SmallVec::from_slice(shape),
            dimensions: dimensions.iter().map(|d| d.as_ref().into()).collect(),
            chunk_shape: None,
            fill_value: fill_value.unwrap_or(FillValue::None),
        };

        // Serialize and write metadata as a JSON line to an object in the store
        // File name is the current unix timestamp in nanoseconds
        self.write_metadata(&metadata).await;
        self.array_metadatas[dataset_index] = Some(Box::new(metadata));

        // If an array is provided, write the array data to the store
        if let Some(array) = array {
            self.write_array::<T>(dataset_index, &array, start).await;
        }
    }

    fn generate_metadata_object_path(&self, timestamp_nanos: i64) -> object_store::path::Path {
        let metadata_dir = self.variable_dir.child(consts::VARIABLE_ARRAY_META_DIR);
        metadata_dir.child(format!("{}.jsonl", timestamp_nanos))
    }

    fn generate_blob_object_path(
        &self,
        start_dataset_index: usize,
        end_dataset_index: usize,
    ) -> object_store::path::Path {
        let blob_dir = self.variable_dir.child(consts::VARIABLE_ARRAY_DIR);
        blob_dir.child(format!("{}_{}", start_dataset_index, end_dataset_index))
    }

    async fn write_metadata(&self, metadata: &ArrayMetadata) {
        let timestamp_nanos = Utc::now().timestamp_nanos();
        let metadata_path = self.generate_metadata_object_path(timestamp_nanos);
        let json_bytes: bytes::Bytes = simd_json::serde::to_vec(&metadata).unwrap().into();
        self.store
            .put(&metadata_path, PutPayload::from_bytes(json_bytes))
            .await
            .unwrap();
    }

    // async fn mutate_in_place<T: ArrayDataType>(
    //     array_blob: ArrayBlob,
    //     allocation_index: usize,
    //     allocated_array_shape: &[usize],
    //     data: &Array<T>,
    //     start: &[usize],
    // ) {
    //     let mut mutable_blob = array_blob.into_mut_blob();
    //     let mut mutable = mutable_blob
    //         .get_array_mut::<T>(allocation_index, allocated_array_shape.into())
    //         .unwrap();

    //     if let Some(mut nd_array) = mutable.as_mut_ndarray() {
    //         // We can slice and mutate the data in place
    //         let slice_args = start
    //             .iter()
    //             .zip(data.shape().iter())
    //             .map(|(&s, &len)| SliceInfoElem::Slice {
    //                 start: s as isize,
    //                 end: Some((s + len) as isize),
    //                 step: 1,
    //             })
    //             .collect::<Vec<_>>();

    //         let mut mutable_nd_array = nd_array.slice_mut(slice_args.as_slice());
    //         mutable_nd_array.assign(&data.as_ndarray());
    //     } else {
    //         panic!(
    //             "In-place mutation is only supported for non variable sized ndarray-compatible arrays"
    //         );
    //     }

    //     mutable_blob
    // }

    pub async fn write_array<T: ArrayDataType>(
        &mut self,
        dataset_index: usize,
        array: &Array<T>,
        start: &[usize],
    ) {
        let array_metadata = self.array_metadatas[dataset_index].as_ref().unwrap();
        assert_eq!(
            array_metadata.data_type,
            T::TYPE,
            "Data type mismatch when writing array"
        );

        if let Some(allocation_index) = array_metadata.allocation_index {
            if matches!(T::TYPE, DataType::Utf8) {
                // Requires rewriting the entire blob as UTF-8 data can vary in size
            } else {
                // Write directly to the existing allocation
            }
        } else {
            // Create new blob allocation
            let new_blob_index = self.array_blob_metadatas.len();
            let blob_path = self.generate_blob_object_path(dataset_index, dataset_index + 1);
            let array_blob_metadata = ArrayBlobMetadata {
                start_dataset_index: dataset_index,
                end_dataset_index: dataset_index + 1,
                num_allocations: 1,
                path: blob_path,
            };
            self.array_blob_metadatas.push(array_blob_metadata);
            self.array_metadatas[dataset_index]
                .as_mut()
                .unwrap()
                .allocation_index = Some(new_blob_index);
            if let Some(array_metadata) = &self.array_metadatas[dataset_index] {
                self.write_metadata(array_metadata).await;
            }
        }

        todo!()
    }
}

use compact_str::CompactString;
use smallvec::SmallVec;

use crate::{dtype::DataType, variable::array::fill_value::FillValue};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArrayMetadata {
    pub dataset_index: usize,
    pub allocation_index: Option<usize>,
    pub data_type: DataType,
    pub shape: SmallVec<[usize; 4]>,
    pub chunk_shape: Option<Vec<usize>>,
    pub dimensions: SmallVec<[CompactString; 4]>,
    pub fill_value: FillValue,
}

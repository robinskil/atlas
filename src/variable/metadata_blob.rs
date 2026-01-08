use crate::dtype::DataType;

pub struct MetadataBlob {
    pub start: usize,
    pub end: usize,
    pub variable_metadata_blobs: Vec<VariableMetadataBlob>,
}

pub struct VariableMetadataBlob {
    pub starting_blob_index: usize,
    pub datatype: DataType,
}

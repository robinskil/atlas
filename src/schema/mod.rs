use crate::dtype::DataType;

pub mod reader;
pub mod wal;
pub mod writer;

pub struct Schema {
    pub variables: Vec<VariableSchema>,
    pub global_attributes: Vec<AttributeSchema>,
}

pub struct VariableSchema {
    pub name: String,                     // variable name
    pub dtype: DataType,                  // variable data type
    pub attributes: Vec<AttributeSchema>, // variable attributes
}

pub struct AttributeSchema {
    pub name: String,
    pub dtype: DataType,
}

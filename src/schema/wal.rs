use crate::dtype::DataType;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct WalEntry {
    pub dataset_index: usize,
    pub operation: Operation,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum Operation {
    CreateVariable {
        name: String,
        dtype: DataType,
    },
    UpdateVariable {
        name: String,
        dtype: DataType,
    },
    DeleteVariable {
        name: String,
    },
    CreateVariableAttribute {
        variable_name: String,
        attribute_name: String,
        dtype: DataType,
    },
    UpdateVariableAttribute {
        variable_name: String,
        attribute_name: String,
        dtype: DataType,
    },
    DeleteVariableAttribute {
        variable_name: String,
        attribute_name: String,
    },
}

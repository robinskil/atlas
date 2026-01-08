#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DataType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Utf8,
    Timestamp,
}

impl DataType {
    pub const fn is_variable_sized(&self) -> bool {
        matches!(self, DataType::Utf8)
    }
}

use crate::dtype::DataType;

pub struct Schema {
    pub fields: Vec<SchemaField>,
    pub datasets: Vec<String>, // dataset names (can be used as IDs and in combination with field names to identify variables)
}

impl Schema {
    pub fn new_empty() -> Self {
        Self {
            fields: Vec::new(),
            datasets: Vec::new(),
        }
    }
}

pub struct SchemaField {
    pub name: String, // variable name
    pub dtype: DataType,
    pub dataset_active: Vec<bool>, // one per dataset to determine if field (variable) is active in that dataset
}

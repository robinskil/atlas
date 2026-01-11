use compact_str::CompactString;
use serde::{Deserialize, Serialize};

use crate::dtype::DataType;

pub type StrId = u32; // interned string
pub type DatasetId = u32; // dataset index
pub type VarId = u16; // global variable id
pub type AttrId = u16; // global attribute id
pub type VarAttrLayoutId = u32; // interned variable-attribute layout

#[derive(Clone, Serialize, Deserialize)]
pub struct SchemaSnapshot {
    /// Interned string table
    /// StrId -> String
    pub strings: Vec<CompactString>,

    /// Global variable definitions
    /// VarId -> VariableDef
    pub variables: Vec<VariableDef>,

    /// Global attribute definitions
    /// AttrId -> AttributeDef
    pub attributes: Vec<AttributeDef>,

    /// Interned layouts for variable attributes
    /// VarAttrLayoutId -> VarAttrLayout
    pub var_attr_layouts: Vec<VarAttrLayout>,

    /// Dataset schemas
    pub datasets: DatasetSchemas,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDef {
    pub name: StrId,
    pub data_type: DataType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeDef {
    pub name: StrId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarAttrLayout {
    pub attrs: Vec<DatasetAttr>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DatasetVar {
    pub var_id: VarId,
    pub layout_id: VarAttrLayoutId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAttr {
    pub attr_id: AttrId,
    pub dtype: DataType,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DatasetEntry {
    /// Attributes on the dataset itself
    pub attributes: Vec<DatasetAttr>,

    /// Variables in this dataset
    pub vars: Vec<DatasetVar>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DatasetSchemas {
    /// All dataset entries concatenated
    pub entries: Vec<DatasetEntry>,

    /// offsets.len() == num_datasets + 1
    pub offsets: Vec<u32>,
}

impl DatasetSchemas {
    pub fn dataset(&self, dataset: DatasetId) -> &DatasetEntry {
        let i = dataset as usize;
        let start = self.offsets[i] as usize;
        &self.entries[start]
    }
}

impl SchemaSnapshot {
    pub fn variable_attrs(&self, dv: &DatasetVar) -> &[DatasetAttr] {
        &self.var_attr_layouts[dv.layout_id as usize].attrs
    }
}

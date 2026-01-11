use crate::{
    dtype::DataType,
    schema::raw::{AttrId, DatasetId, DatasetVar, VarId},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOp {
    /* ---------------- Dataset lifecycle ---------------- */
    /// Create a new dataset with an initial variable list
    AddDataset {
        dataset_id: DatasetId,
        vars: Vec<DatasetVar>,
    },

    /// Remove a dataset entirely
    RemoveDataset { dataset_id: DatasetId },

    /* ---------------- Variable in dataset ---------------- */
    /// Add a variable to a dataset
    AddVariableToDataset {
        dataset_id: DatasetId,
        var_id: VarId,
    },

    /// Remove a variable from a dataset
    RemoveVariableFromDataset {
        dataset_id: DatasetId,
        var_id: VarId,
    },

    ChangeVariableType {
        dataset_id: DatasetId,
        var_id: VarId,
        new_dtype: DataType,
    },

    /// Change the base data type of a variable (global definition)
    ChangeGlobalVariableType { var_id: VarId, new_dtype: DataType },

    /// Add a new attribute to a dataset
    AddDatasetAttribute {
        dataset_id: DatasetId,
        attr_id: AttrId,
        dtype: DataType,
    },

    /// Remove an attribute from a dataset
    RemoveDatasetAttribute {
        dataset_id: DatasetId,
        attr_id: AttrId,
    },

    /* ---------------- Attribute in dataset variable ---------------- */
    /// Add an attribute to a variable in a dataset
    AddAttributeToDatasetVar {
        dataset_id: DatasetId,
        var_id: VarId,
        attr_id: AttrId,
        dtype: DataType,
    },

    /// Remove an attribute from a variable in a dataset
    RemoveAttributeFromDatasetVar {
        dataset_id: DatasetId,
        var_id: VarId,
        attr_id: AttrId,
    },

    /// Change the data type of an attribute in a dataset variable
    ChangeAttributeType {
        dataset_id: DatasetId,
        var_id: VarId,
        attr_id: AttrId,
        new_dtype: DataType,
    },
}

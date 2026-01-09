//! Variable attributes module.
//!
//! This module contains functionality for reading and writing variable attributes,
//!
//! ## Layout
//!
//! ```text
//! collection.atlas/
//!         ├── datasets/
//!         ├── variables/
//!              ├── variable_0/
//!                  variable.json
//!                  array_meta/
//!                  ...
//!                  attributes/
//!                      attribute_name/
//!                          0_10.json
//!                          12_12.json
//!                      attribute_name2/
//!                          0_5.json
//! ```
//!
//! ### JSON Attribute Files
//! Object keys are dataset indices, and values are attribute values.
//! ```json
//! {
//!    "1": "attribute_value",
//!    "2": 42,
//!    "3": 3.14,
//!    "4": true,
//!    "5": null
//! }
//!
//! ```
//!

pub(crate) mod discovery;
pub mod reader;
pub mod writer;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Attribute {
    pub name: String,
    pub value: AttributeValue,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    Null,
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

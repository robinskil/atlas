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
//!                          0_10.jsonl
//!                          12_12.jsonl
//!                      attribute_name2/
//!                          0_5.jsonl

pub mod reader;
pub mod writer;

pub struct Attribute {
    pub name: String,
    pub value: AttributeValue,
}

pub enum AttributeValue {
    Null,
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

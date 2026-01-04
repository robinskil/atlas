use smol_str::SmolStr;

use crate::{
    schema::Schema,
    variable::{Variable, VariableMut},
};

pub struct Dataset<'l> {
    pub name: &'l str,
    pub variables: Vec<Option<Variable<'l>>>,
}

pub struct DatasetMut<'l> {
    pub collection_schema: &'l mut Schema,
    pub name: String,
    pub variables: Vec<Option<VariableMut>>,
}

impl<'l> DatasetMut<'l> {
    pub fn new(name: String, collection_schema: &'l mut Schema) -> Self {
        collection_schema.datasets.push(name.clone());
        let mut variables = Vec::with_capacity(collection_schema.datasets.len());
        for _ in 0..collection_schema.datasets.len() {
            variables.push(None);
        }

        Self {
            name,
            collection_schema,
            variables,
        }
    }

    // pub fn add_variable<T>(
    //     &mut self,
    //     name: &'l str,
    //     shape: &'l [usize],
    //     chunk_shape: Option<&'l [usize]>,
    //     dimensions: &'l [SmolStr],
    // ) -> VariableMut<'l> {
    //     let chunk_shape = if let Some(cs) = chunk_shape {
    //         cs
    //     } else {
    //         shape
    //     };

    //     // Placeholder implementation
    //     unimplemented!()
    // }

    // pub fn get_variable(&'l self, name: &str) -> Option<VariableMut<'_>> {
    //     let variable_index = self
    //         .collection_schema
    //         .datasets
    //         .iter()
    //         .position(|dataset_name| dataset_name == name)?;

    //     let var_mut = self.variables[variable_index].clone();
    // }

    pub fn remove_variable(&mut self, name: &str) {
        let variable_index = self
            .collection_schema
            .datasets
            .iter()
            .position(|dataset_name| dataset_name == name)
            .expect("Variable not found in schema");
        let dataset_index = self
            .collection_schema
            .datasets
            .iter()
            .position(|dataset_name| dataset_name == &self.name)
            .expect("Dataset not found in schema");

        self.collection_schema.fields[variable_index].dataset_active[dataset_index] = false;
    }
}

use std::sync::Arc;

use smallvec::SmallVec;
use smol_str::SmolStr;

use crate::{
    array::{Array, ArrayMut},
    dtype::DataType,
    io::{AllocationHandle, IOHandler},
};

pub struct Variable<'l> {
    arena_allocator: Arc<IOHandler>,
    name: SmolStr,
    shape: &'l [usize],
    chunk_shape: &'l [usize],
    array_chunk_allocations: &'l [AllocationHandle],
    dimensions: &'l [SmolStr],
    dtype: DataType,
}

impl<'l> Variable<'l> {
    pub fn read_array<T>(&self, slice: Option<&[usize]>) -> Array<'l, T> {
        // Placeholder implementation
        unimplemented!()
    }
}

pub struct VariableMut {
    arena_allocator: Arc<IOHandler>,
    name: SmolStr,
    shape: SmallVec<[usize; 4]>,
    chunk_shape: SmallVec<[usize; 4]>,
    array_chunk_allocations: SmallVec<[AllocationHandle; 4]>,
    dimensions: SmallVec<[SmolStr; 4]>,
    dtype: DataType,
}

impl VariableMut {
    pub fn new(
        arena_allocator: Arc<IOHandler>,
        name: SmolStr,
        shape: &[usize],
        chunk_shape: &[usize],
        dimensions: &[SmolStr],
        dtype: DataType,
    ) -> Self {
        // Placeholder implementation
        unimplemented!()
    }

    pub fn array_mut<T>(&mut self, slice: Option<&[usize]>) -> ArrayMut<'_, T> {
        // Placeholder implementation
        unimplemented!()
    }
}

use crate::io::{AllocationHandle, BufferArena, IOHandler, IOProvider};

pub async fn alloc(io_handle: &IOHandler, field_index: usize, bytes: &[u8]) -> AllocationHandle {
    // Find a free space in the arena and allocate the space for the bytes.

    // Then write the bytes to the IO provider at the correct offset.

    // Return an AllocationHandle representing the allocation.
    unimplemented!()
}

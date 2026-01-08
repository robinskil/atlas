use bytes::Bytes;
use object_store::ObjectMeta;

use crate::variable::array::{Array, datatype::ArrayDataType};

pub struct ArrayBlobMetadata {
    pub start_dataset_index: usize,
    pub end_dataset_index: usize,
    pub num_allocations: usize,
    pub object_meta: ObjectMeta,
}

pub struct ArrayBlob {
    pub bytes: Bytes,
    pub allocations: Vec<Allocation>,
}

impl ArrayBlob {
    pub fn new_from_bytes(bytes: Bytes) -> Self {
        assert!(
            bytes.len() >= 8,
            "ArrayBlob bytes too small to contain header"
        );
        // Parse the allocations from the bytes
        let mut allocations = Vec::new();
        let num_allocations = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let mut forward_offset = 8;
        let mut downward_offset = bytes.len();
        for _ in 0..num_allocations {
            // Allocation sizes are stored upwards
            let size = u64::from_le_bytes(
                bytes[forward_offset..forward_offset + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            forward_offset += 8;

            // The allocations itself are stored downwards
            let allocation = Allocation {
                offset: downward_offset - size,
                size,
            };
            allocations.push(allocation);
            downward_offset -= size;
        }

        Self { bytes, allocations }
    }

    pub fn get_allocation(&self, index: usize) -> Option<&Allocation> {
        self.allocations.get(index)
    }

    pub fn get_allocation_bytes(&self, index: usize) -> Option<Bytes> {
        let allocation = self.get_allocation(index)?;
        Some(
            self.bytes
                .slice(allocation.offset..allocation.offset + allocation.size),
        )
    }

    pub fn get_array<T: ArrayDataType>(
        &self,
        index: usize,
        shape: smallvec::SmallVec<[usize; 4]>,
    ) -> Option<Array<T>> {
        let allocation_bytes = self.get_allocation_bytes(index)?;
        Some(unsafe { Array::new_unchecked(allocation_bytes, shape) })
    }
}

pub struct Allocation {
    pub offset: usize,
    pub size: usize,
}

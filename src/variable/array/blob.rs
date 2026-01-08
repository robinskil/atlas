use bytes::{BufMut, Bytes, BytesMut};
use object_store::{ObjectStore, PutPayload};

use crate::{
    consts,
    variable::array::{Array, Immutable, MutableRef, datatype::ArrayDataType},
};

pub struct ArrayBlobMetadata {
    pub start_dataset_index: usize,
    pub end_dataset_index: usize,
    pub num_allocations: usize,
    pub path: object_store::path::Path,
}

pub struct Committed {
    object_path: object_store::path::Path,
}

pub struct Uncommitted;

pub struct ArrayBlob<T, B> {
    pub state: T,
    pub buffer: B,
    pub allocations: Vec<Allocation>,
}

impl<T, B> ArrayBlob<T, B> {
    pub const BLOB_ALIGNMENT: usize = 8;
    pub async fn from_existing<S: ObjectStore>(
        store: &S,
        path: object_store::path::Path,
    ) -> ArrayBlob<Committed, Bytes> {
        let get_result = store.get(&path).await.unwrap();
        let bytes = get_result.bytes().await.unwrap();
        ArrayBlob::<Committed, Bytes>::new_from_bytes(bytes, Committed { object_path: path })
    }

    pub async fn from_existing_mut<S: ObjectStore>(
        store: &S,
        path: object_store::path::Path,
    ) -> ArrayBlob<Committed, BytesMut> {
        let immutable_blob = Self::from_existing(store, path).await;
        immutable_blob.into_mut_blob()
    }

    pub fn new_mut() -> ArrayBlob<Uncommitted, BytesMut> {
        let mut buffer = BytesMut::new();
        // Write magic bytes for uncompressed header
        buffer.put_slice(consts::MAGIC_NUMBER_UNCOMPRESSED);
        // Reserve space for the header (number of allocations)
        buffer.put_u64_le(0); // Placeholder for number of allocations

        ArrayBlob {
            state: Uncommitted,
            buffer,
            allocations: Vec::new(),
        }
    }

    fn find_next_aligned_ptr(offset: usize, alignment: usize) -> usize {
        let align_mask = alignment - 1;
        (offset + align_mask) & !align_mask
    }

    pub fn new_from_bytes<ST, LB: AsRef<[u8]>>(bytes_buffer: LB, state: ST) -> ArrayBlob<ST, LB> {
        // Buffer bytes should be atleast 16 bytes (8 for magic number + 8 for number of allocations)
        assert!(bytes_buffer.as_ref().len() >= 16);
        let magic_bytes_read = &bytes_buffer.as_ref()[0..8];
        assert!(magic_bytes_read == consts::MAGIC_NUMBER_UNCOMPRESSED);
        let num_allocations =
            u64::from_le_bytes(bytes_buffer.as_ref()[8..16].try_into().unwrap()) as usize;

        let mut offset_ptr = 16;
        let mut allocations = Vec::new();
        for _ in 0..num_allocations {
            // align the offset pointer to 8 bytes
            offset_ptr = Self::find_next_aligned_ptr(offset_ptr, Self::BLOB_ALIGNMENT);
            let size = u64::from_le_bytes(
                bytes_buffer.as_ref()[offset_ptr..offset_ptr + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            allocations.push(Allocation {
                offset: offset_ptr + 8,
                size,
            });
            offset_ptr += size;
        }

        ArrayBlob {
            buffer: bytes_buffer,
            allocations,
            state,
        }
    }
}

impl ArrayBlob<Committed, Bytes> {
    pub async fn commit<S: ObjectStore>(&self, store: S) -> object_store::Result<()> {
        let put_payload = PutPayload::from(self.buffer.clone());
        store.put(&self.state.object_path, put_payload).await?;
        Ok(())
    }
}

impl ArrayBlob<Uncommitted, Bytes> {
    pub async fn commit<S: ObjectStore>(
        self,
        store: S,
        path: object_store::path::Path,
    ) -> object_store::Result<ArrayBlob<Committed, Bytes>> {
        let put_payload = PutPayload::from(self.buffer.clone());
        store.put(&path, put_payload).await?;
        Ok(ArrayBlob {
            state: Committed { object_path: path },
            buffer: self.buffer,
            allocations: self.allocations,
        })
    }
}

impl<T> ArrayBlob<T, Bytes> {
    pub fn into_mut_blob(self) -> ArrayBlob<T, BytesMut> {
        match self.buffer.try_into_mut() {
            Ok(bytes_mut) => ArrayBlob {
                state: self.state,
                buffer: bytes_mut,
                allocations: self.allocations,
            },
            Err(copyable) => {
                // Create a copy of the bytes
                let bytes_mut = bytes::BytesMut::from(copyable.as_ref());
                ArrayBlob {
                    state: self.state,
                    buffer: bytes_mut,
                    allocations: self.allocations,
                }
            }
        }
    }
}

impl<T> ArrayBlob<T, Bytes> {
    pub fn get_allocation(&self, index: usize) -> Option<&Allocation> {
        self.allocations.get(index)
    }

    pub fn get_allocation_bytes(&self, index: usize) -> Option<Bytes> {
        let allocation = self.get_allocation(index)?;
        Some(
            self.buffer
                .slice(allocation.offset..allocation.offset + allocation.size),
        )
    }

    pub fn get_array<A: ArrayDataType>(
        &self,
        index: usize,
        shape: smallvec::SmallVec<[usize; 4]>,
    ) -> Option<Array<A>> {
        let allocation_bytes = self.get_allocation_bytes(index)?;
        Some(unsafe { Array::new_unchecked(Immutable(allocation_bytes), shape) })
    }
}

impl<T> ArrayBlob<T, BytesMut> {
    pub fn into_immutable_blob(self) -> ArrayBlob<T, Bytes> {
        ArrayBlob {
            state: self.state,
            buffer: self.buffer.freeze(),
            allocations: self.allocations,
        }
    }

    pub fn write_array<A: ArrayDataType>(&mut self, array: &Array<A>) -> usize {
        // Find next aligned offset
        let aligned_offset = Self::find_next_aligned_ptr(self.buffer.len(), Self::BLOB_ALIGNMENT);
        // Pad buffer to aligned offset
        while self.buffer.len() < aligned_offset {
            self.buffer.put_u8(0);
        }
        // Write size of the allocation
        let data_as_bytes = array.data.as_ref();
        self.buffer.put_u64_le(data_as_bytes.len() as u64);
        // Write the actual data
        self.buffer.put_slice(data_as_bytes);
        self.allocations.push(Allocation {
            offset: aligned_offset + 8,
            size: data_as_bytes.len(),
        });
        self.allocations.len() - 1 // return the index of the new allocation
    }

    pub fn get_array_mut<A: ArrayDataType>(
        &'_ mut self,
        index: usize,
        shape: smallvec::SmallVec<[usize; 4]>,
    ) -> Option<Array<A, MutableRef<'_>>> {
        let allocation = self.allocations.get(index)?;
        let allocation_bytes =
            &mut self.buffer[allocation.offset..allocation.offset + allocation.size];

        Some(unsafe { Array::new_unchecked(MutableRef(allocation_bytes), shape) })
    }
}

pub struct Allocation {
    pub offset: usize,
    pub size: usize,
}

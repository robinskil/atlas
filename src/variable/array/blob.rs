//! Array blob storage.
//!
//! This module defines [`ArrayBlob`], a compact binary container that stores multiple array
//! payloads (“allocations”) in a single byte buffer.
//!
//! ## Format
//! - 8 bytes: magic number (currently `MAGIC_NUMBER_UNCOMPRESSED`)
//! - 8 bytes: `u64` little-endian allocation count
//! - repeated allocations:
//!   - zero padding to align the next allocation header to 8 bytes
//!   - 8 bytes: `u64` little-endian payload size
//!   - `size` bytes: payload
//!
//! [`ArrayBlob::try_from_bytes`] validates that all allocations are in-bounds.
//!
//! ## Typed access and safety
//! The blob stores *raw bytes*. [`ArrayBlob::get_array`] / [`ArrayBlob::get_array_mut`] interpret
//! an allocation as a typed [`crate::variable::array::Array`] using `Array::new_unchecked`, which
//! skips validation. Use `Array::try_new` if you need validation.
//!
//! ## Writing
//! [`ArrayBlob::write_array`] appends an allocation by copying the array’s bytes into the blob
//! buffer (even if the source array borrows its data, e.g. from an `ndarray`).
//!

use bytes::{BufMut, Bytes, BytesMut};
use object_store::{ObjectStore, PutPayload};

use crate::{
    consts,
    variable::array::{Array, Immutable, MutableRef, datatype::ArrayDataType},
};

/// Errors returned when loading or parsing an [`ArrayBlob`].
#[derive(Debug, thiserror::Error)]
pub enum ArrayBlobError {
    #[error(transparent)]
    ObjectStore(#[from] object_store::Error),
    #[error("blob too short: expected at least {min} bytes, got {actual}")]
    TooShort { min: usize, actual: usize },
    #[error("invalid blob magic bytes")]
    InvalidMagic,
    #[error("truncated blob while reading allocation header")]
    TruncatedAllocationHeader,
    #[error("truncated blob while reading allocation data")]
    TruncatedAllocationData,
    #[error("allocation size overflow")]
    AllocationSizeOverflow,
}

pub struct ArrayBlobMetadata {
    pub start_dataset_index: usize,
    pub end_dataset_index: usize,
    pub num_allocations: usize,
    pub path: object_store::path::Path,
}

/// Marker state for a blob that is persisted in an object store.
pub struct Committed {
    object_path: object_store::path::Path,
}

/// Marker state for a blob that is not yet persisted.
pub struct Uncommitted;

/// A collection of allocations stored in a single backing buffer.
///
/// `state` is a marker type describing whether the blob is committed.
/// `buffer` is either [`Bytes`] (immutable) or [`BytesMut`] (mutable).
pub struct ArrayBlob<T, B> {
    pub state: T,
    pub buffer: B,
    pub allocations: Vec<Allocation>,
}

impl<T, B> ArrayBlob<T, B> {
    pub const BLOB_ALIGNMENT: usize = 8;

    /// Load an existing blob from an [`ObjectStore`].
    pub async fn try_from_existing<S: ObjectStore>(
        store: &S,
        path: object_store::path::Path,
    ) -> Result<ArrayBlob<Committed, Bytes>, ArrayBlobError> {
        let get_result = store.get(&path).await?;
        let bytes = get_result.bytes().await?;
        Ok(ArrayBlob::<Committed, Bytes>::try_from_bytes(
            bytes,
            Committed { object_path: path },
        )?)
    }

    /// Load an existing blob from an [`ObjectStore`] and return a mutable buffer.
    pub async fn try_from_existing_mut<S: ObjectStore>(
        store: &S,
        path: object_store::path::Path,
    ) -> Result<ArrayBlob<Committed, BytesMut>, ArrayBlobError> {
        let immutable_blob = Self::try_from_existing(store, path).await?;
        Ok(immutable_blob.into_mut_blob())
    }

    /// Load an existing blob from an [`ObjectStore`], panicking on error.
    ///
    /// Prefer [`Self::try_from_existing`] in production code.
    pub async fn from_existing<S: ObjectStore>(
        store: &S,
        path: object_store::path::Path,
    ) -> ArrayBlob<Committed, Bytes> {
        Self::try_from_existing(store, path)
            .await
            .expect("failed to load ArrayBlob")
    }

    /// Load an existing blob into a mutable buffer, panicking on error.
    ///
    /// Prefer [`Self::try_from_existing_mut`] in production code.
    pub async fn from_existing_mut<S: ObjectStore>(
        store: &S,
        path: object_store::path::Path,
    ) -> ArrayBlob<Committed, BytesMut> {
        Self::try_from_existing_mut(store, path)
            .await
            .expect("failed to load ArrayBlob")
    }

    /// Create a new, empty blob with a mutable buffer.
    ///
    /// The header is initialized with the magic bytes and a zero allocation count.
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

    /// Parse a blob from bytes.
    ///
    /// This validates the header and that all allocations are in-bounds.
    pub fn try_from_bytes<ST, LB: AsRef<[u8]>>(
        bytes_buffer: LB,
        state: ST,
    ) -> Result<ArrayBlob<ST, LB>, ArrayBlobError> {
        let buf = bytes_buffer.as_ref();
        if buf.len() < 16 {
            return Err(ArrayBlobError::TooShort {
                min: 16,
                actual: buf.len(),
            });
        }

        if &buf[0..8] != consts::MAGIC_NUMBER_UNCOMPRESSED {
            return Err(ArrayBlobError::InvalidMagic);
        }

        let num_allocations =
            u64::from_le_bytes(
                buf[8..16]
                    .try_into()
                    .map_err(|_| ArrayBlobError::TooShort {
                        min: 16,
                        actual: buf.len(),
                    })?,
            ) as usize;

        let mut offset_ptr = 16usize;
        let mut allocations = Vec::with_capacity(num_allocations);
        for _ in 0..num_allocations {
            offset_ptr = Self::find_next_aligned_ptr(offset_ptr, Self::BLOB_ALIGNMENT);
            if offset_ptr
                .checked_add(8)
                .ok_or(ArrayBlobError::AllocationSizeOverflow)?
                > buf.len()
            {
                return Err(ArrayBlobError::TruncatedAllocationHeader);
            }

            let size_u64 = u64::from_le_bytes(
                buf[offset_ptr..offset_ptr + 8]
                    .try_into()
                    .map_err(|_| ArrayBlobError::TruncatedAllocationHeader)?,
            );
            let size: usize = size_u64
                .try_into()
                .map_err(|_| ArrayBlobError::AllocationSizeOverflow)?;

            let data_offset = offset_ptr + 8;
            let end = data_offset
                .checked_add(size)
                .ok_or(ArrayBlobError::AllocationSizeOverflow)?;
            if end > buf.len() {
                return Err(ArrayBlobError::TruncatedAllocationData);
            }

            allocations.push(Allocation {
                offset: data_offset,
                size,
            });
            offset_ptr = end;
        }

        Ok(ArrayBlob {
            buffer: bytes_buffer,
            allocations,
            state,
        })
    }

    /// Backwards-compatible constructor that panics on invalid blobs.
    pub fn new_from_bytes<ST, LB: AsRef<[u8]>>(bytes_buffer: LB, state: ST) -> ArrayBlob<ST, LB> {
        Self::try_from_bytes(bytes_buffer, state).expect("invalid ArrayBlob bytes")
    }
}

impl ArrayBlob<Committed, Bytes> {
    /// Persist the blob back to its committed object path.
    pub async fn commit<S: ObjectStore>(&self, store: S) -> object_store::Result<()> {
        let put_payload = PutPayload::from(self.buffer.clone());
        store.put(&self.state.object_path, put_payload).await?;
        Ok(())
    }
}

impl ArrayBlob<Uncommitted, Bytes> {
    /// Persist an uncommitted blob to `path` and return a committed blob handle.
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
    /// Convert an immutable blob to a mutable blob.
    ///
    /// This is zero-copy when the underlying [`Bytes`] can be converted to [`BytesMut`].
    /// Otherwise, the data is copied.
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
    /// Return the allocation metadata for `index`.
    pub fn get_allocation(&self, index: usize) -> Option<&Allocation> {
        self.allocations.get(index)
    }

    /// Return the raw bytes for allocation `index`.
    pub fn get_allocation_bytes(&self, index: usize) -> Option<Bytes> {
        let allocation = self.get_allocation(index)?;
        Some(
            self.buffer
                .slice(allocation.offset..allocation.offset + allocation.size),
        )
    }

    /// Interpret allocation `index` as a typed [`Array`].
    ///
    /// # Safety / invariants
    /// This method does **not** validate that the stored bytes match `A` and `shape`.
    /// If you need validation, construct via `Array::try_new` instead.
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
    /// Freeze the blob into an immutable buffer.
    pub fn into_immutable_blob(self) -> ArrayBlob<T, Bytes> {
        ArrayBlob {
            state: self.state,
            buffer: self.buffer.freeze(),
            allocations: self.allocations,
        }
    }

    /// Append an [`Array`] payload to the blob and return its allocation index.
    ///
    /// The bytes are copied into the internal buffer. The allocation count in the
    /// header is updated.
    pub fn write_array<A: ArrayDataType, B: AsRef<[u8]>>(&mut self, array: &Array<A, B>) -> usize {
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

        // Update allocation count in header.
        let count = self.allocations.len() as u64;
        self.buffer[8..16].copy_from_slice(&count.to_le_bytes());

        self.allocations.len() - 1 // return the index of the new allocation
    }

    /// Get a mutable typed view of allocation `index`.
    ///
    /// # Safety / invariants
    /// This method does **not** validate that the stored bytes match `A` and `shape`.
    /// Mutating bytes to an invalid representation may cause later consumers to fail.
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

/// A single allocation stored in a blob.
///
/// `offset` points to the start of the payload bytes (not the 8-byte size header).
pub struct Allocation {
    pub offset: usize,
    pub size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::array::Array;

    #[test]
    fn new_mut_writes_header() {
        let blob = ArrayBlob::<Uncommitted, BytesMut>::new_mut();
        assert_eq!(&blob.buffer[0..8], consts::MAGIC_NUMBER_UNCOMPRESSED);
        assert_eq!(
            u64::from_le_bytes(blob.buffer[8..16].try_into().unwrap()),
            0
        );
        assert!(blob.allocations.is_empty());
    }

    #[test]
    fn write_array_updates_count_and_roundtrips() {
        let arr = Array::<u8>::try_new(
            Immutable(Bytes::from(vec![1u8, 2, 3, 4])),
            smallvec::smallvec![4],
        )
        .unwrap();

        let mut blob = ArrayBlob::<Uncommitted, BytesMut>::new_mut();
        let idx0 = blob.write_array(&arr);
        let idx1 = blob.write_array(&arr);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(
            u64::from_le_bytes(blob.buffer[8..16].try_into().unwrap()),
            2
        );

        let frozen = blob.into_immutable_blob().buffer;
        let parsed = ArrayBlob::<Uncommitted, Bytes>::try_from_bytes(frozen, Uncommitted).unwrap();
        assert_eq!(parsed.allocations.len(), 2);

        let a0 = parsed.get_array::<u8>(0, smallvec::smallvec![4]).unwrap();
        let got0: Vec<u8> = a0.as_ndarray().iter().copied().collect();
        assert_eq!(got0, vec![1, 2, 3, 4]);
    }

    #[test]
    fn write_ndarray_multidim_roundtrips() {
        let nd = ndarray::Array::from_shape_vec((2, 2), vec![10i32, 20, 30, 40])
            .unwrap()
            .into_dyn();
        let arr = Array::<i32>::try_from_ndarray(&nd).unwrap();

        let mut blob = ArrayBlob::<Uncommitted, BytesMut>::new_mut();
        let idx = blob.write_array(&arr);
        assert_eq!(idx, 0);

        let frozen = blob.into_immutable_blob().buffer;
        let parsed = ArrayBlob::<Uncommitted, Bytes>::try_from_bytes(frozen, Uncommitted).unwrap();
        assert_eq!(parsed.allocations.len(), 1);

        let got = parsed
            .get_array::<i32>(0, smallvec::smallvec![2, 2])
            .unwrap();
        let values: Vec<i32> = got.as_ndarray().iter().copied().collect();
        assert_eq!(values, vec![10, 20, 30, 40]);
    }

    #[test]
    fn write_multiple_ndarrays_roundtrip_and_alignment() {
        let a_u16 = ndarray::Array::from_shape_vec((3,), vec![1u16, 2, 500])
            .unwrap()
            .into_dyn();
        let a_f32 = ndarray::Array::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.5, 4.25, 5.0, 6.0])
            .unwrap()
            .into_dyn();

        let arr_u16 = Array::<u16>::try_from_ndarray(&a_u16).unwrap();
        let arr_f32 = Array::<f32>::try_from_ndarray(&a_f32).unwrap();

        let mut blob = ArrayBlob::<Uncommitted, BytesMut>::new_mut();
        let idx0 = blob.write_array(&arr_u16);
        let idx1 = blob.write_array(&arr_f32);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(
            u64::from_le_bytes(blob.buffer[8..16].try_into().unwrap()),
            2
        );

        // Data offsets should be 8-byte aligned.
        assert_eq!(
            blob.allocations[0].offset % ArrayBlob::<Uncommitted, BytesMut>::BLOB_ALIGNMENT,
            0
        );
        assert_eq!(
            blob.allocations[1].offset % ArrayBlob::<Uncommitted, BytesMut>::BLOB_ALIGNMENT,
            0
        );

        let frozen = blob.into_immutable_blob().buffer;
        let parsed = ArrayBlob::<Uncommitted, Bytes>::try_from_bytes(frozen, Uncommitted).unwrap();
        assert_eq!(parsed.allocations.len(), 2);

        let got0 = parsed.get_array::<u16>(0, smallvec::smallvec![3]).unwrap();
        let got0_vals: Vec<u16> = got0.as_ndarray().iter().copied().collect();
        assert_eq!(got0_vals, vec![1, 2, 500]);

        let got1 = parsed
            .get_array::<f32>(1, smallvec::smallvec![2, 3])
            .unwrap();
        let got1_vals: Vec<f32> = got1.as_ndarray().iter().copied().collect();
        assert_eq!(got1_vals, vec![1.0, 2.0, 3.5, 4.25, 5.0, 6.0]);
    }

    #[test]
    fn try_from_bytes_rejects_invalid_magic() {
        let mut buf = vec![0u8; 16];
        buf[0..8].copy_from_slice(b"BADMAGIC");
        assert!(matches!(
            ArrayBlob::<Uncommitted, Bytes>::try_from_bytes(Bytes::from(buf), Uncommitted),
            Err(ArrayBlobError::InvalidMagic)
        ));
    }

    #[test]
    fn try_from_bytes_rejects_truncated_allocation() {
        let mut blob = ArrayBlob::<Uncommitted, BytesMut>::new_mut();
        // Set allocation count to 1 but don't provide allocation header.
        blob.buffer[8..16].copy_from_slice(&1u64.to_le_bytes());
        let frozen = blob.into_immutable_blob().buffer;
        assert!(matches!(
            ArrayBlob::<Uncommitted, Bytes>::try_from_bytes(frozen, Uncommitted),
            Err(ArrayBlobError::TruncatedAllocationHeader)
        ));
    }

    #[tokio::test]
    async fn from_existing_roundtrip_in_memory_store() {
        use object_store::memory::InMemory;
        use object_store::path::Path;

        let arr = Array::<u8>::try_new(
            Immutable(Bytes::from(vec![9u8, 8, 7])),
            smallvec::smallvec![3],
        )
        .unwrap();

        let mut blob = ArrayBlob::<Uncommitted, BytesMut>::new_mut();
        blob.write_array(&arr);
        let frozen = blob.into_immutable_blob().buffer;

        let store = InMemory::new();
        let path = Path::from("blob.bin");
        store
            .put(&path, PutPayload::from(frozen.clone()))
            .await
            .unwrap();

        let loaded = ArrayBlob::<Committed, Bytes>::try_from_existing(&store, path.clone())
            .await
            .unwrap();
        assert_eq!(loaded.allocations.len(), 1);
        let a0 = loaded.get_array::<u8>(0, smallvec::smallvec![3]).unwrap();
        let got: Vec<u8> = a0.as_ndarray().iter().copied().collect();
        assert_eq!(got, vec![9, 8, 7]);
    }
}

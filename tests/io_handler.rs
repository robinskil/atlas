// use std::sync::Arc;

// use bytes::Bytes;
// use futures::future::BoxFuture;

// use atlas::io::{AtlasIoError, IOHandler, IOProvider};

// #[derive(Default)]
// struct MemIOProvider {
//     data: parking_lot::RwLock<Vec<u8>>,
// }

// impl MemIOProvider {
//     fn new() -> Self {
//         Self::default()
//     }
// }

// impl IOProvider for MemIOProvider {
//     fn read_at(&self, offset: u64, len: usize) -> BoxFuture<'_, std::io::Result<Bytes>> {
//         Box::pin(async move {
//             let offset = offset as usize;
//             let data = self.data.read();
//             let end = offset.saturating_add(len);
//             if end > data.len() {
//                 return Err(std::io::Error::new(
//                     std::io::ErrorKind::UnexpectedEof,
//                     "short read",
//                 ));
//             }
//             Ok(Bytes::copy_from_slice(&data[offset..end]))
//         })
//     }

//     fn write_at(&self, bytes: Bytes, offset: u64) -> BoxFuture<'_, std::io::Result<()>> {
//         Box::pin(async move {
//             let offset = offset as usize;
//             let end = offset.saturating_add(bytes.len());
//             let mut data = self.data.write();
//             if end > data.len() {
//                 data.resize(end, 0);
//             }
//             data[offset..end].copy_from_slice(&bytes);
//             Ok(())
//         })
//     }
// }

// #[tokio::test]
// async fn alloc_and_get_allocation_roundtrip() {
//     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::new());
//     let io = IOHandler::new(provider, 0);

//     let handle = io.alloc(0, b"hello").await.unwrap();
//     let got = io.get_allocation(handle).await.unwrap();

//     assert_eq!(&got[..], b"hello");
// }

// #[tokio::test]
// async fn realloc_same_size_overwrites_in_place() {
//     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::new());
//     let io = IOHandler::new(provider, 0);

//     let handle = io.alloc(0, b"abcd").await.unwrap();
//     let handle2 = io.realloc(&handle, b"WXYZ").await.unwrap();

//     assert_eq!(handle.field_index, handle2.field_index);
//     assert_eq!(handle.arena_index, handle2.arena_index);
//     assert_eq!(handle.offset, handle2.offset);
//     assert_eq!(handle.size, handle2.size);

//     let got = io.get_allocation(handle2).await.unwrap();
//     assert_eq!(&got[..], b"WXYZ");
// }

// #[tokio::test]
// async fn realloc_new_size_allocates_new_chunk() {
//     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::new());
//     let io = IOHandler::new(provider, 0);

//     let handle = io.alloc(0, b"aa").await.unwrap();
//     let handle2 = io.realloc(&handle, b"bbbb").await.unwrap();

//     assert_eq!(handle2.size, 4);

//     let got = io.get_allocation(handle2).await.unwrap();
//     assert_eq!(&got[..], b"bbbb");
// }

// #[tokio::test]
// async fn invalid_handle_returns_error() {
//     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::new());
//     let io = IOHandler::new(provider, 0);

//     let bad = atlas::io::AllocationHandle {
//         field_index: 0,
//         arena_index: 999,
//         offset: 0,
//         size: 1,
//     };

//     let err = io.get_allocation(bad).await.unwrap_err();
//     assert!(matches!(err, AtlasIoError::InvalidHandle));
// }

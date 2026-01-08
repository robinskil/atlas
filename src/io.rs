// //! IO + arena allocator primitives.
// //!
// //! This module provides a small, random-access IO abstraction (`IOProvider`) plus an arena-based
// //! allocator (`IOHandler`) used to store variable-sized byte allocations in a file-like backing
// //! store.
// //!
// //! **High level design**
// //! - The file is partitioned into *arenas*. Each arena is a contiguous region on disk with a fixed
// //!   logical size.
// //! - Arenas are tracked per field (variable) as `Vec<Vec<Arc<BufferArena>>>`.
// //! - An allocation is addressed by an [`AllocationHandle`], which contains:
// //!   `(field_index, arena_index, offset, size)`.
// //!
// //! **On-disk layout (conceptual)**
// //! ```text
// //! 0                                                       EOF
// //! +----------------------+-----------------------------------------------+
// //! | MAGIC_NUMBER bytes   | arena data...                                  |
// //! +----------------------+-----------------------------------------------+
// //!                        ^
// //!                        `IOHandler::new(..., arena_offset)` typically points here
// //!                         (often `MAGIC_NUMBER.len()`)
// //! ```
// //!
// //! Arenas are stored as raw byte regions in the backing store. An arena can be:
// //! - uncompressed: a fixed-size region reserved on disk
// //! - compressed: a variable-size blob whose *logical* size is described by its `inner` metadata
// //!
// //! ```text
// //! Example after some allocations (not to scale)
// //! +----------------------+--------------------+-------------------+-------+
// //! | MAGIC_NUMBER         | Arena A (raw)      | Arena B (zstd)    | ...   |
// //! +----------------------+--------------------+-------------------+-------+
// //!                        ^ offset_A           ^ offset_B
// //!                        size = 1MiB          size = compressed_len
// //! ```
// //!
// //! **How handles map to bytes**
// //! ```text
// //! field_arenas[field_index][arena_index] -> BufferArena (metadata)
// //!                                      -> on-disk bytes at `arena.offset`
// //! AllocationHandle { offset, size } slices within the *uncompressed* arena bytes:
// //!   arena_bytes[offset .. offset + size]
// //! ```
// //!
// //! **Compaction / rewrite (GC + compression)**
// //! [`IOHandler::gc_and_compress`] rewrites arenas in ascending on-disk offset order into a
// //! compact region starting at `MAGIC_NUMBER.len()`:
// //! ```text
// //! Before (sparse / mixed)
// //! +----------------------+-----A-----+--free--+---C---+--free--+--B--+----+
// //! | MAGIC_NUMBER         |raw/uncomp |        |zstd    |        |raw  |    |
// //! +----------------------+-----------+--------+--------+--------+-----+----+
// //!                        ^ old offsets are arbitrary / non-contiguous
// //!
// //! After (compact + compressed)
// //! +----------------------+--A'--+---C'---+--B'--+
// //! | MAGIC_NUMBER         |zstd  |zstd    |zstd  |
// //! +----------------------+----- +--------+------+
// //!                        ^ write_cursor starts here and advances sequentially
// //! ```
// //!
// //! Arena indices are preserved: if an arena is marked GC, its slot becomes `None` in the
// //! returned metadata instead of shifting indices.
// //!
// //! **Caching**
// //! - `IOHandler` caches *uncompressed* arena bytes in memory.
// //! - Writes invalidate the corresponding cache entry.
// //!
// //! **Compression / GC**
// //! - Arenas may be stored compressed (`BufferArena::Compressed`) or uncompressed
// //!   (`BufferArena::Uncompressed`).
// //! - [`IOHandler::gc_and_compress`] rewrites all non-GC arenas sequentially and ensures the
// //!   returned arenas are compressed, while preserving arena indices (GC'd arenas become `None`).
// //!
// //! **Important invariants**
// //! - Allocation handles are validated on read: out-of-bounds or missing arenas return
// //!   [`AtlasIoError::InvalidHandle`].
// //! - This allocator is append-only within an arena; it does not currently reclaim freed space.
// use std::os::unix::fs::FileExt;
// use std::sync::Arc;
// use std::sync::atomic::{AtomicBool, AtomicU64};

// use bytes::{Bytes, BytesMut};
// use futures::future::BoxFuture;
// use thiserror::Error;

// use crate::header::HeaderArenaHandle;

// /// Errors returned by the arena allocator / IO layer.
// ///
// /// This error type intentionally groups:
// /// - underlying IO failures (`std::io::Error`),
// /// - validation failures (e.g. invalid handles),
// /// - compression / cache errors.
// #[derive(Debug, Error)]
// pub enum AtlasIoError {
//     #[error(transparent)]
//     Io(#[from] std::io::Error),

//     #[error("invalid allocation handle")]
//     InvalidHandle,

//     #[error("allocations onto compressed arenas are not supported")]
//     CompressedArenaUnsupported,

//     #[error("compression error: {0}")]
//     Compression(String),

//     #[error("cache error: {0}")]
//     Cache(String),
// }

// /// Result alias used throughout the IO/allocator subsystem.
// pub type AtlasResult<T> = Result<T, AtlasIoError>;

// /// Abstraction over random-access reads and writes used by `IOHandler`.
// ///
// /// Implementations must support:
// /// - reading exactly `len` bytes from an absolute file offset,
// /// - writing the full buffer to an absolute file offset.
// ///
// /// Semantics:
// /// - `read_at(offset, len)` returns an error on short reads (e.g. EOF before `len`).
// /// - `write_at(bytes, offset)` returns an error on short writes.
// ///
// /// The interface is async via boxed futures so `IOHandler` can use different backends
// /// (in-memory, OS files, etc.) without committing to a single async IO type.
// pub trait IOProvider {
//     /// Read exactly `len` bytes from `offset`.
//     fn read_at(&self, offset: u64, len: usize) -> BoxFuture<'_, std::io::Result<Bytes>>;

//     /// Write all `bytes` to `offset`.
//     fn write_at(&self, bytes: Bytes, offset: u64) -> BoxFuture<'_, std::io::Result<()>>;

//     /// File size in bytes, if known.
//     fn file_size(&self) -> BoxFuture<'_, std::io::Result<u64>>;
// }

// /// `IOProvider` implementation backed by a `std::fs::File`.
// ///
// /// Uses `std::os::unix::fs::FileExt::{read_at, write_at}` for positioned IO.
// /// The syscalls are executed inside `tokio::spawn_blocking` so they do not block
// /// the async runtime.
// pub struct AtlasStdFile {
//     file: Arc<std::fs::File>,
//     tokio_handle: tokio::runtime::Handle,
// }

// // impl IO provider for wrapping std::fs::File in an Arc
// impl IOProvider for AtlasStdFile {
//     fn read_at(&self, offset: u64, len: usize) -> BoxFuture<'_, std::io::Result<Bytes>> {
//         let file = self.file.clone();
//         let handle = self.tokio_handle.clone();

//         Box::pin(async move {
//             let bytes = handle
//                 .spawn_blocking(move || {
//                     let mut buf = vec![0u8; len];
//                     let mut read_total = 0usize;
//                     while read_total < buf.len() {
//                         let n = file.read_at(&mut buf[read_total..], offset + read_total as u64)?;
//                         if n == 0 {
//                             return Err(std::io::Error::new(
//                                 std::io::ErrorKind::UnexpectedEof,
//                                 "short read",
//                             ));
//                         }
//                         read_total += n;
//                     }
//                     Ok::<Bytes, std::io::Error>(Bytes::from(buf))
//                 })
//                 .await
//                 .map_err(|e| std::io::Error::other(e.to_string()))??;

//             Ok(bytes)
//         })
//     }

//     fn write_at(&self, bytes: Bytes, offset: u64) -> BoxFuture<'_, std::io::Result<()>> {
//         let file = self.file.clone();
//         let handle = self.tokio_handle.clone();

//         Box::pin(async move {
//             handle
//                 .spawn_blocking(move || {
//                     let mut written_total = 0usize;
//                     while written_total < bytes.len() {
//                         let n =
//                             file.write_at(&bytes[written_total..], offset + written_total as u64)?;
//                         if n == 0 {
//                             return Err(std::io::Error::new(
//                                 std::io::ErrorKind::WriteZero,
//                                 "short write",
//                             ));
//                         }
//                         written_total += n;
//                     }
//                     Ok::<(), std::io::Error>(())
//                 })
//                 .await
//                 .map_err(|e| std::io::Error::other(e.to_string()))??;

//             Ok(())
//         })
//     }

//     fn file_size(&self) -> BoxFuture<'_, std::io::Result<u64>> {
//         let file = self.file.clone();
//         let handle = self.tokio_handle.clone();

//         Box::pin(async move {
//             handle
//                 .spawn_blocking(move || {
//                     let metadata = file.metadata()?;
//                     Ok::<u64, std::io::Error>(metadata.len())
//                 })
//                 .await
//                 .map_err(|e| std::io::Error::other(e.to_string()))?
//         })
//     }
// }

// pub struct IOHandler {
//     io_provider: Arc<dyn IOProvider + Send + Sync>,
//     io_buffer_cache: moka::future::Cache<BufferArenaKey, Bytes>, // Always stores bytes uncompressed.
//     file_offset: AtomicU64,
//     field_arenas_indexes: parking_lot::RwLock<Vec<Vec<usize>>>, // one vec per field. Each vec contains arena indices for that field(variable).
//     arenas: parking_lot::RwLock<Vec<Option<BufferArena>>>,      // flat list of all arenas.
//     field_arenas: parking_lot::RwLock<Vec<Vec<Arc<BufferArena>>>>, // one vec per field. Each vec contains arenas for that field(variable).
// }

// #[derive(Clone)]
// struct ArenaEntry {
//     field_index: usize,
//     arena_index: usize,
//     arena: Arc<BufferArena>,
//     offset: u64,
// }

// impl IOHandler {
//     const ARENA_SIZE: usize = 1024 * 1024; // 1 MiB
//     const FILE_OFFSET_START: u64 =
//         crate::consts::MAGIC_NUMBER.len() as u64 + std::mem::size_of::<HeaderArenaHandle>() as u64;

//     /// Create a new `IOHandler` writing arenas starting at `arena_offset`.
//     ///
//     /// - `f` is the underlying storage backend.
//     /// - `arena_offset` is the absolute file offset where new arenas begin.
//     ///
//     /// Internals:
//     /// - Maintains an in-memory index of arenas per field.
//     /// - Maintains a cache of *uncompressed* arena bytes keyed by `(field_index, arena_index)`.
//     ///   The cache is invalidated on writes.
//     pub fn new<F: Into<Arc<dyn IOProvider + Send + Sync>>>(f: F) -> Self {
//         let cache = moka::future::Cache::builder()
//             .max_capacity(1024 * 1024 * 1024) // 1 GiB
//             .build();

//         Self {
//             file_offset: AtomicU64::new(Self::FILE_OFFSET_START),
//             io_provider: f.into(),
//             io_buffer_cache: cache,
//             arenas: parking_lot::RwLock::new(Vec::new()),
//             field_arenas_indexes: parking_lot::RwLock::new(Vec::new()),
//             field_arenas: parking_lot::RwLock::new(Vec::new()),
//         }
//     }

//     /// Borrow the underlying `IOProvider`.
//     ///
//     /// This is primarily useful for tests or higher-level code that needs to perform
//     /// its own IO against the same backing store.
//     pub fn io_provider(&self) -> &Arc<dyn IOProvider + Send + Sync> {
//         &self.io_provider
//     }

//     fn arena_offset(arena: &BufferArena) -> u64 {
//         match arena {
//             BufferArena::Compressed(compressed) => compressed.offset,
//             BufferArena::Uncompressed(uncompressed) => uncompressed.offset,
//         }
//     }

//     fn arena_is_gc(arena: &BufferArena) -> bool {
//         match arena {
//             BufferArena::Compressed(compressed) => compressed
//                 .inner
//                 .gc
//                 .load(std::sync::atomic::Ordering::SeqCst),
//             BufferArena::Uncompressed(uncompressed) => {
//                 uncompressed.gc.load(std::sync::atomic::Ordering::SeqCst)
//             }
//         }
//     }

//     fn snapshot_arena_entries(&self) -> (Vec<Vec<Option<BufferArena>>>, Vec<ArenaEntry>) {
//         let arenas_guard = self.field_arenas.read();

//         let mut output_field_arenas: Vec<Vec<Option<BufferArena>>> =
//             Vec::with_capacity(arenas_guard.len());
//         let mut entries: Vec<ArenaEntry> = Vec::new();

//         for (field_index, field_arenas) in arenas_guard.iter().enumerate() {
//             output_field_arenas.push(vec![None; field_arenas.len()]);

//             for (arena_index, arena) in field_arenas.iter().enumerate() {
//                 entries.push(ArenaEntry {
//                     field_index,
//                     arena_index,
//                     offset: Self::arena_offset(arena.as_ref()),
//                     arena: arena.clone(),
//                 });
//             }
//         }

//         (output_field_arenas, entries)
//     }

//     async fn rewrite_compressed_arena(
//         &self,
//         arena: &BufferArenaCompressed,
//         write_cursor: u64,
//     ) -> AtlasResult<BufferArenaCompressed> {
//         if write_cursor != arena.offset {
//             let arena_bytes = self
//                 .io_provider
//                 .read_at(arena.offset, arena.size as usize)
//                 .await?;
//             self.io_provider.write_at(arena_bytes, write_cursor).await?;
//         }

//         let mut new_arena = arena.clone();
//         new_arena.offset = write_cursor;
//         Ok(new_arena)
//     }

//     async fn compress_uncompressed_arena(
//         &self,
//         arena: &BufferArenaUncompressed,
//         write_cursor: u64,
//     ) -> AtlasResult<BufferArenaCompressed> {
//         let arena_bytes = self
//             .io_provider
//             .read_at(arena.offset, arena.size as usize)
//             .await?;

//         let compressed_buf = zstd::bulk::compress(&arena_bytes, 0)
//             .map_err(|e| AtlasIoError::Compression(format!("failed to compress arena: {}", e)))?;
//         let compressed_len = compressed_buf.len() as u64;

//         self.io_provider
//             .write_at(Bytes::from(compressed_buf), write_cursor)
//             .await?;

//         Ok(BufferArenaCompressed {
//             offset: write_cursor,
//             size: compressed_len,
//             inner: arena.clone(),
//         })
//     }

//     async fn gc_and_compress_one(
//         &self,
//         arena: &BufferArena,
//         write_cursor: u64,
//     ) -> AtlasResult<(BufferArena, u64)> {
//         let compressed = match arena {
//             BufferArena::Compressed(arena) => {
//                 self.rewrite_compressed_arena(arena, write_cursor).await?
//             }
//             BufferArena::Uncompressed(arena) => {
//                 self.compress_uncompressed_arena(arena, write_cursor)
//                     .await?
//             }
//         };

//         let size = compressed.size;
//         Ok((BufferArena::Compressed(compressed), size))
//     }

//     /// Compact and compress all arenas that aren't compressed yet. Write them back to disk.
//     /// Returns
//     /// - A vector of vectors of BufferArenas, one per field(variable).
//     /// - The new file offset after writing all arenas.
//     ///
//     /// This method:
//     /// - Consumes this `IOHandler` (so the in-memory arena cache and bookkeeping are dropped).
//     /// - Iterates arenas in ascending on-disk offset order and rewrites them sequentially
//     ///   starting at `crate::consts::MAGIC_NUMBER.len()`.
//     /// - Ensures *all* returned non-GC arenas are `BufferArena::Compressed`.
//     /// - Does **not** shift arena indices within a field: the returned structure has the same
//     ///   shape as the input `field_arenas`, but entries may become `None` if the arena is GC'd.
//     ///
//     /// GC semantics:
//     /// - If an arena has its `gc` flag set, the corresponding `(field_index, arena_index)` slot
//     ///   is returned as `None` and the arena is omitted from the rewritten output.
//     ///
//     /// Notes:
//     /// - Allocation handles remain index-based (field/arena/offset/size). When consuming the
//     ///   result, callers must consult the returned `Vec<Vec<Option<BufferArena>>>` to resolve
//     ///   handles; a `None` entry means the handle is no longer valid.
//     pub async fn gc_and_compress(self) -> AtlasResult<(Vec<Vec<Option<BufferArena>>>, u64)> {
//         let (mut output_field_arenas, mut entries) = self.snapshot_arena_entries();

//         // Rewrite in on-disk order so we write sequentially.
//         entries.sort_by_key(|e| e.offset);

//         let mut write_cursor = crate::consts::MAGIC_NUMBER.len() as u64; // Start after magic number.

//         for entry in entries {
//             let arena = entry.arena.as_ref();
//             if Self::arena_is_gc(arena) {
//                 continue;
//             }

//             let (out_arena, written_size) = self.gc_and_compress_one(arena, write_cursor).await?;
//             output_field_arenas[entry.field_index][entry.arena_index] = Some(out_arena);
//             write_cursor += written_size;
//         }

//         Ok((output_field_arenas, write_cursor))
//     }

//     /// Allocate `bytes` into the arena list for `field_index` and return an `AllocationHandle`.
//     ///
//     /// Behavior:
//     /// - Finds an existing uncompressed arena with enough remaining capacity.
//     /// - Otherwise creates a new arena and appends it to that field's arena list.
//     /// - Writes `bytes` into the chosen arena and returns a handle pointing at the
//     ///   written region.
//     ///
//     /// Notes:
//     /// - This allocator is append-only per arena; it does not currently reuse freed space.
//     /// - Allocations onto compressed arenas are not supported and return
//     ///   `AtlasIoError::CompressedArenaUnsupported`.
//     pub async fn alloc(&self, field_index: usize, bytes: &[u8]) -> AtlasResult<AllocationHandle> {
//         // Step 1: under lock, ensure the field exists and try to find a free arena.
//         let maybe_existing: Option<(Arc<BufferArena>, usize)> = {
//             let mut arenas_guard = self.field_arenas.write();
//             if arenas_guard.len() <= field_index {
//                 arenas_guard.resize_with(field_index + 1, Vec::new);
//             }
//             let field_arenas = &arenas_guard[field_index];
//             Self::find_free_arena(field_arenas, bytes.len())
//         };

//         // Step 2: if none found, create a new arena without holding the lock.
//         let (allocatable_arena, arena_index) = match maybe_existing {
//             Some((arena, index)) => (arena, index),
//             None => {
//                 let arena =
//                     Self::create_arena(self.io_provider.clone(), &self.file_offset, bytes.len())
//                         .await?;

//                 // Re-acquire lock only to append the arena and get its index.
//                 let mut arenas_guard = self.field_arenas.write();
//                 if arenas_guard.len() <= field_index {
//                     arenas_guard.resize_with(field_index + 1, Vec::new);
//                 }
//                 let field_arenas = &mut arenas_guard[field_index];
//                 field_arenas.push(arena.clone());
//                 let index = field_arenas.len() - 1;
//                 (arena, index)
//             }
//         };

//         match allocatable_arena.as_ref() {
//             BufferArena::Uncompressed(uncompressed_arena) => {
//                 return self
//                     .allocate_onto_uncompressed_buffer_arena(
//                         uncompressed_arena,
//                         field_index,
//                         arena_index,
//                         bytes,
//                     )
//                     .await;
//             }
//             BufferArena::Compressed(_) => {
//                 // For compressed arenas, we would need to decompress, allocate onto new arena and push to list.
//                 Err(AtlasIoError::CompressedArenaUnsupported)
//             }
//         }
//     }

//     /// Reallocate an existing allocation to contain `bytes`.
//     ///
//     /// Semantics:
//     /// - If `bytes.len()` matches `handle.size` and the underlying arena is uncompressed,
//     ///   the bytes are overwritten in-place and the same handle is returned.
//     /// - Otherwise, a new allocation is created via `alloc`, the bytes are written there,
//     ///   and the old arena's allocation counter is decremented.
//     /// - If the old arena's allocation counter reaches zero, it is marked `gc = true`.
//     ///
//     /// Notes:
//     /// - This does not copy the old allocation contents; the caller supplies the full
//     ///   new contents.
//     pub async fn realloc(
//         &self,
//         handle: &AllocationHandle,
//         bytes: &[u8],
//     ) -> AtlasResult<AllocationHandle> {
//         let new_size = bytes.len() as u64;
//         let field_index = handle.field_index;

//         // Fast path: if size is unchanged and the arena is uncompressed, overwrite in place.
//         if handle.size == new_size {
//             let arena = self.get_arena(field_index, handle.arena_index)?;

//             if let BufferArena::Uncompressed(uncompressed_arena) = arena.as_ref() {
//                 let file_offset = uncompressed_arena.offset + handle.offset;
//                 self.io_provider
//                     .write_at(Bytes::copy_from_slice(bytes), file_offset)
//                     .await?;

//                 uncompressed_arena
//                     .dirty
//                     .store(true, std::sync::atomic::Ordering::SeqCst);

//                 self.io_buffer_cache
//                     .invalidate(&BufferArenaKey {
//                         field_index,
//                         arena_index: handle.arena_index,
//                     })
//                     .await;

//                 return Ok(*handle);
//             }
//         }

//         // Allocate a new chunk and write the new bytes.
//         let new_handle = self.alloc(field_index, bytes).await?;

//         // Decrease the allocation count on the old arena; mark for GC if it reached zero.
//         let old_arena = self.get_arena(field_index, handle.arena_index)?;

//         if let BufferArena::Uncompressed(uncompressed_arena) = old_arena.as_ref() {
//             let remaining = saturating_atomic_decrement(&uncompressed_arena.num_allocations);
//             if remaining == 0 {
//                 uncompressed_arena
//                     .gc
//                     .store(true, std::sync::atomic::Ordering::SeqCst);
//             }
//         }

//         Ok(new_handle)
//     }

//     fn get_arena(&self, field_index: usize, arena_index: usize) -> AtlasResult<Arc<BufferArena>> {
//         let arenas_guard = self.field_arenas.read();
//         let field_arenas = arenas_guard
//             .get(field_index)
//             .ok_or(AtlasIoError::InvalidHandle)?;
//         let arena = field_arenas
//             .get(arena_index)
//             .ok_or(AtlasIoError::InvalidHandle)?;
//         Ok(arena.clone())
//     }

//     fn find_free_arena(
//         arenas: &[Arc<BufferArena>],
//         size: usize,
//     ) -> Option<(Arc<BufferArena>, usize)> {
//         for (index, arena) in arenas.iter().enumerate() {
//             if let BufferArena::Uncompressed(uncompressed_arena) = arena.as_ref() {
//                 let free_space = uncompressed_arena
//                     .free_space
//                     .load(std::sync::atomic::Ordering::SeqCst);
//                 if free_space as usize >= size {
//                     return Some((arena.clone(), index));
//                 }
//             }
//         }
//         None
//     }

//     async fn create_arena(
//         io_provider: Arc<dyn IOProvider + Send + Sync>,
//         file_offset: &AtomicU64,
//         preallocated_size: usize,
//     ) -> AtlasResult<Arc<BufferArena>> {
//         let arena_size = preallocated_size.max(Self::ARENA_SIZE);
//         let offset = file_offset.fetch_add(arena_size as u64, std::sync::atomic::Ordering::SeqCst);
//         // Write zeros to the file to reserve space
//         io_provider
//             .write_at(Bytes::from(vec![0u8; arena_size]), offset)
//             .await?;
//         // Add the new arena to the list of arenas of the corresponding field(variable).
//         let arena = BufferArenaUncompressed {
//             offset,
//             size: arena_size as u64,
//             free_space: AtomicU64::new(arena_size as u64),
//             num_allocations: AtomicU64::new(0),
//             dirty: AtomicBool::new(false),
//             gc: AtomicBool::new(false),
//         };

//         Ok(Arc::new(BufferArena::Uncompressed(arena)))
//     }

//     async fn allocate_onto_uncompressed_buffer_arena(
//         &self,
//         arena: &BufferArenaUncompressed,
//         field_index: usize,
//         arena_index: usize,
//         bytes: &[u8],
//     ) -> AtlasResult<AllocationHandle> {
//         let top_offset = arena
//             .free_space
//             .fetch_sub(bytes.len() as u64, std::sync::atomic::Ordering::SeqCst);
//         let offset = arena.size - top_offset;

//         let file_offset = arena.offset + offset;

//         self.io_provider
//             .write_at(Bytes::copy_from_slice(bytes), file_offset)
//             .await?; // Write the bytes to the IO provider at the correct offset.

//         arena
//             .num_allocations
//             .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
//         arena.dirty.store(true, std::sync::atomic::Ordering::SeqCst);

//         self.io_buffer_cache
//             .invalidate(&BufferArenaKey {
//                 field_index,
//                 arena_index,
//             })
//             .await;

//         // Return an AllocationHandle representing the allocation.
//         Ok(AllocationHandle {
//             field_index,
//             arena_index,
//             offset,
//             size: bytes.len() as u64,
//         })
//     }

//     /// Fetch the bytes for a previously returned `AllocationHandle`.
//     ///
//     /// Behavior:
//     /// - Loads the entire arena backing the handle (from cache or IO).
//     /// - For compressed arenas, decompresses the arena bytes.
//     /// - Validates the handle range, then returns a `Bytes` slice of the allocation.
//     ///
//     /// Errors:
//     /// - Returns `AtlasIoError::InvalidHandle` if the `(field_index, arena_index)` are out of
//     ///   range, or if `offset + size` exceeds the arena length.
//     pub async fn get_allocation(&self, handle: AllocationHandle) -> AtlasResult<Bytes> {
//         let arena_index = handle.arena_index;
//         let field_index = handle.field_index;
//         let arena_cache_key = BufferArenaKey {
//             field_index,
//             arena_index,
//         };
//         let buffer_arena = self.get_arena(field_index, arena_index)?;

//         let io_provider = self.io_provider.clone();

//         let arena_bytes = self
//             .io_buffer_cache
//             .try_get_with(arena_cache_key, async move {
//                 Self::get_arena_bytes(io_provider, &buffer_arena).await
//             })
//             .await
//             .map_err(|e| AtlasIoError::Cache(e.to_string()))?;

//         if (handle.offset + handle.size) as usize > arena_bytes.len() {
//             return Err(AtlasIoError::InvalidHandle);
//         }

//         // Slice out the allocation from the arena bytes
//         let range = handle.offset as usize..(handle.offset + handle.size) as usize;
//         Ok(arena_bytes.slice(range))
//     }

//     async fn get_arena_bytes(
//         io_provider: Arc<dyn IOProvider + Send + Sync>,
//         arena: &BufferArena,
//     ) -> AtlasResult<Bytes> {
//         match arena {
//             BufferArena::Compressed(compressed_arena) => {
//                 Self::get_compressed_arena_bytes(io_provider, compressed_arena).await
//             }
//             BufferArena::Uncompressed(uncompressed_arena) => {
//                 Self::get_uncompressed_arena_bytes(io_provider, uncompressed_arena).await
//             }
//         }
//     }

//     async fn get_compressed_arena_bytes(
//         io_provider: Arc<dyn IOProvider + Send + Sync>,
//         arena: &BufferArenaCompressed,
//     ) -> AtlasResult<Bytes> {
//         let compressed_buf = io_provider
//             .read_at(arena.offset, arena.size as usize)
//             .await?;

//         let uncompressed_len = arena.inner.size as usize;
//         let mut arena_buffer = BytesMut::with_capacity(uncompressed_len);
//         arena_buffer.resize(uncompressed_len, 0);
//         zstd::bulk::decompress_to_buffer(compressed_buf.as_ref(), arena_buffer.as_mut())
//             .map_err(|e| AtlasIoError::Compression(e.to_string()))?;

//         Ok(arena_buffer.freeze())
//     }

//     async fn get_uncompressed_arena_bytes(
//         io_provider: Arc<dyn IOProvider + Send + Sync>,
//         arena: &BufferArenaUncompressed,
//     ) -> AtlasResult<Bytes> {
//         Ok(io_provider
//             .read_at(arena.offset, arena.size as usize)
//             .await?)
//     }
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// struct BufferArenaKey {
//     field_index: usize,
//     arena_index: usize,
// }

// #[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
// pub enum BufferArena {
//     Compressed(BufferArenaCompressed),
//     Uncompressed(BufferArenaUncompressed),
// }

// /// A compressed arena stored on disk.
// ///
// /// - `offset`/`size` describe the compressed blob on disk.
// /// - `inner` describes the *logical* uncompressed arena layout (including flags and
// ///   the uncompressed `size`).
// ///
// /// Allocation handles are always interpreted against the uncompressed layout.
// #[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
// pub struct BufferArenaCompressed {
//     offset: u64,
//     size: u64,
//     inner: BufferArenaUncompressed,
// }

// /// An uncompressed arena stored on disk.
// ///
// /// The allocator treats each arena as a fixed-size append-only region.
// /// `free_space` is tracked as bytes remaining, and allocations are placed from low offsets
// /// upward.
// #[derive(Debug, bincode::Encode, bincode::Decode)]
// pub struct BufferArenaUncompressed {
//     offset: u64, // Should never change after creation
//     size: u64,   // Should never change after creation
//     free_space: AtomicU64,
//     num_allocations: AtomicU64,
//     dirty: AtomicBool, // Whether the arena has been modified since last write. If so, it needs to be written back to disk.
//     gc: AtomicBool,
// }

// impl Clone for BufferArenaUncompressed {
//     fn clone(&self) -> Self {
//         Self {
//             offset: self.offset,
//             size: self.size,
//             free_space: AtomicU64::new(self.free_space.load(std::sync::atomic::Ordering::SeqCst)),
//             num_allocations: AtomicU64::new(
//                 self.num_allocations
//                     .load(std::sync::atomic::Ordering::SeqCst),
//             ),
//             dirty: AtomicBool::new(self.dirty.load(std::sync::atomic::Ordering::SeqCst)),
//             gc: AtomicBool::new(self.gc.load(std::sync::atomic::Ordering::SeqCst)),
//         }
//     }
// }

// #[derive(Debug, Clone, Copy, bincode::Encode, bincode::Decode)]
// pub struct AllocationHandle {
//     /// Field (variable) index this allocation belongs to.
//     pub field_index: usize, // Index of the field(variable) this allocation belongs to

//     /// Arena index within the field's arena list.
//     pub arena_index: usize, // Index of the buffer arena in the allocator

//     /// Offset within the arena (in bytes).
//     pub offset: u64, // Offset in the arena block size

//     /// Size of this allocation (in bytes).
//     pub size: u64, // Size of the chunk
// }

// fn saturating_atomic_decrement(value: &AtomicU64) -> u64 {
//     loop {
//         let current = value.load(std::sync::atomic::Ordering::SeqCst);
//         if current == 0 {
//             return 0;
//         }

//         match value.compare_exchange(
//             current,
//             current - 1,
//             std::sync::atomic::Ordering::SeqCst,
//             std::sync::atomic::Ordering::SeqCst,
//         ) {
//             Ok(_) => return current - 1,
//             Err(_) => continue,
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::sync::Arc;

//     #[derive(Default)]
//     struct MemIOProvider {
//         data: parking_lot::RwLock<Vec<u8>>,
//     }

//     impl IOProvider for MemIOProvider {
//         fn read_at(&self, offset: u64, len: usize) -> BoxFuture<'_, std::io::Result<Bytes>> {
//             Box::pin(async move {
//                 let offset = offset as usize;
//                 let data = self.data.read();
//                 let end = offset.saturating_add(len);
//                 if end > data.len() {
//                     return Err(std::io::Error::new(
//                         std::io::ErrorKind::UnexpectedEof,
//                         "short read",
//                     ));
//                 }
//                 Ok(Bytes::copy_from_slice(&data[offset..end]))
//             })
//         }

//         fn write_at(&self, bytes: Bytes, offset: u64) -> BoxFuture<'_, std::io::Result<()>> {
//             Box::pin(async move {
//                 let offset = offset as usize;
//                 let end = offset.saturating_add(bytes.len());
//                 let mut data = self.data.write();
//                 if end > data.len() {
//                     data.resize(end, 0);
//                 }
//                 data[offset..end].copy_from_slice(&bytes);
//                 Ok(())
//             })
//         }

//         fn file_size(&self) -> BoxFuture<'_, std::io::Result<u64>> {
//             Box::pin(async move {
//                 let data = self.data.read();
//                 Ok(data.len() as u64)
//             })
//         }
//     }

//     // #[test]
//     // fn saturating_atomic_decrement_never_underflows() {
//     //     let value = AtomicU64::new(0);
//     //     assert_eq!(super::saturating_atomic_decrement(&value), 0);
//     //     assert_eq!(value.load(std::sync::atomic::Ordering::SeqCst), 0);

//     //     value.store(2, std::sync::atomic::Ordering::SeqCst);
//     //     assert_eq!(super::saturating_atomic_decrement(&value), 1);
//     //     assert_eq!(super::saturating_atomic_decrement(&value), 0);
//     //     assert_eq!(super::saturating_atomic_decrement(&value), 0);
//     // }

//     // #[tokio::test]
//     // async fn alloc_and_get_allocation_roundtrip_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let io = IOHandler::new(provider, 0);

//     //     let handle = io.alloc(0, b"hello").await.unwrap();
//     //     let got = io.get_allocation(handle).await.unwrap();
//     //     assert_eq!(&got[..], b"hello");
//     // }

//     // #[tokio::test]
//     // async fn alloc_grows_field_list_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let io = IOHandler::new(provider, 0);

//     //     let h = io.alloc(3, b"x").await.unwrap();
//     //     assert_eq!(h.field_index, 3);

//     //     let guard = io.field_arenas.read();
//     //     assert_eq!(guard.len(), 4);
//     //     assert!(!guard[3].is_empty());
//     // }

//     // #[tokio::test]
//     // async fn alloc_multiple_handles_roundtrip_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let io = IOHandler::new(provider, 0);

//     //     let h1 = io.alloc(0, b"aaa").await.unwrap();
//     //     let h2 = io.alloc(0, b"bbbb").await.unwrap();
//     //     let h3 = io.alloc(0, b"cc").await.unwrap();

//     //     assert_eq!(&io.get_allocation(h1).await.unwrap()[..], b"aaa");
//     //     assert_eq!(&io.get_allocation(h2).await.unwrap()[..], b"bbbb");
//     //     assert_eq!(&io.get_allocation(h3).await.unwrap()[..], b"cc");

//     //     // Sanity: offsets should not overlap for allocations within the same arena.
//     //     // (They may be in different arenas if rollover happens.)
//     //     if h1.arena_index == h2.arena_index {
//     //         assert_ne!(h1.offset, h2.offset);
//     //     }
//     // }

//     // #[tokio::test]
//     // async fn alloc_rolls_over_to_new_arena_when_full_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let io = IOHandler::new(provider, 0);

//     //     // Force a rollover by exhausting most of the default 1MiB arena.
//     //     let a = vec![1u8; 700_000];
//     //     let b = vec![2u8; 700_000];

//     //     let h1 = io.alloc(0, &a).await.unwrap();
//     //     let h2 = io.alloc(0, &b).await.unwrap();

//     //     assert_ne!(h1.arena_index, h2.arena_index);
//     //     assert_eq!(io.get_allocation(h1).await.unwrap().len(), a.len());
//     //     assert_eq!(io.get_allocation(h2).await.unwrap().len(), b.len());
//     // }

//     // #[tokio::test]
//     // async fn realloc_same_size_overwrites_cached_arena_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let io = IOHandler::new(provider, 0);

//     //     let h = io.alloc(0, b"abcd").await.unwrap();

//     //     // Populate cache.
//     //     let before = io.get_allocation(h).await.unwrap();
//     //     assert_eq!(&before[..], b"abcd");

//     //     // Realloc with same length should overwrite in place and invalidate cache.
//     //     let h2 = io.realloc(&h, b"WXYZ").await.unwrap();
//     //     assert_eq!(h.field_index, h2.field_index);
//     //     assert_eq!(h.arena_index, h2.arena_index);
//     //     assert_eq!(h.offset, h2.offset);
//     //     assert_eq!(h.size, h2.size);

//     //     let after = io.get_allocation(h2).await.unwrap();
//     //     assert_eq!(&after[..], b"WXYZ");
//     // }

//     // #[tokio::test]
//     // async fn realloc_new_size_marks_old_arena_for_gc_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let io = IOHandler::new(provider, 0);

//     //     // Make the original allocation nearly fill the arena so a larger realloc cannot
//     //     // fit in the same arena, ensuring we allocate a new arena.
//     //     let original = vec![b'a'; IOHandler::ARENA_SIZE - 8];
//     //     let grown = vec![b'b'; IOHandler::ARENA_SIZE - 4];

//     //     let h = io.alloc(0, &original).await.unwrap();
//     //     let old_arena = io.get_arena(h.field_index, h.arena_index).unwrap();

//     //     let h2 = io.realloc(&h, &grown).await.unwrap();
//     //     assert_ne!(h.size, h2.size);
//     //     assert_ne!(h.arena_index, h2.arena_index);
//     //     assert_eq!(io.get_allocation(h2).await.unwrap().len(), grown.len());

//     //     // Old arena should have allocation count decremented to zero and be marked for GC.
//     //     if let BufferArena::Uncompressed(uncompressed) = old_arena.as_ref() {
//     //         assert_eq!(
//     //             uncompressed
//     //                 .num_allocations
//     //                 .load(std::sync::atomic::Ordering::SeqCst),
//     //             0
//     //         );
//     //         assert!(uncompressed.gc.load(std::sync::atomic::Ordering::SeqCst));
//     //     } else {
//     //         panic!("expected uncompressed arena");
//     //     }
//     // }

//     // #[tokio::test]
//     // async fn get_allocation_out_of_bounds_returns_error_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let io = IOHandler::new(provider, 0);

//     //     let h = io.alloc(0, b"hello").await.unwrap();
//     //     let arena = io.get_arena(h.field_index, h.arena_index).unwrap();
//     //     let arena_size = match arena.as_ref() {
//     //         BufferArena::Uncompressed(uncompressed) => uncompressed.size,
//     //         BufferArena::Compressed(compressed) => compressed.inner.size,
//     //     };

//     //     // Create a handle whose (offset + size) exceeds the arena size.
//     //     let bad = AllocationHandle {
//     //         field_index: h.field_index,
//     //         arena_index: h.arena_index,
//     //         offset: arena_size - 1,
//     //         size: 2,
//     //     };

//     //     let err = io.get_allocation(bad).await.unwrap_err();
//     //     assert!(matches!(err, AtlasIoError::InvalidHandle));
//     // }

//     // #[tokio::test]
//     // async fn realloc_invalid_handle_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let io = IOHandler::new(provider, 0);

//     //     let bad = AllocationHandle {
//     //         field_index: 0,
//     //         arena_index: 123,
//     //         offset: 0,
//     //         size: 1,
//     //     };

//     //     let err = io.realloc(&bad, b"x").await.unwrap_err();
//     //     assert!(matches!(err, AtlasIoError::InvalidHandle));
//     // }

//     // async fn read_allocation_via_metadata(
//     //     provider: &Arc<dyn IOProvider + Send + Sync>,
//     //     arenas: &[Vec<Option<BufferArena>>],
//     //     handle: AllocationHandle,
//     // ) -> Bytes {
//     //     let arena = arenas[handle.field_index][handle.arena_index]
//     //         .as_ref()
//     //         .expect("expected arena to exist");

//     //     let arena_bytes = match arena {
//     //         BufferArena::Compressed(compressed) => {
//     //             let compressed_buf = provider
//     //                 .read_at(compressed.offset, compressed.size as usize)
//     //                 .await
//     //                 .unwrap();

//     //             let uncompressed_len = compressed.inner.size as usize;
//     //             let mut buf = BytesMut::with_capacity(uncompressed_len);
//     //             buf.resize(uncompressed_len, 0);
//     //             zstd::bulk::decompress_to_buffer(compressed_buf.as_ref(), buf.as_mut()).unwrap();
//     //             buf.freeze()
//     //         }
//     //         BufferArena::Uncompressed(uncompressed) => provider
//     //             .read_at(uncompressed.offset, uncompressed.size as usize)
//     //             .await
//     //             .unwrap(),
//     //     };

//     //     arena_bytes.slice(handle.offset as usize..(handle.offset + handle.size) as usize)
//     // }

//     // #[tokio::test]
//     // async fn gc_and_compress_preserves_allocation_bytes_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let provider_clone = provider.clone();
//     //     let io = IOHandler::new(provider, 0);

//     //     // Force two arenas.
//     //     let a = vec![7u8; 700_000];
//     //     let b = vec![9u8; 700_000];
//     //     let h1 = io.alloc(0, &a).await.unwrap();
//     //     let h2 = io.alloc(0, &b).await.unwrap();

//     //     let (new_arenas, new_end) = io.gc_and_compress().await.unwrap();

//     //     // All non-GC arenas should be compressed.
//     //     for field in &new_arenas {
//     //         for arena in field.iter().flatten() {
//     //             assert!(matches!(arena, BufferArena::Compressed(_)));
//     //         }
//     //     }

//     //     // Allocation bytes can be retrieved via returned metadata.
//     //     let got1 = read_allocation_via_metadata(&provider_clone, &new_arenas, h1).await;
//     //     let got2 = read_allocation_via_metadata(&provider_clone, &new_arenas, h2).await;
//     //     assert_eq!(&got1[..], &a[..]);
//     //     assert_eq!(&got2[..], &b[..]);

//     //     // Layout sanity: end offset matches max(offset + size).
//     //     let mut max_end = crate::consts::MAGIC_NUMBER.len() as u64;
//     //     for field in &new_arenas {
//     //         for arena in field {
//     //             if let Some(BufferArena::Compressed(c)) = arena {
//     //                 max_end = max_end.max(c.offset + c.size);
//     //             }
//     //         }
//     //     }
//     //     assert_eq!(new_end, max_end);
//     // }

//     // #[tokio::test]
//     // async fn gc_and_compress_skips_gc_arenas_without_shifting_indices_unit() {
//     //     let provider: Arc<dyn IOProvider + Send + Sync> = Arc::new(MemIOProvider::default());
//     //     let provider_clone = provider.clone();
//     //     let io = IOHandler::new(provider, 0);

//     //     // Force two arenas in the same field.
//     //     let a = vec![1u8; 700_000];
//     //     let b = vec![2u8; 700_000];
//     //     let h1 = io.alloc(0, &a).await.unwrap();
//     //     let h2 = io.alloc(0, &b).await.unwrap();
//     //     assert_ne!(h1.arena_index, h2.arena_index);

//     //     // Mark the first arena for GC.
//     //     let arena = io.get_arena(h1.field_index, h1.arena_index).unwrap();
//     //     if let BufferArena::Uncompressed(uncompressed) = arena.as_ref() {
//     //         uncompressed
//     //             .gc
//     //             .store(true, std::sync::atomic::Ordering::SeqCst);
//     //     } else {
//     //         panic!("expected uncompressed arena");
//     //     }

//     //     let (new_arenas, _new_end) = io.gc_and_compress().await.unwrap();

//     //     // The GC'd arena slot is None (index preserved).
//     //     assert!(new_arenas[h1.field_index][h1.arena_index].is_none());

//     //     // The other arena slot still exists and is readable.
//     //     assert!(new_arenas[h2.field_index][h2.arena_index].is_some());
//     //     let got2 = read_allocation_via_metadata(&provider_clone, &new_arenas, h2).await;
//     //     assert_eq!(&got2[..], &b[..]);
//     // }
// }

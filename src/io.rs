use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64};

use bytes::Bytes;
use futures::future::BoxFuture;

pub trait IOProvider {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> BoxFuture<'_, std::io::Result<()>>;
    fn write_at(&self, buf: &[u8], offset: u64) -> BoxFuture<'_, std::io::Result<()>>;
}

pub struct AtlasStdFile {
    file: Arc<std::fs::File>,
    tokio_handle: tokio::runtime::Handle,
}

// impl IO provider for wrapping std::fs::File in an Arc
impl IOProvider for AtlasStdFile {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> BoxFuture<'_, std::io::Result<()>> {
        unimplemented!()
    }

    fn write_at(&self, buf: &[u8], offset: u64) -> BoxFuture<'_, std::io::Result<()>> {
        unimplemented!()
    }
}

pub struct IOHandler {
    io_provider: Arc<dyn IOProvider + Send + Sync>,
    file_offset: AtomicU64,
    field_arenas: parking_lot::RwLock<Vec<Vec<Arc<BufferArena>>>>, // one vec per field. Each vec contains arenas for that field(variable).
}

impl IOHandler {
    const ARENA_SIZE: usize = 1024 * 1024; // 1 MiB
    pub fn new<F: Into<Arc<dyn IOProvider + Send + Sync>>>(f: F, arena_offset: usize) -> Self {
        Self {
            file_offset: AtomicU64::new(arena_offset as u64),
            io_provider: f.into(),
            field_arenas: parking_lot::RwLock::new(Vec::new()),
        }
    }

    pub fn io_provider(&self) -> &Arc<dyn IOProvider + Send + Sync> {
        &self.io_provider
    }

    pub async fn alloc(&self, field_index: usize, bytes: &[u8]) -> AllocationHandle {
        let mut arenas_guard = self.field_arenas.write();
        let field_arenas = &mut arenas_guard[field_index];
        // Find an arena with enough free space to hold the allocation. If none exists, create a new one.
        let mut allocatable_arena = None;
        let mut arena_index = 0;
        for (index, arena) in field_arenas.iter().enumerate() {
            match arena.as_ref() {
                BufferArena::Uncompressed(uncompressed_arena) => {
                    let free_space = uncompressed_arena
                        .free_space
                        .load(std::sync::atomic::Ordering::SeqCst);
                    if free_space as usize >= bytes.len() {
                        allocatable_arena = Some(arena.clone());
                        arena_index = index;
                        break;
                    }
                }
                BufferArena::Compressed(compressed_arena) => {
                    // Compressed arenas use COW (Copy-on-write semantics) and are not mutable directly.
                    let free_space = compressed_arena
                        .inner
                        .free_space
                        .load(std::sync::atomic::Ordering::SeqCst);
                    if free_space as usize >= bytes.len() {
                        allocatable_arena = Some(arena.clone());
                        arena_index = index;
                        break;
                    }
                }
            }
        }
        match allocatable_arena {
            Some(arena) => {
                match arena.as_ref() {
                    BufferArena::Uncompressed(uncompressed_arena) => {
                        return self
                            .allocate_onto_uncompressed_buffer_arena(
                                uncompressed_arena,
                                field_index,
                                arena_index,
                                bytes,
                            )
                            .await;
                    }
                    BufferArena::Compressed(_compressed_arena) => {
                        // For compressed arenas, we would need to decompress, allocate, and recompress.
                        unimplemented!()
                    }
                }
            }
            None => {
                // Create a new arena
                let new_arena = Self::create_arena(
                    self.io_provider.clone(),
                    &self.file_offset,
                    field_arenas,
                    bytes.len(),
                )
                .await;

                // Allocate onto the new arena
            }
        }

        todo!()
    }

    async fn alloc_onto_arena() {
        unimplemented!()
    }

    async fn create_arena(
        io_provider: Arc<dyn IOProvider + Send + Sync>,
        file_offset: &AtomicU64,
        field_arenas: &mut Vec<Arc<BufferArena>>,
        preallocated_size: usize,
    ) -> Arc<BufferArena> {
        let offset =
            file_offset.fetch_add(Self::ARENA_SIZE as u64, std::sync::atomic::Ordering::SeqCst);
        let arena_size = preallocated_size.max(Self::ARENA_SIZE);
        // Write zeros to the file to reserve space
        io_provider
            .write_at(&vec![0u8; arena_size], offset)
            .await
            .unwrap();
        // Add the new arena to the list of arenas of the corresponding field(variable).
        let arena = BufferArenaUncompressed {
            offset,
            size: Self::ARENA_SIZE as u64,
            free_space: AtomicU64::new(Self::ARENA_SIZE as u64),
            num_allocations: AtomicU64::new(0),
            dirty: AtomicBool::new(false),
            gc: AtomicBool::new(false),
        };
        let arena = Arc::new(BufferArena::Uncompressed(arena));
        field_arenas.push(arena.clone());
        arena
    }

    async fn allocate_onto_uncompressed_buffer_arena(
        &self,
        arena: &BufferArenaUncompressed,
        field_index: usize,
        arena_index: usize,
        bytes: &[u8],
    ) -> AllocationHandle {
        let top_offset = arena
            .free_space
            .fetch_sub(bytes.len() as u64, std::sync::atomic::Ordering::SeqCst);
        let offset = arena.offset + arena.size - top_offset;

        self.io_provider.write_at(bytes, offset).await.unwrap(); // Write the bytes to the IO provider at the correct offset.

        // Return an AllocationHandle representing the allocation.
        AllocationHandle {
            field_index,
            arena_index,
            offset,
            size: bytes.len() as u64,
        }
    }

    pub async fn get_allocation(&self, handle: &AllocationHandle) -> Bytes {
        // Placeholder implementation
        unimplemented!()
    }
}

pub enum BufferArena {
    Compressed(BufferArenaCompressed),
    Uncompressed(BufferArenaUncompressed),
}

pub struct BufferArenaCompressed {
    offset: usize,
    size: usize,
    inner: BufferArenaUncompressed,
}

pub struct BufferArenaUncompressed {
    offset: u64, // Should never change after creation
    size: u64,   // Should never change after creation
    free_space: AtomicU64,
    num_allocations: AtomicU64,
    dirty: AtomicBool, // Whether the arena has been modified since last write. If so, it needs to be written back to disk.
    gc: AtomicBool,
}

#[derive(Debug, Clone, Copy)]
pub struct AllocationHandle {
    pub field_index: usize, // Index of the field(variable) this allocation belongs to
    pub arena_index: usize, // Index of the buffer arena in the allocator
    pub offset: u64,        // Offset in the arena block size
    pub size: u64,          // Size of the chunk
}

use crate::schema::Schema;

#[repr(C)]
pub struct HeaderArenaHandle {
    pub offset: u64,
    pub size: u64,
}

pub struct Header {
    pub version: u32,
    pub title: String,
    pub global_attributes: Vec<(String, String)>,
    pub schema: Schema,
    pub arenas: Vec<ArenaLocation>,
}

pub struct ArenaLocation {
    pub offset: u64,
    pub size: u64,
}

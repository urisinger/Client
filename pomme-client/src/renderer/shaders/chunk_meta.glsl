struct ChunkMeta {
    vec3 aabb_min;
    // The section's arena: its draw count and where its draws start.
    uint arena;
    vec3 aabb_max;
    uint index_count;
    uint first_index;
    int vertex_offset;
    // The fade-in factor, written here each cull and read per instance by
    // the vertex stage.
    uint visibility;
    uint solid_index_count;
    // ivec3 + uint pack into one 16-byte slot, matching the Rust layout.
    ivec3 origin;
    uint water_index_count;
    // The column's slot in the visibility masks, the section's bit in it,
    // and when the section was uploaded (0: no fade).
    uint column;
    uint section;
    uint uploaded_ms;
    uint pad;
    // Where the solid then the cutout indices' face groups end within their
    // layer, the first six of each (`FACE_GROUPS` in mesher.rs); the last
    // ends with the layer.
    uint face_ends[12];
};

struct DrawCmd {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

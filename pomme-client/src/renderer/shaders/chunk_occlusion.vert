#version 450

// Each of the cull's candidate sections as its box, grown by a block,
// depth-tested against the terrain drawn so far; chunk_occlusion.frag marks
// the sections that show.

#include "chunk_meta.glsl"

layout(set = 0, binding = 0) readonly buffer MetaBuf {
    ChunkMeta chunks[];
};

layout(set = 0, binding = 1) readonly buffer CandidateBuf {
    uint candidates[];
};

// The terrain's view projection, eye-relative as chunk.vert draws, the
// camera block and the eye's offset from it, and the cull's stamp.
layout(push_constant) uniform Push {
    mat4 view_proj;
    ivec3 cam;
    uint frame_id;
    vec3 frac;
};

layout(location = 0) flat out uint v_idx;
layout(location = 1) flat out uint v_frame;

const uint DRAWN_BIT = 1u << 31u;
const float EXPANSION = 1.0;

void main() {
    uint idx = candidates[gl_InstanceIndex] & ~DRAWN_BIT;
    ChunkMeta m = chunks[idx];
    vec3 rel = vec3(m.origin - cam) - frac;
    // The index buffer's corners, a bit an axis.
    uint corner = uint(gl_VertexIndex);
    vec3 at = vec3(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
    vec3 lo = m.aabb_min - EXPANSION;
    vec3 hi = m.aabb_max + EXPANSION;
    gl_Position = view_proj * vec4(rel + mix(lo, hi, at), 1.0);
    v_idx = idx;
    v_frame = frame_id;
}

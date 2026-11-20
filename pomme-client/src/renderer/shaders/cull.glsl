// The chunk cull's bindings and helpers, shared by cull.comp (every
// section, before the terrain draws) and cull_late.comp (the occlusion
// pass's candidates, after it).

layout(local_size_x = 64) in;

#include "chunk_meta.glsl"

layout(set = 0, binding = 0) buffer MetaBuf {
    ChunkMeta chunks[];
};

layout(set = 0, binding = 1) uniform FrustumBuf {
    vec4 planes[6];
    // Meta slots to walk: every arena's segment, headroom included.
    uint chunk_count;
    // Camera block position plus the eye's small offset from it; the
    // origins are rebased in integer math so nothing here forms a large
    // float.
    int cam_x;
    int cam_y;
    int cam_z;
    float frac_x;
    float frac_y;
    float frac_z;
    uint now_ms;
    // Whether the occlusion masks apply (the camera view) or not (a shadow
    // view, which draws occluded sections too).
    uint use_visibility;
    // How long a section fades in (`SectionRenderDispatcher` fade-in), 0 for
    // none.
    uint fade_ms;
    // Whether solid and cutout draw only the face groups that can face the
    // camera, in up to `OPAQUE_SLOTS` draws a section, or whole.
    uint use_facing;
    uint pad;
    // This cull's stamp in `seen`, and whether the camera's occlusion pass
    // runs after it (cull_late.comp then draws what it newly sees).
    uint frame_id;
    uint use_occlusion;
    uint pad2;
    uint pad3;
    // Per arena, where its segment of meta slots and draws starts and how
    // many sections it holds; the rest of the segment is headroom.
    uvec4 arena_base[64];
    uvec4 arena_len[64];
};

layout(set = 0, binding = 2) writeonly buffer IndirectBuf {
    DrawCmd draws[];
};

layout(set = 0, binding = 3) buffer CountBuf {
    uint draw_counts[];
};

layout(set = 0, binding = 4) writeonly buffer IndirectCutoutBuf {
    DrawCmd cutout_draws[];
};

layout(set = 0, binding = 5) buffer CutoutCountBuf {
    uint cutout_draw_counts[];
};

layout(set = 0, binding = 6) writeonly buffer IndirectWaterBuf {
    DrawCmd water_draws[];
};

layout(set = 0, binding = 7) buffer WaterCountBuf {
    uint water_draw_counts[];
};

// Per column slot, the occlusion graph's mask of visible sections.
layout(set = 0, binding = 8) readonly buffer VisibilityBuf {
    uint column_visibility[];
};

// The sections the occlusion pass tests, `DRAWN_BIT` set on those drawn
// before it.
layout(set = 0, binding = 9) buffer CandidateBuf {
    uint candidates[];
};

// The occlusion pass's instanced box draw, a box per candidate, then the
// late cull's dispatch over them.
layout(set = 0, binding = 10) buffer OcclusionCmdBuf {
    uint box_index_count;
    uint candidate_count;
    uint box_first_index;
    int box_vertex_offset;
    uint box_first_instance;
    uint late_groups;
    uint late_groups_y;
    uint late_groups_z;
};

// Per meta slot, the `frame_id` of the last cull whose occlusion pass saw it.
layout(set = 0, binding = 11) buffer SeenBuf {
    uint seen[];
};

const uint DRAWN_BIT = 1u << 31u;
// How far around a section's box the eye may be and its box be seen
// without the occlusion pass, whose boxes the near plane would clip; past
// the box pass's growth.
const float NEAR_BOX = 2.0;

const uint SOLID = 0u;
const uint CUTOUT = 1u;
const uint WATER = 2u;
const uint MAX_ARENAS = 256u;
// Draw slots a section takes in the solid and cutout lists with
// `use_facing`: the most runs `facing_mask` leaves.
const uint OPAQUE_SLOTS = 3u;
const uint FACE_GROUPS = 7u;
// How far behind a face group's farthest plane the eye may be and still
// draw it, for a shader pack's waving vertices.
const float FACING_SLACK = 0.25;

// Emit one indexed draw covering [first..first+count) for this section into
// its arena's draws of `layer`. The per-instance attributes (origin + fade)
// come from meta entry `idx`, routed via first_instance.
void emit(ChunkMeta m, uint base, uint count, uint first, uint idx, uint layer) {
    if (count == 0u) return;
    // The arena's index total, after every arena's draw count.
    uint indices = MAX_ARENAS + m.arena;
    DrawCmd d;
    d.index_count = count;
    d.instance_count = 1u;
    d.first_index = first;
    d.vertex_offset = m.vertex_offset;
    d.first_instance = idx;
    if (layer == SOLID) {
        draws[base + atomicAdd(draw_counts[m.arena], 1u)] = d;
        atomicAdd(draw_counts[indices], count);
    } else if (layer == CUTOUT) {
        cutout_draws[base + atomicAdd(cutout_draw_counts[m.arena], 1u)] = d;
        atomicAdd(cutout_draw_counts[indices], count);
    } else {
        water_draws[base + atomicAdd(water_draw_counts[m.arena], 1u)] = d;
        atomicAdd(water_draw_counts[indices], count);
    }
}

// The face groups (down, north, west, unaligned, east, south, up) with a
// plane the eye at the origin can be in front of, given the section's
// box relative to it.
uint facing_mask(vec3 mn, vec3 mx) {
    uint mask = 1u << 3u;
    mask |= uint(mx.y > -FACING_SLACK) << 0u;
    mask |= uint(mx.z > -FACING_SLACK) << 1u;
    mask |= uint(mx.x > -FACING_SLACK) << 2u;
    mask |= uint(mn.x < FACING_SLACK) << 4u;
    mask |= uint(mn.z < FACING_SLACK) << 5u;
    mask |= uint(mn.y < FACING_SLACK) << 6u;
    return mask;
}

// Emit `layer`'s [first..first+count) as the runs of face groups in `mask`,
// its group ends at `ends` in `m.face_ends`; empty groups join either side.
void emit_facing(ChunkMeta m, uint base, uint count, uint first, uint idx, uint layer,
                 uint ends, uint mask) {
    uint run_start = 0u;
    uint run_end = 0u;
    bool open = false;
    for (uint g = 0u; g < FACE_GROUPS; g++) {
        uint start = g == 0u ? 0u : m.face_ends[ends + g - 1u];
        uint end = g == FACE_GROUPS - 1u ? count : m.face_ends[ends + g];
        if ((mask & (1u << g)) != 0u || start == end) {
            if (!open) {
                run_start = start;
                open = true;
            }
            run_end = end;
        } else if (open) {
            emit(m, base, run_end - run_start, first + run_start, idx, layer);
            open = false;
        }
    }
    if (open) {
        emit(m, base, run_end - run_start, first + run_start, idx, layer);
    }
}

// The section's box relative to the eye.
void section_box(ChunkMeta m, out vec3 mn, out vec3 mx) {
    ivec3 cam = ivec3(cam_x, cam_y, cam_z);
    vec3 frac = vec3(frac_x, frac_y, frac_z);
    vec3 rel = vec3(m.origin - cam) - frac;
    mn = rel + m.aabb_min.xyz;
    mx = rel + m.aabb_max.xyz;
}

// The segment start of the section's arena.
uint arena_start(ChunkMeta m) {
    return arena_base[m.arena >> 2][m.arena & 3u];
}

// Emit the section's solid and cutout draws.
void emit_opaque(ChunkMeta m, uint base, uint idx, vec3 mn, vec3 mx) {
    // Indices split solid first, then cutout, then water.
    uint solid_count = m.solid_index_count;
    uint cutout_count = m.index_count - solid_count;
    if (use_facing != 0u) {
        uint mask = facing_mask(mn, mx);
        uint opaque_base = base * OPAQUE_SLOTS;
        emit_facing(m, opaque_base, solid_count, m.first_index, idx, SOLID, 0u, mask);
        emit_facing(m, opaque_base, cutout_count, m.first_index + solid_count, idx, CUTOUT,
                    FACE_GROUPS - 1u, mask);
    } else {
        emit(m, base, solid_count, m.first_index, idx, SOLID);
        emit(m, base, cutout_count, m.first_index + solid_count, idx, CUTOUT);
    }
}

void emit_water(ChunkMeta m, uint base, uint idx) {
    emit(m, base, m.water_index_count, m.first_index + m.index_count, idx, WATER);
}

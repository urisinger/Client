struct GlobalFace {
    vec4 positions[4];
    vec4 uvs[4];
    uvec4 material;
};
struct GlobalCuboid { GlobalFace faces[6]; };
layout(set = 3, binding = 0, std430) readonly buffer MeshData { uint mesh_data[]; };
layout(set = 3, binding = 1, std430) readonly buffer GeometryData { GlobalCuboid cuboids[]; };

vec3 cubeCorner(uint direction, uint corner) {
    // One bit per corner, four bits per direction in Direction::VALUES order.
    // Scalar masks avoid a dynamically indexed local array in every vertex.
    uint bit = direction * 4u + corner;
    return vec3((0xf0c3ccu >> bit) & 1u,
                (0x9999f0u >> bit) & 1u,
                (0x3cf069u >> bit) & 1u);
}

void decodeFace(uint vertex_base, uint vertex, out vec3 position,
                out vec2 uv, out vec3 color, out uint sprite) {
    uint base = vertex_base * 4u; // 16-byte arena units -> uint words
    uint local_face = (vertex & 0x3fffu) / 4u;
    uint corner = vertex % 4u;
    // The section header is eight words. Batch stream pointers are resolved
    // during upload so this path never follows header -> batch -> stream.
    uint batch = base + 8u + (vertex >> 14u) * 8u;
    uint face_word = base + mesh_data[batch] + local_face;
    uint lod = mesh_data[batch + 3u];
    bool fluid = mesh_data[batch + 5u] != 0u;
    uint f = mesh_data[face_word];
    uint reference = f & 0xfffu;
    uint local_block = reference / 6u;
    uint direction = reference % 6u;
    uint block_word = base + mesh_data[batch + 2u] + local_block * 2u;
    uvec2 block = uvec2(mesh_data[block_word], mesh_data[block_word + 1u]);
    vec3 origin = vec3(block.x & 15u, (block.x >> 4u) & 15u, (block.x >> 8u) & 15u);
    uint geometry = (block.x >> 12u) & 0xffffu;
    float scale = float(1u << lod);
    vec4 local;
    if (lod > 0u) local = vec4(cubeCorner(direction, corner) * scale, 1.0);
    else local = cuboids[geometry].faces[direction].positions[corner];
    float height = 1.0;
    if (fluid) {
        uint hc = local.x >= 0.5 ? (local.z >= 0.5 ? 2u : 3u) : (local.z >= 0.5 ? 1u : 0u);
        uint heights = mesh_data[base + mesh_data[batch + 4u] + local_block];
        height = float((heights >> (hc * 4u)) & 15u) / 15.0;
        if (lod > 0u) local.y = local.y / scale * height;
        else if (local.w != 0.0) local.y *= height;
    }
    vec3 model_offset = fluid ? vec3(0.0) : vec3(
        uintBitsToFloat(mesh_data[batch + 6u]),
        uintBitsToFloat(mesh_data[batch + 7u]),
        uintBitsToFloat(mesh_data[batch + 4u]));
    position = origin + local.xyz + model_offset;
    uv = cuboids[geometry].faces[direction].uvs[corner].xy;
    if (lod > 0u) {
        uv *= vec2(scale, fluid && direction >= 2u ? height : scale);
    }
    vec3 tint = vec3((block.x >> 28u) | ((block.y & 15u) << 4u),
        (block.y >> 4u) & 255u, (block.y >> 12u) & 255u) / 255.0;
    color = tint * (float((f >> (12u + corner * 5u)) & 31u) / 31.0);
    sprite = cuboids[geometry].faces[direction].material.x | (lod > 0u ? 0x80000000u : 0u);
}

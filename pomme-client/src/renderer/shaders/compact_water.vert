#version 450

// compact_chunk.vert variant for the water pass: `draw_water` records one draw per
// section CPU-side, so the section origin and fade arrive as push constants.

#include "fog.glsl"
#include "camera_ubo.glsl"
#include "packed_face.glsl"

layout(push_constant) uniform SectionPc {
    vec4 origin_fade;
    uvec4 mesh;
};


layout(location = 0) out vec2 v_sprite_uv;
layout(location = 1) out vec3 v_color;
layout(location = 3) flat out float v_visibility;
layout(location = 4) out vec3 v_fog_color;
layout(location = 5) out float v_fog;
layout(location = 6) flat out uint v_sprite;

void main() {
    vec3 local;
    decodeFace(mesh.x, uint(gl_VertexIndex) - mesh.x, local, v_sprite_uv, v_color, v_sprite);
    vec3 world = origin_fade.xyz + local;
    vec3 rel = world - camera_pos.xyz;
    gl_Position = view_proj * vec4(rel, 1.0);
    v_visibility = origin_fade.w;
    v_fog_color = fog_color.rgb;
    v_fog = total_fog_value(rel, fog_env, camera_pos.w, fog_color.w);
}

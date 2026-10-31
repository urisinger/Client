#version 450
#include "fog.glsl"
#include "camera_ubo.glsl"
#include "packing.glsl"
#include "lightmap.glsl"

layout(push_constant) uniform SectionPc { vec4 origin_fade; };
layout(location = 0) in vec2 in_pos_xy;
layout(location = 1) in float in_pos_z;
layout(location = 2) in uint in_uv;
layout(location = 3) in uint in_sprite;
layout(location = 4) in vec4 in_light_tint;
layout(location = 0) out vec2 v_sprite_uv;
layout(location = 1) out vec3 v_color;
layout(location = 3) flat out float v_visibility;
layout(location = 4) out vec3 v_fog_color;
layout(location = 5) out float v_fog;
layout(location = 6) flat out uint v_sprite;

void main() {
    vec3 local = vec3(in_pos_xy, in_pos_z) * POS_RANGE - POS_BIAS;
    vec3 world = origin_fade.xyz + local;
    vec3 rel = world - camera_pos.xyz;
    gl_Position = view_proj * vec4(rel, 1.0);
    v_sprite_uv = unpack_sprite_uv(in_uv);
#ifdef TERRAIN_LOD
    bool repeat_uv = (in_sprite >> SPRITE_BITS) == 7u;
    if (repeat_uv) v_sprite_uv *= 16.0;
#endif
    v_color = terrain_vertex_color(in_light_tint, in_uv);
    v_visibility = origin_fade.w;
    v_fog_color = fog_color.rgb;
    v_fog = total_fog_value(rel, fog_env, camera_pos.w, fog_color.w);
    v_sprite = in_sprite & ((1u << SPRITE_BITS) - 1u);
#ifdef TERRAIN_LOD
    if (repeat_uv) v_sprite |= 0x80000000u;
#endif
}

#version 450

// Distant terrain's image laid into the frame, fogged like the chunks by
// where the frame's depth puts it, and faded out over the last tenth of its
// distance. Its depth is already the frame's.

#include "camera_ubo.glsl"
#include "fog.glsl"

layout(push_constant) uniform BlitPc {
    // The inverse of the camera's view projection.
    mat4 inverse_view_proj;
    // The fade: `alpha *= 1 - saturate(distance * fade.y + fade.x)`.
    vec4 fade;
};

layout(set = 1, binding = 0) uniform texture2D distant_color;
layout(set = 1, binding = 1) uniform texture2D frame_depth;
layout(set = 1, binding = 2) uniform sampler distant_sampler;

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    // Clear wherever distant terrain lost to the chunks or drew nothing.
    vec4 color = texelFetch(sampler2D(distant_color, distant_sampler), pixel, 0);
    if (color.a == 0.0) {
        discard;
    }
    float depth = texelFetch(sampler2D(frame_depth, distant_sampler), pixel, 0).r;
    vec4 point = inverse_view_proj * vec4(v_uv * 2.0 - 1.0, depth, 1.0);
    vec3 rel = point.xyz / point.w;
    float fog = total_fog_value(rel, fog_env, camera_pos.w, fog_color.w);
    color.rgb = apply_fog(color.rgb, fog, fog_color.rgb);
    color.a *= 1.0 - clamp(length(rel.xz) * fade.y + fade.x, 0.0, 1.0);
    out_color = color;
}

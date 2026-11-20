#version 450

// Each level of the Hi-Z holds the farthest of the four texels under it in
// the level before (level 0 reads the frame's depth): the least, in reverse
// Z. A box whose nearest point lies beyond every texel it covers is hidden.
// A texel at the far plane needs no special case: the min already carries
// it up.

layout(set = 0, binding = 0) uniform texture2D source;
layout(set = 0, binding = 1) uniform sampler source_sampler;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_depth;

void main() {
    vec4 depths = textureGather(sampler2D(source, source_sampler), v_uv, 0);
    out_depth = vec4(min(min(depths.x, depths.y), min(depths.z, depths.w)));
}

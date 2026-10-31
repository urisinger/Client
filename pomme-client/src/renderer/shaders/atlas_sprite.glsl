layout(set = 1, binding = 0) uniform texture2D atlas_texture;
layout(set = 1, binding = 1) uniform sampler atlas_sampler;
// Level-0 sprite rectangles `(x, y, width, height)` in atlas texels, indexed by
// the vertex's sprite id; index 0 is the missing tile.
layout(set = 1, binding = 2) readonly buffer SpriteRects {
    uvec4 sprite_rects[];
};

vec4 sample_atlas_sprite(vec2 sprite_uv, uint sprite) {
#ifdef TERRAIN_LOD
    bool repeat_uv = (sprite & 0x80000000u) != 0u;
    sprite &= 0x7fffffffu;
#endif
    uvec4 atlas_rect = sprite_rects[sprite];
    vec2 atlas_size = vec2(textureSize(sampler2D(atlas_texture, atlas_sampler), 0));
#ifdef TERRAIN_LOD
    if (repeat_uv) {
        vec2 span = vec2(atlas_rect.zw) / atlas_size;
        vec2 dx = dFdx(sprite_uv) * span;
        vec2 dy = dFdy(sprite_uv) * span;
        vec2 uv = (vec2(atlas_rect.xy) + fract(sprite_uv) * vec2(atlas_rect.zw)) / atlas_size;
        return textureGrad(sampler2D(atlas_texture, atlas_sampler), uv, dx, dy);
    }
#endif
    vec2 atlas_uv = (vec2(atlas_rect.xy) + sprite_uv * vec2(atlas_rect.zw)) / atlas_size;
    return texture(sampler2D(atlas_texture, atlas_sampler), atlas_uv);
}

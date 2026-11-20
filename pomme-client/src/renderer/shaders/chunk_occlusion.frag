#version 450

// A fragment of a section's box that passes the depth test marks the section
// seen by this frame's cull.

layout(early_fragment_tests) in;

layout(set = 0, binding = 2) writeonly buffer SeenBuf {
    uint seen[];
};

layout(location = 0) flat in uint v_idx;
layout(location = 1) flat in uint v_frame;

void main() {
    seen[v_idx] = v_frame;
}

// Upscale-composite pass for the re-entry plasma. The plasma is raymarched into a
// half-resolution HDR buffer (premultiplied-over); this samples that buffer with
// bilinear filtering and blends it over the full-resolution scene with the same
// premultiplied-over blend, so the heavy volumetric march runs at a quarter of the
// pixels while the soft glow upscales cleanly.

@group(0) @binding(0) var plasma_tex: texture_2d<f32>;
@group(0) @binding(1) var plasma_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VsOut;
    o.pos = vec4<f32>(p[vi], 0.0, 1.0);
    // map clip [-1,1] to texture uv [0,1] (flip y for texture space)
    o.uv = vec2<f32>(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // already premultiplied (rgb carries rgb*alpha); blend state does over.
    return textureSample(plasma_tex, plasma_samp, in.uv);
}

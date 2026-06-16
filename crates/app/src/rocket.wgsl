// Mesh pipeline for the rocket view: transform world-space triangles by a
// view-projection matrix and shade them with a sun term plus a hemispheric
// (sky/ground) ambient. A logarithmic depth buffer (GPU Gems style) lets the
// metre-scale rocket and the thousands-of-km planet share one depth buffer
// without z-fighting. The render target is sRGB, so no manual gamma here.

struct U {
    viewproj: mat4x4<f32>,
    sun: vec4<f32>,    // world-space sun direction in xyz
    params: vec4<f32>, // x = log-depth Fcoef
    fog: vec4<f32>,    // rgb = horizon haze, w = fog density
};

@group(0) @binding(0) var<uniform> u: U;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) flogz: f32, // 1 + clip.w, for the fragment depth write
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs(
    @location(0) p: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) c: vec3<f32>,
) -> VsOut {
    var o: VsOut;
    var clip = u.viewproj * vec4<f32>(p, 1.0);
    let fcoef = u.params.x;
    let w = max(clip.w, 1e-6);
    // WebGPU clip-space z in [0, 1]; map log of distance into it.
    clip.z = log2(max(1e-6, 1.0 + w)) * fcoef * w;
    o.pos = clip;
    o.flogz = 1.0 + w;
    o.normal = n;
    o.color = c;
    return o;
}

@fragment
fn fs(in: VsOut) -> FsOut {
    let n = normalize(in.normal);
    let s = normalize(u.sun.xyz);
    let diff = max(dot(n, s), 0.0);
    let amb = mix(vec3<f32>(0.18, 0.16, 0.14), vec3<f32>(0.40, 0.45, 0.55), clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let sun_col = vec3<f32>(1.0, 0.97, 0.9);
    var lit = in.color * (amb + sun_col * diff * 0.95);

    // aerial perspective: fade toward horizon haze with view distance.
    let dist = in.flogz - 1.0; // = view-space distance (clip.w)
    let fog = 1.0 - exp(-dist * u.fog.w);
    lit = mix(lit, u.fog.rgb, clamp(fog, 0.0, 1.0));

    var out: FsOut;
    out.color = vec4<f32>(lit, 1.0);
    // exact per-fragment logarithmic depth
    out.depth = log2(in.flogz) * u.params.x;
    return out;
}

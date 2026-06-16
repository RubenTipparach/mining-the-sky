// Mesh pipeline for the rocket view: transform world-space triangles by a
// view-projection matrix and shade them with a sun term plus a hemispheric
// (sky/ground) ambient. The render target is sRGB, so no manual gamma here.

struct U {
    viewproj: mat4x4<f32>,
    sun: vec4<f32>, // world-space sun direction in xyz
};

@group(0) @binding(0) var<uniform> u: U;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs(
    @location(0) p: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) c: vec3<f32>,
) -> VsOut {
    var o: VsOut;
    o.pos = u.viewproj * vec4<f32>(p, 1.0);
    o.normal = n;
    o.color = c;
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let s = normalize(u.sun.xyz);
    let diff = max(dot(n, s), 0.0);
    // hemispheric ambient: brighter facing up (sky), darker facing down (ground)
    let amb = mix(vec3<f32>(0.18, 0.16, 0.14), vec3<f32>(0.40, 0.45, 0.55), clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let sun_col = vec3<f32>(1.0, 0.97, 0.9);
    let lit = in.color * (amb + sun_col * diff * 0.95);
    return vec4<f32>(lit, 1.0);
}

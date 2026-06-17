// Mesh pipeline for the rocket view: transform world-space triangles by a
// view-projection matrix and shade them with a sun term plus a hemispheric
// (sky/ground) ambient. A logarithmic depth buffer (GPU Gems style) lets the
// metre-scale rocket and the thousands-of-km planet share one depth buffer
// without z-fighting. The render target is sRGB, so no manual gamma here.

struct U {
    viewproj: mat4x4<f32>,
    sun: vec4<f32>,    // world-space sun direction in xyz
    params: vec4<f32>, // x = log-depth Fcoef, y = time, z = light count
    fog: vec4<f32>,    // rgb = horizon haze, w = fog density
    lights: array<vec4<f32>, 8>,    // xyz = position (camera-relative), w = range
    light_col: array<vec4<f32>, 8>, // rgb = colour * intensity
};

@group(0) @binding(0) var<uniform> u: U;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) flogz: f32, // 1 + clip.w, for the fragment depth write
    @location(3) wpos: vec3<f32>, // camera-relative position, for point lights
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
    o.wpos = p;
    return o;
}

@fragment
fn fs(in: VsOut) -> FsOut {
    let n = normalize(in.normal);
    let s = normalize(u.sun.xyz);
    let diff = max(dot(n, s), 0.0);
    // Airless (lunar) bodies have no sky-fill: ambient is near-black so the only
    // light is the direct sun, giving stark crater shadows. On worlds with air,
    // use the bluish hemispheric sky/ground ambient.
    let airless = u.sun.w;
    let amb_air = mix(vec3<f32>(0.18, 0.16, 0.14), vec3<f32>(0.40, 0.45, 0.55), clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let amb_moon = vec3<f32>(0.05, 0.05, 0.055);
    let amb = mix(amb_air, amb_moon, airless);
    let sun_col = mix(vec3<f32>(1.0, 0.97, 0.9), vec3<f32>(1.25, 1.22, 1.15), airless);
    // Inside the assembly building (interior -> 1) the roof shades the sun, so
    // dim sun + ambient and let the work lights carry the scene.
    let interior = u.params.w;
    let sf = 1.0 - 0.9 * interior;
    let af = 1.0 - 0.68 * interior;
    var lit = in.color * (amb * af + sun_col * diff * 0.95 * sf);

    // interior point lights (work lights in the assembly building): per-fragment
    // diffuse with inverse-square-ish falloff, so they pool light on nearby
    // geometry and fade out before reaching the distant terrain.
    let nlights = i32(u.params.z);
    for (var i = 0; i < nlights; i = i + 1) {
        let d = u.lights[i].xyz - in.wpos;
        let range = u.lights[i].w;
        let dist = length(d);
        let ld = d / max(dist, 1e-3);
        let atten = 1.0 / (1.0 + (dist * dist) / (range * range));
        lit = lit + in.color * u.light_col[i].rgb * (max(dot(n, ld), 0.0) * atten);
    }

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

// Re-entry plasma as a procedural glow mesh. A shell that hugs the vehicle hull
// (built on the CPU from the vehicle SDF, smeared downstream into a wake) is
// rasterized as ordinary depth-tested geometry; this shader shades it with a
// white -> orange -> red cooling ramp keyed to the per-vertex "cool" coordinate
// (color.x, 0 at the windward face .. 1 at the wake), plus triangle-wave
// turbulence for the boil and a fresnel rim for a soft, volumetric-looking edge.
// Cost is geometry-bound and it occludes correctly against terrain/the vehicle.
//
// Turbulence is the triangle-wave noise from nimitz's "Re-entry" (Shadertoy
// 4dGyRh), CC BY-NC-SA 3.0 - attribution retained.

struct U {
    viewproj: mat4x4<f32>,
    sun: vec4<f32>,
    params: vec4<f32>, // x = log-depth Fcoef, y = time, z = light count
    fog: vec4<f32>,
    lights: array<vec4<f32>, 8>,
    light_col: array<vec4<f32>, 8>,
    detail: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

// Look knobs for the cooling ramp + opacity.
const WHITE_END:  f32 = 0.10;
const YELLOW_END: f32 = 0.22;
const ORANGE_END: f32 = 0.45;
const COL_RED:    vec3<f32> = vec3<f32>(0.40, 0.05, 0.02);
const COL_ORANGE: vec3<f32> = vec3<f32>(1.00, 0.36, 0.07);
const COL_YELLOW: vec3<f32> = vec3<f32>(1.00, 0.78, 0.40);
const COL_WHITE:  vec3<f32> = vec3<f32>(1.40, 1.32, 1.22);
const COL_SHEEN:  vec3<f32> = vec3<f32>(0.30, 0.45, 0.70);
const ALPHA_CAP:  f32 = 0.72;
const FLOW_AMP:   f32 = 0.9;   // metres of animated ripple (x layer) - flowing gas
const SCROLL:     f32 = 1.8;   // turbulence scroll speed (higher = flames fly faster)
const TAIL_FADE:  f32 = 0.35;  // cool value below which the wake is fully opaque
const TAIL_END:   f32 = 0.85;  // cool value by which the wake is 100% transparent

fn tri(x: f32) -> f32 { return abs(fract(x) - 0.5) - 0.25; }
fn tri2(x: f32) -> f32 { return abs(fract(x) - 0.5); }
fn tri3(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(tri(p.z + tri(p.y)), tri(p.z + tri(p.x * 1.05)), tri(p.y + tri(p.x * 1.1)));
}
fn tnoise(pin: vec3<f32>, t: f32) -> f32 {
    let m2 = mat2x2<f32>(0.970, 0.242, -0.242, 0.970);
    var p = pin;
    var bp = pin;
    var z = 1.45;
    var rz = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        let dg = tri3(bp);
        p = p + dg + vec3<f32>(t * 0.1 + 10.1);
        bp = bp * 1.65;
        z = z * 1.5;
        p = p * 0.9;
        let pxz = vec2<f32>(p.x, p.z) * m2;
        p.x = pxz.x;
        p.z = pxz.y;
        rz = rz + tri2(p.z + tri2(p.x + tri2(p.y))) / z;
        bp = bp + 0.9;
    }
    return rz;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) cool: f32,
    @location(2) flogz: f32,
    @location(3) wpos: vec3<f32>,
    @location(4) layer: f32,
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
    let t = u.params.y;
    let layer = c.y;
    // Animated flow: ripple the shell along its normal with scrolling noise -
    // gentle on the tight inner shell, big and wispy on the outer layers, so the
    // gas looks like it boils and flows past the hull rather than sitting still.
    let warp = tnoise(p * 0.05 + vec3<f32>(0.0, -t * SCROLL, 0.0), t) - 0.35;
    let pp = p + n * (warp * (0.5 + 3.0 * layer) * FLOW_AMP);
    var clip = u.viewproj * vec4<f32>(pp, 1.0);
    let fcoef = u.params.x;
    let w = max(clip.w, 1e-6);
    clip.z = log2(max(1e-6, 1.0 + w)) * fcoef * w;
    o.pos = clip;
    o.flogz = 1.0 + w;
    o.normal = n;
    o.cool = c.x;
    o.layer = layer;
    o.wpos = pp;
    return o;
}

@fragment
fn fs(in: VsOut) -> FsOut {
    let t = u.params.y;
    let cool = clamp(in.cool, 0.0, 1.0);

    // Boiling turbulence (two octaves) scrolling over the surface; the outer
    // layers get extra wisp so they read as flowing gas, not a hard shell.
    let q = in.wpos * 0.06;
    let tb0 = tnoise(q + vec3<f32>(0.0, -t * SCROLL, 0.0), t);
    let tb1 = tnoise(q * 2.7 + vec3<f32>(5.0, -t * SCROLL * 1.7, 0.0), t);
    let tb = clamp(tb0 * 0.8 + tb1 * (0.5 + 0.5 * in.layer), 0.0, 1.4);
    let fil = smoothstep(0.55, 0.95, tb1);

    // Steep cooling ramp keyed to the downstream "cool" coordinate.
    let whitef = smoothstep(WHITE_END, 0.0, cool);
    let yellowf = smoothstep(YELLOW_END, WHITE_END * 0.4, cool);
    let orangef = smoothstep(ORANGE_END, WHITE_END, cool);
    var col = COL_RED;
    col = mix(col, COL_ORANGE, orangef);
    col = mix(col, COL_YELLOW, yellowf);
    col = mix(col, COL_WHITE, whitef);
    col = col + COL_SHEEN * fil * whitef;

    // Fresnel: gas reads denser at grazing angles (silhouette), giving the flat
    // shell a volumetric feel. eye is the origin in camera-relative space.
    let nrm = normalize(in.normal);
    let view = normalize(-in.wpos);
    let graze = 1.0 - abs(dot(nrm, view));
    let rim = pow(clamp(graze, 0.0, 1.0), 1.6);

    // density: hot windward gas is opaque, the cool tail stays wispy; modulated
    // by turbulence and the fresnel rim (denser at grazing angles -> volumetric).
    let hot = max(whitef, yellowf * 0.7);
    let dens = clamp((0.30 + 1.1 * tb + 0.5 * fil) * (0.45 + 0.8 * rim), 0.0, 1.0);
    // Blend the wake tail fully out to NOTHING well before the mesh's geometric
    // end, so the long tail dissolves into space instead of ending on a hard
    // edge. Fades across most of the wake (more fade), 100% transparent by
    // TAIL_END. The outer wisp layers are fainter so they read as soft gas.
    let tailfade = smoothstep(TAIL_END, TAIL_FADE, cool);
    let layerfade = 1.0 - 0.5 * in.layer;
    var a = dens * (0.45 + 1.3 * hot) * tailfade * layerfade;
    a = clamp(a, 0.0, ALPHA_CAP);

    var out: FsOut;
    // premultiplied-over (rgb carries rgb*alpha); blend state does the over.
    out.color = vec4<f32>(col * a, a);
    out.depth = log2(in.flogz) * u.params.x;
    return out;
}

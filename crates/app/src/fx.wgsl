// Thruster FX pipeline: emissive flame + smoke-particle billboards for the
// rocket view. One pipeline with premultiplied-alpha blending serves both:
//  - flame (kind 0): additive (alpha 0, bright rgb) - a procedural fire ramp
//    with a white-hot core, an orange body and a flickering tip.
//  - smoke (kind 1): soft round premultiplied-over puffs that fade with age.
// Depth is tested (so geometry in front occludes the FX) but never written, so
// overlapping billboards blend. Shares the rocket mesh's uniform (viewproj +
// log-depth Fcoef in params.x, animation time in params.y).

struct U {
    viewproj: mat4x4<f32>,
    sun: vec4<f32>,
    params: vec4<f32>, // x = log-depth Fcoef, y = time
    fog: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) flogz: f32,
    @location(3) kind: f32,
};

@vertex
fn vs(
    @location(0) p: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) c: vec4<f32>,
    @location(3) kind: f32,
) -> VsOut {
    var o: VsOut;
    var clip = u.viewproj * vec4<f32>(p, 1.0);
    let fcoef = u.params.x;
    let w = max(clip.w, 1e-6);
    clip.z = log2(max(1e-6, 1.0 + w)) * fcoef * w;
    o.pos = clip;
    o.flogz = 1.0 + w;
    o.uv = uv;
    o.color = c;
    o.kind = kind;
    return o;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.547);
}

@fragment
fn fs(in: VsOut) -> FsOut {
    var out: FsOut;
    out.depth = log2(in.flogz) * u.params.x;
    let t = u.params.y;

    if (in.kind < 0.5) {
        // ---- flame ----  uv.x across the width (0..1), uv.y nozzle(0) -> tip(1)
        let across = 1.0 - abs(in.uv.x - 0.5) * 2.0;
        let along = in.uv.y;
        let seed = in.color.x;
        let intensity = in.color.y;
        // turbulent flicker
        let fl = 0.72 + 0.28 * sin(t * 46.0 + along * 11.0 + seed * 6.28)
                              * sin(t * 27.0 + seed * 3.14 + across * 4.0);
        var a = pow(max(across, 0.0), 1.5) * (1.0 - along);
        a = a * a * fl;
        // colour ramp: white-hot core -> yellow -> orange -> deep-red tip
        let cool = mix(vec3<f32>(1.0, 0.55, 0.12), vec3<f32>(0.85, 0.12, 0.03), along);
        let body = mix(vec3<f32>(1.0, 0.97, 0.85), cool, pow(along, 0.5));
        let core = pow(max(across, 0.0), 5.0) * (1.0 - along);
        let rgb = mix(body, vec3<f32>(1.3, 1.25, 1.1), core);
        out.color = vec4<f32>(rgb * a * intensity * 1.7, 0.0); // additive
    } else {
        // ---- smoke ----  soft round puff, premultiplied over
        let r = length(in.uv * 2.0 - 1.0);
        let n = 0.8 + 0.2 * hash(floor(in.uv * 7.0) + vec2<f32>(in.color.x, 0.0));
        let mask = smoothstep(1.0, 0.1, r) * n;
        let a = in.color.a * mask;
        out.color = vec4<f32>(in.color.rgb * a, a);
    }
    return out;
}

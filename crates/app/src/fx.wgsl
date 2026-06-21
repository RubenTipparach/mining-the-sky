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

// Triangle-wave turbulence from nimitz's "Re-entry" (Shadertoy 4dGyRh),
// CC BY-NC-SA 3.0. Used here to make the reentry plasma boil like real
// hypersonic shock-layer gas. Ported to WGSL.
fn tri(x: f32) -> f32 { return abs(fract(x) - 0.5) - 0.25; }
fn tri2(x: f32) -> f32 { return abs(fract(x) - 0.5); }
fn tri3(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        tri(p.z + tri(p.y)),
        tri(p.z + tri(p.x * 1.05)),
        tri(p.y + tri(p.x * 1.1)),
    );
}
fn tri_noise3d(pin: vec3<f32>, spd: f32, t: f32) -> f32 {
    let m2 = mat2x2<f32>(0.970, 0.242, -0.242, 0.970);
    var p = pin;
    var bp = pin;
    var z = 1.45;
    var rz = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        let dg = tri3(bp);
        p = p + dg + vec3<f32>(t * spd + 10.1);
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
    } else if (in.kind < 1.5) {
        // ---- smoke ----  soft round puff, premultiplied over
        let r = length(in.uv * 2.0 - 1.0);
        let n = 0.8 + 0.2 * hash(floor(in.uv * 7.0) + vec2<f32>(in.color.x, 0.0));
        let mask = smoothstep(1.0, 0.1, r) * n;
        let a = in.color.a * mask;
        out.color = vec4<f32>(in.color.rgb * a, a);
    } else if (in.kind < 2.5) {
        // ---- RCS jet (kind 2) ----  a short, cool blue-white attitude puff
        let across = 1.0 - abs(in.uv.x - 0.5) * 2.0;
        let along = in.uv.y;
        let seed = in.color.x;
        let intensity = in.color.y;
        let fl = 0.7 + 0.3 * sin(t * 80.0 + seed * 6.28 + along * 9.0);
        var a = pow(max(across, 0.0), 1.4) * (1.0 - along);
        a = a * a * fl;
        // cool white core fading to pale blue at the tip
        let rgb = mix(vec3<f32>(0.85, 0.93, 1.0), vec3<f32>(0.45, 0.6, 1.0), along);
        out.color = vec4<f32>(rgb * a * intensity * 1.6, 0.0); // additive
    } else if (in.kind < 3.5) {
        // ---- reentry plasma shock (kind 3) ----
        // colour-graded after nimitz's "Re-entry": a cool blue-white compression
        // front over a deep-orange incandescent body fading to a violet wake,
        // boiling with triangle-noise turbulence. uv.x across the sheet, uv.y the
        // leading shock (0) -> streaming wake (1). color.x=seed, color.y=heat,
        // color.z = role (0 = bow-shock cap, 1 = trailing streak).
        let across = 1.0 - abs(in.uv.x - 0.5) * 2.0;
        let along = in.uv.y;
        let seed = in.color.x;
        let heat = in.color.y;
        let role = in.color.z;
        // boiling turbulence in a synthesized 3D field; the wake streams downwind
        let np = vec3<f32>(in.uv.x * 3.2 + seed * 11.0, in.uv.y * 4.6 - t * 2.1, seed * 7.0);
        let turb = tri_noise3d(np, 0.7, t); // ~0..0.8
        // sheet shape: a broad bow-shock cap, or a thin licking streak
        var a: f32;
        if (role < 0.5) {
            a = pow(max(across, 0.0), 1.3) * pow(1.0 - along, 1.4);
        } else {
            a = pow(max(across, 0.0), 2.2) * (1.0 - along) * smoothstep(0.0, 0.12, along);
        }
        // turbulence carves the sheet into flame tongues
        a = a * (0.22 + 1.95 * turb) * clamp(heat, 0.0, 1.3);
        // colour grade: front (along ~ 0) cool, body orange, tail violet
        let front = pow(1.0 - along, 2.0);
        var rgb = vec3<f32>(0.72, 0.26, 0.03) * 1.35;          // deep orange body
        rgb = rgb + vec3<f32>(0.55, 0.77, 0.95) * 1.5 * front;  // blue-white shock front
        // incandescent white-hot sliver at the stagnation line
        let stag = pow(max(across, 0.0), 6.0) * front;
        rgb = rgb + vec3<f32>(1.0, 0.9, 0.6) * stag * 1.3;
        // cooling violet wake
        rgb = mix(rgb, vec3<f32>(0.5, 0.22, 0.62), smoothstep(0.55, 1.0, along) * 0.8);
        out.color = vec4<f32>(rgb * a * 1.8, 0.0); // additive
    } else {
        // ---- re-entry sparks (kind 4): tiny bright embers, additive ----
        // color.x = seed, color.y = brightness (fades with age).
        let seed = in.color.x;
        let bright = in.color.y;
        let r = length(in.uv * 2.0 - 1.0);
        let core = smoothstep(1.0, 0.0, r);             // round ember
        let flick = 0.7 + 0.3 * sin(t * 55.0 + seed * 6.28);
        // white-hot core grading to gold embers
        let rgb = mix(vec3<f32>(2.2, 1.8, 1.1), vec3<f32>(2.2, 0.95, 0.35), seed);
        let a = core * core * bright * flick;
        out.color = vec4<f32>(rgb * a * 3.5, 0.0); // additive
    }
    return out;
}

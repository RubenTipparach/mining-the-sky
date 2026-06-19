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
    } else {
        // ---- reentry plasma shock (kind 3) ----
        // uv.x across the sheet (0..1), uv.y leading shock (0, hottest) ->
        // streaming wake (1, cooler). color.x = seed, color.y = heat (0..1.3),
        // color.z = layer role (0 = bow-shock cap, 1 = trailing plasma streak).
        let across = 1.0 - abs(in.uv.x - 0.5) * 2.0;
        let along = in.uv.y;
        let seed = in.color.x;
        let heat = in.color.y;
        let role = in.color.z;
        // fast, fine turbulence so the sheet boils like real plasma
        let turb = 0.62 + 0.38 * sin(t * 34.0 + along * 16.0 + seed * 6.28)
                              * sin(t * 21.0 + across * 7.0 + seed * 2.0)
                              * sin(t * 13.0 - along * 5.0);
        // shape: a bright leading edge that streams back into a tapering tail
        var a: f32;
        var ramp: f32; // 0 at the hot shock front, 1 at the cool wake tip
        if (role < 0.5) {
            // bow-shock cap: hottest at the windward apex (along ~ 0), broad
            a = pow(max(across, 0.0), 1.3) * pow(1.0 - along, 1.4);
            ramp = along;
        } else {
            // trailing plasma streak: a thin tongue licking back off the body
            a = pow(max(across, 0.0), 2.2) * (1.0 - along) * smoothstep(0.0, 0.15, along);
            ramp = clamp(along * 1.1, 0.0, 1.0);
        }
        a = a * turb * clamp(heat, 0.0, 1.3);
        // colour ramp: blue-white compression front -> incandescent orange ->
        // ionised pink/magenta -> violet as the wake cools.
        let core = vec3<f32>(0.75, 0.90, 1.0);   // blue-white shock front
        let hot  = vec3<f32>(1.0, 0.85, 0.55);   // incandescent
        let ion  = vec3<f32>(1.0, 0.40, 0.55);   // ionised pink
        let tail = vec3<f32>(0.55, 0.22, 0.70);  // cooling violet
        var rgb = mix(core, hot, smoothstep(0.0, 0.30, ramp));
        rgb = mix(rgb, ion, smoothstep(0.30, 0.65, ramp));
        rgb = mix(rgb, tail, smoothstep(0.65, 1.0, ramp));
        // a white-hot sliver right at the stagnation line
        let stag = pow(max(across, 0.0), 6.0) * pow(1.0 - along, 2.0);
        rgb = mix(rgb, vec3<f32>(1.2, 1.2, 1.25), stag * heat);
        out.color = vec4<f32>(rgb * a * 1.9, 0.0); // additive
    }
    return out;
}

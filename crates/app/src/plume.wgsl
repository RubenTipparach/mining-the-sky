// Volumetric raymarched engine-exhaust plume. Each active nozzle emits a
// turbulent cone of glowing gas - blue-white at the throat, yellow then orange
// down the plume, fading at the tip, with faint Mach-diamond shock banding.
// Composited additively over the scene (camera-relative scene space, metres).
//
// Turbulence is the triangle-wave noise from nimitz's "Re-entry" (Shadertoy
// 4dGyRh), CC BY-NC-SA 3.0 - attribution retained.

struct P {
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    eye: vec4<f32>,
    center: vec4<f32>, // xyz bounding centre, w = bounding radius
    dir: vec4<f32>,    // xyz exhaust direction (unit), w = plume length
    params: vec4<f32>, // x = tan(fov/2), y = aspect, z = time, w = intensity
    nnoz: vec4<f32>,   // x = nozzle count, y = base radius
    noz: array<vec4<f32>, 12>, // xyz nozzle pos, w = per-nozzle radius scale
};
@group(0) @binding(0) var<uniform> u: P;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VsOut;
    o.pos = vec4<f32>(p[vi], 0.0, 1.0);
    o.uv = p[vi];
    return o;
}

fn tri(x: f32) -> f32 { return abs(fract(x) - 0.5) - 0.25; }
fn tri2(x: f32) -> f32 { return abs(fract(x) - 0.5); }
fn tri3(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(tri(p.z + tri(p.y)), tri(p.z + tri(p.x * 1.05)), tri(p.y + tri(p.x * 1.1)));
}
fn tnoise(pin: vec3<f32>, spd: f32, t: f32) -> f32 {
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

fn sphere_iv(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> vec2<f32> {
    let oc = ro - c;
    let b = dot(oc, rd);
    let cc = dot(oc, oc) - r * r;
    let h = b * b - cc;
    if (h < 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }
    let s = sqrt(h);
    return vec2<f32>(-b - s, -b + s);
}

// Best (densest) cone at a point: returns vec3(density, s-along, perp).
fn plume_at(pos: vec3<f32>) -> vec3<f32> {
    let dir = u.dir.xyz;
    let len = u.dir.w;
    let baseR = u.nnoz.y;
    let n = i32(u.nnoz.x);
    var best = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0; i < n; i = i + 1) {
        let noz = u.noz[i];
        let rel = pos - noz.xyz;
        let s = dot(rel, dir);                  // along the exhaust (0 at throat)
        if (s < -baseR * 0.4 || s > len) {
            continue;
        }
        let perp = length(rel - dir * s);
        let r0 = baseR * noz.w;
        // cone radius: pinched at the throat, flaring then tapering to the tip
        let coneR = r0 * (0.55 + 1.7 * smoothstep(0.0, len * 0.4, s)) * (1.0 - 0.45 * smoothstep(len * 0.45, len, s));
        let radial = smoothstep(coneR, coneR * 0.1, perp);
        let axial = smoothstep(-baseR * 0.4, baseR * 0.3, s) * (1.0 - smoothstep(len * 0.6, len, s));
        let d = radial * axial;
        if (d > best.x) {
            best = vec3<f32>(d, s / len, perp / max(coneR, 0.001));
        }
    }
    return best;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let t = u.params.z;
    let inten = u.params.w;
    let len = u.dir.w;
    let baseR = u.nnoz.y;

    let rd = normalize(u.fwd.xyz
        + u.right.xyz * (in.uv.x * u.params.x * u.params.y)
        + u.up.xyz * (in.uv.y * u.params.x));
    let ro = u.eye.xyz;
    let iv = sphere_iv(ro, rd, u.center.xyz, u.center.w);
    if (iv.y < 0.0) {
        return vec4<f32>(0.0);
    }

    var tt = max(iv.x, 0.0);
    let tmax = iv.y;
    let step = (tmax - tt) / 40.0 + 1e-3;
    var acc = vec3<f32>(0.0);
    for (var i = 0; i < 48; i = i + 1) {
        if (tt > tmax) {
            break;
        }
        let pos = ro + rd * tt;
        let pl = plume_at(pos);
        let d = pl.x;
        if (d > 0.003) {
            let frac = pl.y;       // 0 throat -> 1 tip
            let rn = pl.z;         // 0 axis -> 1 cone edge
            // boiling turbulence along the jet
            let q = pos / baseR * 0.5 - u.dir.xyz * (t * 7.0);
            let tb = tnoise(q, 0.1, t);
            var dens = d * (0.35 + 1.4 * tb);
            // Mach diamonds: bright shock cells near the throat, on-axis
            let diamonds = 0.5 + 0.5 * sin(frac * 38.0);
            let shock = smoothstep(0.5, 0.0, frac) * smoothstep(0.5, 0.0, rn) * diamonds;
            // colour: blue-white core/throat -> yellow -> orange -> fade
            let core = smoothstep(0.7, 0.0, rn);
            var c = mix(vec3<f32>(0.55, 0.72, 1.0), vec3<f32>(1.0, 0.9, 0.55), smoothstep(0.0, 0.22, frac));
            c = mix(c, vec3<f32>(1.0, 0.5, 0.16), smoothstep(0.22, 0.65, frac));
            c = c + vec3<f32>(0.5, 0.65, 1.0) * core * (1.0 - smoothstep(0.0, 0.35, frac)); // hot blue core
            c = c + vec3<f32>(1.0, 0.95, 0.85) * shock * 1.2;                                // mach diamonds
            // fade toward the tip; additive emission
            let emit = dens * (1.0 - smoothstep(0.6, 1.0, frac)) * (0.55 + 0.7 * core);
            acc = acc + c * emit;
        }
        tt = tt + step;
    }
    return vec4<f32>(acc * inten * 1.6, 0.0); // additive (alpha 0)
}

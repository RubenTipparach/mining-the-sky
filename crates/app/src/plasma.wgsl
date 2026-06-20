// Volumetric re-entry plasma driven by the vehicle SDF. The whole vehicle is a
// union of round-cone SDF primitives (one or more per part); this pass raymarches
// the shock layer that hugs the windward surfaces of that real geometry plus a
// downwind wake, in camera-relative scene space (metres), composited
// premultiplied-over on top of the scene.
//
// Density turbulence + colour grade adapted from nimitz's "Re-entry"
// (Shadertoy 4dGyRh), License CC BY-NC-SA 3.0 - attribution retained.

const MAX_PRIMS: u32 = 24u;

struct P {
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    eye: vec4<f32>,
    center: vec4<f32>,  // xyz vehicle centre, w = bounding radius
    flow: vec4<f32>,    // xyz airflow/velocity dir (unit), w = vehicle radius
    head: vec4<f32>,    // xyz windward leading point, w = vehicle length
    params: vec4<f32>,  // x = tan(fov/2), y = aspect, z = time, w = heat
    nprims: vec4<f32>,  // x = primitive count
    prims: array<vec4<f32>, 48>, // MAX_PRIMS*2: [a.xyz,r1] then [b.xyz,r2]
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

// --- triangle-wave turbulence (nimitz, "Re-entry") ---
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

// Round-cone SDF between a (radius r1) and b (radius r2). (iq, MIT.)
fn sd_round_cone(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r1: f32, r2: f32) -> f32 {
    let ba = b - a;
    let l2 = dot(ba, ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;
    let pa = p - a;
    let y = dot(pa, ba);
    let z = y - l2;
    let d2v = pa * l2 - ba * y;
    let x2 = dot(d2v, d2v);
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) {
        return sqrt(x2 + z2) * il2 - r2;
    }
    if (sign(y) * a2 * y2 < k) {
        return sqrt(x2 + y2) * il2 - r1;
    }
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

// Signed distance to the whole vehicle (union of the round-cone primitives).
fn vehicle_sdf(p: vec3<f32>) -> f32 {
    var d = 1.0e9;
    let n = i32(u.nprims.x);
    for (var i = 0; i < n; i = i + 1) {
        let pa = u.prims[i * 2];
        let pb = u.prims[i * 2 + 1];
        d = min(d, sd_round_cone(p, pa.xyz, pb.xyz, pa.w, pb.w));
    }
    return d;
}

// Ray/sphere interval (near, far) around a centre; far < 0 means a miss.
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

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let t = u.params.z;
    let heat = clamp(u.params.w, 0.0, 1.4);
    let center = u.center.xyz;
    let bound = u.center.w;
    let vhat = normalize(u.flow.xyz);   // airflow / velocity direction
    let vrad = max(u.flow.w, 1.0);      // vehicle radius scale

    let rd = normalize(u.fwd.xyz
        + u.right.xyz * (in.uv.x * u.params.x * u.params.y)
        + u.up.xyz * (in.uv.y * u.params.x));
    let ro = u.eye.xyz;

    let iv = sphere_iv(ro, rd, center, bound);
    if (iv.y < 0.0) {
        return vec4<f32>(0.0);
    }

    let lv = max(u.head.w, vrad * 2.0); // extent along the airflow (windward gate)
    let vsize = max(u.nprims.y, vrad * 2.0); // vehicle size (sets the wake length)
    let shell = vrad * 1.3;             // shock-layer thickness off the surface
    let smear_len = vsize * 1.4;        // how far the hot gas smears downstream

    var tt = max(iv.x, 0.0);
    let tmax = iv.y;
    var rz = vec4<f32>(0.0);
    let step = bound / 60.0;
    for (var i = 0; i < 96; i = i + 1) {
        if (rz.a > 0.99 || tt > tmax) {
            break;
        }
        let pos = ro + rd * tt;

        // SDF smear: walk a few samples UPSTREAM along the airflow. A point lights
        // up if a *windward* surface sits upstream of it (band hugging the SDF on
        // the leading side), so the windward shock smears straight downstream into
        // the wake, fading + cooling with distance. This works at any attitude
        // (the SDF + flow define it) and auto-rejects the leeward side (its
        // upstream is the body interior, where the band is killed).
        var glow = 0.0;
        var smear = smear_len; // downstream distance of the lit surface (cooling)
        // dither the sample offsets per pixel so the smear reads as soft streaks
        // rather than regular bands.
        let jit = fract(sin(dot(in.uv, vec2<f32>(127.1, 311.7)) + t) * 43758.5);
        for (var j = 0; j < 16; j = j + 1) {
            let k = (f32(j) + jit) / 16.0 * smear_len;
            let q = pos + vhat * k;
            let dv = vehicle_sdf(q);
            let band = smoothstep(shell, 0.0, dv) * smoothstep(-shell * 0.45, 0.1, dv);
            let windward = smoothstep(-0.18 * lv, 0.22 * lv, dot(q - center, vhat));
            let fade = exp(-k / (smear_len * 0.5));
            let g = band * windward * fade;
            if (g > glow) {
                glow = g;
                smear = k;
            }
        }

        if (glow > 0.0025) {
            // multi-octave boiling turbulence streaming down the smear: a coarse
            // roll plus a finer wisp layer for detail.
            let flow_t = vhat * (t * 3.2);
            let tb0 = tnoise(pos / vrad * 0.7 - flow_t, 0.1, t);
            let tb1 = tnoise(pos / vrad * 2.3 - flow_t * 1.7 + vec3<f32>(5.0), 0.15, t);
            let tb = clamp(tb0 * 0.8 + tb1 * 0.5, 0.0, 1.4);
            // fine filaments running along the streak (sharpened high band)
            let fil = smoothstep(0.55, 0.95, tb1);
            var dens = clamp(glow * (0.22 + 1.3 * tb + 0.5 * fil), 0.0, 1.0);

            // STEEP cooling gradient keyed to downstream distance, so the
            // white-hot glow stays pinned to the windward surface (smear ~ 0) and
            // drops fast to orange, then a long deep-red tail.
            let cool = clamp(smear / smear_len, 0.0, 1.0);
            let whitef = smoothstep(0.10, 0.0, cool);   // only right at the surface
            let yellowf = smoothstep(0.22, 0.04, cool);
            let orangef = smoothstep(0.45, 0.10, cool);
            var c = vec3<f32>(0.40, 0.05, 0.02);                  // deep-red tail
            c = mix(c, vec3<f32>(1.0, 0.36, 0.07), orangef);      // orange
            c = mix(c, vec3<f32>(1.0, 0.78, 0.40), yellowf);      // yellow
            c = mix(c, vec3<f32>(1.4, 1.32, 1.22), whitef);       // white-hot windward
            // faint pink/violet ionised fringe in the mid-temperature gas
            c = c + vec3<f32>(0.32, 0.05, 0.20) * orangef * (1.0 - yellowf) * 0.7;
            // cool blue-white sheen on the hottest filaments (windward only)
            c = c + vec3<f32>(0.3, 0.45, 0.7) * fil * whitef;

            // alpha biased to the hot windward glow; the cool tail stays wispy.
            let hot = max(whitef, yellowf * 0.65);
            let a = dens * dens * (0.3 + 1.1 * hot) + fil * whitef * 0.2;
            var col = vec4<f32>(c, a);
            col = vec4<f32>(col.rgb * col.a, col.a);
            rz = rz + col * (1.0 - rz.a);
        }
        tt = tt + step;
    }
    rz = rz * heat;
    rz.a = min(rz.a, 0.42);
    return clamp(rz, vec4<f32>(0.0), vec4<f32>(1.0));
}

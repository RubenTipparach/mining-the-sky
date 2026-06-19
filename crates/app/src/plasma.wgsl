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

    let shell = vrad * 0.95;            // shock-layer thickness off the surface
    var tt = max(iv.x, 0.0);
    let tmax = iv.y;
    var rz = vec4<f32>(0.0);
    let step = bound / 40.0;
    for (var i = 0; i < 80; i = i + 1) {
        if (rz.a > 0.99 || tt > tmax) {
            break;
        }
        let pos = ro + rd * tt;
        let dv = vehicle_sdf(pos);
        let rel = pos - center;
        let along = dot(rel, vhat);              // + = windward (leading) side
        let perp = length(rel - vhat * along);

        // shock layer hugging the windward surfaces (a band just outside, killed
        // inside the body so it doesn't paint over the leeward hull).
        let band = smoothstep(shell, 0.0, dv) * smoothstep(-shell * 0.4, 0.15, dv);
        let windward = smoothstep(-0.10 * bound, 0.34 * bound, along);
        let sheath = band * windward;

        // downwind wake: a translucent tail streaming behind, fading as it cools.
        let s = -along;                          // downwind distance from centre
        let wake = smoothstep(0.0, 0.12 * bound, s)
            * exp(-max(s, 0.0) / (0.42 * bound))
            * smoothstep(vrad * 4.0, 0.0, perp);

        let f = max(sheath, wake * 0.65);
        if (f > 0.001) {
            // boiling turbulence (sampled in vehicle radii)
            let q = pos / vrad - vhat * (t * 3.0);
            let tb = tnoise(q * 0.6, 0.1, t);
            let dens = clamp(f * (0.3 + 1.5 * tb), 0.0, 1.0);
            // windward density gradient -> cool blue-white compression front
            let gr = clamp((dv - vehicle_sdf(pos - vhat * shell * 0.6)) / shell, 0.0, 1.0);
            // cooling along the wake: white-hot/orange head -> deep red, fading
            let cool = clamp(max(s, 0.0) / (0.5 * bound), 0.0, 1.0);
            var lg = mix(vec3<f32>(1.0, 0.6, 0.2), vec3<f32>(0.55, 0.1, 0.04), cool);
            lg = lg + 1.5 * vec3<f32>(0.55, 0.77, 0.95) * gr * (1.0 - cool);
            var col = vec4<f32>(lg, dens * dens * 1.25);
            col = vec4<f32>(col.rgb * col.a, col.a);
            rz = rz + col * (1.0 - rz.a);
        }
        tt = tt + step;
    }
    rz = rz * heat;
    rz.a = min(rz.a, 0.72);
    return clamp(rz, vec4<f32>(0.0), vec4<f32>(1.0));
}

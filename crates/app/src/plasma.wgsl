// Volumetric re-entry plasma. The fireball volume is raymarched in a local frame
// anchored at the rocket (+Y = downwind / airflow direction) and composited
// premultiplied-over on top of the scene, so the vehicle glows inside a real
// boiling shock layer rather than behind flat billboards.
//
// The volume density, turbulence and colour grade are adapted from nimitz's
// "Re-entry" (Shadertoy 4dGyRh), License CC BY-NC-SA 3.0 - attribution retained.

struct P {
    right: vec4<f32>,   // camera basis (camera-relative scene space)
    up: vec4<f32>,
    fwd: vec4<f32>,
    eye: vec4<f32>,     // camera position (same space)
    center: vec4<f32>,  // plasma centre; w = metres per plasma unit (scale)
    axis: vec4<f32>,    // +Y plasma axis = downwind (airflow) direction
    side: vec4<f32>,    // +X plasma axis (perpendicular)
    params: vec4<f32>,  // x = tan(fov/2), y = aspect, z = time, w = heat
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

fn map2(p: vec3<f32>) -> f32 { return length(p) - 1.3; }

fn gradm(p: vec3<f32>) -> f32 {
    let e = 0.06;
    var d = map2(vec3<f32>(p.x, p.y - e, p.z)) - map2(vec3<f32>(p.x, p.y + e, p.z));
    d = d + map2(vec3<f32>(p.x - e, p.y, p.z)) - map2(vec3<f32>(p.x + e, p.y, p.z));
    d = d + map2(vec3<f32>(p.x, p.y, p.z - e)) - map2(vec3<f32>(p.x, p.y, p.z + e));
    return d;
}

// Fireball density at a plasma-space point (after nimitz's mapVol).
fn map_vol(pin: vec3<f32>, spd: f32, t: f32) -> f32 {
    var p = pin;
    // The dense bright shock sits on the windward (low-y, fall-direction) side
    // and thins out into a faint, transparent wake streaming up (+y) the body.
    let f = smoothstep(0.0, 1.25, 2.2 - (p.y + dot(p.xz, p.xz) * 0.62));
    let g = p.y;
    p.y = p.y * 0.27;
    p.z = p.z + gradm(p * 0.73) * 3.5;
    p.y = p.y + t * 6.0;
    var d = tnoise(p * vec3<f32>(0.3, 0.27, 0.3), spd * 0.7, t) * 1.4 + 0.01;
    d = d + max(g * 0.12, 0.0); // faint streaming wake (kept thin/transparent)
    d = d * f;
    return clamp(d, 0.0, 1.0);
}

// Ray/sphere interval (near, far) around the origin; far < 0 means a miss.
fn sphere_iv(ro: vec3<f32>, rd: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - r * r;
    let h = b * b - c;
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

    // per-pixel camera ray, then into the plasma's local frame
    let rd = normalize(u.fwd.xyz
        + u.right.xyz * (in.uv.x * u.params.x * u.params.y)
        + u.up.xyz * (in.uv.y * u.params.x));
    let yax = u.axis.xyz;
    let xax = u.side.xyz;
    let zax = cross(xax, yax);
    let scl = u.center.w;
    let o = u.eye.xyz - u.center.xyz;
    let op = vec3<f32>(dot(o, xax), dot(o, yax), dot(o, zax)) / scl;
    let dp = vec3<f32>(dot(rd, xax), dot(rd, yax), dot(rd, zax)); // unit (orthonormal basis)

    let RB = 2.8;
    let iv = sphere_iv(op, dp, RB);
    if (iv.y < 0.0) {
        return vec4<f32>(0.0);
    }
    var tt = max(iv.x, 0.0);
    let tmax = iv.y;
    var rz = vec4<f32>(0.0);
    for (var i = 0; i < 48; i = i + 1) {
        if (rz.a > 0.99 || tt > tmax) {
            break;
        }
        let pos = op + dp * tt;
        let r = map_vol(pos, 0.1, t);
        // windward (-y) density gradient -> the cool blue-white compression
        // front sits on the leading face the air hits, not the trailing wake.
        let gr = clamp((r - map_vol(pos - vec3<f32>(0.0, 0.7, 0.0), 0.1, t)) / 0.3, 0.0, 1.0);
        let lg = vec3<f32>(0.72, 0.28, 0.0) * 1.2 + 1.3 * vec3<f32>(0.55, 0.77, 0.9) * gr;
        // thinner per-step opacity so the shock layer reads as translucent gas
        var col = vec4<f32>(lg, r * r * r * 1.1);
        col.a = col.a * smoothstep(0.4, 1.2, 0.7 - map2(vec3<f32>(pos.x, pos.y * 0.17, pos.z)));
        col = vec4<f32>(col.rgb * col.a, col.a);
        rz = rz + col * (1.0 - rz.a);
        tt = tt + 0.16;
    }
    // warm the accumulation as in the reference, then scale by heat and keep it
    // translucent overall (alpha capped so the scene shows through).
    rz.g = rz.g * (rz.w * 0.9 + 0.12);
    rz.r = rz.r * (rz.w * 0.5 + 0.48);
    rz = rz * heat;
    rz.a = min(rz.a, 0.72);
    return clamp(rz, vec4<f32>(0.0), vec4<f32>(1.0));
}

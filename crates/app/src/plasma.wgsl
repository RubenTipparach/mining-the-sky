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

// Comet-style plasma density. The local frame has +Y = downwind (the trail
// direction), so the bright head sits at the windward leading face (s ~ 0) and
// a long boiling tail streams up +Y, fading out as it cools. `s` is the
// distance downwind from the head; the tail widens and thins downstream.
fn map_vol(pin: vec3<f32>, spd: f32, t: f32) -> f32 {
    var p = pin;
    let s = p.y;                 // 0 at the windward head, grows up the tail
    let rad = length(p.xz);
    // axial profile: rises sharply at the head, exponential cooling up the tail
    let axial = smoothstep(-0.9, 0.15, s) * exp(-max(s, 0.0) * 0.42);
    // the tail spreads a little downstream
    let width = 0.38 + max(s, 0.0) * 0.16;
    let radialf = smoothstep(width, 0.0, rad);
    let f = axial * radialf;
    var q = p;
    q.y = q.y * 0.5 + t * 5.0;   // scroll the boil up the tail
    var d = tnoise(q * vec3<f32>(0.5, 0.32, 0.5), spd * 0.7, t) * 1.6 + 0.02;
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

    let RB = 5.5;
    let iv = sphere_iv(op, dp, RB);
    if (iv.y < 0.0) {
        return vec4<f32>(0.0);
    }
    var tt = max(iv.x, 0.0);
    let tmax = iv.y;
    var rz = vec4<f32>(0.0);
    for (var i = 0; i < 64; i = i + 1) {
        if (rz.a > 0.99 || tt > tmax) {
            break;
        }
        let pos = op + dp * tt;
        let r = map_vol(pos, 0.1, t);
        // windward (-y) density gradient -> a cool blue-white compression front
        // on the leading face the air hits.
        let gr = clamp((r - map_vol(pos - vec3<f32>(0.0, 0.7, 0.0), 0.1, t)) / 0.3, 0.0, 1.0);
        // cooling along the tail: white-hot/orange head -> deep red, fading out.
        let cool = clamp(max(pos.y, 0.0) * 0.22, 0.0, 1.0);
        var lg = mix(vec3<f32>(1.0, 0.58, 0.18), vec3<f32>(0.58, 0.11, 0.04), cool);
        lg = lg + 1.5 * vec3<f32>(0.55, 0.77, 0.95) * gr * (1.0 - cool); // blue-white front at the head
        // translucent gas: thin per-step opacity, a touch denser at the hot head
        let dens = r * r * r * (1.5 + 1.2 * (1.0 - cool));
        var col = vec4<f32>(lg, dens);
        col = vec4<f32>(col.rgb * col.a, col.a);
        rz = rz + col * (1.0 - rz.a);
        tt = tt + 0.16;
    }
    // scale by heat and keep it translucent overall (alpha capped).
    rz = rz * heat;
    rz.a = min(rz.a, 0.72);
    return clamp(rz, vec4<f32>(0.0), vec4<f32>(1.0));
}

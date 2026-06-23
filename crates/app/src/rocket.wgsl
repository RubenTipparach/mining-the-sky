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
    detail: vec4<f32>,              // xyz = body centre (camera-rel), w = radius (0=off)
};

@group(0) @binding(0) var<uniform> u: U;

// --- procedural surface detail (value noise + fbm), used for airless bodies ---
fn dhash(p: vec3<f32>) -> f32 {
    let q = fract(p * 0.3183099 + vec3<f32>(0.1, 0.2, 0.3));
    let r = q * 17.0;
    return fract(r.x * r.y * r.z * (r.x + r.y + r.z));
}
fn dnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let c000 = dhash(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = dhash(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = dhash(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = dhash(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = dhash(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = dhash(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = dhash(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = dhash(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, w.x);
    let x10 = mix(c010, c110, w.x);
    let x01 = mix(c001, c101, w.x);
    let x11 = mix(c011, c111, w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}
fn dfbm(p: vec3<f32>) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var pp = p;
    for (var i = 0; i < 4; i = i + 1) {
        s = s + a * (dnoise(pp) * 2.0 - 1.0);
        pp = pp * 2.0;
        a = a * 0.5;
    }
    return s;
}
// 3 random values in [0,1) for an integer cell (for cellular crater placement).
fn dhash3(c: vec3<f32>) -> vec3<f32> {
    let q = vec3<f32>(
        dot(c, vec3<f32>(127.1, 311.7, 74.7)),
        dot(c, vec3<f32>(269.5, 183.3, 246.1)),
        dot(c, vec3<f32>(113.5, 271.9, 124.6)),
    );
    return fract(sin(q) * 43758.5453);
}

// Cellular crater field at a surface direction: scatter one impact per grid
// cell, each a depressed bowl with a raised rim, and sum the contributions of
// the nearby cells. `freq` sets the crater scale, `depth` their relief.
fn crater_layer(dir: vec3<f32>, freq: f32, depth: f32) -> f32 {
    let pp = dir * freq;
    let ip = floor(pp);
    let fp = pp - ip;
    var h = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let g = vec3<f32>(f32(dx), f32(dy), f32(dz));
                let r = dhash3(ip + g);
                // feature point inside the cell; r.z gates presence + varies size
                let feat = g + r - fp;
                let d = length(feat);
                let present = step(0.30, r.z);
                let cr = 0.34 + 0.30 * r.x; // crater radius (cell units)
                let s = d / cr;
                if (present > 0.5 && s < 1.35) {
                    let dep = depth * (0.55 + 0.7 * r.y);
                    var c = 0.0;
                    if (s < 1.0) {
                        c = c - dep * (1.0 - s * s); // parabolic bowl floor
                    }
                    // sharp raised rim ring + a little ejecta
                    c = c + 0.45 * dep * exp(-pow((s - 1.0) / 0.16, 2.0));
                    h = h + c;
                }
            }
        }
    }
    return h;
}

// Surface relief at a direction (unit): three crater scales + fine roughness.
fn detail_height(dir: vec3<f32>) -> f32 {
    var h = crater_layer(dir, 7.0, 1.0);
    h = h + crater_layer(dir, 17.0, 0.45);
    h = h + crater_layer(dir, 44.0, 0.18); // small craters (close-up)
    h = h + 0.10 * dfbm(dir * 90.0);       // fine regolith grain
    return h;
}

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
    var n = normalize(in.normal);
    let s = normalize(u.sun.xyz);

    // Procedural surface detail: perturb the (smooth) geometry normal with the
    // gradient of a regolith height field, and add a micro self-shadow toward
    // the sun. This is a continuous function of the body-space direction, so it
    // is identical across LOD patch seams and brings crater/rubble relief at
    // altitude without needing fine geometry.
    var ao = 1.0;
    if (u.detail.w > 0.5) {
        let bp = in.wpos - u.detail.xyz;       // body-space position
        let dir = normalize(bp);
        // tangent frame on the sphere
        var t = cross(select(vec3<f32>(0.0,0.0,1.0), vec3<f32>(1.0,0.0,0.0), abs(dir.x) < 0.9), dir);
        t = normalize(t);
        let b = cross(dir, t);
        // Sample step shrinks as we close in, so fine craters resolve on final
        // approach while staying coarse (anti-aliased) at distance.
        let fade = clamp(1.0 - (in.flogz - 1.0) / (u.detail.w * 10.0), 0.25, 1.0);
        let e = mix(0.020, 0.0035, fade);
        let h0 = detail_height(dir);
        let ht = detail_height(normalize(dir + t * e));
        let hb = detail_height(normalize(dir + b * e));
        // surface-tangent gradient -> tilt the normal (the smaller step near the
        // surface makes the slope steeper, so scale it back down by 1/e)
        let grad = ((ht - h0) * t + (hb - h0) * b) * (0.010 / e);
        // Ground-level smoothing: near fragments (the area around a landed
        // craft) taper the detail back to smooth, settled regolith over a wide
        // band; the far surface and horizon keep their craters.
        let near = smoothstep(20.0, 260.0, in.flogz - 1.0);
        n = normalize(n - grad * (1.6 * fade * near));
        // free ambient occlusion: crater floors (low h0) sit in shadow / hold
        // more regolith, so they read darker (also eased out up close).
        ao = mix(1.0, clamp(1.0 + 0.7 * min(h0, 0.0), 0.45, 1.0), near);
    }

    let diff = max(dot(n, s), 0.0) * ao;
    let sun_vis = step(0.0001, dot(n, s)); // 1 on the sun-facing hemisphere
    // Airless (lunar) bodies have no sky-fill: ambient is near-black so the only
    // light is the direct sun, giving stark crater shadows. On worlds with air,
    // use a richer bluish hemispheric sky/ground ambient.
    let airless = u.sun.w;
    let amb_air = mix(vec3<f32>(0.22, 0.20, 0.17), vec3<f32>(0.48, 0.54, 0.64), clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let amb_moon = vec3<f32>(0.09, 0.09, 0.10);
    let amb = mix(amb_air, amb_moon, airless);
    let sun_col = mix(vec3<f32>(1.05, 1.0, 0.92), vec3<f32>(1.25, 1.22, 1.15), airless);

    // Soft sun specular (Blinn-Phong): a sheen on the car, buildings and metal so
    // surfaces catch the light instead of reading as flat matte blocks.
    let vdir = normalize(-in.wpos); // camera sits at the origin in camera-rel space
    let hvec = normalize(s + vdir);
    let spec = pow(max(dot(n, hvec), 0.0), 26.0) * 0.20 * ao * sun_vis;

    // Inside the assembly building (interior -> 1) the roof shades the sun, so
    // dim sun + ambient and let the work lights carry the scene.
    let interior = u.params.w;
    let sf = 1.0 - 0.9 * interior;
    let af = 1.0 - 0.68 * interior;
    var lit = in.color * (amb * af + sun_col * diff * 1.05 * sf) + sun_col * spec * sf;

    // A subtle sky reflection on grazing faces (worlds with air only): lifts the
    // edges of buildings/vehicles with a hint of sky colour for some depth.
    let fres = pow(1.0 - max(dot(n, vdir), 0.0), 4.0);
    lit = lit + vec3<f32>(0.45, 0.52, 0.64) * fres * 0.08 * (1.0 - airless);

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

// Atmosphere for the rocket view: a fullscreen pass that ray-marches a true
// spherical single-scattering (Rayleigh + Mie) density field wrapping the
// to-scale planet. Drawn behind the terrain (depth write off, compare always),
// so it paints the sky, the limb glow and space; the terrain draws over the
// ground. Because it is evaluated in planet-centred world coords, the sky goes
// blue at the surface and fades to black with a bright atmospheric limb as the
// rocket climbs to orbit - the seamless ground-to-orbit transition.

struct U {
    right: vec4<f32>, // world camera basis
    up: vec4<f32>,
    fwd: vec4<f32>,
    sun: vec4<f32>,   // world sun direction
    cam: vec4<f32>,   // camera position relative to planet centre (metres)
    params: vec4<f32>, // x tan(fov/2), y aspect, z planet radius, w atmosphere top
};

@group(0) @binding(0) var<uniform> u: U;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(p[vi], 0.0, 1.0);
    o.ndc = p[vi];
    return o;
}

const PI: f32 = 3.14159265;
const HR: f32 = 8000.0;   // Rayleigh scale height (m)
const HM: f32 = 1200.0;   // Mie scale height (m)
const G: f32 = 0.76;      // Mie anisotropy
// Rayleigh / Mie extinction at sea level (1/m). Rayleigh tuned to the
// wavelengthsInv4 = (5.602, 9.473, 19.644) ratio for a blue sky.
const BETA_R: vec3<f32> = vec3<f32>(5.5e-6, 13.0e-6, 22.4e-6);
const BETA_M: vec3<f32> = vec3<f32>(21e-6, 21e-6, 21e-6);
const SUN_I: f32 = 22.0;
const VIEW_N: i32 = 12;
const LIGHT_N: i32 = 5;

// Near intersection distance of a ray with a sphere of `radius` centred at the
// origin, or -1 if it misses / is behind. Returns vec2(t_near, t_far).
fn ray_sphere(orig: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(orig, dir);
    let c = dot(orig, orig) - radius * radius;
    let d = b * b - c;
    if (d < 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }
    let s = sqrt(d);
    return vec2<f32>(-b - s, -b + s);
}

// Single-scattering in-scatter along the view ray from `orig` in `dir`.
fn atmosphere(orig: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>, rp: f32, ra: f32) -> vec3<f32> {
    let atm = ray_sphere(orig, dir, ra);
    if (atm.y < 0.0) {
        return vec3<f32>(0.0); // ray never reaches the atmosphere shell
    }
    var tmin = max(atm.x, 0.0);
    var tmax = atm.y;
    // a hit on the solid planet ends the view ray at the surface
    let pl = ray_sphere(orig, dir, rp);
    if (pl.x > 0.0) {
        tmax = min(tmax, pl.x);
    }
    let seg = (tmax - tmin) / f32(VIEW_N);
    if (seg <= 0.0) {
        return vec3<f32>(0.0);
    }

    var sum_r = vec3<f32>(0.0);
    var sum_m = vec3<f32>(0.0);
    var od_r = 0.0; // accumulated view-ray optical depth
    var od_m = 0.0;
    var t = tmin;
    for (var i = 0; i < VIEW_N; i = i + 1) {
        let p = orig + dir * (t + seg * 0.5);
        let h = length(p) - rp;
        let hr = exp(-h / HR) * seg;
        let hm = exp(-h / HM) * seg;
        od_r = od_r + hr;
        od_m = od_m + hm;

        // optical depth from the sample toward the sun
        let ls = ray_sphere(p, sun, ra).y;
        let segl = ls / f32(LIGHT_N);
        var odl_r = 0.0;
        var odl_m = 0.0;
        var blocked = false;
        var tl = 0.0;
        for (var j = 0; j < LIGHT_N; j = j + 1) {
            let pl2 = p + sun * (tl + segl * 0.5);
            let hl = length(pl2) - rp;
            if (hl < 0.0) {
                blocked = true;
                break;
            }
            odl_r = odl_r + exp(-hl / HR) * segl;
            odl_m = odl_m + exp(-hl / HM) * segl;
            tl = tl + segl;
        }
        if (!blocked) {
            let tau = BETA_R * (od_r + odl_r) + BETA_M * 1.1 * (od_m + odl_m);
            let att = exp(-tau);
            sum_r = sum_r + att * hr;
            sum_m = sum_m + att * hm;
        }
        t = t + seg;
    }

    let mu = dot(dir, sun);
    let phase_r = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    let g2 = G * G;
    let phase_m = 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu))
        / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * G * mu, 1.5));
    return SUN_I * (sum_r * BETA_R * phase_r + sum_m * BETA_M * phase_m);
}

// Cheap hash-based starfield: quantise the ray direction into cells and light a
// pixel when it lands near a per-cell random star position. Airless (lunar) sky.
fn hash21(p: vec2<f32>) -> f32 {
    var h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn starfield(dir: vec3<f32>) -> vec3<f32> {
    // Map the direction to spherical-ish UV cells.
    let uv = vec2<f32>(atan2(dir.z, dir.x), asin(clamp(dir.y, -1.0, 1.0)));
    let scale = 180.0;
    let cell = floor(uv * scale);
    let f = fract(uv * scale);
    let r1 = hash21(cell);
    let r2 = hash21(cell + vec2<f32>(41.3, 7.7));
    let r3 = hash21(cell + vec2<f32>(13.1, 91.7));
    // Only some cells contain a star.
    if (r3 > 0.93) {
        let star = vec2<f32>(r1, r2);
        let d = length(f - star);
        let b = smoothstep(0.06, 0.0, d) * (0.4 + 0.6 * r3);
        let tint = mix(vec3<f32>(0.8, 0.85, 1.0), vec3<f32>(1.0, 0.95, 0.85), r1);
        return tint * b;
    }
    return vec3<f32>(0.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let tan = u.params.x;
    let aspect = u.params.y;
    let rp = u.params.z;
    let ra = u.params.w;
    let ray = normalize(
        u.fwd.xyz
        + in.ndc.x * tan * aspect * u.right.xyz
        + in.ndc.y * tan * u.up.xyz
    );
    let sun = normalize(u.sun.xyz);
    let cam = u.cam.xyz;

    // Airless body (the moon): pure black space, a starfield, and a hard sun
    // disk. No atmospheric scattering, no limb glow.
    if (u.cam.w >= 0.5) {
        let pl = ray_sphere(cam, ray, rp);
        let space = select(1.0, 0.0, pl.x > 0.0);
        var lc = starfield(ray) * space;
        let sd = max(dot(ray, sun), 0.0);
        let disk = smoothstep(0.9994, 0.9998, sd) * 1.0;
        let glow = pow(sd, 3000.0) * 0.4;
        lc = lc + vec3<f32>(1.0, 0.98, 0.92) * (disk + glow) * space;
        return vec4<f32>(lc, 1.0);
    }

    var col = atmosphere(cam, ray, sun, rp, ra);

    // Sun disk + glow, only when the line of sight to the sun is not blocked by
    // the planet body.
    let pl = ray_sphere(cam, ray, rp);
    let sun_vis = select(1.0, 0.0, pl.x > 0.0);
    let sd = max(dot(ray, sun), 0.0);
    let disk = smoothstep(0.9996, 0.9999, sd) * 14.0;
    let glow = pow(sd, 1500.0) * 6.0;
    col = col + vec3<f32>(1.0, 0.96, 0.86) * (disk + glow) * sun_vis;

    // Tonemap (HDR -> display) and a faint space tint so deep space is near black.
    col = vec3<f32>(1.0) - exp(-col * 1.4);
    return vec4<f32>(col, 1.0);
}

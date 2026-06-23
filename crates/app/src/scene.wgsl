// Perspective "system view": a fullscreen raymarch of the home world + its
// moon, positioned in world space (units of 1000 km so f32 stays well
// conditioned at interplanetary distances). This is the seed of the real 3D
// renderer - a free perspective camera that can frame more than one body, as
// opposed to the orthographic single-disk surface view.

struct Scene {
    cam_pos: vec4<f32>, // xyz camera position (Mm)
    cam_x: vec4<f32>,   // camera right
    cam_y: vec4<f32>,   // camera up
    cam_z: vec4<f32>,   // camera forward (toward the scene)
    sun: vec4<f32>,     // xyz world sun direction
    home: vec4<f32>,    // xyz centre, w radius (Mm)
    moon: vec4<f32>,    // xyz centre, w radius (Mm)
    sunbody: vec4<f32>,  // star A: xyz centre, w radius (Mm)
    sunbody2: vec4<f32>, // star B: xyz centre, w radius (Mm)
    params: vec4<f32>,   // x=tan(fov/2), y=aspect, z=time, w=planet count
    res: vec4<f32>,      // x,y = resolution
    planets: array<vec4<f32>, 16>,    // xyz centre, w radius (Mm)
    planet_col: array<vec4<f32>, 16>, // rgb colour
    moons: array<vec4<f32>, 8>,       // nearest moons: xyz centre, w radius
};


@group(0) @binding(0) var<uniform> s: Scene;
@group(0) @binding(1) var home_tex: texture_2d<f32>;
@group(0) @binding(2) var home_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    out.uv = p[vi];
    return out;
}

const PI: f32 = 3.14159265;

fn hash3(p: vec3<f32>) -> f32 {
    let q = fract(p * 0.3183099 + vec3<f32>(0.1, 0.2, 0.3));
    let r = q + vec3<f32>(dot(q, q.yzx + 19.19));
    return fract((r.x + r.y) * r.z);
}

fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let c000 = hash3(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, w.x);
    let x10 = mix(c010, c110, w.x);
    let x01 = mix(c001, c101, w.x);
    let x11 = mix(c011, c111, w.x);
    let y0 = mix(x00, x10, w.y);
    let y1 = mix(x01, x11, w.y);
    return mix(y0, y1, w.z);
}

// Scattered jitter vector in 0..1 for a Voronoi cell, so feature points sit at
// irregular positions inside their integer cells.
fn hash33(p: vec3<f32>) -> vec3<f32> {
    let q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6)),
    );
    return fract(sin(q) * 43758.5453);
}

// Brightness of a single light point at squared distance `d2` (cell units),
// reaching zero past `radius` so points stay small and separated.
fn dot_bright(d2: f32, radius: f32) -> f32 {
    let d = sqrt(d2);
    return pow(clamp(1.0 - d / radius, 0.0, 1.0), 1.4);
}

// Gated Voronoi point field. Each cell holds one feature point ("a town"), but it
// is only lit if the cell's random rank falls below the local `density`. So a
// dense core (density >= 1) lights every point and reads as a packed cluster,
// while a low-density fringe lights only a scattered few - isolated dots with
// dark gaps between them, like real city lights from orbit. Returns the
// brightness of the nearest lit point to `p`.
fn scatter_points(p: vec3<f32>, density: f32, radius: f32) -> f32 {
    let ip = floor(p);
    let fp = fract(p);
    var best = 1.0e9;
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let g = vec3<f32>(f32(x), f32(y), f32(z));
                let cell = ip + g;
                let rank = hash3(cell + vec3<f32>(0.7, 2.3, 4.1));
                if (rank < density) {
                    let o = hash33(cell);
                    let r = g + o - fp;
                    best = min(best, dot(r, r));
                }
            }
        }
    }
    return dot_bright(best, radius);
}

// Nearest positive root of the ray/sphere intersection, or -1.0 on a miss.
fn hit_sphere(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let oc = center - ro;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    let t = b - sqrt(disc);
    return t;
}

fn shade_star(p: vec3<f32>, center: vec3<f32>, rd: vec3<f32>, tint: vec3<f32>) -> vec3<f32> {
    let n = normalize(p - center);
    // limb darkening + granulation; emissive and bright (bloom via tonemap)
    let limb = pow(clamp(dot(n, -rd), 0.0, 1.0), 0.45);
    let gran = 0.85 + 0.15 * vnoise(n * 40.0);
    return tint * (0.6 + 0.7 * limb) * gran;
}

fn shade_planet(p: vec3<f32>, center: vec3<f32>, color: vec3<f32>) -> vec3<f32> {
    let n = normalize(p - center);
    let sun = normalize(s.sun.xyz - p);
    let ndl = max(dot(n, sun), 0.0);
    // faint banding for variety
    let band = 0.92 + 0.08 * sin(n.y * 18.0);
    return color * band * (0.05 + 0.95 * ndl);
}

// Night-side city lights. The baked emission map (texel.a) is treated as a
// POPULATION DENSITY, not a glow: it gates a gated Voronoi point field so the
// surface lights up as discrete settlements - packed clusters in the dense
// cores fading to scattered isolated dots toward the fringes, with dark country
// between - rather than a blurry blob. Two scales give big-city dots plus a
// finer scatter of small towns. Highways (their own faint density) read as
// dotted strings of light between the cities.
fn city_lights(n: vec3<f32>, emission: f32, time: f32) -> vec3<f32> {
    if (emission < 0.004) {
        return vec3<f32>(0.0);
    }
    // local density drives how many feature points light up (cores >= 1 fill in;
    // fringes light only a scattered few).
    let density = clamp(emission * 1.4, 0.0, 1.45);
    let coarse = scatter_points(n * 125.0, density, 0.58);
    let fine = scatter_points(n * 350.0 + vec3<f32>(11.3, 4.1, 7.7), density, 0.5);
    let dots = max(coarse, 0.8 * fine);
    // a faint connective bloom only in the brightest cores (kept small so the
    // outskirts stay dotty, not smeared).
    let glow = 0.10 * emission * emission;
    // warm/cool district variation + a gentle twinkle on the points.
    let wc = vnoise(n * 200.0 + vec3<f32>(7.3, 1.1, 4.2));
    let tw = 0.82 + 0.18 * sin(time * 2.2 + wc * 120.0);
    let inten = (glow + dots * (0.45 + 0.95 * emission)) * tw;
    let cool = smoothstep(0.72, 0.94, wc);
    let warm = vec3<f32>(1.0, 0.72, 0.38);
    let white = vec3<f32>(0.85, 0.90, 1.0);
    return mix(warm, white, cool * 0.6) * inten;
}

fn shade_home(p: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    let n = normalize(p - s.home.xyz);
    let sun = normalize(s.sun.xyz - p);

    let lon = atan2(n.z, n.x);
    let lat = asin(clamp(n.y, -1.0, 1.0));
    let tuv = vec2<f32>((lon + PI) / (2.0 * PI), (PI * 0.5 - lat) / PI);
    let texel = textureSampleLevel(home_tex, home_samp, tuv, 0.0);
    let albedo = texel.rgb;
    let emission = texel.a;

    let ndl = dot(n, sun);
    let day = smoothstep(-0.06, 0.16, ndl);
    let diffuse = day * (0.12 + 0.88 * max(ndl, 0.0));
    var col = albedo * vec3<f32>(1.05, 1.02, 0.95) * diffuse;

    // detailed city lights on the dark side (fade in through the terminator)
    let night = 1.0 - day;
    col = col + city_lights(n, emission, s.params.z) * night * 2.6;

    // atmospheric limb (rim toward the camera)
    let viewdir = -rd;
    let rim = pow(1.0 - max(dot(n, viewdir), 0.0), 3.0);
    col = col + vec3<f32>(0.3, 0.5, 1.0) * rim * (0.7 * day + 0.05);
    return col;
}

fn shade_moon_at(p: vec3<f32>, center: vec3<f32>) -> vec3<f32> {
    let n = normalize(p - center);
    let sun = normalize(s.sun.xyz - p);
    // grey regolith with darker maria from noise
    let maria = vnoise(n * 6.0) * 0.5 + vnoise(n * 18.0) * 0.25;
    let base = mix(0.32, 0.62, smoothstep(0.35, 0.75, maria));
    let ndl = max(dot(n, sun), 0.0);
    let amb = 0.04;
    return vec3<f32>(base) * (amb + ndl);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let aspect = s.params.y;
    let fs_scale = s.params.x;
    let uv = in.uv;
    let rd = normalize(
        s.cam_z.xyz + (uv.x * aspect * fs_scale) * s.cam_x.xyz + (uv.y * fs_scale) * s.cam_y.xyz
    );
    let ro = s.cam_pos.xyz;

    var col = vec3<f32>(0.0);
    var hit = false;
    var best = 1e30;

    let th = hit_sphere(ro, rd, s.home.xyz, s.home.w);
    if (th > 0.0 && th < best) { best = th; col = shade_home(ro + rd * th, rd); hit = true; }
    let tm = hit_sphere(ro, rd, s.moon.xyz, s.moon.w);
    if (s.moon.w > 0.0 && tm > 0.0 && tm < best) { best = tm; col = shade_moon_at(ro + rd * tm, s.moon.xyz); hit = true; }
    // nearest moons as lit spheres (so they read as real bodies up close)
    let mcount = i32(s.res.z);
    for (var k = 0; k < mcount; k = k + 1) {
        let mb = s.moons[k];
        let tmo = hit_sphere(ro, rd, mb.xyz, mb.w);
        if (tmo > 0.0 && tmo < best) {
            best = tmo;
            col = shade_moon_at(ro + rd * tmo, mb.xyz);
            hit = true;
        }
    }
    // binary stars
    let ta = hit_sphere(ro, rd, s.sunbody.xyz, s.sunbody.w);
    if (ta > 0.0 && ta < best) { best = ta; col = shade_star(ro + rd * ta, s.sunbody.xyz, rd, vec3<f32>(1.6, 1.3, 0.8)); hit = true; }
    let tb = hit_sphere(ro, rd, s.sunbody2.xyz, s.sunbody2.w);
    if (tb > 0.0 && tb < best) { best = tb; col = shade_star(ro + rd * tb, s.sunbody2.xyz, rd, vec3<f32>(1.5, 0.55, 0.4)); hit = true; }
    // circumbinary planets
    let pcount = i32(s.params.w);
    for (var k = 0; k < pcount; k = k + 1) {
        let pl = s.planets[k];
        let tp = hit_sphere(ro, rd, pl.xyz, pl.w);
        if (tp > 0.0 && tp < best) {
            best = tp;
            col = shade_planet(ro + rd * tp, pl.xyz, s.planet_col[k].rgb);
            hit = true;
        }
    }

    if (!hit) {
        // faint starfield, plus a thin atmospheric halo around the home limb
        let star = step(0.9975, hash3(floor(rd * 1400.0)));
        col = vec3<f32>(star) * 0.7;

        // corona/glow when looking near either star
        let to_sun = normalize(s.sunbody.xyz - ro);
        let sa = max(dot(rd, to_sun), 0.0);
        col = col + vec3<f32>(1.4, 1.1, 0.7) * (pow(sa, 220.0) * 1.5 + pow(sa, 12.0) * 0.10);

        // the red companion (star B) glows with its own ruddy corona
        let to_sun2 = normalize(s.sunbody2.xyz - ro);
        let sb = max(dot(rd, to_sun2), 0.0);
        col = col + vec3<f32>(1.5, 0.45, 0.30) * (pow(sb, 240.0) * 1.3 + pow(sb, 12.0) * 0.09);

        let oc = s.home.xyz - ro;
        let tca = dot(oc, rd);
        if (tca > 0.0) {
            let d = sqrt(max(dot(oc, oc) - tca * tca, 0.0));
            let r = s.home.w;
            let halo = smoothstep(r * 1.06, r, d) * smoothstep(r * 0.985, r, d);
            let sun = normalize(s.sun.xyz - s.home.xyz);
            let lit = clamp(dot(normalize(oc), sun) * 0.5 + 0.6, 0.0, 1.0);
            col = col + vec3<f32>(0.3, 0.5, 1.0) * halo * lit * 0.9;
        }
    }

    col = vec3<f32>(1.0) - exp(-col * 1.2);
    return vec4<f32>(col, 1.0);
}

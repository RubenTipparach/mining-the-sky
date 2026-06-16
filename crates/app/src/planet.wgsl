// Live planet rendered from the baked worldgen texture (RGB = albedo,
// A = city-light emission). Orthographic raymarch of a sphere with day/night
// terminator, atmospheric limb, ocean glint, and dark-side city lights. This
// is the GPU counterpart to the worldgen CPU preview.

struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
    sun: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var planet_tex: texture_2d<f32>;
@group(0) @binding(2) var planet_samp: sampler;

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

const PI: f32 = 3.14159265;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    var uv = in.uv;
    uv.x = uv.x * aspect;
    uv = uv * 1.25; // margin so the disk + atmosphere fit

    let sun = normalize(u.sun.xyz);
    let r2 = dot(uv, uv);

    // slow rotation of the planet about its axis
    let ang = u.time * 0.04;
    let ca = cos(ang);
    let sa = sin(ang);

    var col = vec3<f32>(0.0);

    if (r2 <= 1.0) {
        let nz = sqrt(1.0 - r2);
        let n = vec3<f32>(uv.x, uv.y, nz); // view-space normal
        let pdir = normalize(vec3<f32>(ca * n.x + sa * n.z, n.y, -sa * n.x + ca * n.z));

        // sample the baked world (equirectangular)
        let lon = atan2(pdir.z, pdir.x);
        let lat = asin(clamp(pdir.y, -1.0, 1.0));
        let tuv = vec2<f32>((lon + PI) / (2.0 * PI), (PI * 0.5 - lat) / PI);
        let texel = textureSampleLevel(planet_tex, planet_samp, tuv, 0.0);
        let albedo = texel.rgb;
        let emission = texel.a;

        let ndl = dot(n, sun);
        let day = smoothstep(-0.06, 0.16, ndl);
        let diffuse = day * (0.12 + 0.88 * max(ndl, 0.0));
        col = albedo * vec3<f32>(1.05, 1.02, 0.95) * diffuse;

        // ocean glint where albedo is blue-dominant
        let oceanish = clamp((albedo.b - max(albedo.r, albedo.g)) * 4.0, 0.0, 1.0);
        let half = normalize(sun + vec3<f32>(0.0, 0.0, 1.0));
        let spec = pow(max(dot(n, half), 0.0), 60.0) * day * oceanish;
        col = col + vec3<f32>(0.8, 0.8, 0.72) * spec;

        // city lights on the dark side
        let night = 1.0 - day;
        col = col + vec3<f32>(1.0, 0.82, 0.5) * emission * night * 1.7;

        // atmosphere limb
        let rim = pow(1.0 - nz, 3.0);
        col = col + vec3<f32>(0.3, 0.5, 1.0) * rim * (0.6 * day + 0.04);
    } else {
        let r = sqrt(r2);
        if (r < 1.06) {
            let ln = normalize(vec3<f32>(uv.x, uv.y, 0.0));
            let ndl = max(dot(ln, sun), 0.0);
            let d = (r - 1.0) / 0.06;
            let glow = pow(clamp(1.0 - d, 0.0, 1.0), 2.0);
            col = vec3<f32>(0.3, 0.5, 1.0) * glow * (ndl * 0.9 + 0.05);
        } else {
            let s = vnoise(vec3<f32>(in.uv * 800.0, 1.0));
            col = vec3<f32>(step(0.985, s));
        }
    }

    col = vec3<f32>(1.0) - exp(-col * 1.1);
    col = pow(col, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(col, 1.0);
}

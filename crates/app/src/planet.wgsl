// Live planet rendered from the baked worldgen texture (RGB = albedo,
// A = city-light emission). Orthographic raymarch of a sphere with day/night
// terminator, atmospheric limb, ocean glint, and dark-side city lights. The
// camera is a free orbit camera: `c*` are the columns of the world-from-view
// rotation, so view-space normals are lit and sampled in world space and the
// terminator stays fixed on the planet as you orbit around it.

struct Uniforms {
    resolution: vec2<f32>,
    scale: f32,   // view-plane half-extent: smaller = zoomed in
    time: f32,
    sun: vec4<f32>, // world-space sun direction in xyz
    cx: vec4<f32>,  // world-from-view rotation, column 0 (view right)
    cy: vec4<f32>,  // column 1 (view up)
    cz: vec4<f32>,  // column 2 (toward the viewer)
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

const PI: f32 = 3.14159265;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    var uv = in.uv;
    uv.x = uv.x * aspect;
    uv = uv * u.scale;

    let sun = normalize(u.sun.xyz);
    let rot = mat3x3<f32>(u.cx.xyz, u.cy.xyz, u.cz.xyz);
    let viewer = u.cz.xyz; // world direction toward the camera
    let r2 = dot(uv, uv);

    var col = vec3<f32>(0.0);

    if (r2 <= 1.0) {
        let nz = sqrt(1.0 - r2);
        let n = vec3<f32>(uv.x, uv.y, nz); // view-space normal
        let pdir = rot * n;                // world-space normal

        // sample the baked world (equirectangular)
        let lon = atan2(pdir.z, pdir.x);
        let lat = asin(clamp(pdir.y, -1.0, 1.0));
        let tuv = vec2<f32>((lon + PI) / (2.0 * PI), (PI * 0.5 - lat) / PI);
        let texel = textureSampleLevel(planet_tex, planet_samp, tuv, 0.0);
        let albedo = texel.rgb;
        let emission = texel.a;

        let ndl = dot(pdir, sun);
        let day = smoothstep(-0.06, 0.16, ndl);
        let diffuse = day * (0.12 + 0.88 * max(ndl, 0.0));
        col = albedo * vec3<f32>(1.05, 1.02, 0.95) * diffuse;

        // ocean glint where albedo is blue-dominant
        let oceanish = clamp((albedo.b - max(albedo.r, albedo.g)) * 4.0, 0.0, 1.0);
        let half = normalize(sun + viewer);
        let spec = pow(max(dot(pdir, half), 0.0), 60.0) * day * oceanish;
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
            // approximate the world normal at the limb for the glow's day side
            let pdir = rot * ln;
            let ndl = max(dot(pdir, sun), 0.0);
            let d = (r - 1.0) / 0.06;
            let glow = pow(clamp(1.0 - d, 0.0, 1.0), 2.0);
            col = vec3<f32>(0.3, 0.5, 1.0) * glow * (ndl * 0.9 + 0.05);
        }
    }

    // Exposure/tonemap only. The render target is sRGB, so it applies the
    // gamma encode on store - do NOT also gamma-correct here (that double-
    // encodes and washes the image out).
    col = vec3<f32>(1.0) - exp(-col * 1.2);
    return vec4<f32>(col, 1.0);
}

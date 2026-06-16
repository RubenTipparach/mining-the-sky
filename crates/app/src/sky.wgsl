// Sky for the rocket view: a fullscreen pass that reconstructs the per-pixel
// world-space view ray and shades a gradient sky with a sun disk + glow. Drawn
// behind the terrain (depth write off, compare always), then the terrain draws
// over it. The horizon colour matches the terrain's aerial-perspective fog so
// the surface fades seamlessly into the sky.

struct U {
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    sun: vec4<f32>,
    params: vec4<f32>,  // x = tan(fov/2), y = aspect
    horizon: vec4<f32>, // rgb haze colour
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

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let tan = u.params.x;
    let aspect = u.params.y;
    let ray = normalize(
        u.fwd.xyz
        + in.ndc.x * tan * aspect * u.right.xyz
        + in.ndc.y * tan * u.up.xyz
    );
    let sun = normalize(u.sun.xyz);

    let up_t = clamp(ray.y, 0.0, 1.0);
    let zenith = vec3<f32>(0.18, 0.40, 0.78);
    let horizon = u.horizon.rgb;
    var col = mix(horizon, zenith, pow(up_t, 0.55));
    // below the horizon line, settle to haze (mostly covered by terrain)
    if (ray.y < 0.0) {
        col = mix(horizon, vec3<f32>(0.45, 0.42, 0.40), clamp(-ray.y * 3.0, 0.0, 1.0));
    }

    // sun glow + disk
    let sd = max(dot(ray, sun), 0.0);
    let glow = pow(sd, 8.0) * 0.25 + pow(sd, 350.0) * 1.2;
    let disk = smoothstep(0.9995, 0.9998, sd) * 6.0;
    col = col + vec3<f32>(1.0, 0.95, 0.82) * (glow + disk);

    return vec4<f32>(col, 1.0);
}

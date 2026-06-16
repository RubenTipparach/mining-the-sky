// Trajectory / rocket overlay. Vertices are already projected to clip space on
// the CPU (the planet uses an orthographic camera, so projecting a world point
// is a rotate-and-drop-Z that is cheap to do per frame), with a per-vertex
// colour. Drawn as a line list on top of the planet, no depth test.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs(@location(0) p: vec2<f32>, @location(1) color: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

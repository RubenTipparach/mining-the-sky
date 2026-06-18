//! Cube-sphere mapping: 6 cube faces, each parameterised by (u, v) in [-1, 1],
//! mapped to the unit sphere with an equal-area-ish correction so patches stay
//! well shaped near the cube corners.

use glam::DVec3;

/// A point on cube face `face` at (u, v) in [-1, 1], as a point on the [-1, 1]^3
/// cube.
pub fn face_point(face: u8, u: f64, v: f64) -> DVec3 {
    match face {
        0 => DVec3::new(1.0, v, -u),  // +X
        1 => DVec3::new(-1.0, v, u),  // -X
        2 => DVec3::new(u, 1.0, -v),  // +Y
        3 => DVec3::new(u, -1.0, v),  // -Y
        4 => DVec3::new(u, v, 1.0),   // +Z
        _ => DVec3::new(-u, v, -1.0), // -Z
    }
}

/// Map a cube point in [-1, 1]^3 onto the unit sphere (Cobe-style correction so
/// the projection is closer to equal-area than a plain normalize).
pub fn cube_to_sphere(p: DVec3) -> DVec3 {
    let (x2, y2, z2) = (p.x * p.x, p.y * p.y, p.z * p.z);
    DVec3::new(
        p.x * (1.0 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3.0).sqrt(),
        p.y * (1.0 - z2 * 0.5 - x2 * 0.5 + z2 * x2 / 3.0).sqrt(),
        p.z * (1.0 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3.0).sqrt(),
    )
}

/// Unit sphere direction for a face/uv coordinate.
pub fn face_dir(face: u8, u: f64, v: f64) -> DVec3 {
    cube_to_sphere(face_point(face, u, v)).normalize()
}

/// A local east/north tangent basis at a unit direction (north biased to +Y).
pub fn tangent_basis(dir: DVec3) -> (DVec3, DVec3) {
    let up = dir.normalize();
    let world_up = if up.y.abs() > 0.999 { DVec3::Z } else { DVec3::Y };
    let east = world_up.cross(up).normalize();
    let north = up.cross(east).normalize();
    (east, north)
}

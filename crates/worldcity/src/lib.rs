//! Unified procedural city system.
//!
//! A single deterministic generator turns a compact [`CityDesc`] (a city's world
//! address: lon/lat, population, seed) into a full [`CityLayout`] - the block,
//! road and building data in the city's own local tangent frame. That one layout
//! is the source for every level of detail:
//!
//! * near (< a few km): the 3D building meshes,
//! * far: flat concrete block + road footprints,
//! * orbit: the summed light emission,
//!
//! so the lights you see on the ground and the lights you see from space finally
//! come from the same data instead of two disconnected systems.
//!
//! Layouts are content cached on disk ([`CityCache`], native only) and looked up
//! through a spatial [`CityIndex`] keyed by world address, so a layout is
//! generated once at world-load time and retrieved cheaply afterwards.

use glam::{DVec3, Vec2};

mod cache;
mod index;
pub use cache::CityCache;
pub use index::CityIndex;

/// 0..1 hash from two integer keys - the deterministic noise the generator runs
/// on. Bit-identical to the app's `rocket::hash01`, so `generate` reproduces
/// exactly the building layout the prototype's hand-rolled `city()` drew (the
/// near buildings and the far footprints therefore line up).
#[inline]
pub fn hash01(i: i32, j: i32) -> f32 {
    let mut h = (i as u32)
        .wrapping_mul(0x1657_4d2b)
        .wrapping_add((j as u32).wrapping_mul(0x9e37_79b1))
        .wrapping_add(0x85eb_ca6b);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    (h & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}

/// A city's "world address": where it is on the planet and the seed its layout is
/// generated from. POD so an array of these is the on-disk city index (cities.bin).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CityDesc {
    /// Stable id (index into the world's city list); also the cache key.
    pub id: u32,
    /// 1 = major (coastal metro), 0 = minor (inland town).
    pub kind: u32,
    /// Longitude, radians in [-pi, pi].
    pub lon: f32,
    /// Latitude, radians in [-pi/2, pi/2].
    pub lat: f32,
    /// Population (people).
    pub pop: f32,
    /// Layout seed.
    pub seed: u32,
    /// Built-up footprint radius (metres) - the LOD / load query radius.
    pub radius_m: f32,
    pub _pad: u32,
}

impl CityDesc {
    /// Unit direction of the city on the sphere (planet-centred world frame).
    pub fn dir(&self) -> DVec3 {
        let (lat, lon) = (self.lat as f64, self.lon as f64);
        DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
    }
}

/// One building in a city's local tangent frame (x east, z north, metres), with
/// its footprint, height and look. POD so the building list serialises directly.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Building {
    pub cx: f32,
    pub cz: f32,
    /// Footprint half-extents (metres).
    pub fw: f32,
    pub fd: f32,
    pub height: f32,
    /// Facade palette index.
    pub pal: u32,
    /// 1 if the building has lit windows at night.
    pub lit: u32,
    /// 1 = warm window light, 0 = cool.
    pub warm: u32,
}

/// One street segment in the city's local frame (x east, z north, metres). The
/// road network is stored explicitly (not derived from a grid) so a city can be
/// laid out organically - curving rings and radiating avenues - instead of a
/// rigid Manhattan grid. POD so the segment list serialises directly.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RoadSeg {
    pub ax: f32,
    pub az: f32,
    pub bx: f32,
    pub bz: f32,
    /// Full road width (m).
    pub width: f32,
    pub _pad: f32,
}

impl RoadSeg {
    pub fn a(&self) -> Vec2 {
        Vec2::new(self.ax, self.az)
    }
    pub fn b(&self) -> Vec2 {
        Vec2::new(self.bx, self.bz)
    }
}

/// A generated city, laid out organically as a radial-concentric plan: avenues
/// radiate from the centre and are crossed by (irregular, jittered) ring roads,
/// so the streets curve and the blocks are wedge-shaped rather than a square
/// grid. Both the road network and the buildings are stored explicitly, and
/// every level of detail (near massing, far footprints, orbital glow, kerbside
/// lamps) is derived from this one layout. All coordinates are local metres,
/// centred on the city origin (placed in the world via the city's `dir`).
#[derive(Clone, Debug, PartialEq)]
pub struct CityLayout {
    /// Layout seed.
    pub seed: u32,
    /// Built-up radius (m): the LOD / paving / glow extent.
    pub radius: f32,
    pub roads: Vec<RoadSeg>,
    pub buildings: Vec<Building>,
}

impl CityLayout {
    /// Half-extent of the built-up area (m) - the city is roughly a disc, so the
    /// same value spans both axes.
    pub fn half(&self) -> f32 {
        self.radius
    }
    /// Street intersection points (local x,z) - the endpoints of the road
    /// network, deduplicated, for kerbside lamps.
    pub fn intersections(&self) -> Vec<Vec2> {
        let mut v: Vec<Vec2> = Vec::new();
        for r in &self.roads {
            for p in [r.a(), r.b()] {
                if !v.iter().any(|q| (*q - p).length_squared() < 1.0) {
                    v.push(p);
                }
            }
        }
        v
    }
}

/// Facade palette shared by every city (so the look is consistent everywhere).
pub const PALETTE: [[f32; 3]; 6] = [
    [0.58, 0.58, 0.61],
    [0.63, 0.58, 0.49],
    [0.42, 0.53, 0.63],
    [0.70, 0.71, 0.74],
    [0.50, 0.52, 0.57],
    [0.66, 0.62, 0.55],
];

/// Deterministically generate a city's layout from its descriptor. Bigger
/// populations get more rings (a larger city); everything else is hashed from
/// `desc.seed`, so the same descriptor always yields exactly the same city (the
/// property the cache relies on).
///
/// The plan is radial-concentric: `spokes` avenues radiate from the centre,
/// crossed by `rings` concentric ring roads. Every ring node is pushed off its
/// ideal circle by hashed jitter (the rings buckle, the avenues wander), so the
/// streets curve and the blocks between them are irregular wedges - an organic
/// town, not a grid. Outer blocks are dropped by a noisy mask for a ragged edge.
pub fn generate(desc: &CityDesc) -> CityLayout {
    let seed = desc.seed as i32;
    let h = |a: i32, b: i32| hash01(a.wrapping_add(seed), b.wrapping_sub(seed));
    let jit = |a: i32, b: i32| h(a, b) * 2.0 - 1.0; // [-1, 1]

    let pm = (desc.pop / 1.0e6).max(0.1);
    // 3..7 rings out from the centre, 7..11 radiating avenues.
    let rings = (3.0 + 1.7 * pm.sqrt()).clamp(3.0, 7.0) as i32;
    let spokes = (7.0 + (4.0 * h(11, 23)).floor()).clamp(7.0, 11.0) as i32;
    let ring0 = 36.0f32; // radius of the innermost ring (a central plaza)
    let ring_gap = 30.0f32; // nominal radial spacing between rings
    let street = 13.0f32;

    // node[k][s]: the intersection of ring k and avenue s, jittered off its
    // ideal circle so the network buckles organically. s wraps modulo `spokes`.
    let node = |k: i32, s: i32| -> Vec2 {
        let sm = ((s % spokes) + spokes) % spokes;
        let ang = (sm as f32 / spokes as f32) * std::f32::consts::TAU
            + 0.20 * jit(sm * 13 + 7, 31) // each avenue leans a fixed amount
            + 0.17 * jit(k * 17 + 3, sm * 5 + 9); // and wanders ring to ring
        // each ring buckles in and out so it is never a clean circle/polygon
        let rad = ring0
            + k as f32 * ring_gap * (0.82 + 0.30 * h(sm * 11 + 5, 17))
            + ring_gap * 0.55 * jit(k * 23 + 1, sm * 7 + 2);
        Vec2::new(ang.cos() * rad, ang.sin() * rad)
    };

    // A block sits between rings k..k+1 and avenues s..s+1. Drop outer blocks by
    // a noisy mask so the built-up area frays out instead of ending on a clean
    // circle. The centre is always developed.
    let developed = |k: i32, s: i32| -> bool {
        if k < 0 || k >= rings {
            return false;
        }
        let edge = k as f32 / rings as f32; // 0 centre .. ~1 rim
        h(k * 9 + s * 3 + 5, k * 5 + s * 7 + 1) > edge * 0.85 - 0.12
    };

    let mut roads: Vec<RoadSeg> = Vec::new();
    let mut push = |a: Vec2, b: Vec2, w: f32| {
        roads.push(RoadSeg { ax: a.x, az: a.y, bx: b.x, bz: b.y, width: w, _pad: 0.0 });
    };
    // Ring roads (the arcs) and radial avenues, kept only where they border a
    // developed block, so the network follows the ragged outline.
    for k in 0..=rings {
        for s in 0..spokes {
            // ring arc between avenue s and s+1, on ring k
            if developed(k, s) || developed(k - 1, s) {
                push(node(k, s), node(k, s + 1), street);
            }
            // radial avenue between ring k and k+1, on avenue s
            if k < rings && (developed(k, s) || developed(k, s - 1)) {
                // avenues are a touch wider than the ring streets
                push(node(k, s), node(k + 1, s), street * 1.15);
            }
        }
    }

    // Buildings: fill each developed block with a few footprints, placed by
    // bilinear interpolation of the block's four (warped) corners so they sit
    // inside the wedge. Taller toward the centre for a downtown massing.
    let mut buildings = Vec::new();
    for k in 0..rings {
        for s in 0..spokes {
            if !developed(k, s) {
                continue;
            }
            let p00 = node(k, s);
            let p01 = node(k, s + 1);
            let p10 = node(k + 1, s);
            let p11 = node(k + 1, s + 1);
            let bil = |u: f32, v: f32| -> Vec2 {
                let top = p00.lerp(p01, u);
                let bot = p10.lerp(p11, u);
                top.lerp(bot, v)
            };
            // block scale, to size + count the buildings that fit
            let du = (p01 - p00).length().min((p11 - p10).length());
            let dv = (p10 - p00).length().min((p11 - p01).length());
            let central = (1.0 - k as f32 / rings as f32).clamp(0.0, 1.0);
            // up to 2x2 buildings, fewer out at the rim
            let nu = if du > 46.0 { 2 } else { 1 };
            let nv = if dv > 46.0 { 2 } else { 1 };
            for iu in 0..nu {
                for iv in 0..nv {
                    let hk = h(k * 41 + s * 7 + iu * 3 + 1, s * 29 + k * 11 + iv * 5 + 2);
                    if hk < 0.18 {
                        continue; // a few empty lots
                    }
                    let u = (iu as f32 + 0.5) / nu as f32 + 0.12 * jit(k * 7 + iu, s * 9 + iv);
                    let v = (iv as f32 + 0.5) / nv as f32 + 0.12 * jit(s * 7 + iv, k * 9 + iu);
                    let c = bil(u.clamp(0.18, 0.82), v.clamp(0.18, 0.82));
                    let h1 = h(k * 31 + iu, s * 17 + iv + 7);
                    let h2 = h(k * 13 + 5, s * 29 + iu + iv);
                    let fw = (du / nu as f32) * 0.30 * (0.7 + 0.3 * h1);
                    let fd = (dv / nv as f32) * 0.30 * (0.7 + 0.3 * h2);
                    let height = 8.0 + central * (16.0 + 60.0 * h1) + (1.0 - central) * 10.0 * h2;
                    let pal =
                        ((h(k + iu, s + iv) * PALETTE.len() as f32) as u32).min(PALETTE.len() as u32 - 1);
                    let lit = (h(k * 3 + iu, s * 9 + iv + 1) > 0.32) as u32;
                    let warm = (h(k * 7 + iu, s * 5 + iv + 3) > 0.5) as u32;
                    buildings.push(Building {
                        cx: c.x,
                        cz: c.y,
                        fw: fw.max(3.0),
                        fd: fd.max(3.0),
                        height,
                        pal,
                        lit,
                        warm,
                    });
                }
            }
        }
    }

    let radius = ring0 + rings as f32 * ring_gap * 1.45 + 20.0;
    CityLayout { seed: desc.seed, radius, roads, buildings }
}

// --- POD (de)serialisation of one layout: a small header + the building array.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayoutHeader {
    magic: u32,
    version: u32,
    seed: u32,
    radius: f32,
    n_roads: u32,
    n_buildings: u32,
}

const LAYOUT_MAGIC: u32 = 0x4C54_4943; // "CITL"
const LAYOUT_VERSION: u32 = 4;

/// Serialise a layout to bytes (header + roads + buildings).
pub fn layout_to_bytes(l: &CityLayout) -> Vec<u8> {
    let hdr = LayoutHeader {
        magic: LAYOUT_MAGIC,
        version: LAYOUT_VERSION,
        seed: l.seed,
        radius: l.radius,
        n_roads: l.roads.len() as u32,
        n_buildings: l.buildings.len() as u32,
    };
    let mut out = bytemuck::bytes_of(&hdr).to_vec();
    out.extend_from_slice(bytemuck::cast_slice(&l.roads));
    out.extend_from_slice(bytemuck::cast_slice(&l.buildings));
    out
}

/// Parse a layout from bytes, or `None` if the header is wrong / truncated. Reads
/// each record unaligned, so it works on a borrowed `include_bytes!` blob (which
/// has no alignment guarantee) as well as a file buffer.
pub fn layout_from_bytes(bytes: &[u8]) -> Option<CityLayout> {
    let hsz = std::mem::size_of::<LayoutHeader>();
    if bytes.len() < hsz {
        return None;
    }
    let hdr: LayoutHeader = bytemuck::pod_read_unaligned(&bytes[..hsz]);
    if hdr.magic != LAYOUT_MAGIC || hdr.version != LAYOUT_VERSION {
        return None;
    }
    let rsz = std::mem::size_of::<RoadSeg>();
    let bsz = std::mem::size_of::<Building>();
    let need = hdr.n_roads as usize * rsz + hdr.n_buildings as usize * bsz;
    if bytes.len() < hsz + need {
        return None;
    }
    let roads: Vec<RoadSeg> = (0..hdr.n_roads as usize)
        .map(|i| bytemuck::pod_read_unaligned(&bytes[hsz + i * rsz..hsz + (i + 1) * rsz]))
        .collect();
    let boff = hsz + hdr.n_roads as usize * rsz;
    let buildings: Vec<Building> = (0..hdr.n_buildings as usize)
        .map(|i| bytemuck::pod_read_unaligned(&bytes[boff + i * bsz..boff + (i + 1) * bsz]))
        .collect();
    Some(CityLayout { seed: hdr.seed, radius: hdr.radius, roads, buildings })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(id: u32, pop: f32) -> CityDesc {
        CityDesc { id, kind: 1, lon: 0.4, lat: -0.1, pop, seed: 1009 * id + 7, radius_m: 600.0, _pad: 0 }
    }

    #[test]
    fn generation_is_deterministic() {
        let d = desc(3, 8.0e6);
        let a = generate(&d);
        let b = generate(&d);
        assert_eq!(a, b, "same descriptor must yield the same layout");
        assert!(!a.buildings.is_empty());
    }

    #[test]
    fn population_scales_grid() {
        let small = generate(&desc(1, 0.4e6));
        let big = generate(&desc(2, 18.0e6));
        assert!(big.radius >= small.radius);
        assert!(big.buildings.len() > small.buildings.len());
        assert!(!small.roads.is_empty() && !big.roads.is_empty());
    }

    #[test]
    fn layout_roundtrips_through_bytes() {
        let l = generate(&desc(5, 6.0e6));
        let bytes = layout_to_bytes(&l);
        let back = layout_from_bytes(&bytes).expect("parse");
        assert_eq!(l, back);
    }

    #[test]
    fn bad_bytes_rejected() {
        assert!(layout_from_bytes(&[0u8; 4]).is_none());
        assert!(layout_from_bytes(&[9u8; 64]).is_none());
    }
}

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
/// populations get a larger grid; everything else is hashed from `desc.seed`, so
/// the same descriptor always yields exactly the same city (the property the
/// cache relies on).
///
/// The plan is an *irregular grid*: orthogonal streets like a real city grid,
/// but with non-uniform block sizes (so the roads are different lengths), a mild
/// per-node warp (so the streets wobble instead of being mechanically straight),
/// and a ragged round-ish outline (so the city is not a filled square). That
/// reads as an organic town while keeping the legible grid structure, rather
/// than the random-looking radial plan it replaces.
pub fn generate(desc: &CityDesc) -> CityLayout {
    let seed = desc.seed as i32;
    let h = |a: i32, b: i32| hash01(a.wrapping_add(seed), b.wrapping_sub(seed));
    let jit = |a: i32, b: i32| h(a, b) * 2.0 - 1.0; // [-1, 1]

    let pm = (desc.pop / 1.0e6).max(0.1);
    // 5..10 blocks a side, scaling with sqrt(population).
    let cols = (5.0 + 1.8 * pm.sqrt()).clamp(5.0, 10.0) as i32;
    let rows = (5.0 + 1.8 * pm.sqrt()).clamp(5.0, 10.0) as i32;
    let street = 13.0f32;
    let base = 42.0f32; // nominal block size

    // Non-uniform grid-line positions: each block's width/depth is hashed, so
    // adjacent streets sit at varying distances and the road segments between
    // intersections come out different lengths - the organic part of the grid.
    let mut xs = vec![0.0f32; (cols + 1) as usize];
    for i in 1..=cols as usize {
        let w = base * (0.55 + 0.9 * h(i as i32 * 7 + 3, 11));
        xs[i] = xs[i - 1] + w + street;
    }
    let mut zs = vec![0.0f32; (rows + 1) as usize];
    for j in 1..=rows as usize {
        let d = base * (0.55 + 0.9 * h(j as i32 * 7 + 5, 23));
        zs[j] = zs[j - 1] + d + street;
    }
    let cx = xs[cols as usize] * 0.5; // centre the grid on the origin
    let cz = zs[rows as usize] * 0.5;

    // Grid node, with a small hashed warp so the streets are not mechanically
    // perfect (kept well under a block so they stay clearly orthogonal).
    let warp = base * 0.10;
    let node = |i: i32, j: i32| -> Vec2 {
        let (ic, jc) = (i.clamp(0, cols) as usize, j.clamp(0, rows) as usize);
        Vec2::new(
            xs[ic] - cx + warp * jit(i * 13 + 1, j * 7 + 2),
            zs[jc] - cz + warp * jit(i * 5 + 9, j * 17 + 4),
        )
    };

    let (nc, nr) = (cols as f32, rows as f32);
    // Whether block (i, j) is built up: a ragged round-ish mask drops the corners
    // and frays the edge, so the city is an organic blob, not a filled square.
    let developed = |i: i32, j: i32| -> bool {
        if i < 0 || i >= cols || j < 0 || j >= rows {
            return false;
        }
        let dx = (i as f32 + 0.5 - nc * 0.5) / (nc * 0.5);
        let dz = (j as f32 + 0.5 - nr * 0.5) / (nr * 0.5);
        (dx * dx + dz * dz).sqrt() <= 0.94 + 0.46 * h(i * 7 + 91, j * 7 + 43)
    };

    // Roads: the four boundary edges of every developed block, so the street grid
    // follows the ragged outline (shared interior edges simply overlap).
    let mut roads: Vec<RoadSeg> = Vec::new();
    let mut push = |a: Vec2, b: Vec2, w: f32| {
        roads.push(RoadSeg { ax: a.x, az: a.y, bx: b.x, bz: b.y, width: w, _pad: 0.0 });
    };
    for i in 0..cols {
        for j in 0..rows {
            if !developed(i, j) {
                continue;
            }
            let (p00, p10) = (node(i, j), node(i + 1, j));
            let (p01, p11) = (node(i, j + 1), node(i + 1, j + 1));
            push(p00, p10, street); // south edge
            push(p01, p11, street); // north edge
            push(p00, p01, street); // west edge
            push(p10, p11, street); // east edge
        }
    }

    // Buildings: fill each developed block, taller toward the centre. Sub-divided
    // into up to 2x2 footprints, placed by bilinear interpolation of the block's
    // four (warped) corners so they sit inside the block.
    let mut buildings = Vec::new();
    for i in 0..cols {
        for j in 0..rows {
            if !developed(i, j) {
                continue;
            }
            let (p00, p10) = (node(i, j), node(i + 1, j));
            let (p01, p11) = (node(i, j + 1), node(i + 1, j + 1));
            let bil = |u: f32, v: f32| -> Vec2 {
                let lo = p00.lerp(p10, u);
                let hi = p01.lerp(p11, u);
                lo.lerp(hi, v)
            };
            let bw = (p10 - p00).length(); // block width (x)
            let bd = (p01 - p00).length(); // block depth (z)
            let dx = (i as f32 + 0.5 - nc * 0.5) / (nc * 0.5);
            let dz = (j as f32 + 0.5 - nr * 0.5) / (nr * 0.5);
            let central = (1.0 - (dx * dx + dz * dz).sqrt()).clamp(0.0, 1.0);
            let nu = if bw > 40.0 { 2 } else { 1 };
            let nv = if bd > 40.0 { 2 } else { 1 };
            for iu in 0..nu {
                for iv in 0..nv {
                    if h(i * 41 + iu * 3 + 1, j * 29 + iv * 5 + 2) < 0.12 {
                        continue; // a few empty lots
                    }
                    let u = (iu as f32 + 0.5) / nu as f32;
                    let v = (iv as f32 + 0.5) / nv as f32;
                    let c = bil(u, v);
                    let h1 = h(i * 31 + iu, j * 17 + iv + 7);
                    let h2 = h(i * 13 + 5, j * 29 + iu + iv);
                    let fw = (bw / nu as f32) * 0.34 * (0.7 + 0.3 * h1);
                    let fd = (bd / nv as f32) * 0.34 * (0.7 + 0.3 * h2);
                    let height = 8.0 + central * (16.0 + 60.0 * h1) + (1.0 - central) * 10.0 * h2;
                    let pal =
                        ((h(i + iu, j + iv) * PALETTE.len() as f32) as u32).min(PALETTE.len() as u32 - 1);
                    let lit = (h(i * 3 + iu, j * 9 + iv + 1) > 0.32) as u32;
                    let warm = (h(i * 7 + iu, j * 5 + iv + 3) > 0.5) as u32;
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

    let radius = cx.hypot(cz) + street; // disc covers the whole grid, corners too
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
const LAYOUT_VERSION: u32 = 5;

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

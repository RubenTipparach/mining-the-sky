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

/// A generated city. The street grid (and thus road/block footprints and the
/// kerbside lamp positions) is regular and derived from `cols/rows/block/street`;
/// only the buildings need explicit storage. All coordinates are local metres,
/// centred on the city origin (place it in the world via the city's `dir`).
#[derive(Clone, Debug, PartialEq)]
pub struct CityLayout {
    pub cols: u32,
    pub rows: u32,
    /// Block footprint (m) and street width (m).
    pub block: f32,
    pub street: f32,
    /// Layout seed (so the far-footprint builder can re-derive the organic
    /// built-up mask and match the buildings).
    pub seed: u32,
    pub buildings: Vec<Building>,
}

/// Whether a grid cell is built up. Cities are NOT perfect squares: cells are
/// dropped toward the edges by a noisy radial mask, so the built-up area has an
/// organic, ragged outline with the corners cut. Shared by `generate` and the
/// far-footprint builder, so buildings and footprints always agree.
pub fn cell_developed(seed: i32, cols: u32, rows: u32, ix: i32, iz: i32) -> bool {
    let h = |a: i32, b: i32| hash01(a.wrapping_add(seed), b.wrapping_sub(seed));
    let dx = (ix as f32 - (cols as f32 - 1.0) * 0.5) / (cols as f32 * 0.5);
    let dz = (iz as f32 - (rows as f32 - 1.0) * 0.5) / (rows as f32 * 0.5);
    let r = (dx * dx + dz * dz).sqrt();
    r <= 0.82 + 0.62 * h(ix * 7 + 91, iz * 7 + 43)
}

impl CityLayout {
    /// Block-to-block spacing (m).
    pub fn span(&self) -> f32 {
        self.block + self.street
    }
    /// Half-extent of the built-up area along x (m).
    pub fn half_x(&self) -> f32 {
        self.cols as f32 * self.span() * 0.5
    }
    /// Half-extent of the built-up area along z (m).
    pub fn half_z(&self) -> f32 {
        self.rows as f32 * self.span() * 0.5
    }
    /// Street-grid intersection points (local x,z), for kerbside lamps / road
    /// footprints - derived, not stored.
    pub fn intersections(&self) -> Vec<Vec2> {
        let (span, hx, hz) = (self.span(), self.half_x(), self.half_z());
        let mut v = Vec::new();
        for ix in 0..=self.cols {
            for iz in 0..=self.rows {
                v.push(Vec2::new(-hx + ix as f32 * span, -hz + iz as f32 * span));
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
/// populations get larger grids; everything else is hashed from `desc.seed`, so
/// the same descriptor always yields exactly the same city (the property the
/// cache relies on).
pub fn generate(desc: &CityDesc) -> CityLayout {
    let seed = desc.seed as i32;
    let h = |a: i32, b: i32| hash01(a.wrapping_add(seed), b.wrapping_sub(seed));

    let pm = (desc.pop / 1.0e6).max(0.1);
    // 4..11 blocks a side, scaling with sqrt(population).
    let n = (4.0 + 2.4 * pm.sqrt()).clamp(4.0, 11.0) as u32;
    let (cols, rows) = (n, n);
    let block = 46.0f32;
    let street = 14.0f32;
    let span = block + street;
    let half_x = cols as f32 * span * 0.5;
    let half_z = rows as f32 * span * 0.5;

    let mut buildings = Vec::new();
    for ix in 0..cols as i32 {
        for iz in 0..rows as i32 {
            let bx0 = -half_x + (ix as f32 + 0.5) * span;
            let bz0 = -half_z + (iz as f32 + 0.5) * span;
            let dx = (ix as f32 - (cols as f32 - 1.0) * 0.5) / (cols as f32 * 0.5);
            let dz = (iz as f32 - (rows as f32 - 1.0) * 0.5) / (rows as f32 * 0.5);
            // organic outline: skip cells dropped by the built-up mask, so the
            // city is a ragged blob rather than a perfect square.
            if !cell_developed(seed, cols, rows, ix, iz) {
                continue;
            }
            let central = (1.0 - (dx * dx + dz * dz).sqrt()).clamp(0.0, 1.0);
            let count = 1 + (h(ix, iz) * 3.99) as i32;
            for k in 0..count {
                let sx = if k % 2 == 0 { -1.0 } else { 1.0 };
                let sz = if k < 2 { -1.0 } else { 1.0 };
                let h1 = h(ix * 31 + k, iz * 17 + 7);
                let h2 = h(ix * 13 + 5, iz * 29 + k);
                let fw = block * 0.21 * (0.7 + 0.3 * h1);
                let fd = block * 0.21 * (0.7 + 0.3 * h2);
                let height = 8.0 + central * (16.0 + 60.0 * h1) + (1.0 - central) * 10.0 * h2;
                let pal = ((h(ix + k, iz) * PALETTE.len() as f32) as u32).min(PALETTE.len() as u32 - 1);
                let cx = bx0 + sx * block * 0.25;
                let cz = bz0 + sz * block * 0.25;
                let lit = (h(ix * 3 + k, iz * 9 + 1) > 0.32) as u32;
                let warm = (h(ix * 7 + k, iz * 5 + 3) > 0.5) as u32;
                buildings.push(Building { cx, cz, fw, fd, height, pal, lit, warm });
            }
        }
    }
    CityLayout { cols, rows, block, street, seed: desc.seed, buildings }
}

// --- POD (de)serialisation of one layout: a small header + the building array.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayoutHeader {
    magic: u32,
    version: u32,
    cols: u32,
    rows: u32,
    block: f32,
    street: f32,
    n_buildings: u32,
    seed: u32,
}

const LAYOUT_MAGIC: u32 = 0x4C54_4943; // "CITL"
const LAYOUT_VERSION: u32 = 3;

/// Serialise a layout to bytes (header + buildings).
pub fn layout_to_bytes(l: &CityLayout) -> Vec<u8> {
    let hdr = LayoutHeader {
        magic: LAYOUT_MAGIC,
        version: LAYOUT_VERSION,
        cols: l.cols,
        rows: l.rows,
        block: l.block,
        street: l.street,
        n_buildings: l.buildings.len() as u32,
        seed: l.seed,
    };
    let mut out = bytemuck::bytes_of(&hdr).to_vec();
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
    let bsz = std::mem::size_of::<Building>();
    let need = hdr.n_buildings as usize * bsz;
    if bytes.len() < hsz + need {
        return None;
    }
    let buildings: Vec<Building> = (0..hdr.n_buildings as usize)
        .map(|i| bytemuck::pod_read_unaligned(&bytes[hsz + i * bsz..hsz + (i + 1) * bsz]))
        .collect();
    Some(CityLayout { cols: hdr.cols, rows: hdr.rows, block: hdr.block, street: hdr.street, seed: hdr.seed, buildings })
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
        assert!(big.cols >= small.cols);
        assert!(big.buildings.len() > small.buildings.len());
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

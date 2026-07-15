//! Spatial index over the world's cities (the baked `cities.bin` asset).
//!
//! Maps a world address - a direction / lon-lat on the planet - to the city
//! descriptors near it, so the renderer can ask "which cities are within N km of
//! the camera?" and then pull their layouts from the [`CityCache`](crate::CityCache).

use crate::CityDesc;
use glam::DVec3;

const INDEX_MAGIC: u32 = 0x4954_4943; // "CITI"

/// The world's city list, loaded from `cities.bin`.
pub struct CityIndex {
    cities: Vec<CityDesc>,
}

impl CityIndex {
    pub fn from_descs(cities: Vec<CityDesc>) -> Self {
        Self { cities }
    }

    pub fn cities(&self) -> &[CityDesc] {
        &self.cities
    }

    pub fn len(&self) -> usize {
        self.cities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cities.is_empty()
    }

    /// Serialise the index: `[magic][count][CityDesc; count]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.cities.len() * std::mem::size_of::<CityDesc>());
        out.extend_from_slice(&INDEX_MAGIC.to_le_bytes());
        out.extend_from_slice(&(self.cities.len() as u32).to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice(&self.cities));
        out
    }

    /// Parse an index from bytes (the `cities.bin` asset), or `None` if invalid.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        if magic != INDEX_MAGIC {
            return None;
        }
        let count = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
        let sz = std::mem::size_of::<CityDesc>();
        if bytes.len() < 8 + count * sz {
            return None;
        }
        // Read unaligned so a borrowed include_bytes! blob (alignment 1) is fine.
        let cities: Vec<CityDesc> = (0..count)
            .map(|i| bytemuck::pod_read_unaligned(&bytes[8 + i * sz..8 + (i + 1) * sz]))
            .collect();
        Some(Self { cities })
    }

    /// Cities whose centre is within `radius_m` great-circle metres of unit
    /// direction `dir`, nearest first. `planet_radius_m` converts the angular
    /// separation to metres.
    ///
    /// Brute force over the list (fine for the current city counts); the layout
    /// is a flat POD array, so swapping in a lon/lat bucket grid later is a local
    /// change behind this method.
    pub fn near(&self, dir: DVec3, radius_m: f64, planet_radius_m: f64) -> Vec<CityDesc> {
        let dir = dir.normalize();
        let max_ang = (radius_m / planet_radius_m).min(std::f64::consts::PI);
        let cos_max = max_ang.cos();
        let mut hits: Vec<(f64, CityDesc)> = self
            .cities
            .iter()
            .filter_map(|c| {
                let d = c.dir().dot(dir); // larger = nearer
                if d >= cos_max {
                    Some((d, *c))
                } else {
                    None
                }
            })
            .collect();
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        hits.into_iter().map(|(_, c)| c).collect()
    }

    /// The single nearest city to a direction, or `None` if the index is empty.
    pub fn nearest(&self, dir: DVec3) -> Option<CityDesc> {
        let dir = dir.normalize();
        self.cities
            .iter()
            .copied()
            .max_by(|a, b| a.dir().dot(dir).partial_cmp(&b.dir().dot(dir)).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(id: u32, lon: f32, lat: f32) -> CityDesc {
        CityDesc { id, kind: 0, lon, lat, pop: 1.0e6, seed: id, radius_m: 400.0, _pad: 0 }
    }

    #[test]
    fn index_roundtrips_and_queries() {
        let cities = vec![mk(0, 0.0, 0.0), mk(1, 0.02, 0.0), mk(2, 1.5, 0.5)];
        let idx = CityIndex::from_descs(cities);
        let bytes = idx.to_bytes();
        let back = CityIndex::from_bytes(&bytes).expect("parse");
        assert_eq!(back.len(), 3);

        let r = 6.2e6f64; // planet radius
        let at = DVec3::new(1.0, 0.0, 0.0); // lon 0, lat 0
        let near = back.near(at, 200_000.0, r); // 200 km
        assert_eq!(near.len(), 2, "two cities within 200 km, the far one excluded");
        assert_eq!(near[0].id, 0, "nearest first");
        assert_eq!(back.nearest(at).unwrap().id, 0);
    }

    #[test]
    fn bad_index_bytes_rejected() {
        assert!(CityIndex::from_bytes(&[0u8; 4]).is_none());
        assert!(CityIndex::from_bytes(&[1u8; 16]).is_none());
    }
}

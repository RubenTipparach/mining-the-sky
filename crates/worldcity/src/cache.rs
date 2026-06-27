//! On-disk content cache for generated city layouts.
//!
//! A layout is generated once and stored as `<id>.clb`; later loads read it back.
//! Native only in practice (wasm has no filesystem) - on wasm the reads/writes
//! simply fail and `get` falls back to generating in-memory, which is correct,
//! just uncached.

use crate::{generate, layout_from_bytes, layout_to_bytes, CityDesc, CityLayout};
use std::path::PathBuf;

/// A directory of cached city layouts, keyed by city id.
pub struct CityCache {
    dir: PathBuf,
}

impl CityCache {
    /// Open (creating if needed) a cache directory.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        let _ = std::fs::create_dir_all(&dir);
        Self { dir }
    }

    fn path(&self, id: u32) -> PathBuf {
        self.dir.join(format!("{id:08x}.clb"))
    }

    /// True if a (valid-looking) cached layout already exists for this id.
    pub fn is_cached(&self, id: u32) -> bool {
        self.path(id).exists()
    }

    /// The layout for a city: load it from the cache if present, otherwise
    /// generate it, store it, and return it.
    pub fn get(&self, desc: &CityDesc) -> CityLayout {
        if let Ok(bytes) = std::fs::read(self.path(desc.id)) {
            if let Some(l) = layout_from_bytes(&bytes) {
                return l;
            }
        }
        let l = generate(desc);
        let _ = std::fs::write(self.path(desc.id), layout_to_bytes(&l));
        l
    }

    /// Generate + store any descriptors not already cached (world-load warmup).
    /// Returns how many were freshly generated.
    pub fn warm(&self, descs: &[CityDesc]) -> usize {
        let mut n = 0;
        for d in descs {
            if !self.is_cached(d.id) {
                let _ = std::fs::write(self.path(d.id), layout_to_bytes(&generate(d)));
                n += 1;
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CityDesc;

    fn tmpdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        // A per-process-ish unique dir so parallel test runs don't collide.
        p.push(format!("worldcity_test_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn desc(id: u32) -> CityDesc {
        CityDesc { id, kind: 1, lon: 0.2, lat: 0.1, pop: 5.0e6, seed: 17 * id + 3, radius_m: 500.0, _pad: 0 }
    }

    #[test]
    fn cache_generates_then_reloads() {
        let dir = tmpdir("rw");
        let cache = CityCache::new(&dir);
        let d = desc(42);
        assert!(!cache.is_cached(d.id));
        let first = cache.get(&d); // generates + stores
        assert!(cache.is_cached(d.id));
        let second = cache.get(&d); // reads from disk
        assert_eq!(first, second);
        assert_eq!(first, generate(&d), "cached layout matches a fresh generate");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn warm_only_generates_missing() {
        let dir = tmpdir("warm");
        let cache = CityCache::new(&dir);
        let descs: Vec<CityDesc> = (0..5).map(desc).collect();
        assert_eq!(cache.warm(&descs), 5, "first warm generates all");
        assert_eq!(cache.warm(&descs), 0, "second warm is a no-op");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

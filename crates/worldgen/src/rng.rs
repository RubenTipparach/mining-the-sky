//! Tiny deterministic RNG wrapper plus a procedural place-name generator.

use rand::{Rng as _, SeedableRng};
use rand_pcg::Pcg64Mcg;

pub struct Rng(Pcg64Mcg);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(Pcg64Mcg::seed_from_u64(seed))
    }
    pub fn unit(&mut self) -> f64 {
        self.0.gen::<f64>()
    }
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.0.gen::<f64>()
    }
    pub fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.0.gen::<f64>() * n as f64) as usize % n
        }
    }
}

const ONSET: [&str; 24] = [
    "Ka", "Or", "Ve", "Ta", "Sol", "Mar", "Lun", "Cir", "No", "Hel", "Ar", "By",
    "Cra", "Del", "Fen", "Gly", "Hesp", "Ith", "Jor", "Kep", "Mira", "Nyx", "Ost", "Pyr",
];
const CODA: [&str; 20] = [
    "ton", "vik", "mar", "dis", "polis", "gard", "haven", "reach", "fall", "mont",
    "shoal", "delta", "port", "stad", "ridge", "burg", "mouth", "crest", "hollow", "bay",
];

pub fn city_name(rng: &mut Rng) -> String {
    let a = ONSET[rng.index(ONSET.len())];
    let b = CODA[rng.index(CODA.len())];
    format!("{a}{b}")
}

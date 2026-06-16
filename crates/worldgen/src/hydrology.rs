//! D8 downhill flow + accumulation. This drives where rivers and, crucially,
//! coastal river deltas form -- the prime sites for major cities.

use crate::grid::Grid;

pub struct Hydrology {
    /// log(flow accumulation); higher = bigger river.
    pub flow: Grid<f32>,
    /// 1 where the cell is below sea level.
    pub is_ocean: Grid<u8>,
}

const NEIGH8: [(i64, i64); 8] = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
];

pub fn compute(elev: &Grid<f32>, sea_level: f32) -> Hydrology {
    let w = elev.w;
    let h = elev.h;
    let n = w * h;

    let mut is_ocean = Grid::<u8>::new(w, h);
    for i in 0..n {
        is_ocean.data[i] = (elev.data[i] <= sea_level) as u8;
    }

    // Steepest-descent neighbour for each land cell.
    let mut downhill = vec![-1i64; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if is_ocean.data[i] == 1 {
                continue;
            }
            let mut best = elev.data[i];
            let mut bi = -1i64;
            for (dx, dy) in NEIGH8 {
                let nx = (x as i64 + dx).rem_euclid(w as i64) as usize;
                let ny = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                let ni = ny * w + nx;
                if elev.data[ni] < best {
                    best = elev.data[ni];
                    bi = ni as i64;
                }
            }
            downhill[i] = bi;
        }
    }

    // Accumulate precipitation from high terrain to low by processing land
    // cells in descending elevation order.
    let mut order: Vec<usize> = (0..n).filter(|&i| is_ocean.data[i] == 0).collect();
    order.sort_by(|&a, &b| elev.data[b].partial_cmp(&elev.data[a]).unwrap());
    let mut acc = vec![1.0f32; n];
    for &i in &order {
        let d = downhill[i];
        if d >= 0 {
            acc[d as usize] += acc[i];
        }
    }

    let mut flow = Grid::<f32>::new(w, h);
    for i in 0..n {
        flow.data[i] = acc[i].max(1.0).ln() as f32;
    }

    Hydrology { flow, is_ocean }
}

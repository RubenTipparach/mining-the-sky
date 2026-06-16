//! Inter-city road network: a great-circle-routed MST plus short regional links.

use crate::grid::haversine;
use crate::sites::City;
use glam::DVec3;

pub struct Road {
    /// Unit-sphere points along the route.
    pub pts: Vec<DVec3>,
    pub major: bool,
}

pub fn build_roads(cities: &[City]) -> Vec<Road> {
    let n = cities.len();
    if n < 2 {
        return Vec::new();
    }
    let dist = |a: usize, b: usize| {
        haversine(cities[a].lon, cities[a].lat, cities[b].lon, cities[b].lat)
    };

    // Prim's minimum spanning tree -> a connected backbone (no isolated cities).
    let mut in_tree = vec![false; n];
    let mut best: Vec<(f64, usize)> = (0..n).map(|j| (dist(0, j), 0usize)).collect();
    in_tree[0] = true;
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for _ in 1..n {
        let mut mj = usize::MAX;
        let mut md = f64::INFINITY;
        for j in 0..n {
            if !in_tree[j] && best[j].0 < md {
                md = best[j].0;
                mj = j;
            }
        }
        if mj == usize::MAX {
            break;
        }
        in_tree[mj] = true;
        edges.push((best[mj].1, mj));
        for j in 0..n {
            if !in_tree[j] {
                let d = dist(mj, j);
                if d < best[j].0 {
                    best[j] = (d, mj);
                }
            }
        }
    }

    // Extra short-haul links between nearby cities for a denser regional grid.
    let mut extra: Vec<(usize, usize)> = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            if dist(a, b) < 0.22
                && !edges.iter().any(|&(u, v)| (u == a && v == b) || (u == b && v == a))
            {
                extra.push((a, b));
            }
        }
    }

    let mut roads = Vec::new();
    for (a, b) in edges {
        roads.push(route(cities[a].dir, cities[b].dir, true));
    }
    for (a, b) in extra {
        roads.push(route(cities[a].dir, cities[b].dir, false));
    }
    roads
}

fn route(a: DVec3, b: DVec3, major: bool) -> Road {
    let ang = a.dot(b).clamp(-1.0, 1.0).acos();
    let steps = (ang / 0.008).ceil().max(2.0) as usize;
    let mut pts = Vec::with_capacity(steps + 1);
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        pts.push(slerp(a, b, t));
    }
    Road { pts, major }
}

fn slerp(a: DVec3, b: DVec3, t: f64) -> DVec3 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    let th = dot.acos();
    if th < 1e-6 {
        return a;
    }
    let s = th.sin();
    (((th * (1.0 - t)).sin() / s) * a + ((th * t).sin() / s) * b).normalize()
}

//! Equirectangular grid storage and spherical coordinate helpers.
//!
//! The whole planet surface is stored as an equirectangular (lon/lat) grid.
//! `x` wraps in longitude, `y` clamps at the poles. Geometry is done in f64
//! unit-sphere space; grids store f32 to keep memory modest at 2048x1024+.

use glam::DVec3;
use std::f64::consts::PI;

#[derive(Clone)]
pub struct Grid<T> {
    pub w: usize,
    pub h: usize,
    pub data: Vec<T>,
}

impl<T: Copy + Default> Grid<T> {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![T::default(); w * h] }
    }
}

impl<T: Copy> Grid<T> {
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.data[y * self.w + x]
    }
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: T) {
        self.data[y * self.w + x] = v;
    }
    /// Fetch with longitude wrap and latitude clamp.
    #[inline]
    pub fn get_wrap(&self, x: i64, y: i64) -> T {
        let xx = x.rem_euclid(self.w as i64) as usize;
        let yy = y.clamp(0, self.h as i64 - 1) as usize;
        self.get(xx, yy)
    }
}

impl Grid<f32> {
    /// Bilinear sample by a unit direction (lon wraps, lat clamps).
    pub fn sample_dir(&self, d: DVec3) -> f32 {
        let (lon, lat) = dir_to_lonlat(d);
        let u = (lon + PI) / (2.0 * PI) * self.w as f64 - 0.5;
        let v = (PI * 0.5 - lat) / PI * self.h as f64 - 0.5;
        let x0 = u.floor();
        let y0 = v.floor();
        let fx = (u - x0) as f32;
        let fy = (v - y0) as f32;
        let x0i = x0 as i64;
        let y0i = y0 as i64;
        let a = self.get_wrap(x0i, y0i);
        let b = self.get_wrap(x0i + 1, y0i);
        let c = self.get_wrap(x0i, y0i + 1);
        let e = self.get_wrap(x0i + 1, y0i + 1);
        let top = a + (b - a) * fx;
        let bot = c + (e - c) * fx;
        top + (bot - top) * fy
    }
}

/// Longitude/latitude (radians) to a unit direction with +Y up.
pub fn lonlat_to_dir(lon: f64, lat: f64) -> DVec3 {
    let cl = lat.cos();
    DVec3::new(cl * lon.cos(), lat.sin(), cl * lon.sin())
}

/// Unit direction to (lon, lat) in radians.
pub fn dir_to_lonlat(d: DVec3) -> (f64, f64) {
    let lat = d.y.clamp(-1.0, 1.0).asin();
    let lon = d.z.atan2(d.x);
    (lon, lat)
}

/// Pixel center to (lon, lat) in radians.
pub fn pixel_to_lonlat(x: usize, y: usize, w: usize, h: usize) -> (f64, f64) {
    let lon = ((x as f64 + 0.5) / w as f64) * 2.0 * PI - PI;
    let lat = PI * 0.5 - ((y as f64 + 0.5) / h as f64) * PI;
    (lon, lat)
}

/// (lon, lat) radians to nearest pixel (lon wraps, lat clamps).
pub fn lonlat_to_pixel(lon: f64, lat: f64, w: usize, h: usize) -> (usize, usize) {
    let u = (lon + PI) / (2.0 * PI);
    let v = (PI * 0.5 - lat) / PI;
    let x = ((u * w as f64).floor() as i64).rem_euclid(w as i64) as usize;
    let y = ((v * h as f64).floor() as i64).clamp(0, h as i64 - 1) as usize;
    (x, y)
}

/// Great-circle (angular) distance in radians between two lon/lat points.
pub fn haversine(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat * 0.5).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon * 0.5).sin().powi(2);
    2.0 * a.sqrt().clamp(-1.0, 1.0).asin()
}

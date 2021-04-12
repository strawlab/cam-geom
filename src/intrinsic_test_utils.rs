//! Utilities for testing `cam_geom` implementations.
use super::*;
use nalgebra::{
    base::{dimension::Dynamic, VecStorage},
    convert,
};

pub(crate) fn generate_uv_raw<R: RealField>(
    width: usize,
    height: usize,
    step: usize,
    border: usize,
) -> Pixels<R, Dynamic, VecStorage<R, Dynamic, U2>> {
    let mut uv_raws: Vec<[R; 2]> = Vec::new();
    for row in num_iter::range_step(border, height - border, step) {
        for col in num_iter::range_step(border, width - border, step) {
            uv_raws.push([convert(col as f64), convert(row as f64)]);
        }
    }

    let mut data = nalgebra::OMatrix::<R, Dynamic, U2>::from_element(uv_raws.len(), convert(0.0));
    for i in 0..uv_raws.len() {
        for j in 0..2 {
            data[(i, j)] = uv_raws[i][j];
        }
    }
    Pixels { data }
}

/// Test roundtrip projection from pixels to camera rays for an intrinsic camera model.
///
/// Generate pixel coordinates, project them to rays, convert to points on the
/// rays, convert the points back to pixels, and then compare with the original
/// pixel coordinates.
pub fn roundtrip_intrinsics<R, CAM>(
    cam: &CAM,
    width: usize,
    height: usize,
    step: usize,
    border: usize,
    eps: R,
) where
    R: RealField,
    CAM: IntrinsicParameters<R>,
    CAM::BundleType: Bundle<R>,
{
    let pixels = generate_uv_raw(width, height, step, border);

    let camcoords = cam.pixel_to_camera(&pixels);
    let camera_coords_points = camcoords.point_on_ray();

    // project back to pixel coordinates
    let pixel_actual = cam.camera_to_pixel(&camera_coords_points);
    approx::assert_abs_diff_eq!(pixels.data, pixel_actual.data, epsilon = convert(eps));
}

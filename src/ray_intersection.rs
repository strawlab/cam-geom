#![deny(unsafe_code, missing_docs)]

use nalgebra as na;

use na::{
    allocator::Allocator, base::storage::Owned, DefaultAllocator, Dim, Dyn, Matrix3, OMatrix,
    RealField, SMatrix, Vector3, U1, U3,
};

use itertools::izip;

use crate::{CoordinateSystem, Error, Points, Ray};

/// Iter::Sum is not implemented for R.
macro_rules! sum_as_f64 {
    ($iter:expr,$R:ty) => {{
        $iter
            .map(|x| na::try_convert::<$R, f64>(x.clone()).unwrap())
            .sum()
    }};
}
// just like above but without * dereference.
macro_rules! refsum_as_f64 {
    ($iter:expr,$R:ty) => {{
        $iter.map(|x| na::try_convert::<$R, f64>(x).unwrap()).sum()
    }};
}

/// convert from specialized type (e.g. f64) into generic type $R
macro_rules! despecialize {
    ($a:expr, $R:ty, $rows:expr, $cols:expr) => {{
        SMatrix::<$R, $rows, $cols>::from_iterator(
            $a.as_slice().into_iter().map(|x| na::convert(*x)),
        )
    }};
}

/// A single point. (This is a `Points` instance defined to have only one point).
pub type SinglePoint<Coords, R> = Points<Coords, R, U1, Owned<R, U1, U3>>;

/// Return the 3D point which is the best intersection of rays.
#[allow(non_snake_case)]
pub fn best_intersection_of_rays<Coords, R>(
    rays: &[Ray<Coords, R>],
) -> Result<SinglePoint<Coords, R>, Error>
where
    Coords: CoordinateSystem,
    R: RealField,
    DefaultAllocator: Allocator<R, Dyn, U3>,
    DefaultAllocator: Allocator<R, U1, Dyn>,
    DefaultAllocator: Allocator<R, U1, U3>,
{
    if rays.len() < 2 {
        return Err(Error::MinimumTwoRaysNeeded);
    }

    let npts = Dyn(rays.len());
    let u1 = U1::from_usize(1);
    let u3 = U3::from_usize(3);

    let mut line_dirs = OMatrix::<R, Dyn, U3>::zeros_generic(npts, u3);
    let mut line_points = OMatrix::<R, Dyn, U3>::zeros_generic(npts, u3);

    for i in 0..rays.len() {
        let ray_wc = rays.get(i).unwrap();
        let d = &ray_wc.direction;

        // Normalize the vector length.
        let dir = nalgebra::base::Unit::new_normalize(Vector3::new(
            d.data.0[0][0].clone(),
            d.data.0[1][0].clone(),
            d.data.0[2][0].clone(),
        ));

        line_dirs[(i, 0)] = dir[0].clone();
        line_dirs[(i, 1)] = dir[1].clone();
        line_dirs[(i, 2)] = dir[2].clone();

        line_points[(i, 0)] = ray_wc.center.data.0[0][0].clone();
        line_points[(i, 1)] = ray_wc.center.data.0[1][0].clone();
        line_points[(i, 2)] = ray_wc.center.data.0[2][0].clone();
    }

    let mut xxm1 = OMatrix::<R, U1, Dyn>::zeros_generic(u1, npts);
    let mut yym1 = OMatrix::<R, U1, Dyn>::zeros_generic(u1, npts);
    let mut zzm1 = OMatrix::<R, U1, Dyn>::zeros_generic(u1, npts);
    let mut xy = OMatrix::<R, U1, Dyn>::zeros_generic(u1, npts);
    let mut xz = OMatrix::<R, U1, Dyn>::zeros_generic(u1, npts);
    let mut yz = OMatrix::<R, U1, Dyn>::zeros_generic(u1, npts);

    // TODO element-wise add, mul with nalgebra matrices

    let nx = line_dirs.column(0);
    let ny = line_dirs.column(1);
    let nz = line_dirs.column(2);

    let minusone: R = na::convert(-1.0);

    for (x, xxm1) in nx.into_iter().zip(xxm1.iter_mut()) {
        *xxm1 = x.clone().powi(2).add(minusone.clone());
    }

    for (y, yym1) in ny.into_iter().zip(yym1.iter_mut()) {
        *yym1 = y.clone().powi(2).add(minusone.clone());
    }

    for (z, zzm1) in nz.into_iter().zip(zzm1.iter_mut()) {
        *zzm1 = z.clone().powi(2).add(minusone.clone());
    }

    for (x, y, xy) in izip!(nx.into_iter(), ny.into_iter(), xy.iter_mut()) {
        *xy = x.clone() * y.clone();
    }

    for (x, z, xz) in izip!(nx.into_iter(), nz.into_iter(), xz.iter_mut()) {
        *xz = x.clone() * z.clone();
    }

    for (y, z, yz) in izip!(ny.into_iter(), nz.into_iter(), yz.iter_mut()) {
        *yz = y.clone() * z.clone();
    }

    let SXX: f64 = sum_as_f64!(xxm1.iter(), R);
    let SYY: f64 = sum_as_f64!(yym1.iter(), R);
    let SZZ: f64 = sum_as_f64!(zzm1.iter(), R);
    let SXY: f64 = sum_as_f64!(xy.iter(), R);
    let SXZ: f64 = sum_as_f64!(xz.iter(), R);
    let SYZ: f64 = sum_as_f64!(yz.iter(), R);

    let S = Matrix3::new(SXX, SXY, SXZ, SXY, SYY, SYZ, SXZ, SYZ, SZZ);

    let px = line_points.column(0);
    let py = line_points.column(1);
    let pz = line_points.column(2);

    let xt1 = px
        .iter()
        .zip(xxm1.iter())
        .map(|(a, b)| a.clone() * b.clone());
    let xt2 = py.iter().zip(xy.iter()).map(|(a, b)| a.clone() * b.clone());
    let xt3 = pz.iter().zip(xz.iter()).map(|(a, b)| a.clone() * b.clone());
    let CX: f64 = refsum_as_f64!(izip!(xt1, xt2, xt3).map(|(t1, t2, t3)| t1 + t2 + t3), R);
    let CX: R = na::convert(CX);

    let yt1 = px.iter().zip(xy.iter()).map(|(a, b)| a.clone() * b.clone());
    let yt2 = py
        .iter()
        .zip(yym1.iter())
        .map(|(a, b)| a.clone() * b.clone());
    let yt3 = pz.iter().zip(yz.iter()).map(|(a, b)| a.clone() * b.clone());
    let CY: f64 = refsum_as_f64!(izip!(yt1, yt2, yt3).map(|(t1, t2, t3)| t1 + t2 + t3), R);
    let CY: R = na::convert(CY);

    let zt1 = px.iter().zip(xz.iter()).map(|(a, b)| a.clone() * b.clone());
    let zt2 = py.iter().zip(yz.iter()).map(|(a, b)| a.clone() * b.clone());
    let zt3 = pz
        .iter()
        .zip(zzm1.iter())
        .map(|(a, b)| a.clone() * b.clone());
    let CZ: f64 = refsum_as_f64!(izip!(zt1, zt2, zt3).map(|(t1, t2, t3)| t1 + t2 + t3), R);
    let CZ: R = na::convert(CZ);

    let C = Vector3::new(CX, CY, CZ);

    let s_f64_pinv = my_pinv(&S)?;
    let s_pinv = despecialize!(s_f64_pinv, R, 3, 3);
    let r: Vector3<R> = s_pinv * C;

    let mut result = crate::Points::new(nalgebra::OMatrix::<R, U1, U3>::zeros_generic(u1, u3));
    for j in 0..3 {
        result.data[(0, j)] = r[j].clone();
    }
    Ok(result)
}

fn my_pinv<R: RealField>(m: &SMatrix<R, 3, 3>) -> Result<SMatrix<R, 3, 3>, Error> {
    Ok(
        na::linalg::SVD::try_new(m.clone(), true, true, na::convert(1e-7), 100)
            .ok_or(Error::SvdFailed)?
            .pseudo_inverse(na::convert(1.0e-7))
            .unwrap(),
    )
    // The unwrap() above is safe because, as of nalgebra 0.19, this could only error if the SVD has not been computed.
    // But this is not possible here, so the unwrap() should never result in a panic.
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use crate::{Camera, ExtrinsicParameters, IntrinsicParametersPerspective, PerspectiveParams};

    use super::*;

    #[test]
    fn roundtrip() {
        let i1: IntrinsicParametersPerspective<_> = PerspectiveParams {
            fx: 100.0,
            fy: 102.0,
            skew: 0.1,
            cx: 321.0,
            cy: 239.9,
        }
        .into();
        let e1 = ExtrinsicParameters::from_view(
            &Vector3::new(1.2, 3.4, 5.6),                                // camcenter
            &Vector3::new(2.2, 3.4, 5.6),                                // lookat
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)), // up
        );
        let cam1 = Camera::new(i1, e1);

        let i2: IntrinsicParametersPerspective<_> = PerspectiveParams {
            fx: 200.0,
            fy: 202.0,
            skew: 0.01,
            cx: 321.0,
            cy: 239.9,
        }
        .into();
        let e2 = ExtrinsicParameters::from_view(
            &Vector3::new(3.4, 1.2, 5.6),                                // camcenter
            &Vector3::new(2.2, 3.4, 5.6),                                // lookat
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)), // up
        );
        let cam2 = Camera::new(i2, e2);

        let expected_point = Points::new(SMatrix::<_, 1, 3>::new(2.21, 3.41, 5.61));

        let image1 = cam1.world_to_pixel(&expected_point);

        let image2 = cam2.world_to_pixel(&expected_point);

        let ray1 = cam1.pixel_to_world(&image1).to_single_ray();
        let ray2 = cam2.pixel_to_world(&image2).to_single_ray();
        let rays = [ray1, ray2];

        let point3d = best_intersection_of_rays(&rays).unwrap();

        // check reprojection for cam1
        let image1_actual = cam1.world_to_pixel(&point3d);
        approx::assert_abs_diff_eq!(
            image1_actual.data,
            image1.data,
            epsilon = nalgebra::convert(1e-10)
        );

        // check reprojection for cam2
        let image2_actual = cam2.world_to_pixel(&point3d);
        approx::assert_abs_diff_eq!(
            image2_actual.data,
            image2.data,
            epsilon = nalgebra::convert(1e-10)
        );

        // check 3d reconstruction
        approx::assert_abs_diff_eq!(
            point3d.data,
            expected_point.data,
            epsilon = nalgebra::convert(1e-10)
        );
    }
}

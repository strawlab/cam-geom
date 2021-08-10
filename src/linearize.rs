//! Linearize camera models by computing the Jacobian matrix.

use crate::{Camera, IntrinsicParametersPerspective, Points, WorldFrame};
use nalgebra::{storage::Storage, RealField, SMatrix, U1, U3};

/// Required data required for finding Jacobian of perspective camera models.
///
/// Create this with the [`new()`](struct.JacobianPerspectiveCache.html#method.new) method.
pub struct JacobianPerspectiveCache<R: RealField> {
    m: SMatrix<R, 3, 4>,
}

impl<R: RealField> JacobianPerspectiveCache<R> {
    /// Create a new `JacobianPerspectiveCache` from a `Camera` with a perspective model.
    pub fn new(cam: &Camera<R, IntrinsicParametersPerspective<R>>) -> Self {
        let m = {
            let p33 = cam.intrinsics().as_intrinsics_matrix();
            p33 * cam.extrinsics().matrix()
        };

        // flip sign if focal length < 0
        let m = if m[(0, 0)] < nalgebra::zero() { -m } else { m };

        let m = m.clone() / m[(2, 3)].clone(); // normalize

        Self { m }
    }

    /// Linearize camera model by evaluating around input point `p`.
    ///
    /// Returns Jacobian matrix `A` (shape 2x3) such that `Ao = (u,v)` where `o`
    /// is 3D world coords offset from `p` and `(u,v)` are the shift in pixel
    /// coords from the projected location of `p`. In other words, for a camera
    /// model `F(x)`, if `F(p) = (a,b)` and `F(p+o) = (a,b)
    /// + Ao = (a,b) + (u,v) = (a+u,b+v)`, this function returns `A`.
    pub fn linearize_at<STORAGE>(&self, p: &Points<WorldFrame, R, U1, STORAGE>) -> SMatrix<R, 2, 3>
    where
        STORAGE: Storage<R, U1, U3>,
    {
        let pt3d = &p.data;

        // See pinhole_jacobian_demo.py in flydra for the original source of this. It has
        // been manually factored it a bit futher.
        // https://github.com/strawlab/flydra/blob/3ab1b5843b095d73f796bf707e6680b923993899/flydra_core/sympy_demo/pinhole_jacobian_demo.py
        let x = pt3d[(0, 0)].clone();
        let y = pt3d[(0, 1)].clone();
        let z = pt3d[(0, 2)].clone();

        let p = &self.m;
        let denom = p[(2, 0)].clone() * x.clone()
            + p[(2, 1)].clone() * y.clone()
            + p[(2, 2)].clone() * z.clone()
            + p[(2, 3)].clone();
        let denom_sqrt = denom.clone().powi(-2);

        let factor_u = p[(0, 0)].clone() * x.clone()
            + p[(0, 1)].clone() * y.clone()
            + p[(0, 2)].clone() * z.clone()
            + p[(0, 3)].clone();
        let ux = -p[(2, 0)].clone() * denom_sqrt.clone() * factor_u.clone()
            + p[(0, 0)].clone() / denom.clone();
        let uy = -p[(2, 1)].clone() * denom_sqrt.clone() * factor_u.clone()
            + p[(0, 1)].clone() / denom.clone();
        let uz = -p[(2, 2)].clone() * denom_sqrt.clone() * factor_u.clone()
            + p[(0, 2)].clone() / denom.clone();

        let factor_v = p[(1, 0)].clone() * x.clone()
            + p[(1, 1)].clone() * y.clone()
            + p[(1, 2)].clone() * z.clone()
            + p[(1, 3)].clone();
        let vx = -p[(2, 0)].clone() * denom_sqrt.clone() * factor_v.clone()
            + p[(1, 0)].clone() / denom.clone();
        let vy = -p[(2, 1)].clone() * denom_sqrt.clone() * factor_v.clone()
            + p[(1, 1)].clone() / denom.clone();
        let vz = -p[(2, 2)].clone() * denom_sqrt.clone() * factor_v.clone()
            + p[(1, 2)].clone() / denom.clone();

        SMatrix::<R, 2, 3>::new(ux, uy, uz, vx, vy, vz)
    }
}

#[test]
fn test_jacobian_perspective() {
    use nalgebra::{OMatrix, RowVector2, RowVector3, Unit, Vector3};

    use super::*;
    use crate::{Camera, ExtrinsicParameters, IntrinsicParametersPerspective};

    // create a perspective camera
    let params = PerspectiveParams {
        fx: 100.0,
        fy: 102.0,
        skew: 0.1,
        cx: 321.0,
        cy: 239.9,
    };

    let intrinsics: IntrinsicParametersPerspective<_> = params.into();

    let camcenter = Vector3::new(10.0, 0.0, 10.0);
    let lookat = Vector3::new(0.0, 0.0, 0.0);
    let up = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
    let pose = ExtrinsicParameters::from_view(&camcenter, &lookat, &up);

    let cam = Camera::new(intrinsics, pose);

    // cache the required data to compute a jacobian
    let cam_jac = JacobianPerspectiveCache::new(&cam);

    // We are going to linearize around this center 3D center point
    let center = Points::new(RowVector3::new(0.01, 0.02, 0.03));

    // We will test a 3D point at this offset from the center
    let offset = Vector3::new(0.0, 0.0, 0.01);

    // Get the 2D projection (in pixels) of our center.
    let center_projected: OMatrix<f64, U1, U2> = cam.world_to_pixel(&center).data;

    // Linearize the camera model around the center 3D point.
    let linearized_cam = cam_jac.linearize_at(&center);

    // Compute the 3D point which we now want to view.
    let new_point = Points::new(center.data + offset.transpose());

    // Get the 2D projection (in pixels) using the original, non-linear camera model
    let nonlin = cam.world_to_pixel(&new_point).data;

    // Get the 2D projection (in pixels) point with the linearized camera model
    let o = linearized_cam * offset;
    let linear_prediction = RowVector2::new(center_projected.x + o[0], center_projected.y + o[1]);

    // Check both approaches are equal
    approx::assert_relative_eq!(linear_prediction, nonlin, epsilon = nalgebra::convert(1e-4));
}

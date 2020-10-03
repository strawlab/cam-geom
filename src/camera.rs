use nalgebra::{
    allocator::Allocator,
    base::storage::{Owned, Storage},
    convert,
    geometry::{Point3, Rotation3, UnitQuaternion},
    DefaultAllocator, Dim, Matrix, Matrix3, MatrixMN, RealField, Vector3, U1, U2, U3, U4,
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::{
    coordinate_system::WorldFrame,
    intrinsics_perspective::{IntrinsicParametersPerspective, PerspectiveParams},
    Bundle, Error, ExtrinsicParameters, IntrinsicParameters, Pixels, Points, RayBundle,
};

/// A camera model that can convert world coordinates into pixel coordinates.
///
/// # Examples
///
/// Creates a new perspective camera:
///
/// ```
/// use cam_geom::*;
/// use nalgebra::*;
///
/// // perepective parameters - focal length of 100, no skew, pixel center at (640,480)
/// let intrinsics = IntrinsicParametersPerspective::from(PerspectiveParams {
///     fx: 100.0,
///     fy: 100.0,
///     skew: 0.0,
///     cx: 640.0,
///     cy: 480.0,
/// });
///
/// // Set extrinsic parameters - camera at (10,0,10), looing at (0,0,0), up (0,0,1)
/// let camcenter = Vector3::new(10.0, 0.0, 10.0);
/// let lookat = Vector3::new(0.0, 0.0, 0.0);
/// let up = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
/// let pose = ExtrinsicParameters::from_view(&camcenter, &lookat, &up);
///
/// // Create camera with both intrinsic and extrinsic parameters.
/// let cam = Camera::new(intrinsics, pose);
/// ```
///
/// Creates a new orthographic camera:
///
/// ```
/// use cam_geom::*;
/// use nalgebra::*;
///
/// // orthographic parameters - scale of 100, pixel center at (640,480)
/// let intrinsics = IntrinsicParametersOrthographic::from(OrthographicParams {
///     sx: 100.0,
///     sy: 100.0,
///     cx: 640.0,
///     cy: 480.0,
/// });
///
/// // Set extrinsic parameters - camera at (10,0,10), looing at (0,0,0), up (0,0,1)
/// let camcenter = Vector3::new(10.0, 0.0, 10.0);
/// let lookat = Vector3::new(0.0, 0.0, 0.0);
/// let up = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
/// let pose = ExtrinsicParameters::from_view(&camcenter, &lookat, &up);
///
/// // Create camera with both intrinsic and extrinsic parameters.
/// let cam = Camera::new(intrinsics, pose);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Camera<R, I>
where
    I: IntrinsicParameters<R>,
    R: RealField,
{
    intrinsics: I,
    extrinsics: ExtrinsicParameters<R>,
}

impl<R, I> Camera<R, I>
where
    I: IntrinsicParameters<R>,
    R: RealField,
{
    /// Create a new camera from intrinsic and extrinsic parameters.
    ///
    /// # Arguments
    /// Intrinsic parameters and extrinsic parameters
    #[inline]
    pub fn new(intrinsics: I, extrinsics: ExtrinsicParameters<R>) -> Self {
        Self {
            intrinsics,
            extrinsics,
        }
    }

    /// Return a reference to the extrinsic parameters.
    #[inline]
    pub fn extrinsics(&self) -> &ExtrinsicParameters<R> {
        &self.extrinsics
    }

    /// Return a reference to the intrinsic parameters.
    #[inline]
    pub fn intrinsics(&self) -> &I {
        &self.intrinsics
    }

    /// take 3D coordinates in world frame and convert to pixel coordinates
    pub fn world_to_pixel<NPTS, InStorage>(
        &self,
        world: &Points<WorldFrame, R, NPTS, InStorage>,
    ) -> Pixels<R, NPTS, Owned<R, NPTS, U2>>
    where
        NPTS: Dim,
        InStorage: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<R, NPTS, U3>,
        DefaultAllocator: Allocator<R, NPTS, U2>,
    {
        let camera_frame = self.extrinsics.world_to_camera(&world);
        self.intrinsics.camera_to_pixel(&camera_frame)
    }

    /// take pixel coordinates and project to 3D in world frame
    ///
    /// output arguments:
    /// `camera` - camera frame coordinate rays
    /// `world` - world frame coordinate rays
    ///
    /// Note that the camera frame coordinates are returned as they must
    /// be computed anyway, so this additional data is "free".
    pub fn pixel_to_world<IN, NPTS>(
        &self,
        pixels: &Pixels<R, NPTS, IN>,
    ) -> RayBundle<WorldFrame, I::BundleType, R, NPTS, Owned<R, NPTS, U3>>
    where
        I::BundleType: Bundle<R>,
        IN: Storage<R, NPTS, U2>,
        NPTS: Dim,
        I::BundleType: Bundle<R>,
        DefaultAllocator: Allocator<R, U1, U2>,
        DefaultAllocator: Allocator<R, NPTS, U2>,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        // get camera frame rays
        let camera = self.intrinsics.pixel_to_camera(&pixels);

        // get world frame rays
        self.extrinsics.ray_camera_to_world(&camera)
    }
}

impl<R: RealField> Camera<R, IntrinsicParametersPerspective<R>> {
    /// Create a `Camera` from a 3x4 perspective projection matrix.
    pub fn from_perspective_matrix<S>(pmat: &Matrix<R, U3, U4, S>) -> Result<Self, Error>
    where
        S: Storage<R, U3, U4> + Clone,
    {
        let m = pmat.clone().remove_column(3);
        let (rquat, k) = rq_decomposition(m)?;

        let k22: R = k[(2, 2)];

        let one: R = convert(1.0);

        let k = k * (one / k22); // normalize

        let params = PerspectiveParams {
            fx: k[(0, 0)],
            fy: k[(1, 1)],
            skew: k[(0, 1)],
            cx: k[(0, 2)],
            cy: k[(1, 2)],
        };

        let camcenter = pmat2cam_center(pmat);
        let extrinsics = ExtrinsicParameters::from_rotation_and_camcenter(rquat, camcenter);

        Ok(Self::new(params.into(), extrinsics))
    }

    /// Create a 3x4 perspective projection matrix modeling this camera.
    pub fn as_camera_matrix(&self) -> MatrixMN<R, U3, U4> {
        let m = {
            let p33 = self.intrinsics().as_intrinsics_matrix();
            p33 * self.extrinsics().cache.qt
        };

        // flip sign if focal length < 0
        let m = if m[(0, 0)] < nalgebra::convert(0.0) {
            -m
        } else {
            m
        };

        m / m[(2, 3)] // normalize
    }
}

#[cfg(test)]
pub fn roundtrip_camera<R, I>(
    cam: Camera<R, I>,
    width: usize,
    height: usize,
    step: usize,
    border: usize,
    eps: R,
) where
    R: RealField,
    I: IntrinsicParameters<R>,
    I::BundleType: Bundle<R>,
{
    let pixels = crate::intrinsic_test_utils::generate_uv_raw(width, height, step, border);

    let world_coords = cam.pixel_to_world(&pixels);
    let world_coords_points = world_coords.point_on_ray();

    // project back to pixel coordinates
    let pixel_actual = cam.world_to_pixel(&world_coords_points);
    approx::assert_abs_diff_eq!(pixels.data, pixel_actual.data, epsilon = convert(eps));
}

#[allow(non_snake_case)]
fn rq<R: RealField>(A: Matrix3<R>) -> (Matrix3<R>, Matrix3<R>) {
    let zero: R = convert(0.0);
    let one: R = convert(1.0);

    // see https://math.stackexchange.com/a/1640762
    let P = Matrix3::<R>::new(
        zero, zero, one, // row 1
        zero, one, zero, // row 2
        one, zero, zero, // row 3
    );
    let Atilde = P * A;

    let (Qtilde, Rtilde) = {
        let qrm = nalgebra::linalg::QR::new(Atilde.transpose());
        (qrm.q(), qrm.r())
    };
    let Q = P * Qtilde.transpose();
    let R = P * Rtilde.transpose() * P;
    (R, Q)
}

/// perform RQ decomposition and return results as right-handed quaternion and intrinsics matrix
fn rq_decomposition<R: RealField>(
    orig: Matrix3<R>,
) -> Result<(UnitQuaternion<R>, Matrix3<R>), Error> {
    let (mut intrin, mut q) = rq(orig);
    let zero: R = convert(0.0);
    for i in 0..3 {
        if intrin[(i, i)] < zero {
            for j in 0..3 {
                intrin[(j, i)] = -intrin[(j, i)];
                q[(i, j)] = -q[(i, j)];
            }
        }
    }

    match right_handed_rotation_quat_new(&q) {
        Ok(rquat) => Ok((rquat, intrin)),
        Err(error) => {
            match error {
                Error::InvalidRotationMatrix => {
                    // convert left-handed rotation to right-handed rotation
                    let q = -q;
                    let intrin = -intrin;
                    let rquat = right_handed_rotation_quat_new(&q)?;
                    Ok((rquat, intrin))
                }
                e => Err(e),
            }
        }
    }
}

/// convert a 3x3 matrix into a valid right-handed rotation
fn right_handed_rotation_quat_new<R: RealField>(
    orig: &Matrix3<R>,
) -> Result<UnitQuaternion<R>, Error> {
    let r1 = *orig;
    let rotmat = Rotation3::from_matrix_unchecked(r1);
    let rquat = UnitQuaternion::from_rotation_matrix(&rotmat);
    if !is_right_handed_rotation_quat(&rquat) {
        return Err(Error::InvalidRotationMatrix);
    }
    Ok(rquat)
}

/// Check for valid right-handed rotation.
///
/// Converts quaternion to rotation matrix and back again to quat then comparing
/// quats. Probably there is a much faster and better way.
pub(crate) fn is_right_handed_rotation_quat<R: RealField>(rquat: &UnitQuaternion<R>) -> bool {
    let rotmat2 = rquat.to_rotation_matrix();
    let rquat2 = UnitQuaternion::from_rotation_matrix(&rotmat2);
    let delta = rquat.rotation_to(&rquat2);
    let angle = my_quat_angle(&delta);
    let epsilon = R::default_epsilon() * convert(1e5);
    angle.abs() <= epsilon
}

/// get the camera center from a 3x4 camera projection matrix
fn pmat2cam_center<R, S>(p: &Matrix<R, U3, U4, S>) -> Point3<R>
where
    R: RealField + Clone,
    S: Storage<R, U3, U4> + Clone,
{
    let x = p.clone().remove_column(0).determinant();
    let y = -p.clone().remove_column(1).determinant();
    let z = p.clone().remove_column(2).determinant();
    let w = -p.clone().remove_column(3).determinant();
    Point3::from(Vector3::new(x / w, y / w, z / w))
}

// Calculate angle of quaternion
///
/// This is the implementation from prior to
/// https://github.com/rustsim/nalgebra/commit/74aefd9c23dadd12ee654c7d0206b0a96d22040c
fn my_quat_angle<R: RealField>(quat: &nalgebra::UnitQuaternion<R>) -> R {
    let w = quat.quaternion().scalar().abs();

    // Handle inaccuracies that make break `.acos`.
    if w >= R::one() {
        R::zero()
    } else {
        w.acos() * convert(2.0f64)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "serde-serialize")]
    fn test_serde() {
        use nalgebra::{Unit, Vector3};

        use super::PerspectiveParams;
        use crate::{Camera, ExtrinsicParameters, IntrinsicParametersPerspective};

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

        let expected = Camera::new(intrinsics, pose);

        let buf = serde_json::to_string(&expected).unwrap();
        let actual: Camera<_, _> = serde_json::from_str(&buf).unwrap();
        assert!(expected == actual);
    }
}

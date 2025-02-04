#![cfg_attr(not(feature = "std"), no_std)]
#![deny(rust_2018_idioms, unsafe_code, missing_docs)]
#![cfg_attr(not(doctest), doc = include_str!("../README.md"))]
#![cfg_attr(doc_cfg, feature(doc_cfg))]

//! # Examples
//!
//! ## Example - projecting 3D world coordinates to 2D pixel coordinates.
//!
//! ```
//! use cam_geom::*;
//! use nalgebra::{Matrix2x3, Unit, Vector3};
//!
//! // Create two points in the world coordinate frame.
//! let world_coords = Points::new(Matrix2x3::new(
//!     1.0, 0.0, 0.0, // point 1
//!     0.0, 1.0, 0.0, // point 2
//! ));
//!
//! // perepective parameters - focal length of 100, no skew, pixel center at (640,480)
//! let intrinsics = IntrinsicParametersPerspective::from(PerspectiveParams {
//!     fx: 100.0,
//!     fy: 100.0,
//!     skew: 0.0,
//!     cx: 640.0,
//!     cy: 480.0,
//! });
//!
//! // Set extrinsic parameters - camera at (10,0,0), looking at (0,0,0), up (0,0,1)
//! let camcenter = Vector3::new(10.0, 0.0, 0.0);
//! let lookat = Vector3::new(0.0, 0.0, 0.0);
//! let up = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
//! let pose = ExtrinsicParameters::from_view(&camcenter, &lookat, &up);
//!
//! // Create a `Camera` with both intrinsic and extrinsic parameters.
//! let camera = Camera::new(intrinsics, pose);
//!
//! // Project the original 3D coordinates to 2D pixel coordinates.
//! let pixel_coords = camera.world_to_pixel(&world_coords);
//!
//! // Print the results.
//! for i in 0..world_coords.data.nrows() {
//!     let wc = world_coords.data.row(i);
//!     let pix = pixel_coords.data.row(i);
//!     println!("{} -> {}", wc, pix);
//! }
//! ```
//!
//! This will print:
//!
//! ```text
//!   ┌       ┐
//!   │ 1 0 0 │
//!   └       ┘
//!
//!  ->
//!   ┌         ┐
//!   │ 640 480 │
//!   └         ┘
//!
//!
//!
//!   ┌       ┐
//!   │ 0 1 0 │
//!   └       ┘
//!
//!  ->
//!   ┌         ┐
//!   │ 650 480 │
//!   └         ┘
//! ```
//!
//!
//! ## Example - intersection of rays
//!
//! ```
//! use cam_geom::*;
//! use nalgebra::RowVector3;
//!
//! // Create the first ray.
//!     let ray1 = Ray::<WorldFrame, _>::new(
//!     RowVector3::new(1.0, 0.0, 0.0), // origin
//!     RowVector3::new(0.0, 1.0, 0.0), // direction
//! );
//!
//! // Create the second ray.
//! let ray2 = Ray::<WorldFrame, _>::new(
//!     RowVector3::new(0.0, 1.0, 0.0), // origin
//!     RowVector3::new(1.0, 0.0, 0.0), // direction
//! );
//!
//! // Compute the best intersection.
//! let result = best_intersection_of_rays(&[ray1, ray2]).unwrap();
//!
//! // Print the result.
//! println!("result: {}", result.data);
//! ```
//!
//! This will print:
//!
//! ```text
//! result:
//!   ┌       ┐
//!   │ 1 1 0 │
//!   └       ┘
//! ```

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use nalgebra::{
    allocator::Allocator,
    storage::{Owned, Storage},
    DefaultAllocator, Dim, DimName, Isometry3, Matrix, Point3, RealField, SMatrix, Vector3, U1, U2,
    U3,
};

#[cfg(feature = "std")]
pub mod intrinsic_test_utils;

mod intrinsics_perspective;
pub use intrinsics_perspective::{IntrinsicParametersPerspective, PerspectiveParams};

mod intrinsics_orthographic;
pub use intrinsics_orthographic::{IntrinsicParametersOrthographic, OrthographicParams};

mod extrinsics;
pub use extrinsics::ExtrinsicParameters;

mod camera;
pub use camera::Camera;

/// Defines the different possible types of ray bundles.
pub mod ray_bundle_types;

#[cfg(feature = "alloc")]
mod ray_intersection;
#[cfg(feature = "alloc")]
pub use ray_intersection::best_intersection_of_rays;

pub mod linearize;

/// All possible errors.
#[cfg_attr(feature = "std", derive(Debug))]
#[non_exhaustive]
pub enum Error {
    /// Invalid input.
    InvalidInput,
    /// Singular Value Decomposition did not converge.
    SvdFailed,
    /// At least two rays are needed to compute their intersection.
    MinimumTwoRaysNeeded,
    /// Invalid rotation matrix
    InvalidRotationMatrix,
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[cfg(feature = "std")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

/// 2D pixel locations on the image sensor.
///
/// These pixels are "distorted" - with barrel and pincushion distortion - if
/// the camera model incorporates such. (Undistorted pixels are handled
/// internally within the camera model.)
///
/// This is a newtype wrapping an `nalgebra::Matrix`.
#[derive(Clone)]
pub struct Pixels<R: RealField, NPTS: Dim, STORAGE> {
    /// The matrix storing pixel locations.
    pub data: nalgebra::Matrix<R, NPTS, U2, STORAGE>,
}

impl<R: RealField, NPTS: Dim, STORAGE> Pixels<R, NPTS, STORAGE> {
    /// Create a new Pixels instance
    #[inline]
    pub fn new(data: nalgebra::Matrix<R, NPTS, U2, STORAGE>) -> Self {
        Self { data }
    }
}

/// A coordinate system in which points and rays can be defined.
pub trait CoordinateSystem {}

/// Implementations of [`CoordinateSystem`](trait.CoordinateSystem.html).
pub mod coordinate_system {

    #[cfg(feature = "serde-serialize")]
    use serde::{Deserialize, Serialize};

    /// Coordinates in the camera coordinate system.
    ///
    /// The camera center is at (0,0,0) at looking at (0,0,1) with up as
    /// (0,-1,0) in this coordinate frame.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
    pub struct CameraFrame {}
    impl crate::CoordinateSystem for CameraFrame {}

    /// Coordinates in the world coordinate system.
    ///
    /// The camera center is may be located at an arbitrary position and pointed
    /// in an arbitrary direction in this coordinate frame.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
    pub struct WorldFrame {}
    impl crate::CoordinateSystem for WorldFrame {}
}
pub use coordinate_system::{CameraFrame, WorldFrame};

/// 3D points. Can be in any [`CoordinateSystem`](trait.CoordinateSystem.html).
///
/// This is a newtype wrapping an `nalgebra::Matrix`.
pub struct Points<Coords: CoordinateSystem, R: RealField, NPTS: Dim, STORAGE> {
    coords: std::marker::PhantomData<Coords>,
    /// The matrix storing point locations.
    pub data: nalgebra::Matrix<R, NPTS, U3, STORAGE>,
}

#[cfg(feature = "std")]
impl<Coords: CoordinateSystem, R: RealField, NPTS: Dim, STORAGE: std::fmt::Debug> std::fmt::Debug
    for Points<Coords, R, NPTS, STORAGE>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Points")
            .field("coords", &self.coords)
            .field("data", &self.data)
            .finish()
    }
}

impl<Coords, R, NPTS, STORAGE> Points<Coords, R, NPTS, STORAGE>
where
    Coords: CoordinateSystem,
    R: RealField,
    NPTS: Dim,
{
    /// Create a new Points instance from the underlying storage.
    #[inline]
    pub fn new(data: nalgebra::Matrix<R, NPTS, U3, STORAGE>) -> Self {
        Self {
            coords: std::marker::PhantomData,
            data,
        }
    }
}

/// 3D rays. Can be in any [`CoordinateSystem`](trait.CoordinateSystem.html).
///
/// Any given `RayBundle` will have a particular bundle type, which implements
/// the [`Bundle`](trait.Bundle.html) trait.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct RayBundle<Coords, BType, R, NPTS, StorageMultiple>
where
    Coords: CoordinateSystem,
    BType: Bundle<R>,
    R: RealField,
    NPTS: Dim,
    StorageMultiple: Storage<R, NPTS, U3>,
{
    coords: std::marker::PhantomData<Coords>,
    /// The matrix storing the ray data.
    pub data: Matrix<R, NPTS, U3, StorageMultiple>,
    bundle_type: BType,
}

impl<Coords, BType, R, NPTS, StorageMultiple> RayBundle<Coords, BType, R, NPTS, StorageMultiple>
where
    Coords: CoordinateSystem,
    BType: Bundle<R>,
    R: RealField,
    NPTS: DimName,
    StorageMultiple: Storage<R, NPTS, U3>,
    DefaultAllocator: Allocator<NPTS, U3>,
{
    /// get directions of each ray in bundle
    #[inline]
    pub fn directions(&self) -> Matrix<R, NPTS, U3, Owned<R, NPTS, U3>> {
        self.bundle_type.directions(&self.data)
    }

    /// get centers (origins) of each ray in bundle
    #[inline]
    pub fn centers(&self) -> Matrix<R, NPTS, U3, Owned<R, NPTS, U3>> {
        self.bundle_type.centers(&self.data)
    }
}

impl<Coords, BType, R> RayBundle<Coords, BType, R, U1, Owned<R, U1, U3>>
where
    Coords: CoordinateSystem,
    BType: Bundle<R>,
    R: RealField,
{
    /// Return the single ray from the RayBundle with exactly one ray.
    #[inline]
    pub fn to_single_ray(&self) -> Ray<Coords, R> {
        self.bundle_type.to_single_ray(&self.data)
    }
}

impl<Coords, R, NPTS, StorageMultiple>
    RayBundle<Coords, crate::ray_bundle_types::SharedOriginRayBundle<R>, R, NPTS, StorageMultiple>
where
    Coords: CoordinateSystem,
    R: RealField,
    NPTS: Dim,
    StorageMultiple: Storage<R, NPTS, U3>,
{
    /// Create a new RayBundle instance in which all rays share origin at zero.
    ///
    /// The number of points allocated is given by the `npts` parameter, which
    /// should agree with the `NPTS` type. The coordinate system is given by the
    /// `Coords` type.
    pub fn new_shared_zero_origin(data: Matrix<R, NPTS, U3, StorageMultiple>) -> Self {
        let bundle_type = crate::ray_bundle_types::SharedOriginRayBundle::new_shared_zero_origin();
        Self::new(bundle_type, data)
    }
}

impl<Coords, R, NPTS, StorageMultiple>
    RayBundle<
        Coords,
        crate::ray_bundle_types::SharedDirectionRayBundle<R>,
        R,
        NPTS,
        StorageMultiple,
    >
where
    Coords: CoordinateSystem,
    R: RealField,
    NPTS: Dim,
    StorageMultiple: Storage<R, NPTS, U3>,
{
    /// Create a new RayBundle instance in which all rays share +z direction.
    ///
    /// The number of points allocated is given by the `npts` parameter, which
    /// should agree with the `NPTS` type. The coordinate system is given by the
    /// `Coords` type.
    pub fn new_shared_plusz_direction(data: Matrix<R, NPTS, U3, StorageMultiple>) -> Self {
        let bundle_type =
            crate::ray_bundle_types::SharedDirectionRayBundle::new_plusz_shared_direction();
        Self::new(bundle_type, data)
    }
}

impl<Coords, BType, R, NPTS, StorageMultiple> RayBundle<Coords, BType, R, NPTS, StorageMultiple>
where
    Coords: CoordinateSystem,
    BType: Bundle<R>,
    R: RealField,
    NPTS: Dim,
    StorageMultiple: Storage<R, NPTS, U3>,
{
    /// Create a new RayBundle instance from the underlying storage.
    ///
    /// The coordinate system is given by the `Coords` type and the bundle type
    /// (e.g. shared origin or shared direction) is given by the `BType`.
    #[inline]
    fn new(bundle_type: BType, data: nalgebra::Matrix<R, NPTS, U3, StorageMultiple>) -> Self {
        Self {
            coords: std::marker::PhantomData,
            data,
            bundle_type,
        }
    }

    /// get a 3D point on the ray, obtained by adding the direction(s) to the origin(s)
    ///
    /// The distance of the point from the ray bundle center is not definted and
    /// can be arbitrary.
    #[inline]
    pub fn point_on_ray(&self) -> Points<Coords, R, NPTS, Owned<R, NPTS, U3>>
    where
        DefaultAllocator: Allocator<NPTS, U3>,
    {
        self.bundle_type.point_on_ray(&self.data)
    }

    /// get a 3D point on the ray at a defined distance from the origin(s)
    #[inline]
    pub fn point_on_ray_at_distance(
        &self,
        distance: R,
    ) -> Points<Coords, R, NPTS, Owned<R, NPTS, U3>>
    where
        DefaultAllocator: Allocator<NPTS, U3>,
    {
        self.bundle_type
            .point_on_ray_at_distance(&self.data, distance)
    }

    #[inline]
    fn to_pose<OutFrame>(
        &self,
        pose: Isometry3<R>,
    ) -> RayBundle<OutFrame, BType, R, NPTS, Owned<R, NPTS, U3>>
    where
        R: RealField,
        NPTS: Dim,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<NPTS, U3>,
    {
        self.bundle_type.to_pose(pose, &self.data)
    }
}

/// A single ray. Can be in any [`CoordinateSystem`](trait.CoordinateSystem.html).
///
/// A `RayBundle` with only one ray can be converted to this with
/// `RayBundle::to_single_ray()`.
pub struct Ray<Coords, R: RealField> {
    /// The center (origin) of the ray.
    pub center: SMatrix<R, 1, 3>,
    /// The direction of the ray.
    pub direction: SMatrix<R, 1, 3>,
    c: std::marker::PhantomData<Coords>,
}

impl<Coords, R: RealField> Ray<Coords, R> {
    /// Create a new ray from center (origin) and direction.
    #[inline]
    pub fn new(center: SMatrix<R, 1, 3>, direction: SMatrix<R, 1, 3>) -> Self {
        Self {
            center,
            direction,
            c: std::marker::PhantomData,
        }
    }
}

/// Specifies operations which any RayBundle must implement.
pub trait Bundle<R>
where
    R: RealField,
{
    /// Return a single ray from a `RayBundle` with exactly one ray.
    fn to_single_ray<Coords>(&self, self_data: &SMatrix<R, 1, 3>) -> Ray<Coords, R>
    where
        Coords: CoordinateSystem;

    /// Get directions of each ray in bundle.
    ///
    /// This can be inefficient, because when not every ray has a different
    /// direction (which is the case for the `SharedDirectionRayBundle` type),
    /// this will nevertheless copy the single direction `NPTS` times.
    fn directions<NPTS, StorageIn>(
        &self,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> Matrix<R, NPTS, U3, Owned<R, NPTS, U3>>
    where
        NPTS: DimName,
        StorageIn: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<NPTS, U3>;

    /// Get centers of each ray in bundle.
    ///
    /// This can be inefficient, because when not every ray has a different
    /// center (which is the case for the `SharedOriginRayBundle` type),
    /// this will nevertheless copy the single center `NPTS` times.
    fn centers<NPTS, StorageIn>(
        &self,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> Matrix<R, NPTS, U3, Owned<R, NPTS, U3>>
    where
        NPTS: DimName,
        StorageIn: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<NPTS, U3>;

    /// Return points on on the input rays.
    ///
    /// The distance of the point from the ray bundle center is not definted and
    /// can be arbitrary.
    fn point_on_ray<NPTS, StorageIn, OutFrame>(
        &self,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> Points<OutFrame, R, NPTS, Owned<R, NPTS, U3>>
    where
        Self: Sized,
        NPTS: Dim,
        StorageIn: Storage<R, NPTS, U3>,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<NPTS, U3>;

    /// Return points on on the input rays at a defined distance from the origin(s).
    fn point_on_ray_at_distance<NPTS, StorageIn, OutFrame>(
        &self,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
        distance: R,
    ) -> Points<OutFrame, R, NPTS, Owned<R, NPTS, U3>>
    where
        Self: Sized,
        NPTS: Dim,
        StorageIn: Storage<R, NPTS, U3>,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<NPTS, U3>;

    /// Convert the input rays by the pose given.
    fn to_pose<NPTS, StorageIn, OutFrame>(
        &self,
        pose: Isometry3<R>,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> RayBundle<OutFrame, Self, R, NPTS, Owned<R, NPTS, U3>>
    where
        Self: Sized,
        R: RealField,
        NPTS: Dim,
        StorageIn: Storage<R, NPTS, U3>,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<NPTS, U3>;
}

/// A geometric model of camera coordinates to pixels (and vice versa).
pub trait IntrinsicParameters<R>: std::fmt::Debug + Clone
where
    R: RealField,
{
    /// What type of ray bundle is returned when projecting pixels to rays.
    type BundleType;

    /// project pixels to camera coords
    fn pixel_to_camera<IN, NPTS>(
        &self,
        pixels: &Pixels<R, NPTS, IN>,
    ) -> RayBundle<coordinate_system::CameraFrame, Self::BundleType, R, NPTS, Owned<R, NPTS, U3>>
    where
        Self::BundleType: Bundle<R>,
        IN: Storage<R, NPTS, U2>,
        NPTS: Dim,
        DefaultAllocator: Allocator<U1, U2>, // needed to make life easy for implementors
        DefaultAllocator: Allocator<NPTS, U2>, // needed to make life easy for implementors
        DefaultAllocator: Allocator<NPTS, U3>;

    /// project camera coords to pixel coordinates
    fn camera_to_pixel<IN, NPTS>(
        &self,
        camera: &Points<coordinate_system::CameraFrame, R, NPTS, IN>,
    ) -> Pixels<R, NPTS, Owned<R, NPTS, U2>>
    where
        IN: Storage<R, NPTS, U3>,
        NPTS: Dim,
        DefaultAllocator: Allocator<NPTS, U2>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::convert;

    #[cfg(not(feature = "std"))]
    compile_error!("tests require std");

    #[test]
    fn rays_shared_origin() {
        // Create rays in world coorindates all with a shared origin at zero.
        let b1 =
            RayBundle::<WorldFrame, _, _, _, _>::new_shared_zero_origin(SMatrix::<_, 2, 3>::new(
                1.0, 2.0, 3.0, // ray 1
                4.0, 5.0, 6.0, // ray 2
            ));

        // Get points on rays at a specific distance.
        let actual_dist1 = b1.point_on_ray_at_distance(1.0).data;

        {
            // Manually compte what these points should be.
            let r1m = (1.0_f64 + 4.0 + 9.0).sqrt();
            let r2m = (16.0_f64 + 25.0 + 36.0).sqrt();
            let expected = SMatrix::<_, 2, 3>::new(
                1.0 / r1m,
                2.0 / r1m,
                3.0 / r1m, // ray 1
                4.0 / r2m,
                5.0 / r2m,
                6.0 / r2m, // ray 2
            );

            // Check the points vs the manually computed versions.
            approx::assert_abs_diff_eq!(actual_dist1, expected, epsilon = 1e-10);
        }

        // Get points on rays at a specific distance.
        let actual_dist10 = b1.point_on_ray_at_distance(10.0).data;

        // Get points on rays at arbitrary distance.
        let actual = b1.point_on_ray().data;

        for i in 0..actual_dist1.nrows() {
            assert_on_line(actual_dist1.row(i), actual_dist10.row(i), actual.row(i));
        }
    }

    #[test]
    fn rays_shared_direction() {
        // Create rays in world coorindates all with a shared direction (+z).
        let b1 = RayBundle::<WorldFrame, _, _, _, _>::new_shared_plusz_direction(
            SMatrix::<_, 2, 3>::new(
                1.0, 2.0, 0.0, // ray 1
                3.0, 4.0, 0.0, // ray 2
            ),
        );

        // Get points on rays at a specific distance.
        let actual_dist10 = b1.point_on_ray_at_distance(10.0).data;

        {
            // Manually compte what these points should be.
            let expected_dist10 = SMatrix::<_, 2, 3>::new(
                1.0, 2.0, 10.0, // ray 1
                3.0, 4.0, 10.0, // ray 2
            );

            // Check the points vs the manually computed versions.
            approx::assert_abs_diff_eq!(actual_dist10, expected_dist10, epsilon = 1e-10);
        }

        // Get points on rays at a specific distance.
        let actual_dist0 = b1.point_on_ray_at_distance(0.0).data;

        {
            // Manually compte what these points should be.
            let expected_dist0 = SMatrix::<_, 2, 3>::new(
                1.0, 2.0, 0.0, // ray 1
                3.0, 4.0, 0.0, // ray 2
            );

            // Check the points vs the manually computed versions.
            approx::assert_abs_diff_eq!(actual_dist0, expected_dist0, epsilon = 1e-10);
        }

        // Get points on rays at arbitrary distance.
        let actual = b1.point_on_ray().data;

        for i in 0..actual_dist0.nrows() {
            assert_on_line(actual_dist0.row(i), actual_dist10.row(i), actual.row(i));
        }
    }

    fn assert_on_line<R, S1, S2, S3>(
        line_a: Matrix<R, U1, U3, S1>,
        line_b: Matrix<R, U1, U3, S2>,
        test_pt: Matrix<R, U1, U3, S3>,
    ) where
        R: RealField,
        S1: Storage<R, U1, U3>,
        S2: Storage<R, U1, U3>,
        S3: Storage<R, U1, U3>,
    {
        let dir = &line_b - &line_a;
        let testx = &test_pt - &line_a;
        let mag_dir = (dir[0].clone() * dir[0].clone()
            + dir[1].clone() * dir[1].clone()
            + dir[2].clone() * dir[2].clone())
        .sqrt();
        let mag_testx = (testx[0].clone() * testx[0].clone()
            + testx[1].clone() * testx[1].clone()
            + testx[2].clone() * testx[2].clone())
        .sqrt();
        let scale = mag_dir / mag_testx;

        for j in 0..3 {
            approx::assert_abs_diff_eq!(
                testx[j].clone() * scale.clone(),
                dir[j].clone(),
                epsilon = convert(1e-10)
            );
        }
    }

    #[test]
    #[cfg(feature = "serde-serialize")]
    fn test_ray_bundle_serde() {
        let expected =
            RayBundle::<WorldFrame, _, _, _, _>::new_shared_plusz_direction(
                SMatrix::<_, 2, 3>::new(
                    1.0, 2.0, 0.0, // ray 1
                    3.0, 4.0, 0.0, // ray 2
                ),
            );

        let buf = serde_json::to_string(&expected).unwrap();
        let actual: RayBundle<_, _, _, _, _> = serde_json::from_str(&buf).unwrap();
        assert!(expected == actual);
    }
}

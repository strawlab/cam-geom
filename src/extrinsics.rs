use nalgebra::geometry::{Isometry3, Point3, Rotation3, Translation, UnitQuaternion};
use nalgebra::{
    allocator::Allocator,
    storage::{Owned, Storage},
    DefaultAllocator, RealField,
};
use nalgebra::{convert, Dim, MatrixMN, Unit, Vector3, U3, U4};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::{
    coordinate_system::{CameraFrame, WorldFrame},
    Bundle, Points, RayBundle,
};

/// Defines the pose of a camera in the world coordinate system.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize))]
pub struct ExtrinsicParameters<R: RealField> {
    pub(crate) rquat: UnitQuaternion<R>,
    pub(crate) camcenter: Point3<R>,
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) cache: ExtrinsicsCache<R>,
}

impl<R: RealField> std::fmt::Debug for ExtrinsicParameters<R> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // This should match the auto derived Debug implementation but not print
        // the cache field.
        fmt.debug_struct("ExtrinsicParameters")
            .field("rquat", &self.rquat)
            .field("camcenter", &self.camcenter)
            .finish()
    }
}

#[derive(Clone, PartialEq)]
pub(crate) struct ExtrinsicsCache<R: RealField> {
    pub(crate) q: Rotation3<R>,
    pub(crate) translation: Point3<R>,
    pub(crate) qt: MatrixMN<R, U3, U4>,
    pub(crate) q_inv: Rotation3<R>,
    pub(crate) camcenter_z0: Point3<R>,
    pub(crate) pose: Isometry3<R>,
    pub(crate) pose_inv: Isometry3<R>,
}

impl<R: RealField> std::fmt::Debug for ExtrinsicsCache<R> {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // do not show cache
        Ok(())
    }
}

impl<R: RealField> ExtrinsicParameters<R> {
    /// Create a new instance from a rotation and a camera center.
    pub fn from_rotation_and_camcenter(rotation: UnitQuaternion<R>, camcenter: Point3<R>) -> Self {
        let q = rotation.to_rotation_matrix();
        let translation = -(q * camcenter);
        #[rustfmt::skip]
        let qt = {
            let q = q.matrix();
            MatrixMN::<R,U3,U4>::new(
                q[(0,0)], q[(0,1)], q[(0,2)], translation[0],
                q[(1,0)], q[(1,1)], q[(1,2)], translation[1],
                q[(2,0)], q[(2,1)], q[(2,2)], translation[2],
            )
        };
        let q_inv = q.inverse();
        let camcenter_z0 = Point3::from(Vector3::new(camcenter[0], camcenter[1], convert(0.0)));
        let pose = Isometry3::from_parts(
            Translation {
                vector: translation.coords,
            },
            rotation,
        );
        let pose_inv = pose.inverse();
        let cache = ExtrinsicsCache {
            q,
            translation,
            qt,
            q_inv,
            camcenter_z0,
            pose,
            pose_inv,
        };

        // TODO: ensure that our rotation is right handed. The commented out
        // code below is not robust enough. For example, the following code
        // would cause it to panic:
        //
        //     let camcenter = Vector3::new(10.0, 0.0, 10.0);
        //     let lookat = Vector3::new(0.0, 0.0, 0.0);
        //     let up = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
        //     ExtrinsicParameters::from_view(&camcenter, &lookat, &up);

        // if !crate::camera::is_right_handed_rotation_quat(&rotation) {
        //     panic!("rotation is not right handed.");
        // }

        Self {
            rquat: rotation,
            camcenter,
            cache,
        }
    }

    /// Create a new instance from an [`nalgebra::Isometry3`](https://docs.rs/nalgebra/latest/nalgebra/geometry/type.Isometry3.html).
    pub fn from_pose(pose: &Isometry3<R>) -> Self {
        let rquat = pose.rotation;
        let translation = pose.translation.vector;
        let q = rquat.inverse().to_rotation_matrix();
        let camcenter = -(q * translation);
        let cc = Point3 { coords: camcenter };
        Self::from_rotation_and_camcenter(rquat, cc)
    }

    /// Create a new instance from a camera center, a lookat vector, and an up vector.
    pub fn from_view(camcenter: &Vector3<R>, lookat: &Vector3<R>, up: &Unit<Vector3<R>>) -> Self {
        let dir = lookat - camcenter;
        let dir_unit = nalgebra::Unit::new_normalize(dir);
        let q = UnitQuaternion::look_at_lh(dir_unit.as_ref(), up.as_ref());
        let pi: R = convert(std::f64::consts::PI);

        let q2 = UnitQuaternion::from_axis_angle(&dir_unit, pi);
        let q3 = q * q2;

        Self::from_rotation_and_camcenter(q3, Point3 { coords: *camcenter })
    }

    /// Return the camera center
    #[inline]
    pub fn camcenter(&self) -> &Point3<R> {
        &self.camcenter
    }

    /// Return the camera pose
    #[inline]
    pub fn pose(&self) -> &Isometry3<R> {
        &self.cache.pose
    }

    /// Return the pose as a 3x4 matrix
    #[inline]
    pub fn matrix(&self) -> &MatrixMN<R, U3, U4> {
        &self.cache.qt
    }

    /// Return the rotation part of the pose
    #[inline]
    pub fn rotation(&self) -> &Rotation3<R> {
        &self.cache.q
    }

    /// Return the translation part of the pose
    #[inline]
    pub fn translation(&self) -> &Point3<R> {
        &self.cache.translation
    }

    /// Return a unit vector aligned along our look (+Z) direction.
    pub fn forward(&self) -> Unit<Vector3<R>> {
        let pt_cam = Point3::new(R::zero(), R::zero(), R::one());
        self.lookdir(&pt_cam)
    }

    /// Return a unit vector aligned along our up (-Y) direction.
    pub fn up(&self) -> Unit<Vector3<R>> {
        let pt_cam = Point3::new(R::zero(), -R::one(), R::zero());
        self.lookdir(&pt_cam)
    }

    /// Return a unit vector aligned along our right (+X) direction.
    pub fn right(&self) -> Unit<Vector3<R>> {
        let pt_cam = Point3::new(R::one(), R::zero(), R::zero());
        self.lookdir(&pt_cam)
    }

    /// Return a world coords unit vector aligned along the given direction
    ///
    /// `pt_cam` is specified in camera coords.
    #[inline]
    fn lookdir(&self, pt_cam: &Point3<R>) -> Unit<Vector3<R>> {
        let cc = self.camcenter();
        let pt = self.cache.pose_inv.transform_point(&pt_cam) - cc;
        nalgebra::Unit::new_normalize(pt)
    }

    /// Convert points in camera coordinates to world coordinates.
    pub fn camera_to_world<NPTS, InStorage>(
        &self,
        cam_coords: &Points<CameraFrame, R, NPTS, InStorage>,
    ) -> Points<WorldFrame, R, NPTS, Owned<R, NPTS, U3>>
    where
        NPTS: Dim,
        InStorage: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        let mut world = Points::new(MatrixMN::zeros_generic(
            NPTS::from_usize(cam_coords.data.nrows()),
            U3::from_usize(3),
        ));

        // Potential optimization: remove for loops
        let in_mult = &cam_coords.data;
        let out_mult = &mut world.data;

        for i in 0..in_mult.nrows() {
            let tmp = self.cache.pose_inv.transform_point(&Point3::new(
                in_mult[(i, 0)],
                in_mult[(i, 1)],
                in_mult[(i, 2)],
            ));
            for j in 0..3 {
                out_mult[(i, j)] = tmp[j];
            }
        }
        world
    }

    /// Convert rays in camera coordinates to world coordinates.
    pub fn ray_camera_to_world<BType, NPTS, StorageCamera>(
        &self,
        camera: &RayBundle<CameraFrame, BType, R, NPTS, StorageCamera>,
    ) -> RayBundle<WorldFrame, BType, R, NPTS, Owned<R, NPTS, U3>>
    where
        BType: Bundle<R>,
        NPTS: Dim,
        StorageCamera: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        camera.to_pose(self.cache.pose_inv)
    }

    /// Convert points in world coordinates to camera coordinates.
    pub fn world_to_camera<NPTS, InStorage>(
        &self,
        world: &Points<WorldFrame, R, NPTS, InStorage>,
    ) -> Points<CameraFrame, R, NPTS, Owned<R, NPTS, U3>>
    where
        NPTS: Dim,
        InStorage: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        let mut cam_coords = Points::new(MatrixMN::zeros_generic(
            NPTS::from_usize(world.data.nrows()),
            U3::from_usize(3),
        ));

        // Potential optimization: remove for loops
        let in_mult = &world.data;
        let out_mult = &mut cam_coords.data;

        for i in 0..in_mult.nrows() {
            let tmp = self.cache.pose.transform_point(&Point3::new(
                in_mult[(i, 0)],
                in_mult[(i, 1)],
                in_mult[(i, 2)],
            ));
            for j in 0..3 {
                out_mult[(i, j)] = tmp[j];
            }
        }
        cam_coords
    }
}

// So far, I could not figure out how to get serde derive to construct a cache
// only after rquat and camcenter are created. Instead serde derive wants the
// struct to be deserialized to implement the Default trait.
#[cfg(feature = "serde-serialize")]
impl<'de, R: RealField + serde::Deserialize<'de>> serde::Deserialize<'de>
    for ExtrinsicParameters<R>
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de;
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            RQuat,
            CamCenter,
        };

        struct ExtrinsicParametersVisitor<'de, R2: RealField + serde::Deserialize<'de>>(
            std::marker::PhantomData<&'de R2>,
        );

        impl<'de, R2: RealField + serde::Deserialize<'de>> serde::de::Visitor<'de>
            for ExtrinsicParametersVisitor<'de, R2>
        {
            type Value = ExtrinsicParameters<R2>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("struct ExtrinsicParameters")
            }

            fn visit_seq<V>(
                self,
                mut seq: V,
            ) -> std::result::Result<ExtrinsicParameters<R2>, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                let rquat = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let camcenter = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(ExtrinsicParameters::from_rotation_and_camcenter(
                    rquat, camcenter,
                ))
            }

            fn visit_map<V>(
                self,
                mut map: V,
            ) -> std::result::Result<ExtrinsicParameters<R2>, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut rquat = None;
                let mut camcenter = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::RQuat => {
                            if rquat.is_some() {
                                return Err(de::Error::duplicate_field("rquat"));
                            }
                            rquat = Some(map.next_value()?);
                        }
                        Field::CamCenter => {
                            if camcenter.is_some() {
                                return Err(de::Error::duplicate_field("camcenter"));
                            }
                            camcenter = Some(map.next_value()?);
                        }
                    }
                }
                let rquat = rquat.ok_or_else(|| de::Error::missing_field("rquat"))?;
                let camcenter = camcenter.ok_or_else(|| de::Error::missing_field("camcenter"))?;
                Ok(ExtrinsicParameters::from_rotation_and_camcenter(
                    rquat, camcenter,
                ))
            }
        }

        const FIELDS: &'static [&'static str] = &["rquat", "camcenter"];
        deserializer.deserialize_struct(
            "ExtrinsicParameters",
            FIELDS,
            ExtrinsicParametersVisitor(std::marker::PhantomData),
        )
    }
}

#[cfg(feature = "serde-serialize")]
fn _test_extrinsics_is_serialize() {
    // Compile-time test to ensure ExtrinsicParameters implements Serialize trait.
    fn implements<T: serde::Serialize>() {}
    implements::<ExtrinsicParameters<f64>>();
}

#[cfg(feature = "serde-serialize")]
fn _test_extrinsics_is_deserialize() {
    // Compile-time test to ensure ExtrinsicParameters implements Deserialize trait.
    fn implements<'de, T: serde::Deserialize<'de>>() {}
    implements::<ExtrinsicParameters<f64>>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::convert as c;

    #[test]
    fn roundtrip_f64() {
        roundtrip_generic::<f64>(1e-10)
    }

    #[test]
    fn roundtrip_f32() {
        roundtrip_generic::<f32>(1e-5)
    }

    fn roundtrip_generic<R: RealField>(epsilon: R) {
        let zero = convert(0.0);
        let one = convert(1.0);

        let e1 = ExtrinsicParameters::<R>::from_view(
            &Vector3::new(c(1.2), c(3.4), c(5.6)), // camcenter
            &Vector3::new(c(2.2), c(3.4), c(5.6)), // lookat
            &nalgebra::Unit::new_normalize(Vector3::new(zero, zero, one)), // up
        );

        let cam_coords = Points {
            coords: std::marker::PhantomData,
            data: MatrixMN::<R, U4, U3>::new(
                zero, zero, zero, // at camera center
                zero, zero, one, // one unit in +Z - exactly in camera direction
                one, zero, zero, // one unit in +X - right of camera axis
                zero, one, zero, // one unit in +Y - down from camera axis
            ),
        };

        #[rustfmt::skip]
        let world_expected = MatrixMN::<R, U4, U3>::new(
            c(1.2), c(3.4), c(5.6),
            c(2.2), c(3.4), c(5.6),
            c(1.2), c(2.4), c(5.6),
            c(1.2), c(3.4), c(4.6),
        );

        let world_actual = e1.camera_to_world(&cam_coords);
        approx::assert_abs_diff_eq!(world_expected, world_actual.data, epsilon = epsilon);

        // test roundtrip
        let camera_actual = e1.world_to_camera(&world_actual);
        approx::assert_abs_diff_eq!(cam_coords.data, camera_actual.data, epsilon = epsilon);
    }

    #[test]
    #[cfg(feature = "serde-serialize")]
    fn test_serde() {
        let expected = ExtrinsicParameters::<f64>::from_view(
            &Vector3::new(1.2, 3.4, 5.6),                                // camcenter
            &Vector3::new(2.2, 3.4, 5.6),                                // lookat
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)), // up
        );
        let buf = serde_json::to_string(&expected).unwrap();
        let actual: crate::ExtrinsicParameters<f64> = serde_json::from_str(&buf).unwrap();
        assert!(expected == actual);
    }
}

use nalgebra::{
    allocator::Allocator,
    base::storage::{Owned, Storage},
    convert, DefaultAllocator, Dim, OMatrix, RealField, U2, U3,
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::{
    coordinate_system::CameraFrame, Bundle, IntrinsicParameters, Pixels, Points, RayBundle,
};

use crate::ray_bundle_types::SharedDirectionRayBundle;

// TODO: implement ortho camera with near and far clipping?

/// Parameters defining the intrinsic part of an orthographic camera model.
///
/// These parameters describe the intrinsic parameters, the transformation from
/// camera coordinates to pixel coordinates, for an orthographic camera model.
/// For a full transformation from world coordinates to pixel coordinates, use a
/// [`Camera`](struct.Camera.html), which can be constructed with these intinsic
/// parameters and extrinsic parameters.
///
/// Read more about the [orthographic
/// projection](https://en.wikipedia.org/wiki/Orthographic_projection).
///
/// Can be converted into
/// [`IntrinsicParametersOrthographic`](struct.IntrinsicParametersOrthographic.html)
/// via the `.into()` method like so:
///
/// ```
/// use cam_geom::*;
/// let params = OrthographicParams {
///     sx: 100.0,
///     sy: 100.0,
///     cx: 640.0,
///     cy: 480.0,
/// };
/// let intrinsics: IntrinsicParametersOrthographic<_> = params.into();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct OrthographicParams<R: RealField> {
    /// Horizontal scale.
    pub sx: R,
    /// Vertical scale.
    pub sy: R,
    /// Horizontal component of image center.
    pub cx: R,
    /// Vertical component of image center.
    pub cy: R,
}

impl<R: RealField> From<OrthographicParams<R>> for IntrinsicParametersOrthographic<R> {
    #[inline]
    fn from(params: OrthographicParams<R>) -> Self {
        Self { params }
    }
}

/// An orthographic camera model. Implements [`IntrinsicParameters`](trait.IntrinsicParameters.html).
///
/// Create an `IntrinsicParametersOrthographic` as described for
/// [`OrthographicParams`](struct.OrthographicParams.html) by using `.into()`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct IntrinsicParametersOrthographic<R: RealField> {
    params: OrthographicParams<R>,
}

impl<R> IntrinsicParameters<R> for IntrinsicParametersOrthographic<R>
where
    R: RealField,
{
    type BundleType = SharedDirectionRayBundle<R>;

    fn pixel_to_camera<IN, NPTS>(
        &self,
        pixels: &Pixels<R, NPTS, IN>,
    ) -> RayBundle<CameraFrame, Self::BundleType, R, NPTS, Owned<R, NPTS, U3>>
    where
        Self::BundleType: Bundle<R>,
        IN: Storage<R, NPTS, U2>,
        NPTS: Dim,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        let zero = convert(0.0);

        // allocate zeros, fill later
        let mut result = RayBundle::new_shared_plusz_direction(OMatrix::zeros_generic(
            NPTS::from_usize(pixels.data.nrows()),
            U3::from_usize(3),
        ));

        let origin = &mut result.data;

        // It seems broadcasting is not (yet) supported in nalgebra, so we loop
        // through the data. See
        // https://discourse.nphysics.org/t/array-broadcasting-support/375/3 .

        for i in 0..pixels.data.nrows() {
            let u = pixels.data[(i, 0)];
            let v = pixels.data[(i, 1)];

            let x: R = (u - self.params.cx) / self.params.sx;
            let y: R = (v - self.params.cy) / self.params.sy;

            origin[(i, 0)] = x;
            origin[(i, 1)] = y;
            origin[(i, 2)] = zero;
        }
        result
    }

    fn camera_to_pixel<IN, NPTS>(
        &self,
        camera: &Points<CameraFrame, R, NPTS, IN>,
    ) -> Pixels<R, NPTS, Owned<R, NPTS, U2>>
    where
        IN: Storage<R, NPTS, U3>,
        NPTS: Dim,
        DefaultAllocator: Allocator<R, NPTS, U2>,
    {
        let mut result = Pixels::new(OMatrix::zeros_generic(
            NPTS::from_usize(camera.data.nrows()),
            U2::from_usize(2),
        ));

        // It seems broadcasting is not (yet) supported in nalgebra, so we loop
        // through the data. See
        // https://discourse.nphysics.org/t/array-broadcasting-support/375/3 .

        for i in 0..camera.data.nrows() {
            result.data[(i, 0)] = camera.data[(i, 0)] * self.params.sx + self.params.cx;
            result.data[(i, 1)] = camera.data[(i, 1)] * self.params.sy + self.params.cy;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::{IntrinsicParametersOrthographic, OrthographicParams};
    use crate::camera::{roundtrip_camera, Camera};
    use crate::extrinsics::ExtrinsicParameters;
    use crate::intrinsic_test_utils::roundtrip_intrinsics;

    #[test]
    fn roundtrip() {
        let params = OrthographicParams {
            sx: 100.0,
            sy: 102.0,
            cx: 321.0,
            cy: 239.9,
        };
        let cam: IntrinsicParametersOrthographic<_> = params.into();

        roundtrip_intrinsics(&cam, 640, 480, 5, 0, nalgebra::convert(1e-10));

        let extrinsics = ExtrinsicParameters::from_view(
            &Vector3::new(1.2, 3.4, 5.6),                                // camcenter
            &Vector3::new(2.2, 3.4, 5.6),                                // lookat
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)), // up
        );

        let full_cam = Camera::new(cam, extrinsics);
        roundtrip_camera(full_cam, 640, 480, 5, 0, nalgebra::convert(1e-10));
    }

    #[test]
    #[cfg(feature = "serde-serialize")]
    fn test_serde() {
        let params = OrthographicParams {
            sx: 100.0,
            sy: 102.0,
            cx: 321.0,
            cy: 239.9,
        };
        let expected: IntrinsicParametersOrthographic<_> = params.into();

        let buf = serde_json::to_string(&expected).unwrap();
        let actual: IntrinsicParametersOrthographic<f64> = serde_json::from_str(&buf).unwrap();
        assert!(expected == actual);
    }
}

use nalgebra::{
    allocator::Allocator,
    convert,
    storage::{Owned, Storage},
    DefaultAllocator, Dim, Matrix, OMatrix, RealField, SMatrix, SliceStorage, U1, U2, U3,
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::{
    coordinate_system::CameraFrame, Bundle, Error, IntrinsicParameters, Pixels, Points, RayBundle,
};

use crate::ray_bundle_types::SharedOriginRayBundle;

/// Parameters defining a pinhole perspective camera model.
///
/// These will be used to make the 3x4 intrinsic parameter matrix
/// ```text
/// [[fx, skew, cx, 0],
///  [ 0,   fy, cy, 0],
///  [ 0,    0,  1, 0]]
/// ```
///
/// These parameters describe the intrinsic parameters, the transformation from
/// camera coordinates to pixel coordinates, for a perspective camera model. For
/// a full transformation from world coordinates to pixel coordinates, use a
/// [`Camera`](struct.Camera.html), which can be constructed with these intinsic
/// parameters and extrinsic parameters.
///
/// Read more about the [pinhole perspective
/// projection](https://en.wikipedia.org/wiki/Pinhole_camera_model).
///
/// Can be converted into
/// [`IntrinsicParametersPerspective`](struct.IntrinsicParametersPerspective.html)
/// via the `.into()` method like so:
///
/// ```
/// use cam_geom::*;
/// let params = PerspectiveParams {
///     fx: 100.0,
///     fy: 100.0,
///     skew: 0.0,
///     cx: 640.0,
///     cy: 480.0,
/// };
/// let intrinsics: IntrinsicParametersPerspective<_> = params.into();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct PerspectiveParams<R: RealField> {
    /// Horizontal focal length.
    pub fx: R,
    /// Vertical focal length.
    pub fy: R,
    /// Skew between horizontal and vertical axes.
    pub skew: R,
    /// Horizontal component of the principal point.
    pub cx: R,
    /// Vertical component of the principal point.
    pub cy: R,
}

impl<R: RealField> From<PerspectiveParams<R>> for IntrinsicParametersPerspective<R> {
    fn from(params: PerspectiveParams<R>) -> Self {
        use nalgebra::convert as c;
        #[rustfmt::skip]
        let cache_p = nalgebra::SMatrix::<R,3,4>::new(
            params.fx.clone(), params.skew.clone(), params.cx.clone(), c(0.0),
             c(0.0),     params.fy.clone(), params.cy.clone(), c(0.0),
             c(0.0),        c(0.0), c(1.0), c(0.0),
        );
        Self { params, cache_p }
    }
}

/// A pinhole perspective camera model. Implements [`IntrinsicParameters`](trait.IntrinsicParameters.html).
///
/// Create an `IntrinsicParametersPerspective` as described for
/// [`PerspectiveParams`](struct.PerspectiveParams.html) by using `.into()`.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize))]
pub struct IntrinsicParametersPerspective<R: RealField> {
    params: PerspectiveParams<R>,
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) cache_p: SMatrix<R, 3, 4>,
}

impl<R: RealField> IntrinsicParametersPerspective<R> {
    /// Create a new instance given an intrinsic parameter matrix.
    ///
    /// Returns an error if the intrinsic parameter matrix is not normalized or
    /// otherwise does not represent a perspective camera model.
    pub fn from_normalized_3x4_matrix(p: SMatrix<R, 3, 4>) -> std::result::Result<Self, Error> {
        let params: PerspectiveParams<R> = PerspectiveParams {
            fx: p[(0, 0)].clone(),
            fy: p[(1, 1)].clone(),
            skew: p[(0, 1)].clone(),
            cx: p[(0, 2)].clone(),
            cy: p[(1, 2)].clone(),
        };
        if approx::relative_ne!(p[(0, 3)], nalgebra::convert(0.0)) {
            return Err(Error::InvalidInput);
        }

        if approx::relative_ne!(p[(1, 0)], nalgebra::convert(0.0)) {
            return Err(Error::InvalidInput);
        }

        if approx::relative_ne!(p[(1, 3)], nalgebra::convert(0.0)) {
            return Err(Error::InvalidInput);
        }

        if approx::relative_ne!(p[(2, 0)], nalgebra::convert(0.0)) {
            return Err(Error::InvalidInput);
        }

        if approx::relative_ne!(p[(2, 1)], nalgebra::convert(0.0)) {
            return Err(Error::InvalidInput);
        }

        if approx::relative_ne!(p[(2, 2)], nalgebra::convert(1.0)) {
            return Err(Error::InvalidInput); // camera matrix must be normalized
        }

        if approx::relative_ne!(p[(2, 3)], nalgebra::convert(0.0)) {
            return Err(Error::InvalidInput);
        }

        Ok(params.into())
    }

    /// Get X focal length
    #[inline]
    pub fn fx(&self) -> R {
        self.params.fx.clone()
    }

    /// Get Y focal length
    #[inline]
    pub fn fy(&self) -> R {
        self.params.fy.clone()
    }

    /// Get skew
    #[inline]
    pub fn skew(&self) -> R {
        self.params.skew.clone()
    }

    /// Get X center
    #[inline]
    pub fn cx(&self) -> R {
        self.params.cx.clone()
    }

    /// Get Y center
    #[inline]
    pub fn cy(&self) -> R {
        self.params.cy.clone()
    }

    /// Create a 3x3 projection matrix.
    #[inline]
    pub(crate) fn as_intrinsics_matrix<'a>(
        &'a self,
    ) -> Matrix<R, U3, U3, SliceStorage<'a, R, U3, U3, U1, U3>> {
        // TODO: implement similar functionality for orthographic camera and
        // make a new trait which exposes this functionality. Note that not all
        // intrinsic parameter implementations will be able to implement this
        // hypothetical new trait, because not all cameras are linear.
        self.cache_p.fixed_slice::<3, 3>(0, 0)
    }
}

impl<R> IntrinsicParameters<R> for IntrinsicParametersPerspective<R>
where
    R: RealField,
{
    type BundleType = SharedOriginRayBundle<R>;

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
        let one: R = convert(1.0);

        // allocate zeros, fill later
        let mut result = RayBundle::new_shared_zero_origin(OMatrix::zeros_generic(
            NPTS::from_usize(pixels.data.nrows()),
            U3::from_usize(3),
        ));

        let cam_dir = &mut result.data;

        // It seems broadcasting is not (yet) supported in nalgebra, so we loop
        // through the data. See
        // https://discourse.nphysics.org/t/array-broadcasting-support/375/3 .

        for i in 0..pixels.data.nrows() {
            let u = pixels.data[(i, 0)].clone();
            let v = pixels.data[(i, 1)].clone();

            // point in camcoords at distance 1.0 from image plane
            let y = (v - self.params.cy.clone()) / self.params.fy.clone();
            cam_dir[(i, 0)] = (u - self.params.skew.clone() * y.clone() - self.params.cx.clone())
                / self.params.fx.clone(); // x
            cam_dir[(i, 1)] = y;
            cam_dir[(i, 2)] = one.clone(); // z
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
            let x = nalgebra::Point3::new(
                camera.data[(i, 0)].clone(),
                camera.data[(i, 1)].clone(),
                camera.data[(i, 2)].clone(),
            )
            .to_homogeneous();
            let rst = self.cache_p.clone() * x;
            result.data[(i, 0)] = rst[0].clone() / rst[2].clone();
            result.data[(i, 1)] = rst[1].clone() / rst[2].clone();
        }
        result
    }
}

impl<R: RealField> std::fmt::Debug for IntrinsicParametersPerspective<R> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // This should match the auto derived Debug implementation but not print
        // the cache_p field.
        fmt.debug_struct("IntrinsicParametersPerspective")
            .field("params", &self.params)
            .finish()
    }
}

// See note about serde derive for ExtrinsicParameters Deserialize, which
// applies here, too.
#[cfg(feature = "serde-serialize")]
impl<'de, R: RealField + serde::Deserialize<'de>> serde::Deserialize<'de>
    for IntrinsicParametersPerspective<R>
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
            Params,
        };

        struct IntrinsicParametersPerspectiveVisitor<'de, R2: RealField + serde::Deserialize<'de>>(
            std::marker::PhantomData<&'de R2>,
        );

        impl<'de, R2: RealField + serde::Deserialize<'de>> serde::de::Visitor<'de>
            for IntrinsicParametersPerspectiveVisitor<'de, R2>
        {
            type Value = IntrinsicParametersPerspective<R2>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("struct IntrinsicParametersPerspective")
            }

            fn visit_seq<V>(
                self,
                mut seq: V,
            ) -> std::result::Result<IntrinsicParametersPerspective<R2>, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                let params: PerspectiveParams<_> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                Ok(IntrinsicParametersPerspective::from(params))
            }

            fn visit_map<V>(
                self,
                mut map: V,
            ) -> std::result::Result<IntrinsicParametersPerspective<R2>, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut params = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Params => {
                            if params.is_some() {
                                return Err(de::Error::duplicate_field("params"));
                            }
                            params = Some(map.next_value()?);
                        }
                    }
                }
                let params: PerspectiveParams<_> =
                    params.ok_or_else(|| de::Error::missing_field("params"))?;
                Ok(IntrinsicParametersPerspective::from(params))
            }
        }

        const FIELDS: &'static [&'static str] = &["params"];
        deserializer.deserialize_struct(
            "IntrinsicParametersPerspective",
            FIELDS,
            IntrinsicParametersPerspectiveVisitor(std::marker::PhantomData),
        )
    }
}

#[cfg(feature = "serde-serialize")]
fn _test_is_serialize() {
    // Compile-time test to ensure IntrinsicParametersPerspective implements Serialize trait.
    fn implements<T: serde::Serialize>() {}
    implements::<IntrinsicParametersPerspective<f64>>();
}

#[cfg(feature = "serde-serialize")]
fn _test_is_deserialize() {
    // Compile-time test to ensure IntrinsicParametersPerspective implements Deserialize trait.
    fn implements<'de, T: serde::Deserialize<'de>>() {}
    implements::<IntrinsicParametersPerspective<f64>>();
}

#[cfg(test)]
mod tests {
    use nalgebra::{SMatrix, Vector3};

    use super::{IntrinsicParametersPerspective, PerspectiveParams};
    use crate::camera::{roundtrip_camera, Camera};
    use crate::extrinsics::ExtrinsicParameters;
    use crate::intrinsic_test_utils::roundtrip_intrinsics;
    use crate::Points;

    #[test]
    fn roundtrip() {
        let params = PerspectiveParams {
            fx: 100.0,
            fy: 102.0,
            skew: 0.1,
            cx: 321.0,
            cy: 239.9,
        };

        let cam: IntrinsicParametersPerspective<_> = params.into();
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
    fn reject_invalid_projection_matrix() {
        #[rustfmt::skip]
        let p_valid = nalgebra::SMatrix::<f64,3,4>::new(
            10.0,  0.0, 0.0, 0.0,
             0.0, 10.0, 0.0, 0.0,
             0.0,  0.0, 1.0, 0.0,
        );
        assert!(IntrinsicParametersPerspective::from_normalized_3x4_matrix(p_valid).is_ok());

        let mut p = p_valid.clone();
        p[(2, 2)] = 1.1;
        assert!(IntrinsicParametersPerspective::from_normalized_3x4_matrix(p).is_err());

        let mut p = p_valid.clone();
        p[(0, 3)] = 1.1;
        assert!(IntrinsicParametersPerspective::from_normalized_3x4_matrix(p).is_err());

        let mut p = p_valid.clone();
        p[(1, 0)] = 1.1;
        assert!(IntrinsicParametersPerspective::from_normalized_3x4_matrix(p).is_err());

        let mut p = p_valid.clone();
        p[(1, 3)] = 1.1;
        assert!(IntrinsicParametersPerspective::from_normalized_3x4_matrix(p).is_err());

        let mut p = p_valid.clone();
        p[(2, 0)] = 1.1;
        assert!(IntrinsicParametersPerspective::from_normalized_3x4_matrix(p).is_err());

        let mut p = p_valid.clone();
        p[(2, 1)] = 1.1;
        assert!(IntrinsicParametersPerspective::from_normalized_3x4_matrix(p).is_err());

        let mut p = p_valid.clone();
        p[(2, 2)] = 1.1;
        assert!(IntrinsicParametersPerspective::from_normalized_3x4_matrix(p).is_err());
    }

    fn assert_is_pmat_same(
        cam: &Camera<f64, IntrinsicParametersPerspective<f64>>,
        pmat: &SMatrix<f64, 3, 4>,
    ) {
        let camcoord_pts = Points::new(SMatrix::<f64, 2, 3>::new(
            1.23, 4.56, 7.89, // pt 1
            1.0, 2.0, 3.0, // pt 2
        ));

        // Convert world to pixel using Camera method.
        let pts1 = cam.world_to_pixel(&camcoord_pts);

        for i in 0..pts1.data.nrows() {
            let pt1 = pts1.data.row(i);

            // Convert world to pixel using matrix multiply.
            let cc = camcoord_pts.data.row(i);
            let coords = nalgebra::Point3::new(cc[(0, 0)], cc[(0, 1)], cc[(0, 2)]);
            let cc = pmat * coords.to_homogeneous();
            let pt2 = SMatrix::<f64, 1, 2>::new(cc[0] / cc[2], cc[1] / cc[2]);

            approx::assert_abs_diff_eq!(pt1[0], pt2[0], epsilon = 1e-5);
            approx::assert_abs_diff_eq!(pt1[1], pt2[1], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_to_from_pmat() {
        for (name, cam) in &get_test_cameras() {
            println!("\n\n\ntesting camera {}", name);

            // Get camera matrix from this Camera instance and check it.
            let pmat = cam.as_camera_matrix();
            assert_is_pmat_same(&cam, &pmat);

            // Create a new Camera instance from this matrix and check it.
            let cam2 = Camera::from_perspective_matrix(&pmat).unwrap();
            let pmat2 = cam2.as_camera_matrix();
            assert_is_pmat_same(&cam2, &pmat);
            assert_is_pmat_same(&cam, &pmat2);
        }
    }

    fn get_test_cameras() -> Vec<(String, Camera<f64, IntrinsicParametersPerspective<f64>>)> {
        let mut result = Vec::new();

        // camera 1 - from perspective parameters
        let params = PerspectiveParams {
            fx: 100.0,
            fy: 102.0,
            skew: 0.1,
            cx: 321.0,
            cy: 239.9,
        };

        let cam: IntrinsicParametersPerspective<_> = params.into();
        roundtrip_intrinsics(&cam, 640, 480, 5, 0, nalgebra::convert(1e-10));

        let extrinsics = ExtrinsicParameters::from_view(
            &Vector3::new(1.2, 3.4, 5.6),                                // camcenter
            &Vector3::new(2.2, 3.4, 5.6),                                // lookat
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)), // up
        );

        let from_params = Camera::new(cam, extrinsics);
        result.push(("from-params".into(), from_params));

        // in the future - more cameras

        result
    }

    #[test]
    #[cfg(feature = "serde-serialize")]
    fn test_serde() {
        let params = PerspectiveParams {
            fx: 100.0,
            fy: 102.0,
            skew: 0.1,
            cx: 321.0,
            cy: 239.9,
        };

        let expected: IntrinsicParametersPerspective<_> = params.into();

        let buf = serde_json::to_string(&expected).unwrap();
        let actual: IntrinsicParametersPerspective<f64> = serde_json::from_str(&buf).unwrap();
        assert!(expected == actual);
        assert!(actual.fx() == 100.0);
        assert!(actual.fy() == 102.0);
        assert!(actual.skew() == 0.1);
        assert!(actual.cx() == 321.0);
        assert!(actual.cy() == 239.9);
    }
}

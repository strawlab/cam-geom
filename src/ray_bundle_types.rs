use nalgebra::{OMatrix, SMatrix};

use crate::*;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// A bundle of rays with the same arbitrary origin
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SharedOriginRayBundle<R: RealField> {
    center: Point3<R>,
}

impl<R> SharedOriginRayBundle<R>
where
    R: RealField,
{
    /// Create a new SharedOriginRayBundle with origin (center) at zero.
    #[inline]
    pub fn new_shared_zero_origin() -> Self {
        // center is at (0,0,0)
        let zero: R = nalgebra::convert(0.0);
        Self {
            center: Point3::new(zero.clone(), zero.clone(), zero),
        }
    }
}

impl<R> Bundle<R> for SharedOriginRayBundle<R>
where
    R: RealField,
{
    #[inline]
    fn to_single_ray<Coords>(&self, self_data: &SMatrix<R, 1, 3>) -> Ray<Coords, R>
    where
        Coords: CoordinateSystem,
    {
        Ray {
            direction: self_data.clone(),
            center: self.center.coords.transpose(),
            c: std::marker::PhantomData,
        }
    }

    fn directions<NPTS, StorageIn>(
        &self,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> Matrix<R, NPTS, U3, Owned<R, NPTS, U3>>
    where
        NPTS: nalgebra::DimName,
        StorageIn: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        // TODO: do this more smartly/efficiently
        let mut result = nalgebra::OMatrix::<R, NPTS, U3>::zeros();
        for i in 0..self_data.nrows() {
            for j in 0..3 {
                result[(i, j)] = self_data[(i, j)].clone();
            }
        }
        result
    }

    fn centers<NPTS, StorageIn>(
        &self,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> Matrix<R, NPTS, U3, Owned<R, NPTS, U3>>
    where
        NPTS: nalgebra::DimName,
        StorageIn: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        // TODO: do this more smartly/efficiently
        let mut result = nalgebra::OMatrix::<R, NPTS, U3>::zeros();
        for i in 0..self_data.nrows() {
            for j in 0..3 {
                result[(i, j)] = self.center[j].clone();
            }
        }
        result
    }

    fn point_on_ray<NPTS, StorageIn, OutFrame>(
        &self,
        directions: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> Points<OutFrame, R, NPTS, Owned<R, NPTS, U3>>
    where
        Self: Sized,
        NPTS: Dim,
        StorageIn: Storage<R, NPTS, U3>,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        let mut result = Points::new(OMatrix::zeros_generic(
            NPTS::from_usize(directions.nrows()),
            U3::from_usize(3),
        ));
        let center = [
            self.center[0].clone(),
            self.center[1].clone(),
            self.center[2].clone(),
        ];
        for i in 0..directions.nrows() {
            for j in 0..3 {
                result.data[(i, j)] = center[j].clone() + directions[(i, j)].clone();
            }
        }
        result
    }

    fn point_on_ray_at_distance<NPTS, StorageIn, OutFrame>(
        &self,
        directions: &Matrix<R, NPTS, U3, StorageIn>,
        distance: R,
    ) -> Points<OutFrame, R, NPTS, Owned<R, NPTS, U3>>
    where
        Self: Sized,
        NPTS: Dim,
        StorageIn: Storage<R, NPTS, U3>,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        let mut result = Points::new(OMatrix::zeros_generic(
            NPTS::from_usize(directions.nrows()),
            U3::from_usize(3),
        ));
        let center = [
            self.center[0].clone(),
            self.center[1].clone(),
            self.center[2].clone(),
        ];
        for i in 0..directions.nrows() {
            let dx = directions[(i, 0)].clone();
            let dy = directions[(i, 1)].clone();
            let dz = directions[(i, 2)].clone();
            let mag2 = dx.clone() * dx + dy.clone() * dy + dz.clone() * dz;
            let mag = mag2.sqrt();
            let scale = distance.clone() / mag;
            for j in 0..3 {
                result.data[(i, j)] =
                    center[j].clone() + scale.clone() * directions[(i, j)].clone();
            }
        }
        result
    }

    fn to_pose<NPTS, StorageIn, OutFrame>(
        &self,
        pose: Isometry3<R>,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> RayBundle<OutFrame, Self, R, NPTS, Owned<R, NPTS, U3>>
    where
        NPTS: Dim,
        StorageIn: Storage<R, NPTS, U3>,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        let bundle_type = Self::new_shared_zero_origin();
        let mut reposed = RayBundle::new(
            bundle_type,
            OMatrix::zeros_generic(NPTS::from_usize(self_data.nrows()), U3::from_usize(3)),
        );
        // transform single bundle center point
        let new_center = pose.transform_point(&self.center);

        // transform multiple bundle directions
        for i in 0..self_data.nrows() {
            let orig_vec = self_data.row(i).transpose();
            let new_vec = pose.transform_vector(&orig_vec);
            for j in 0..3 {
                reposed.data[(i, j)] = new_vec[j].clone();
            }
        }

        reposed.bundle_type = SharedOriginRayBundle { center: new_center };
        reposed
    }
}

/// A bundle of rays with the same arbitrary direction
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SharedDirectionRayBundle<R: RealField> {
    direction: Vector3<R>,
}

impl<R: RealField> SharedDirectionRayBundle<R> {
    /// Create a new SharedDirectionRayBundle with direction +z.
    pub fn new_plusz_shared_direction() -> Self {
        // in +z direction
        Self {
            direction: Vector3::new(R::zero(), R::zero(), R::one()),
        }
    }
}

impl<R: RealField> Bundle<R> for SharedDirectionRayBundle<R> {
    fn to_single_ray<Coords>(&self, self_data: &SMatrix<R, 1, 3>) -> Ray<Coords, R>
    where
        Coords: CoordinateSystem,
    {
        Ray {
            direction: self.direction.transpose(),
            center: self_data.clone(),
            c: std::marker::PhantomData,
        }
    }

    fn directions<NPTS, StorageIn>(
        &self,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> Matrix<R, NPTS, U3, Owned<R, NPTS, U3>>
    where
        NPTS: nalgebra::DimName,
        StorageIn: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        // TODO: do this more smartly/efficiently
        let mut result = nalgebra::OMatrix::<R, NPTS, U3>::zeros();
        for i in 0..self_data.nrows() {
            for j in 0..3 {
                result[(i, j)] = self.direction[j].clone();
            }
        }
        result
    }

    fn centers<NPTS, StorageIn>(
        &self,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> Matrix<R, NPTS, U3, Owned<R, NPTS, U3>>
    where
        NPTS: nalgebra::DimName,
        StorageIn: Storage<R, NPTS, U3>,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        // TODO: do this more smartly/efficiently
        let mut result = nalgebra::OMatrix::<R, NPTS, U3>::zeros();
        for i in 0..self_data.nrows() {
            for j in 0..3 {
                result[(i, j)] = self_data[(i, j)].clone();
            }
        }
        result
    }

    fn point_on_ray<NPTS, StorageIn, OutFrame>(
        &self,
        centers: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> Points<OutFrame, R, NPTS, Owned<R, NPTS, U3>>
    where
        Self: Sized,
        NPTS: Dim,
        StorageIn: Storage<R, NPTS, U3>,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        let mut result = Points::new(OMatrix::zeros_generic(
            NPTS::from_usize(centers.nrows()),
            U3::from_usize(3),
        ));
        let direction = [
            self.direction[0].clone(),
            self.direction[1].clone(),
            self.direction[2].clone(),
        ];
        for i in 0..centers.nrows() {
            for j in 0..3 {
                result.data[(i, j)] = direction[j].clone() + centers[(i, j)].clone();
            }
        }
        result
    }

    fn point_on_ray_at_distance<NPTS, StorageIn, OutFrame>(
        &self,
        centers: &Matrix<R, NPTS, U3, StorageIn>,
        distance: R,
    ) -> Points<OutFrame, R, NPTS, Owned<R, NPTS, U3>>
    where
        Self: Sized,
        NPTS: Dim,
        StorageIn: Storage<R, NPTS, U3>,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        let mut result = Points::new(OMatrix::zeros_generic(
            NPTS::from_usize(centers.nrows()),
            U3::from_usize(3),
        ));

        let d = &self.direction;
        let dx = d[0].clone();
        let dy = d[1].clone();
        let dz = d[2].clone();
        let mag2 = dx.clone() * dx.clone() + dy.clone() * dy.clone() + dz.clone() * dz.clone();
        let mag = mag2.sqrt();
        let scale = distance / mag;
        let dist_dir = Vector3::new(scale.clone() * dx, scale.clone() * dy, scale * dz);

        for i in 0..centers.nrows() {
            for j in 0..3 {
                result.data[(i, j)] = dist_dir[j].clone() + centers[(i, j)].clone();
            }
        }
        result
    }

    fn to_pose<NPTS, StorageIn, OutFrame>(
        &self,
        pose: Isometry3<R>,
        self_data: &Matrix<R, NPTS, U3, StorageIn>,
    ) -> RayBundle<OutFrame, Self, R, NPTS, Owned<R, NPTS, U3>>
    where
        NPTS: Dim,
        StorageIn: Storage<R, NPTS, U3>,
        OutFrame: CoordinateSystem,
        DefaultAllocator: Allocator<R, NPTS, U3>,
    {
        let bundle_type = Self::new_plusz_shared_direction();

        let mut reposed = RayBundle::new(
            bundle_type,
            OMatrix::zeros_generic(NPTS::from_usize(self_data.nrows()), U3::from_usize(3)),
        );

        // transform single bundle direction
        let new_direction = pose.transform_vector(&self.direction);

        // transform multiple bundle origins
        for i in 0..self_data.nrows() {
            let orig_point = Point3 {
                coords: self_data.row(i).transpose(),
            };
            let new_point = pose.transform_point(&orig_point);
            for j in 0..3 {
                reposed.data[(i, j)] = new_point[j].clone();
            }
        }

        reposed.bundle_type = SharedDirectionRayBundle {
            direction: new_direction,
        };
        reposed
    }
}

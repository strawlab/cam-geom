# Crate `cam_geom` for the [Rust language](https://www.rust-lang.org/)

<!-- Note: README.md is generated automatically by `cargo readme` -->

📷 📐 Geometric models of cameras for photogrammetry

![pinhole model
image](https://strawlab.org/assets/images/pinhole-model-ladybug.png)

(3D model by
[Adan](https://sketchfab.com/3d-models/lowpoly-lady-bug-90b59b5185b14c52944573f236eb7175),
[CC by 4.0](https://creativecommons.org/licenses/by/4.0/))

## About

The crate implements geometric models of cameras which may be useful for
[photogrammetry](https://en.wikipedia.org/wiki/Photogrammetry).

The crate provides a couple camera models, [the pinhole perspective
camera](https://en.wikipedia.org/wiki/Pinhole_camera_model) and the
[orthographic
camera](https://en.wikipedia.org/wiki/Orthographic_projection). Adding
another camera model entails implementing the
[`IntrinsicParameters`](trait.IntrinsicParameters.html) trait.

Also provided is the function
[`best_intersection_of_rays()`](fn.best_intersection_of_rays.html) which
determines the best 3D point corresponding to the intersection of multiple
rays. Thus, this crate is also useful for multiple view geometry.

Characteristics:

* Extensive use of static typing to ensure no unpleasant runtime surprises
  with coordinate system, matrix dimensions, and so on.
* Serialization and deserialization using [`serde`](https://docs.rs/serde).
  Enable with the `serde-serialize` cargo feature.
* Linear algebra and types from the [`nalgebra`](https://docs.rs/nalgebra)
  crate.
* Possible to create new camera models by implementing the
  [`IntrinsicParameters`](trait.IntrinsicParameters.html) trait. While the
  camera models implemented in this crate are linear, there is no
  requirement that implementations are linear. For example, it is
  anticipated that other implementations may exhibit
  [distortion](https://en.wikipedia.org/wiki/Distortion_(optics)).
* [`ExtrinsicParameters`](struct.ExtrinsicParameters.html) based on the
  [`nalgebra::Isometry3`](https://docs.rs/nalgebra/latest/nalgebra/geometry/type.Isometry3.html)
  type to handle the camera pose.
* No standard library is required (disable the default features to disable
  use of `std`) and no heap allocations. In other words, this can run on a
  bare-metal microcontroller with no OS.
* Extensive documentation and tests.
* Requires rust version 1.40 or greater.

## Testing

### Unit tests

To run all unit tests:

```
cargo test
```

### Test for `no_std`

Since the `thumbv7em-none-eabihf` target does not have `std` available, we
can build for it to check that our crate does not inadvertently pull in std.
The following will fail if a std dependency is present:

```
# install target with: "rustup target add thumbv7em-none-eabihf"
cargo build --no-default-features --target thumbv7em-none-eabihf
```

## Examples

### Example - projecting 3D world coordinates to 2D pixel coordinates.

```rust
use cam_geom::*;
use nalgebra::{Matrix2x3, Unit, Vector3};

// Create two points in the world coordinate frame.
let world_coords = Points::new(Matrix2x3::new(
    1.0, 0.0, 0.0, // point 1
    0.0, 1.0, 0.0, // point 2
));

// perepective parameters - focal length of 100, no skew, pixel center at (640,480)
let intrinsics = IntrinsicParametersPerspective::from(PerspectiveParams {
    fx: 100.0,
    fy: 100.0,
    skew: 0.0,
    cx: 640.0,
    cy: 480.0,
});

// Set extrinsic parameters - camera at (10,0,0), looing at (0,0,0), up (0,0,1)
let camcenter = Vector3::new(10.0, 0.0, 0.0);
let lookat = Vector3::new(0.0, 0.0, 0.0);
let up = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
let pose = ExtrinsicParameters::from_view(&camcenter, &lookat, &up);

// Create a `Camera` with both intrinsic and extrinsic parameters.
let camera = Camera::new(intrinsics, pose);

// Project the original 3D coordinates to 2D pixel coordinates.
let pixel_coords = camera.world_to_pixel(&world_coords);

// Print the results.
for i in 0..world_coords.data.nrows() {
    let wc = world_coords.data.row(i);
    let pix = pixel_coords.data.row(i);
    println!("{} -> {}", wc, pix);
}
```

This will print:

```
  ┌       ┐
  │ 1 0 0 │
  └       ┘

 ->
  ┌         ┐
  │ 640 480 │
  └         ┘



  ┌       ┐
  │ 0 1 0 │
  └       ┘

 ->
  ┌         ┐
  │ 650 480 │
  └         ┘
```


### Example - intersection of rays

```rust
use cam_geom::*;
use nalgebra::RowVector3;

// Create the first ray.
    let ray1 = Ray::<WorldFrame, _>::new(
    RowVector3::new(1.0, 0.0, 0.0), // origin
    RowVector3::new(0.0, 1.0, 0.0), // direction
);

// Create the second ray.
let ray2 = Ray::<WorldFrame, _>::new(
    RowVector3::new(0.0, 1.0, 0.0), // origin
    RowVector3::new(1.0, 0.0, 0.0), // direction
);

// Compute the best intersection.
let result = best_intersection_of_rays(&[ray1, ray2]).unwrap();

// Print the result.
println!("result: {}", result.data);
```

This will print:

```
result:
  ┌       ┐
  │ 1 1 0 │
  └       ┘
```

## Regenerate `README.md`

The `README.md` file can be regenerated with:

```text
cargo readme > README.md
```

## Code of conduct

Anyone who interacts with this software in any space, including but not limited
to this GitHub repository, must follow our [code of
conduct](code_of_conduct.md).

## License

Licensed under either of these:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   https://opensource.org/licenses/MIT)

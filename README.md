[![Crates.io](https://img.shields.io/crates/v/cam-geom.svg)](https://crates.io/crates/cam-geom)
[![Documentation](https://docs.rs/cam-geom/badge.svg)](https://docs.rs/cam-geom/)
[![Crate License](https://img.shields.io/crates/l/cam-geom.svg)](https://crates.io/crates/cam-geom)
[![Dependency status](https://deps.rs/repo/github/strawlab/cam-geom/status.svg)](https://deps.rs/repo/github/strawlab/cam-geom)
[![build](https://github.com/strawlab/cam-geom/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/strawlab/cam-geom/actions?query=branch%3Amain)

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
[`IntrinsicParameters`](trait.IntrinsicParameters.html) trait. See the
[`opencv_ros_camera`](https://crates.io/crates/opencv-ros-camera) crate
for one example.

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
  requirement that implementations are linear. For example, the
  [`opencv_ros_camera`](https://crates.io/crates/opencv-ros-camera) crate
  exhibits [distortion](https://en.wikipedia.org/wiki/Distortion_(optics)).
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

To run the basic unit tests:

```text
cargo test
```

To run all unit tests:

```text
cargo test --features serde-serialize
```

### Test for `no_std`

Since the `thumbv7em-none-eabihf` target does not have `std` available, we
can build for it to check that our crate does not inadvertently pull in
std. The unit tests require std, so cannot be run on a `no_std` platform.
The following will fail if a std dependency is present:

```text
# install target with: "rustup target add thumbv7em-none-eabihf"
cargo build --no-default-features --target thumbv7em-none-eabihf
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

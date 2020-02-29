# Crate `cam_geom` for the [Rust language](https://www.rust-lang.org/)

<!-- Note: README.md is generated automatically by `cargo readme` -->

[![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl]

[ci]: https://img.shields.io/crates/v/cam-geom.svg
[cl]: https://crates.io/crates/cam-geom/

[li]: https://img.shields.io/crates/l/cam-geom.svg?maxAge=2592000

[di]: https://docs.rs/cam-geom/badge.svg
[dl]: https://docs.rs/cam-geom/

{{readme}}

## Altenatives

You may also be interested in
[rust-cv/cv-core](https://github.com/rust-cv/cv-core), which also contains
camera models for photogrammetry. The two crates were developed independently
without knowledge of each other. There are some similarities, such as being
built on nalgebra and having a goal of being no_std and no allocation
compatible. As of cv-core 0.7.3 and cam-geom 0.1.3, the following differences
exist between the camera models:

- cam-geom has an IntrinsicParameters trait and can support different camera
models. The perspective camera model looks identical in terms of
parameterization between the two. cam-geom also has the orthographic model.

- cam-geom can handle transformations of multiple points within a single
function call. Hopefully this will allow vectorized math over many points in
the inner loop, perhaps even automatically using SIMD by the compiler. So far,
however, no such benchmarking or work in this direction has been done.

- cam-geom supports the use serde for serialization of cameras.

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

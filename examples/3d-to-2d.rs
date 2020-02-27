// This is cut-and-pasted from the example in the docs of src/lib.rs.

fn main() {
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
}

// This is cut-and-pasted from the example in the docs of src/lib.rs.

fn main() {
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
}

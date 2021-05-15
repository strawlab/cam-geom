use cam_geom::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{convert, Dynamic, RealField, VecStorage, Vector3, U2};

fn generate_uv_raw<R: RealField>(
    width: usize,
    height: usize,
    step: usize,
    border: usize,
) -> Pixels<R, Dynamic, VecStorage<R, Dynamic, U2>> {
    let mut uv_raws: Vec<[R; 2]> = Vec::new();
    for row in num_iter::range_step(border, height - border, step) {
        for col in num_iter::range_step(border, width - border, step) {
            uv_raws.push([convert(col as f64), convert(row as f64)]);
        }
    }

    let mut data = nalgebra::OMatrix::<R, Dynamic, U2>::from_element(uv_raws.len(), convert(0.0));
    for i in 0..uv_raws.len() {
        for j in 0..2 {
            data[(i, j)] = uv_raws[i][j];
        }
    }
    Pixels { data }
}

fn criterion_benchmark(c: &mut Criterion) {
    let params = PerspectiveParams {
        fx: 100.0,
        fy: 102.0,
        skew: 0.1,
        cx: 321.0,
        cy: 239.9,
    };

    let cam: IntrinsicParametersPerspective<_> = params.into();
    // roundtrip_intrinsics(&cam, 640, 480, 5, 0, nalgebra::convert(1e-10));

    let extrinsics = ExtrinsicParameters::from_view(
        &Vector3::new(1.2, 3.4, 5.6),                                // camcenter
        &Vector3::new(2.2, 3.4, 5.6),                                // lookat
        &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)), // up
    );

    let full_cam = Camera::new(cam, extrinsics);

    let width = 640;
    let height = 480;
    let step = 5;
    let border = 0;
    let pixels = generate_uv_raw(width, height, step, border);

    let world_coords = full_cam.pixel_to_world(&pixels);
    let world_coords_points = world_coords.point_on_ray();
    println!("{} points", world_coords_points.len());

    c.bench_function("world_to_camera", |b| {
        b.iter(|| {
            full_cam
                .extrinsics()
                .world_to_camera(black_box(&world_coords_points))
        });
    });

    c.bench_function("perspective world_to_pixel", |b| {
        b.iter(|| full_cam.world_to_pixel(black_box(&world_coords_points)));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

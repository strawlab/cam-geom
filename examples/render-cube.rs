//! This example renders a cube to an SVG file using both a perspective camera
//! and an orthographic camera.

use cam_geom::*;
use nalgebra::{
    allocator::Allocator, storage::Storage, Const, DefaultAllocator, Dim, Matrix, SMatrix, Unit,
    Vector3,
};

/// Create a perspective camera.
fn get_perspective_cam() -> Camera<f64, IntrinsicParametersPerspective<f64>> {
    // Set intrinsic parameters
    let intrinsics = PerspectiveParams {
        fx: 100.0,
        fy: 100.0,
        skew: 0.0,
        cx: 640.0,
        cy: 480.0,
    };

    // Set extrinsic parameters.
    let camcenter = Vector3::new(10.0, 3.0, 5.0);
    let lookat = Vector3::new(0.0, 0.0, 0.0);
    let up = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
    let pose = ExtrinsicParameters::from_view(&camcenter, &lookat, &up);

    // Create camera with both intrinsic and extrinsic parameters.
    Camera::new(intrinsics.into(), pose)
}

/// Create an orthographic camera.
fn get_ortho_cam() -> Camera<f64, IntrinsicParametersOrthographic<f64>> {
    let intrinsics = OrthographicParams {
        sx: 100.0,
        sy: 102.0,
        cx: 321.0,
        cy: 239.9,
    };

    // Set extrinsic parameters.
    let camcenter = Vector3::new(10.0, 3.0, 5.0);
    let lookat = Vector3::new(0.0, 0.0, 0.0);
    let up = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
    let pose = ExtrinsicParameters::from_view(&camcenter, &lookat, &up);

    // Create camera with both intrinsic and extrinsic parameters.
    Camera::new(intrinsics.into(), pose)
}

/// A simple SVG file writer
struct SvgWriter {
    segs: Vec<((f64, f64), (f64, f64))>,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
}

impl SvgWriter {
    fn new() -> Self {
        Self {
            xmin: std::f64::INFINITY,
            xmax: -std::f64::INFINITY,
            ymin: std::f64::INFINITY,
            ymax: -std::f64::INFINITY,
            segs: Vec::new(),
        }
    }
    fn add_edge<S>(
        &mut self,
        pt0: &Matrix<f64, Const<1>, Const<2>, S>,
        pt1: &Matrix<f64, Const<1>, Const<2>, S>,
    ) where
        S: Storage<f64, Const<1>, Const<2>>,
    {
        self.xmin = self.xmin.min(pt0[0]);
        self.xmin = self.xmin.min(pt1[0]);

        self.ymin = self.ymin.min(pt0[1]);
        self.ymin = self.ymin.min(pt1[1]);

        self.xmax = self.xmax.max(pt0[0]);
        self.xmax = self.xmax.max(pt1[0]);

        self.ymax = self.ymax.max(pt0[1]);
        self.ymax = self.ymax.max(pt1[1]);

        self.segs.push(((pt0[0], pt0[1]), (pt1[0], pt1[1])));
    }
    fn save(&self, fname: &str) -> Result<(), std::io::Error> {
        use std::io::prelude::*;

        let header = "<svg version=\"1.1\" \
     baseProfile=\"full\" \
     width=\"300\" height=\"200\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n";
        let footer = "</svg>\n";

        let width = 300.0;
        let height = 200.0;
        let border = 5.0;
        let mut xscale = (width - 2.0 * border) / (self.xmax - self.xmin);
        let mut yscale = (height - 2.0 * border) / (self.ymax - self.ymin);

        // Keep aspect ratio equal for x and y dimensions.
        if xscale > yscale {
            xscale = yscale;
        } else {
            yscale = xscale;
        }

        let xoffset = -self.xmin * xscale + border;
        let yoffset = -self.ymin * yscale + border;

        let mut file = std::fs::File::create(fname)?;

        file.write_all(header.as_bytes())?;

        let radius = border;
        let stroke_width = 2.0;

        for seg in &self.segs {
            let x1 = (seg.0).0 * xscale + xoffset;
            let x2 = (seg.1).0 * xscale + xoffset;
            let y1 = (seg.0).1 * yscale + yoffset;
            let y2 = (seg.1).1 * yscale + yoffset;

            let buf = format!("<line x1=\"{}\" x2=\"{}\" y1=\"{}\" y2=\"{}\" stroke=\"orange\" stroke-width=\"{}\"/>",
                x1, x2, y1, y2, stroke_width);
            file.write_all(buf.as_bytes())?;
            let buf = format!(
                "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"green\" />\n",
                x1, y1, radius
            );
            file.write_all(buf.as_bytes())?;
            let buf = format!(
                "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"green\" />\n",
                x2, y2, radius
            );
            file.write_all(buf.as_bytes())?;
        }
        file.write_all(footer.as_bytes())?;
        Ok(())
    }
}

// Save a wireframe rendering of the vertices and edges to an SVG file.
fn render_wireframe<NPTS, I, S>(
    verts: &Points<WorldFrame, f64, NPTS, S>,
    edges: &[(usize, usize)],
    cam: &Camera<f64, I>,
    fname: &str,
) -> Result<(), std::io::Error>
where
    NPTS: Dim,
    I: IntrinsicParameters<f64>,
    S: Storage<f64, NPTS, Const<3>>,
    DefaultAllocator: Allocator<f64, NPTS, Const<3>>,
    DefaultAllocator: Allocator<f64, NPTS, Const<2>>,
{
    // Project the original 3D coordinates to 2D pixel coordinates.
    let pixel_coords = cam.world_to_pixel(&verts);

    let mut wtr = SvgWriter::new();

    for edge in edges {
        let (i0, i1) = edge;
        let pt0 = pixel_coords.data.row(*i0);
        let pt1 = pixel_coords.data.row(*i1);
        wtr.add_edge(&pt0, &pt1);
    }
    wtr.save(fname)?;

    Ok(())
}

fn main() -> Result<(), std::io::Error> {
    // Create cube vertices in the world coordinate frame.
    let world_coords = Points::<WorldFrame, _, _, _>::new(SMatrix::<f64, 8, 3>::from_row_slice(&[
        -1.0, -1.0, -1.0, // v1
        1.0, -1.0, -1.0, // v2
        1.0, 1.0, -1.0, // v3
        -1.0, 1.0, -1.0, // v4
        -1.0, -1.0, 1.0, // v5
        1.0, -1.0, 1.0, // v6
        1.0, 1.0, 1.0, // v7
        -1.0, 1.0, 1.0, // v8
    ]));
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    let cam = get_perspective_cam();
    render_wireframe(&world_coords, &edges, &cam, "cube-perspective.svg")?;

    let cam = get_ortho_cam();
    render_wireframe(&world_coords, &edges, &cam, "cube-ortho.svg")?;

    Ok(())
}

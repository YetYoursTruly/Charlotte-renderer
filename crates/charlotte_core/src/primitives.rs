use crate::{MeshData, Vertex};

/// UV sphere centred at the origin.
pub fn uv_sphere(radius: f32, segments: u32, rings: u32) -> MeshData {
    let segments = segments.max(3);
    let rings = rings.max(2);
    let mut vertices = Vec::new();
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        for s in 0..=segments {
            let u = s as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let n = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
            vertices.push(Vertex {
                position: [n[0] * radius, n[1] * radius, n[2] * radius],
                normal: n,
                tangent: [0.0; 4],
                uv: [u, v],
            });
        }
    }
    let mut indices = Vec::new();
    let stride = segments + 1;
    for r in 0..rings {
        for s in 0..segments {
            let a = r * stride + s;
            let b = a + stride;
            indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    let mut mesh = MeshData {
        name: "sphere".into(),
        vertices,
        indices,
    };
    mesh.compute_tangents();
    mesh
}

/// Flat square in the XZ plane facing +Y. UVs repeat `uv_tiles` times.
pub fn plane(half_size: f32, uv_tiles: f32) -> MeshData {
    let h = half_size;
    let v = |x: f32, z: f32, u: f32, w: f32| Vertex {
        position: [x, 0.0, z],
        normal: [0.0, 1.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv: [u, w],
    };
    MeshData {
        name: "plane".into(),
        vertices: vec![
            v(-h, -h, 0.0, 0.0),
            v(h, -h, uv_tiles, 0.0),
            v(h, h, uv_tiles, uv_tiles),
            v(-h, h, 0.0, uv_tiles),
        ],
        // Counter-clockwise seen from +Y.
        indices: vec![0, 2, 1, 0, 3, 2],
    }
}

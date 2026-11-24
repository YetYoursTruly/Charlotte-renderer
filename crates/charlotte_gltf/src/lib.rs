//! Loads `.gltf` / `.glb` files (as exported by Blender) into a [`charlotte_core::Scene`].
//!
//! Supported: meshes (triangles), metallic-roughness materials with base colour / metal-rough /
//! normal / occlusion / emissive textures, `KHR_materials_emissive_strength`, node hierarchy, and
//! `KHR_lights_punctual` (sun / point / spot lights).
//!
//! Not supported yet: skinning, morph targets, cameras, animation, texture transforms, UV sets
//! other than 0, sparse accessors on colour data, KHR_materials_* other than emissive strength.

use std::path::Path;

use anyhow::{bail, Context, Result};
use charlotte_core::glam::{Mat4, Vec3, Vec4};
use charlotte_core::{
    AlphaMode, ImageData, Light, LightKind, Material, MeshData, MeshInstance, Scene, Vertex,
};

pub fn load_gltf(path: impl AsRef<Path>) -> Result<Scene> {
    let path = path.as_ref();
    let (doc, buffers, images) =
        gltf::import(path).with_context(|| format!("importing {}", path.display()))?;
    let mut scene = Scene::default();

    for (i, img) in images.iter().enumerate() {
        scene.images.push(convert_image(i, img));
    }
    for m in doc.materials() {
        scene.materials.push(convert_material(&m));
    }

    // glTF mesh index -> list of (scene mesh index, scene material index)
    let mut mesh_map: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut default_material: Option<usize> = None;
    for mesh in doc.meshes() {
        let mut prims = Vec::new();
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                log::warn!("skipping non-triangle primitive in mesh {:?}", mesh.name());
                continue;
            }
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            let uvs: Option<Vec<[f32; 2]>> =
                reader.read_tex_coords(0).map(|t| t.into_f32().collect());
            let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|t| t.collect());
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            let vertices: Vec<Vertex> = positions
                .iter()
                .enumerate()
                .map(|(i, p)| Vertex {
                    position: *p,
                    normal: normals.as_ref().map_or([0.0, 1.0, 0.0], |n| n[i]),
                    tangent: tangents.as_ref().map_or([0.0; 4], |t| t[i]),
                    uv: uvs.as_ref().map_or([0.0; 2], |u| u[i]),
                })
                .collect();

            let mut data = MeshData {
                name: mesh.name().unwrap_or("mesh").to_string(),
                vertices,
                indices,
            };
            if normals.is_none() {
                data.make_flat_normals();
            }
            if tangents.is_none() && uvs.is_some() {
                data.compute_tangents();
            }

            let material = match prim.material().index() {
                Some(i) => i,
                None => *default_material.get_or_insert_with(|| {
                    scene.materials.push(Material::default());
                    scene.materials.len() - 1
                }),
            };
            scene.meshes.push(data);
            prims.push((scene.meshes.len() - 1, material));
        }
        mesh_map.push(prims);
    }

    let gscene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .context("glTF file contains no scene")?;
    for node in gscene.nodes() {
        visit(&node, Mat4::IDENTITY, &mesh_map, &mut scene);
    }
    if scene.instances.is_empty() {
        bail!("glTF scene has no renderable meshes");
    }
    log::info!(
        "loaded {}: {} meshes, {} instances, {} materials, {} images, {} lights",
        path.display(),
        scene.meshes.len(),
        scene.instances.len(),
        scene.materials.len(),
        scene.images.len(),
        scene.lights.len()
    );
    Ok(scene)
}

fn visit(node: &gltf::Node, parent: Mat4, mesh_map: &[Vec<(usize, usize)>], scene: &mut Scene) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        for &(m, material) in &mesh_map[mesh.index()] {
            scene.instances.push(MeshInstance {
                mesh: m,
                material,
                transform: world,
            });
        }
    }
    if let Some(light) = node.light() {
        use gltf::khr_lights_punctual::Kind;
        let kind = match light.kind() {
            Kind::Directional => LightKind::Directional,
            Kind::Point => LightKind::Point,
            Kind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => LightKind::Spot {
                inner_cone: inner_cone_angle,
                outer_cone: outer_cone_angle,
            },
        };
        scene.lights.push(Light {
            kind,
            color: Vec3::from(light.color()),
            intensity: light.intensity(),
            range: light.range(),
            position: world.transform_point3(Vec3::ZERO),
            direction: world.transform_vector3(Vec3::NEG_Z).normalize_or_zero(),
        });
    }
    for child in node.children() {
        visit(&child, world, mesh_map, scene);
    }
}

fn convert_material(m: &gltf::Material) -> Material {
    let pbr = m.pbr_metallic_roughness();
    let mut mat = Material {
        name: m.name().unwrap_or("material").to_string(),
        base_color: Vec4::from(pbr.base_color_factor()),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emissive: Vec3::from(m.emissive_factor()) * m.emissive_strength().unwrap_or(1.0),
        alpha_mode: match m.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        double_sided: m.double_sided(),
        ..Default::default()
    };
    mat.base_color_image = pbr
        .base_color_texture()
        .map(|t| t.texture().source().index());
    mat.metallic_roughness_image = pbr
        .metallic_roughness_texture()
        .map(|t| t.texture().source().index());
    mat.emissive_image = m.emissive_texture().map(|t| t.texture().source().index());
    if let Some(n) = m.normal_texture() {
        mat.normal_image = Some(n.texture().source().index());
        mat.normal_scale = n.scale();
    }
    if let Some(o) = m.occlusion_texture() {
        mat.occlusion_image = Some(o.texture().source().index());
        mat.occlusion_strength = o.strength();
    }
    mat
}

fn convert_image(index: usize, img: &gltf::image::Data) -> ImageData {
    use gltf::image::Format;
    let n = (img.width * img.height) as usize;
    let mut rgba = Vec::with_capacity(n * 4);
    match img.format {
        Format::R8G8B8A8 => rgba.extend_from_slice(&img.pixels),
        Format::R8G8B8 => {
            for p in img.pixels.chunks_exact(3) {
                rgba.extend_from_slice(&[p[0], p[1], p[2], 255]);
            }
        }
        Format::R8G8 => {
            for p in img.pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[p[0], p[1], 0, 255]);
            }
        }
        Format::R8 => {
            for &p in &img.pixels {
                rgba.extend_from_slice(&[p, p, p, 255]);
            }
        }
        // 16-bit formats: keep the high byte of each little-endian channel.
        Format::R16G16B16A16 => {
            for p in img.pixels.chunks_exact(8) {
                rgba.extend_from_slice(&[p[1], p[3], p[5], p[7]]);
            }
        }
        Format::R16G16B16 => {
            for p in img.pixels.chunks_exact(6) {
                rgba.extend_from_slice(&[p[1], p[3], p[5], 255]);
            }
        }
        Format::R16G16 => {
            for p in img.pixels.chunks_exact(4) {
                rgba.extend_from_slice(&[p[1], p[3], 0, 255]);
            }
        }
        Format::R16 => {
            for p in img.pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[p[1], p[1], p[1], 255]);
            }
        }
        _ => {
            log::warn!("unsupported float image format, using white");
            rgba.resize(n * 4, 255);
        }
    }
    ImageData {
        name: format!("image{index}"),
        width: img.width,
        height: img.height,
        rgba8: rgba,
    }
}

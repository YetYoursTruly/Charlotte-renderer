use std::path::Path;

use anyhow::{bail, Context, Result};
use glam::{Mat4, Vec3, Vec4};
use image::{DynamicImage, ImageFormat};

/// 48-byte vertex, identical layout on the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// xyz = tangent, w = bitangent sign. w == 0 means "no tangent" (normal mapping is skipped).
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, Default)]
pub struct MeshData {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Displaces existing vertices along their normals from a linear height image.
    /// Mesh density is unchanged; subdivide the mesh first for fine detail.
    pub fn displace_from_height(&mut self, image: &ImageData, scale: f32) {
        if image.width == 0 || image.height == 0 || scale == 0.0 {
            return;
        }
        let sample = |uv: [f32; 2]| {
            let x = (uv[0].rem_euclid(1.0) * (image.width - 1) as f32).round() as u32;
            let y = (uv[1].rem_euclid(1.0) * (image.height - 1) as f32).round() as u32;
            let i = ((y * image.width + x) * 4) as usize;
            (image.rgba8[i] as f32 / 255.0) * 0.2126
                + (image.rgba8[i + 1] as f32 / 255.0) * 0.7152
                + (image.rgba8[i + 2] as f32 / 255.0) * 0.0722
        };
        for vertex in &mut self.vertices {
            let height = sample(vertex.uv) - 0.5;
            let normal = Vec3::from(vertex.normal).normalize_or_zero();
            let position = Vec3::from(vertex.position) + normal * height * scale;
            vertex.position = position.to_array();
        }
        let mut normals = vec![Vec3::ZERO; self.vertices.len()];
        for tri in self.indices.chunks_exact(3) {
            let a = tri[0] as usize;
            let b = tri[1] as usize;
            let c = tri[2] as usize;
            let p0 = Vec3::from(self.vertices[a].position);
            let p1 = Vec3::from(self.vertices[b].position);
            let p2 = Vec3::from(self.vertices[c].position);
            let n = (p1 - p0).cross(p2 - p0);
            normals[a] += n;
            normals[b] += n;
            normals[c] += n;
        }
        for (vertex, normal) in self.vertices.iter_mut().zip(normals) {
            if normal.length_squared() > 1e-12 {
                vertex.normal = normal.normalize().to_array();
            }
        }
        self.compute_tangents();
    }

    /// Replaces the mesh with an un-indexed copy that has flat face normals.
    pub fn make_flat_normals(&mut self) {
        let mut out = Vec::with_capacity(self.indices.len());
        for tri in self.indices.chunks_exact(3) {
            let v = [
                self.vertices[tri[0] as usize],
                self.vertices[tri[1] as usize],
                self.vertices[tri[2] as usize],
            ];
            let p0 = Vec3::from(v[0].position);
            let n = (Vec3::from(v[1].position) - p0)
                .cross(Vec3::from(v[2].position) - p0)
                .normalize_or_zero();
            for mut vert in v {
                vert.normal = n.to_array();
                out.push(vert);
            }
        }
        self.indices = (0..out.len() as u32).collect();
        self.vertices = out;
    }

    /// Generates tangents from UVs using the glTF convention (v origin at the top).
    pub fn compute_tangents(&mut self) {
        let n = self.vertices.len();
        let mut tan = vec![Vec3::ZERO; n];
        let mut bit = vec![Vec3::ZERO; n];
        for tri in self.indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let (v0, v1, v2) = (&self.vertices[i0], &self.vertices[i1], &self.vertices[i2]);
            let p0 = Vec3::from(v0.position);
            let e1 = Vec3::from(v1.position) - p0;
            let e2 = Vec3::from(v2.position) - p0;
            let du1 = v1.uv[0] - v0.uv[0];
            let du2 = v2.uv[0] - v0.uv[0];
            // Flip v: glTF uv origin is top-left, normal maps are +Y up.
            let dv1 = -(v1.uv[1] - v0.uv[1]);
            let dv2 = -(v2.uv[1] - v0.uv[1]);
            let det = du1 * dv2 - du2 * dv1;
            if det.abs() < 1e-12 {
                continue;
            }
            let r = 1.0 / det;
            let t = (e1 * dv2 - e2 * dv1) * r;
            let b = (e2 * du1 - e1 * du2) * r;
            for i in [i0, i1, i2] {
                tan[i] += t;
                bit[i] += b;
            }
        }
        for (i, v) in self.vertices.iter_mut().enumerate() {
            let nrm = Vec3::from(v.normal);
            let t = tan[i] - nrm * nrm.dot(tan[i]);
            if t.length_squared() < 1e-20 {
                v.tangent = [0.0; 4];
                continue;
            }
            let t = t.normalize();
            let w = if nrm.cross(t).dot(bit[i]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            v.tangent = [t.x, t.y, t.z, w];
        }
    }
}

/// Decoded 8-bit RGBA image. Whether it is sRGB or linear is decided by how a material uses it.
#[derive(Clone, Debug)]
pub struct ImageData {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Linear-light equirectangular environment used for the sky and image-based lighting.
/// Pixels are RGBA32F and capped to 2048x1024 for predictable memory and upload cost.
#[derive(Clone, Debug)]
pub struct EnvironmentMap {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<[f32; 4]>,
    /// Radiance spherical-harmonic coefficients (L0..L2), integrated over solid angle.
    pub diffuse_sh: [[f32; 4]; 9],
}

impl EnvironmentMap {
    pub fn from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if extension != "exr" && extension != "hdr" {
            bail!(
                "environment must be an OpenEXR (.exr) or Radiance HDR (.hdr) image: {}",
                path.display()
            );
        }
        let image =
            image::open(path).with_context(|| format!("loading environment {}", path.display()))?;
        Self::from_image(image, path.display().to_string())
    }

    pub fn from_exr_bytes(bytes: &[u8], name: impl Into<String>) -> Result<Self> {
        let image = image::load_from_memory_with_format(bytes, ImageFormat::OpenExr)
            .context("decoding bundled OpenEXR environment")?;
        Self::from_image(image, name.into())
    }

    fn from_image(image: DynamicImage, name: String) -> Result<Self> {
        let mut rgba = image.into_rgba32f();
        let (mut width, mut height) = rgba.dimensions();
        if width == 0 || height == 0 {
            bail!("environment image has zero dimensions");
        }
        let scale = (2048.0 / width as f32).min(1024.0 / height as f32).min(1.0);
        if scale < 1.0 {
            width = ((width as f32 * scale).round() as u32).max(1);
            height = ((height as f32 * scale).round() as u32).max(1);
            rgba = image::imageops::resize(
                &rgba,
                width,
                height,
                image::imageops::FilterType::Triangle,
            );
        }
        let pixels: Vec<[f32; 4]> = rgba
            .into_raw()
            .chunks_exact(4)
            .map(|p| {
                [p[0], p[1], p[2], 1.0].map(|v| {
                    if v.is_finite() {
                        v.clamp(0.0, 65504.0)
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let diffuse_sh = compute_diffuse_sh(width, height, &pixels);
        Ok(Self {
            name,
            width,
            height,
            rgba: pixels,
            diffuse_sh,
        })
    }
}

fn compute_diffuse_sh(width: u32, height: u32, pixels: &[[f32; 4]]) -> [[f32; 4]; 9] {
    let mut coeff = [[0.0f32; 4]; 9];
    let d_omega = 2.0 * std::f32::consts::PI.powi(2) / (width * height) as f32;
    for y in 0..height {
        let theta = std::f32::consts::PI * (y as f32 + 0.5) / height as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        for x in 0..width {
            let phi = 2.0 * std::f32::consts::PI * ((x as f32 + 0.5) / width as f32 - 0.5);
            let (sin_phi, cos_phi) = phi.sin_cos();
            let (dx, dy, dz) = (sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
            let basis = [
                0.282095,
                0.488603 * dy,
                0.488603 * dz,
                0.488603 * dx,
                1.092548 * dx * dy,
                1.092548 * dy * dz,
                0.315392 * (3.0 * dy * dy - 1.0),
                1.092548 * dx * dz,
                0.546274 * (dx * dx - dz * dz),
            ];
            let weight = sin_theta * d_omega;
            let p = pixels[(y * width + x) as usize];
            for i in 0..9 {
                for c in 0..3 {
                    coeff[i][c] += p[c] * basis[i] * weight;
                }
            }
        }
    }
    coeff
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    /// Pixels below `alpha_cutoff` are discarded.
    Mask,
    /// Alpha blended with back-to-front sorting on backends that support transparency.
    Blend,
    /// Screen-door coverage for stable, depth-writing fades without sorting.
    Dithered,
}

/// One geometry level in a mesh LOD chain. Levels are ordered from finest to
/// coarsest; `min_screen_fraction` selects the first level whose projected
/// radius fraction is at or above that threshold (the final level is fallback).
#[derive(Clone, Copy, Debug)]
pub struct LodLevel {
    pub mesh: usize,
    pub min_screen_fraction: f32,
}

/// glTF metallic-roughness PBR material. `*_image` fields index into `Scene::images`.
#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub base_color: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: Vec3,
    pub normal_scale: f32,
    /// Height scale for folder-loaded displacement maps, in UV-scaled units.
    pub height_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    /// sRGB colour (alpha in A).
    pub base_color_image: Option<usize>,
    /// Linear. G = roughness, B = metallic (glTF layout).
    pub metallic_roughness_image: Option<usize>,
    /// Linear, +Y up (OpenGL / glTF convention).
    pub normal_image: Option<usize>,
    /// Linear height/displacement map used for parallax occlusion mapping.
    pub height_image: Option<usize>,
    /// Linear. Ambient occlusion in the R channel.
    pub occlusion_image: Option<usize>,
    /// sRGB.
    pub emissive_image: Option<usize>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "default".into(),
            base_color: Vec4::ONE,
            metallic: 0.0,
            roughness: 0.5,
            emissive: Vec3::ZERO,
            normal_scale: 1.0,
            // Conservative default: large parallax offsets make surfaces shear
            // visibly at grazing angles, especially on repeating ground maps.
            height_scale: 0.025,
            occlusion_strength: 1.0,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            base_color_image: None,
            metallic_roughness_image: None,
            normal_image: None,
            height_image: None,
            occlusion_image: None,
            emissive_image: None,
        }
    }
}

impl Material {
    /// Plain coloured material: `Material::solid([0.8, 0.1, 0.1], 0.0, 0.4)`.
    pub fn solid(color: [f32; 3], metallic: f32, roughness: f32) -> Self {
        Self {
            base_color: Vec4::new(color[0], color[1], color[2], 1.0),
            metallic,
            roughness,
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightKind {
    Directional,
    Point,
    Spot { inner_cone: f32, outer_cone: f32 },
}

/// glTF `KHR_lights_punctual` light in world space.
/// Directional intensity is lux, point/spot is candela (arbitrary scale when exported unitless).
#[derive(Clone, Debug)]
pub struct Light {
    pub kind: LightKind,
    pub color: Vec3,
    pub intensity: f32,
    /// 0 / None = infinite range.
    pub range: Option<f32>,
    pub position: Vec3,
    /// Direction the light travels (glTF lights shine down their local -Z).
    pub direction: Vec3,
}

/// Fallback light used for scenes that ship without any.
pub fn default_sun() -> Light {
    Light {
        kind: LightKind::Directional,
        color: Vec3::ONE,
        intensity: 3.0,
        range: None,
        position: Vec3::ZERO,
        direction: Vec3::new(-0.4, -1.0, -0.3).normalize(),
    }
}

#[derive(Clone, Debug)]
pub struct MeshInstance {
    pub mesh: usize,
    pub material: usize,
    pub transform: Mat4,
}

/// A transient transparent draw with an optional per-instance base-color
/// override. Useful for effects that need smooth color and opacity changes
/// without allocating a separate material for every particle.
#[derive(Clone, Debug)]
pub struct DynamicDraw {
    pub mesh: usize,
    pub material: usize,
    pub transform: Mat4,
    pub base_color_factor: Option<Vec4>,
}

impl From<MeshInstance> for DynamicDraw {
    fn from(instance: MeshInstance) -> Self {
        Self {
            mesh: instance.mesh,
            material: instance.material,
            transform: instance.transform,
            base_color_factor: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub meshes: Vec<MeshData>,
    pub images: Vec<ImageData>,
    pub materials: Vec<Material>,
    pub instances: Vec<MeshInstance>,
    /// Mesh LOD chains, indexed by the mesh used in a MeshInstance.
    pub mesh_lods: std::collections::HashMap<usize, Vec<LodLevel>>,
    pub lights: Vec<Light>,
    pub environment: Option<EnvironmentMap>,
}

impl Scene {
    pub fn set_mesh_lods(&mut self, base_mesh: usize, levels: Vec<LodLevel>) -> Result<()> {
        if base_mesh >= self.meshes.len() || levels.is_empty() {
            bail!("LOD chain needs an existing base mesh and at least one level");
        }
        let mut previous = f32::INFINITY;
        for level in &levels {
            if level.mesh >= self.meshes.len()
                || !level.min_screen_fraction.is_finite()
                || level.min_screen_fraction <= 0.0
                || level.min_screen_fraction >= previous
            {
                bail!("LOD levels need valid meshes and descending positive screen thresholds");
            }
            previous = level.min_screen_fraction;
        }
        self.mesh_lods.insert(base_mesh, levels);
        Ok(())
    }

    pub fn select_lod(&self, mesh: usize, screen_fraction: f32) -> usize {
        let Some(levels) = self.mesh_lods.get(&mesh) else { return mesh };
        levels.iter()
            .find(|level| screen_fraction >= level.min_screen_fraction)
            .map_or_else(|| levels.last().unwrap().mesh, |level| level.mesh)
    }

    pub fn add_mesh(&mut self, mesh: MeshData) -> usize {
        self.meshes.push(mesh);
        self.meshes.len() - 1
    }

    pub fn add_material(&mut self, material: Material) -> usize {
        self.materials.push(material);
        self.materials.len() - 1
    }

    pub fn add_image(&mut self, image: ImageData) -> usize {
        self.images.push(image);
        self.images.len() - 1
    }

    pub fn add_instance(&mut self, mesh: usize, material: usize, transform: Mat4) {
        self.instances.push(MeshInstance {
            mesh,
            material,
            transform,
        });
    }

    pub fn add_image_file(&mut self, path: &Path) -> Result<usize> {
        let img = image::open(path)
            .with_context(|| format!("loading image {}", path.display()))?
            .to_rgba8();
        Ok(self.add_image(ImageData {
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            width: img.width(),
            height: img.height(),
            rgba8: img.into_raw(),
        }))
    }

    /// World-space bounding box of all instances. `None` if the scene is empty.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        let mut any = false;
        for inst in &self.instances {
            let Some(mesh) = self.meshes.get(inst.mesh) else {
                continue;
            };
            for v in &mesh.vertices {
                let p = inst.transform.transform_point3(Vec3::from(v.position));
                min = min.min(p);
                max = max.max(p);
                any = true;
            }
        }
        any.then_some((min, max))
    }

    /// The scene's lights, or a default sun when it has none and `add_default` is set.
    pub fn effective_lights(&self, add_default: bool) -> Vec<Light> {
        if !self.lights.is_empty() || !add_default {
            return self.lights.clone();
        }
        vec![default_sun()]
    }

    /// Easy PBR: point at a folder of texture files and get a finished material.
    ///
    /// File roles are detected from words in the file name (case-insensitive), e.g.
    /// `rock_albedo.png`, `rock_normal.png`, `rock_roughness.png`, `rock_metallic.png`,
    /// `rock_ao.png`, `rock_emissive.png`, or a packed `rock_orm.png` (R=AO, G=roughness, B=metal).
    /// Names containing `dx` next to `normal` are treated as DirectX normal maps (green flipped).
    pub fn add_pbr_material_from_dir(&mut self, dir: &Path) -> Result<usize> {
        let mut found: Vec<(Slot, std::path::PathBuf)> = Vec::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let path = entry?.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if !["png", "jpg", "jpeg", "tga"].contains(&ext.as_str()) {
                continue;
            }
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if let Some(slot) = classify(stem) {
                found.push((slot, path));
            }
        }
        if found.is_empty() {
            bail!("no recognisable PBR textures in {}", dir.display());
        }
        found.sort_by(|a, b| a.1.cmp(&b.1));
        let pick = |want: fn(&Slot) -> bool| {
            found
                .iter()
                .find(|(s, _)| want(s))
                .map(|(s, p)| (*s, p.clone()))
        };

        let mut mat = Material {
            name: dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            ..Default::default()
        };

        if let Some((_, p)) = pick(|s| matches!(s, Slot::Base)) {
            mat.base_color_image = Some(self.add_image_file(&p)?);
        }
        if let Some((slot, p)) = pick(|s| matches!(s, Slot::Normal { .. })) {
            let idx = self.add_image_file(&p)?;
            if matches!(slot, Slot::Normal { flip_green: true }) {
                for px in self.images[idx].rgba8.chunks_exact_mut(4) {
                    px[1] = 255 - px[1];
                }
            }
            mat.normal_image = Some(idx);
        }
        if let Some((_, p)) = pick(|s| matches!(s, Slot::Height)) {
            mat.height_image = Some(self.add_image_file(&p)?);
        }
        if let Some((_, p)) = pick(|s| matches!(s, Slot::Emissive)) {
            mat.emissive_image = Some(self.add_image_file(&p)?);
            mat.emissive = Vec3::ONE;
        }
        if let Some((_, p)) = pick(|s| matches!(s, Slot::Orm)) {
            let idx = self.add_image_file(&p)?;
            mat.metallic_roughness_image = Some(idx);
            mat.occlusion_image = Some(idx);
            mat.metallic = 1.0;
            mat.roughness = 1.0;
        } else {
            let rough = pick(|s| matches!(s, Slot::Roughness)).map(|(_, p)| p);
            let metal = pick(|s| matches!(s, Slot::Metallic)).map(|(_, p)| p);
            if rough.is_some() || metal.is_some() {
                let rough_img = rough.as_ref().map(|p| load_gray(p)).transpose()?;
                let metal_img = metal.as_ref().map(|p| load_gray(p)).transpose()?;
                let (w, h) = rough_img
                    .as_ref()
                    .or(metal_img.as_ref())
                    .map(|i| i.dimensions())
                    .unwrap();
                let fit = |i: image::GrayImage| {
                    if i.dimensions() == (w, h) {
                        i
                    } else {
                        image::imageops::resize(&i, w, h, image::imageops::FilterType::Triangle)
                    }
                };
                let rough_img = rough_img.map(fit);
                let metal_img = metal_img.map(fit);
                let mut rgba = vec![255u8; (w * h * 4) as usize];
                for i in 0..(w * h) as usize {
                    if let Some(r) = &rough_img {
                        rgba[i * 4 + 1] = r.as_raw()[i];
                    }
                    if let Some(m) = &metal_img {
                        rgba[i * 4 + 2] = m.as_raw()[i];
                    } else {
                        rgba[i * 4 + 2] = 0;
                    }
                }
                let idx = self.add_image(ImageData {
                    name: format!("{}_packed_mr", mat.name),
                    width: w,
                    height: h,
                    rgba8: rgba,
                });
                mat.metallic_roughness_image = Some(idx);
                mat.metallic = 1.0;
                mat.roughness = 1.0;
            }
        }
        if mat.occlusion_image.is_none() {
            if let Some((_, p)) = pick(|s| matches!(s, Slot::Ao)) {
                mat.occlusion_image = Some(self.add_image_file(&p)?);
            }
        }
        Ok(self.add_material(mat))
    }
}

#[derive(Clone, Copy, Debug)]
enum Slot {
    Base,
    Normal { flip_green: bool },
    Height,
    Roughness,
    Metallic,
    Ao,
    Emissive,
    Orm,
}

fn classify(stem: &str) -> Option<Slot> {
    let lower = stem.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let has = |names: &[&str]| tokens.iter().any(|t| names.contains(t));
    if has(&["orm", "arm", "metallicroughness"]) {
        Some(Slot::Orm)
    } else if has(&["normal", "nor", "nrm", "norm", "normalgl", "normaldx"]) {
        Some(Slot::Normal {
            flip_green: has(&["dx", "normaldx", "directx"]),
        })
    } else if has(&["displacement", "height", "heightmap", "disp"]) {
        Some(Slot::Height)
    } else if has(&["roughness", "rough", "rgh"]) {
        Some(Slot::Roughness)
    } else if has(&["metallic", "metalness", "metal", "met"]) {
        Some(Slot::Metallic)
    } else if has(&["ao", "occlusion", "ambientocclusion"]) {
        Some(Slot::Ao)
    } else if has(&["emissive", "emission", "emit", "glow"]) {
        Some(Slot::Emissive)
    } else if has(&[
        "albedo",
        "basecolor",
        "base",
        "diffuse",
        "diff",
        "color",
        "col",
    ]) {
        Some(Slot::Base)
    } else {
        None
    }
}

fn load_gray(path: &Path) -> Result<image::GrayImage> {
    Ok(image::open(path)
        .with_context(|| format!("loading {}", path.display()))?
        .to_luma8())
}

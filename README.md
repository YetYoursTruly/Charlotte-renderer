# charlotte-renderer (beta)

A Direct3D 12-first PBR renderer for Rust with DXR ray tracing, glTF (Blender) loading and an
experimental Vulkan backend.

> **Status: beta.** The Windows D3D12 path has been built and exercised locally. The public API,
> shader behavior and GPU-driver coverage are still evolving; Vulkan remains experimental.

## Install from crates.io

Applications can add the published beta facade crate with:

```toml
[dependencies]
charlotte-renderer = "0.1.0-beta.2"
```

In Rust code, import `charlotte_renderer`. The renderer facade package and directory are both named
`charlotte-renderer`; internal backend and core crates retain their separate names.

On Windows, the D3D12 dependency downloads Microsoft's pinned DXC NuGet runtime during the first
build and copies `dxcompiler.dll` and `dxil.dll` next to the Cargo-built executable. No manual DLL
copy is needed. The first build needs internet access; set `CHARLOTTE_DXC_DIR` to an existing DLL
folder to use a local copy instead.

## Crates

| crate | what it does |
|---|---|
| `charlotte-renderer` | facade: `create_renderer(Backend::Auto, ..)`, re-exports everything. On Windows `Auto` selects D3D12 and does not silently fall back to Vulkan. Import it as `charlotte_renderer`. |
| `charlotte_core` | scene, materials, lights, camera, settings, `Renderer` trait, easy PBR folder loader |
| `charlotte_gltf` | `.gltf` / `.glb` loader (Blender exports), incl. `KHR_lights_punctual` |
| `charlotte_d3d12` | **main backend**: forward PBR, capability-selected MSAA, DXR 1.1 ray-traced shadows + AO |
| `charlotte_vulkan` | **experimental**: device/swapchain bring-up, clears + presents; does NOT draw scenes yet |
| `apps/viewer` | fly-camera viewer / demo |

## Running (Windows)

1. Install a recent Rust toolchain and up-to-date GPU drivers.
2. ```
   cargo run -p viewer --release                        # demo sphere grid
   cargo run -p viewer --release -- scene.glb           # Blender glTF/GLB export
   cargo run -p viewer --release -- --pbr-dir textures/rock
   cargo run -p viewer --release -- scene.glb --environment sky.exr --ao-samples 24 --shadow-samples 8 --msaa 4 --exposure 0.5
   cargo run -p viewer --release -- scene.glb --fps
   cargo run -p viewer --release -- --backend vulkan    # experimental (clear colour only)
   ```
   Keys: WASD/QE move (Shift fast), hold right mouse to look, `[` `]` exposure, `R` RT shadows, `O` RT AO.

## Features

* **GPU selection**: NVIDIA / AMD / Intel via `RenderSettings::gpu` (D3D12 and Vulkan). `Backend::Auto`
  selects D3D12 on Windows; Vulkan is selected only with `--backend vulkan`. Ray tracing needs DXR tier
  1.1 (RTX 20+, RX 6000+, Arc); without it the renderer runs without ray-traced effects.
* **glTF from Blender**: meshes, node hierarchy, PBR materials, all five texture slots, emissive
  strength, double-sided, alpha mask, and **lights** (`KHR_lights_punctual`: sun, point, spot).
  Tangents are generated when missing. In Blender enable *Include > Punctual Lights* when exporting.
  If lighting looks too bright/dark, adjust exposure (`[` `]` / `--exposure`): Blender's
  "Lighting Mode" export option changes the intensity units.
* **Easy PBR textures**:
  ```rust
  let mat = scene.add_pbr_material_from_dir(Path::new("textures/rock"))?;
  scene.add_instance(mesh, mat, Mat4::IDENTITY);
  ```
  Files are matched by name: `*_albedo|basecolor|diffuse|color`, `*_normal` (add `dx` for DirectX-style),
  `*_roughness`, `*_metallic`, `*_ao|occlusion`, `*_emissive`, or packed `*_orm`.
  Separate roughness/metallic maps are packed to the glTF layout automatically. Mips are generated.
  Or build a material by hand: `Material::solid([r,g,b], metallic, roughness)`.
* **Anti-aliasing**: D3D12 MSAA request from `RenderSettings::msaa_samples` (1/2/4/8); checks color and depth format support, selects the highest supported level at or below the request, and resolves into the existing flip-model swapchain. Defaults to 4x. `--msaa 1` disables it.
* **Environment lighting**: the viewer embeds `qwantani_dusk_2_puresky_4k.exr` from `charlotte_core` and uses it by default. Override with `--environment sky.exr` (OpenEXR or Radiance HDR). It drives the visible sky, nine-band diffuse irradiance, and roughness-selected HDR reflection mips.
* **Ray tracing**: DXR 1.1 inline `RayQuery` in the pixel shader. BLAS per mesh, TLAS per scene.
  Soft ray-traced shadows sample a configurable angular light radius; 1–16 shadow rays per pixel (`--shadow-samples`). Ray-traced AO uses evenly distributed, distance-weighted rays, enabled by default with 1–32 rays (`--ao-samples`), adjustable range and strength.
* **Viewer performance**: `--fps` shows measured FPS and current AO/shadow ray counts in the window title. After 6 frames above 18.5 ms, the viewer halves DXR ray counts down to AO 4 and shadow 1; after 600 frames below 17.2 ms, it restores the requested counts.
* **PBR correctness**: glTF metallic/roughness channels and linear/sRGB texture handling remain intact; normal transforms now use the inverse-transpose matrix for non-uniform model scale. Direct lighting uses the existing GGX microfacet BRDF and ACES output transform.

## Known limitations (beta)

* HDR environment IBL uses diffuse spherical harmonics and filtered equirectangular mip levels; GGX-prefiltered cubemaps, screen-space reflections and RT reflections are not implemented.
* No screen-space SSAO/GTAO or denoiser. RT AO and soft RT shadows are only available on DXR 1.1 devices.
* D3D12 alpha blending uses back-to-front instance sorting. No skinning/animation, morph targets,
  KHR_texture_transform.
* One frame in flight (waits for the GPU every frame), no culling, one draw call per instance, max 16 lights,
  and a fixed small ray-origin bias.
* Static scenes only: the TLAS is built once in `load_scene`.
* Vulkan backend: see above. D3D11 is not implemented.
* RT AO and soft-shadow sampling use deterministic low-discrepancy rays but have no temporal or spatial denoiser yet, so some structured artifacts can remain.

## Using it from your own app

```rust
let handles = charlotte_renderer::WindowHandles { window, display, width, height }; // raw-window-handle 0.6
let mut settings = charlotte_renderer::RenderSettings::default();
let mut renderer = charlotte_renderer::create_renderer(charlotte_renderer::Backend::Auto, &handles, &settings)?;
renderer.load_scene(&charlotte_renderer::gltf::load_gltf("model.glb")?)?;
renderer.render(&camera, &settings)?; // every frame
```

License: MIT OR Apache-2.0 (as declared in the Cargo manifests; add the license files before publishing).

//! Direct3D 12 backend.
//!
//! * Forward PBR (glTF metallic-roughness), up to 16 punctual lights, ACES tone mapping.
//! * Optional DXR 1.1 inline ray tracing (`RayQuery` in the pixel shader): ray-traced shadows and
//!   ambient occlusion. Requires a GPU with DXR tier 1.1 (NVIDIA RTX / GTX 16 with RT support via
//!   driver, AMD RDNA2+, Intel Arc). Falls back to unshadowed lighting otherwise.
//! * NVIDIA / AMD / Intel adapter selection.
#![cfg(windows)]

mod gpu;
mod shader;

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop};

use anyhow::{anyhow, bail, Context, Result};
use charlotte_core::glam::Mat4;
use charlotte_core::raw_window_handle::RawWindowHandle;
use charlotte_core::{
    default_sun, AdapterInfo, AlphaMode, Camera, GpuPreference, GpuVendor, Light, LightKind,
    Material, RenderSettings, Renderer, Scene, WindowHandles,
};
use windows::core::{s, Interface, PCSTR, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, RECT};
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

use gpu::*;

const FRAME_COUNT: u32 = 2;
const MAX_LIGHTS: usize = 16;
const VERTEX_STRIDE: u32 = 48;
const SRV_PER_MATERIAL: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    position: [f32; 3],
    range: f32,
    direction: [f32; 3],
    kind: u32,
    color: [f32; 3],
    intensity: f32,
    inner_cos: f32,
    outer_cos: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameConstants {
    view_proj: [f32; 16],
    cam_pos: [f32; 3],
    exposure: f32,
    sky: [f32; 3],
    ao_strength: f32,
    ground: [f32; 3],
    ao_radius: f32,
    cam_forward: [f32; 3],
    fov_tan: f32,
    cam_right: [f32; 3],
    aspect: f32,
    cam_up: [f32; 3],
    ibl_strength: f32,
    light_count: u32,
    flags: u32,
    ao_samples: u32,
    shadow_samples: u32,
    env_mips: f32,
    shadow_softness: f32,
    _pad: [u32; 2],
    lights: [GpuLight; MAX_LIGHTS],
    env_sh: [[f32; 4]; 9],
}

/// 32 DWORDs, sent as root constants.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawConstants {
    model: [f32; 16],
    base_color: [f32; 4],
    emissive: [f32; 3],
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32,
    flags: u32,
    _pad: [u32; 3],
}

const _: () = assert!(size_of::<DrawConstants>() == 128);
const _: () = assert!(size_of::<GpuLight>() == 64);
const _: () = assert!(size_of::<FrameConstants>() == 1360);

struct GpuMesh {
    vb: ID3D12Resource,
    ib: ID3D12Resource,
    vertex_bytes: u32,
    index_count: u32,
    blas: Option<ID3D12Resource>,
}

struct GpuMaterial {
    table: D3D12_GPU_DESCRIPTOR_HANDLE,
    draw: DrawConstants,
    double_sided: bool,
}

struct GpuScene {
    meshes: Vec<GpuMesh>,
    materials: Vec<GpuMaterial>,
    instances: Vec<(usize, usize, Mat4)>,
    srv_heap: ID3D12DescriptorHeap,
    _textures: Vec<ID3D12Resource>,
    tlas: Option<ID3D12Resource>,
    lights: Vec<Light>,
    _environment: ID3D12Resource,
    environment_table: D3D12_GPU_DESCRIPTOR_HANDLE,
    environment_enabled: bool,
    environment_mips: u32,
    environment_sh: [[f32; 4]; 9],
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TexKey {
    Image(usize, bool),
    WhiteSrgb,
    WhiteLinear,
    FlatNormal,
}

pub struct D3d12Renderer {
    info: AdapterInfo,
    rt_enabled: bool,
    device: ID3D12Device,
    device5: Option<ID3D12Device5>,
    queue: ID3D12CommandQueue,
    swapchain: IDXGISwapChain3,
    rtv_heap: ID3D12DescriptorHeap,
    rtv_size: u32,
    dsv_heap: ID3D12DescriptorHeap,
    targets: Vec<ID3D12Resource>,
    msaa_target: Option<ID3D12Resource>,
    msaa_rtv_heap: ID3D12DescriptorHeap,
    sample_count: u32,
    depth: Option<ID3D12Resource>,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList4,
    fence: ID3D12Fence,
    fence_value: u64,
    fence_event: HANDLE,
    root_sig: ID3D12RootSignature,
    pso_cull: ID3D12PipelineState,
    pso_no_cull: ID3D12PipelineState,
    frame_cb: ID3D12Resource,
    pso_sky: ID3D12PipelineState,
    width: u32,
    height: u32,
    scene: Option<GpuScene>,
}

impl D3d12Renderer {
    pub fn new(handles: &WindowHandles, settings: &RenderSettings) -> Result<Self> {
        let hwnd = match handles.window {
            RawWindowHandle::Win32(h) => HWND(h.hwnd.get() as *mut c_void),
            other => bail!("D3D12 backend needs a Win32 window handle, got {other:?}"),
        };
        unsafe { Self::new_inner(hwnd, handles.width.max(1), handles.height.max(1), settings) }
    }

    unsafe fn new_inner(
        hwnd: HWND,
        width: u32,
        height: u32,
        settings: &RenderSettings,
    ) -> Result<Self> {
        if settings.debug_layer {
            let mut dbg: Option<ID3D12Debug> = None;
            if D3D12GetDebugInterface(&mut dbg).is_ok() {
                if let Some(d) = dbg {
                    d.EnableDebugLayer();
                }
            }
        }
        let factory_flags = if settings.debug_layer {
            DXGI_CREATE_FACTORY_DEBUG
        } else {
            DXGI_CREATE_FACTORY_FLAGS(0)
        };
        let factory: IDXGIFactory6 = CreateDXGIFactory2(factory_flags)?;

        let (adapter, desc) = pick_adapter(&factory, settings.gpu)?;
        let name = String::from_utf16_lossy(&desc.Description)
            .trim_end_matches('\0')
            .to_string();
        let vendor = GpuVendor::from_pci_id(desc.VendorId);

        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        let device = device.context("D3D12CreateDevice returned no device")?;

        let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        let rt_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS5,
                &mut opts5 as *mut _ as *mut c_void,
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
            )
            .is_ok()
            && opts5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0;
        let device5 = if rt_supported && settings.ray_tracing {
            device.cast::<ID3D12Device5>().ok()
        } else {
            None
        };
        let rt_enabled = device5.is_some();
        let info = AdapterInfo {
            backend: "Direct3D 12",
            name,
            vendor,
            dedicated_vram_mb: (desc.DedicatedVideoMemory / (1024 * 1024)) as u64,
            ray_tracing: rt_supported,
        };
        log::info!(
            "D3D12 adapter: {} ({:?}, {} MB, DXR 1.1: {})",
            info.name,
            info.vendor,
            info.dedicated_vram_mb,
            rt_supported
        );

        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;

        let sc_desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: FRAME_COUNT,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            ..Default::default()
        };
        let swapchain: IDXGISwapChain3 = factory
            .CreateSwapChainForHwnd(&queue, hwnd, &sc_desc, None, None)?
            .cast()?;
        factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER)?;

        let rtv_heap: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: FRAME_COUNT,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsv_heap: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })?;
        let msaa_rtv_heap: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })?;

        let sample_count = select_msaa_samples(&device, settings.msaa_samples);
        if sample_count != settings.msaa_samples.max(1) {
            log::info!(
                "MSAA request {}x unavailable; using {}x",
                settings.msaa_samples,
                sample_count
            );
        }

        let allocator: ID3D12CommandAllocator =
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList4 = device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &allocator,
            None::<&ID3D12PipelineState>,
        )?;
        list.Close()?;

        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let fence_event = CreateEventW(None, false, false, PCWSTR::null())?;

        let root_sig = create_root_signature(&device)?;
        let vs = shader::compile("VSMain", "vs_6_0", rt_enabled)?;
        let ps = shader::compile(
            "PSMain",
            if rt_enabled { "ps_6_5" } else { "ps_6_0" },
            rt_enabled,
        )?;
        let pso_cull = create_pso(
            &device,
            &root_sig,
            &vs,
            &ps,
            D3D12_CULL_MODE_BACK,
            sample_count,
            false,
        )?;
        let pso_no_cull = create_pso(
            &device,
            &root_sig,
            &vs,
            &ps,
            D3D12_CULL_MODE_NONE,
            sample_count,
            false,
        )?;
        let sky_vs = shader::compile("SkyVS", "vs_6_0", false)?;
        let sky_ps = shader::compile("SkyPS", "ps_6_0", false)?;
        let pso_sky = create_pso(
            &device,
            &root_sig,
            &sky_vs,
            &sky_ps,
            D3D12_CULL_MODE_NONE,
            sample_count,
            true,
        )?;

        let cb_size = align_up(size_of::<FrameConstants>() as u64, 256);
        let frame_cb = create_buffer(
            &device,
            cb_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
        )?;

        let mut r = Self {
            info,
            rt_enabled,
            device,
            device5,
            queue,
            swapchain,
            rtv_heap,
            rtv_size,
            dsv_heap,
            targets: Vec::new(),
            msaa_target: None,
            msaa_rtv_heap,
            sample_count,
            depth: None,
            allocator,
            list,
            fence,
            fence_value: 0,
            fence_event,
            root_sig,
            pso_cull,
            pso_no_cull,
            frame_cb,
            pso_sky,
            width,
            height,
            scene: None,
        };
        r.create_targets()?;
        Ok(r)
    }

    unsafe fn create_targets(&mut self) -> Result<()> {
        let base = self.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        self.targets.clear();
        self.msaa_target = None;
        for i in 0..FRAME_COUNT {
            let buf: ID3D12Resource = self.swapchain.GetBuffer(i)?;
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base.ptr + (i * self.rtv_size) as usize,
            };
            let desc = D3D12_RENDER_TARGET_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_RTV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
            };
            self.device
                .CreateRenderTargetView(&buf, Some(&desc), handle);
            self.targets.push(buf);
        }

        if self.sample_count > 1 {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: self.width as u64,
                Height: self.height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: self.sample_count,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            };
            let clear = D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    Color: [0.05, 0.06, 0.08, 1.0],
                },
            };
            let mut target = None;
            self.device.CreateCommittedResource(
                &heap_props(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                Some(&clear),
                &mut target,
            )?;
            let target = target.context("MSAA render target creation failed")?;
            self.device.CreateRenderTargetView(
                &target,
                Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                    ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DMS,
                    Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2DMS: D3D12_TEX2DMS_RTV {
                            UnusedField_NothingToDefine: 0,
                        },
                    },
                }),
                self.msaa_rtv_heap.GetCPUDescriptorHandleForHeapStart(),
            );
            self.msaa_target = Some(target);
        }

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: self.width as u64,
            Height: self.height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: self.sample_count,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        };
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                    Depth: 1.0,
                    Stencil: 0,
                },
            },
        };
        let mut depth: Option<ID3D12Resource> = None;
        self.device.CreateCommittedResource(
            &heap_props(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&clear),
            &mut depth,
        )?;
        let depth = depth.context("depth buffer creation failed")?;
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT,
            ViewDimension: if self.sample_count > 1 {
                D3D12_DSV_DIMENSION_TEXTURE2DMS
            } else {
                D3D12_DSV_DIMENSION_TEXTURE2D
            },
            Flags: D3D12_DSV_FLAG_NONE,
            Anonymous: if self.sample_count > 1 {
                D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DMS: D3D12_TEX2DMS_DSV {
                        UnusedField_NothingToDefine: 0,
                    },
                }
            } else {
                D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_DSV { MipSlice: 0 },
                }
            },
        };
        self.device.CreateDepthStencilView(
            &depth,
            Some(&dsv),
            self.dsv_heap.GetCPUDescriptorHandleForHeapStart(),
        );
        self.depth = Some(depth);
        Ok(())
    }

    unsafe fn wait_idle(&mut self) -> Result<()> {
        self.fence_value += 1;
        self.queue.Signal(&self.fence, self.fence_value)?;
        if self.fence.GetCompletedValue() < self.fence_value {
            self.fence
                .SetEventOnCompletion(self.fence_value, self.fence_event)?;
            WaitForSingleObject(self.fence_event, INFINITE);
        }
        Ok(())
    }

    unsafe fn execute_list(&self) -> Result<()> {
        let list: ID3D12CommandList = self.list.cast()?;
        self.queue.ExecuteCommandLists(&[Some(list)]);
        Ok(())
    }

    unsafe fn load_scene_inner(&mut self, scene: &Scene) -> Result<()> {
        let device = self.device.clone();
        let list = self.list.clone();
        self.allocator.Reset()?;
        list.Reset(&self.allocator, None::<&ID3D12PipelineState>)?;
        let mut keep: Vec<ID3D12Resource> = Vec::new();

        // ---- geometry ----
        let mut meshes = Vec::with_capacity(scene.meshes.len());
        for m in &scene.meshes {
            let vb =
                upload_to_default(&device, &list, bytemuck::cast_slice(&m.vertices), &mut keep)?;
            let ib =
                upload_to_default(&device, &list, bytemuck::cast_slice(&m.indices), &mut keep)?;
            meshes.push(GpuMesh {
                vb,
                ib,
                vertex_bytes: (m.vertices.len() * VERTEX_STRIDE as usize) as u32,
                index_count: m.indices.len() as u32,
                blas: None,
            });
        }

        // ---- materials & textures ----
        let default_material = [Material::default()];
        let materials_src: &[Material] = if scene.materials.is_empty() {
            &default_material
        } else {
            &scene.materials
        };
        let srv_count = (materials_src.len() as u32 * SRV_PER_MATERIAL).max(SRV_PER_MATERIAL) + 1;
        let srv_heap: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: srv_count,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
        let inc = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let cpu_base = srv_heap.GetCPUDescriptorHandleForHeapStart();
        let gpu_base = srv_heap.GetGPUDescriptorHandleForHeapStart();

        let mut cache: HashMap<TexKey, usize> = HashMap::new();
        let mut textures: Vec<(ID3D12Resource, u32, bool)> = Vec::new();
        let mut materials = Vec::with_capacity(materials_src.len());
        for (mi, m) in materials_src.iter().enumerate() {
            let img = |idx: Option<usize>, srgb: bool, default: TexKey| match idx {
                Some(i) if i < scene.images.len() => TexKey::Image(i, srgb),
                _ => default,
            };
            let keys = [
                img(m.base_color_image, true, TexKey::WhiteSrgb),
                img(m.metallic_roughness_image, false, TexKey::WhiteLinear),
                img(m.normal_image, false, TexKey::FlatNormal),
                img(m.emissive_image, true, TexKey::WhiteSrgb),
                img(m.occlusion_image, false, TexKey::WhiteLinear),
            ];
            for (slot, key) in keys.iter().enumerate() {
                let tex_index = match cache.get(key) {
                    Some(&i) => i,
                    None => {
                        let (rgba, w, h, srgb): (Vec<u8>, u32, u32, bool) = match *key {
                            TexKey::Image(i, srgb) => {
                                let im = &scene.images[i];
                                (im.rgba8.clone(), im.width, im.height, srgb)
                            }
                            TexKey::WhiteSrgb => (vec![255; 4], 1, 1, true),
                            TexKey::WhiteLinear => (vec![255; 4], 1, 1, false),
                            TexKey::FlatNormal => (vec![128, 128, 255, 255], 1, 1, false),
                        };
                        let (res, mips) =
                            create_texture(&device, &list, w, h, &rgba, srgb, &mut keep)?;
                        textures.push((res, mips, srgb));
                        cache.insert(*key, textures.len() - 1);
                        textures.len() - 1
                    }
                };
                let (res, mips, srgb) = &textures[tex_index];
                let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: if *srgb {
                        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                    } else {
                        DXGI_FORMAT_R8G8B8A8_UNORM
                    },
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MostDetailedMip: 0,
                            MipLevels: *mips,
                            PlaneSlice: 0,
                            ResourceMinLODClamp: 0.0,
                        },
                    },
                };
                let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu_base.ptr
                        + ((mi as u32 * SRV_PER_MATERIAL + slot as u32) * inc) as usize,
                };
                device.CreateShaderResourceView(res, Some(&srv), dst);
            }
            let draw = DrawConstants {
                model: Mat4::IDENTITY.to_cols_array(),
                base_color: m.base_color.to_array(),
                emissive: m.emissive.to_array(),
                metallic: m.metallic,
                roughness: m.roughness,
                normal_scale: m.normal_scale,
                occlusion_strength: m.occlusion_strength,
                alpha_cutoff: m.alpha_cutoff,
                flags: if m.alpha_mode == AlphaMode::Opaque {
                    0
                } else {
                    1
                },
                _pad: [0; 3],
            };
            materials.push(GpuMaterial {
                table: D3D12_GPU_DESCRIPTOR_HANDLE {
                    ptr: gpu_base.ptr + (mi as u32 * SRV_PER_MATERIAL * inc) as u64,
                },
                draw,
                double_sided: m.double_sided,
            });
        }

        let (environment_texture, environment_mips, environment_sh, environment_enabled) =
            if let Some(env) = &scene.environment {
                let (tex, mip_count) = create_environment_texture(
                    &device, &list, env.width, env.height, &env.rgba, &mut keep,
                )?;
                (tex, mip_count, env.diffuse_sh, true)
            } else {
                let fallback = [[0.0, 0.0, 0.0, 1.0]];
                let (tex, mip_count) =
                    create_environment_texture(&device, &list, 1, 1, &fallback, &mut keep)?;
                (tex, mip_count, [[0.0; 4]; 9], false)
            };
        let env_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: environment_mips,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        let env_slot = materials_src.len() as u32 * SRV_PER_MATERIAL;
        device.CreateShaderResourceView(
            &environment_texture,
            Some(&env_srv),
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_base.ptr + (env_slot * inc) as usize,
            },
        );
        let environment_table = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu_base.ptr + (env_slot * inc) as u64,
        };

        // ---- ray tracing acceleration structures ----
        let instances: Vec<(usize, usize, Mat4)> = scene
            .instances
            .iter()
            .filter(|i| i.mesh < meshes.len() && meshes[i.mesh].index_count > 0)
            .map(|i| (i.mesh, i.material.min(materials.len() - 1), i.transform))
            .collect();
        let mut tlas = None;
        if let (true, Some(device5)) = (self.rt_enabled, self.device5.clone()) {
            for (mesh, src) in meshes.iter_mut().zip(&scene.meshes) {
                if mesh.index_count == 0 {
                    continue;
                }
                mesh.blas = Some(build_blas(
                    &device5,
                    &list,
                    mesh,
                    src.vertices.len() as u32,
                    &mut keep,
                )?);
            }
            uav_barrier_all(&list);
            tlas = Some(build_tlas(&device5, &list, &meshes, &instances, &mut keep)?);
            uav_barrier_all(&list);
        }

        list.Close()?;
        self.execute_list()?;
        self.wait_idle()?;
        drop(keep);

        self.scene = Some(GpuScene {
            meshes,
            materials,
            instances,
            srv_heap,
            _textures: textures.into_iter().map(|t| t.0).collect(),
            tlas,
            lights: scene.lights.clone(),
            _environment: environment_texture,
            environment_table,
            environment_enabled,
            environment_mips,
            environment_sh,
        });
        Ok(())
    }

    unsafe fn render_inner(&mut self, camera: &Camera, settings: &RenderSettings) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Ok(());
        }
        // ---- per-frame constants ----
        let aspect = self.width as f32 / self.height as f32;
        let mut fc: FrameConstants = bytemuck::Zeroable::zeroed();
        fc.view_proj = camera.view_proj(aspect).to_cols_array();
        fc.cam_pos = camera.position.to_array();
        fc.exposure = settings.exposure;
        fc.sky = settings.sky_color;
        fc.ground = settings.ground_color;
        fc.ao_strength = settings.rt_ao_strength;
        fc.ao_radius = settings.rt_ao_radius;
        fc.ao_samples = settings.rt_ao_samples.clamp(1, 32);
        fc.cam_forward = camera.forward().to_array();
        fc.fov_tan = (camera.fov_y * 0.5).tan();
        fc.cam_right = camera.right().to_array();
        fc.aspect = aspect;
        fc.cam_up = camera
            .right()
            .cross(camera.forward())
            .normalize()
            .to_array();
        fc.ibl_strength = settings.ibl_strength;
        fc.shadow_samples = settings.rt_shadow_samples.clamp(1, 16);
        fc.shadow_softness = settings.rt_shadow_softness.max(0.0);
        if let Some(scene) = &self.scene {
            fc.env_sh = scene.environment_sh;
            fc.env_mips = scene.environment_mips.saturating_sub(1) as f32;
            if scene.environment_enabled {
                fc.flags |= 4;
            }
        }
        let has_tlas = self.scene.as_ref().is_some_and(|s| s.tlas.is_some());
        if has_tlas {
            if settings.rt_shadows {
                fc.flags |= 1;
            }
            if settings.rt_ao {
                fc.flags |= 2;
            }
        }
        let mut lights: Vec<Light> = self
            .scene
            .as_ref()
            .map(|s| s.lights.clone())
            .unwrap_or_default();
        if lights.is_empty() && settings.default_light_if_none {
            lights.push(default_sun());
        }
        for (i, l) in lights.iter().take(MAX_LIGHTS).enumerate() {
            let (kind, inner, outer) = match l.kind {
                LightKind::Directional => (0, 0.0, 0.0),
                LightKind::Point => (1, 0.0, 0.0),
                LightKind::Spot {
                    inner_cone,
                    outer_cone,
                } => (2, inner_cone.cos(), outer_cone.cos()),
            };
            fc.lights[i] = GpuLight {
                position: l.position.to_array(),
                range: l.range.unwrap_or(0.0),
                direction: l.direction.to_array(),
                kind,
                color: l.color.to_array(),
                intensity: l.intensity,
                inner_cos: inner,
                outer_cos: outer,
                _pad: [0.0; 2],
            };
            fc.light_count += 1;
        }
        write_to_buffer(&self.frame_cb, 0, bytemuck::bytes_of(&fc))?;

        // ---- record ----
        self.allocator.Reset()?;
        let list = self.list.clone();
        list.Reset(&self.allocator, &self.pso_cull)?;

        let frame = self.swapchain.GetCurrentBackBufferIndex() as usize;
        let target = self.targets[frame].clone();
        let render_target = self.msaa_target.as_ref().unwrap_or(&target).clone();
        if self.sample_count == 1 {
            transition(
                &list,
                &target,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
        } else {
            transition(
                &list,
                &render_target,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
        }

        let rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.rtv_heap.GetCPUDescriptorHandleForHeapStart().ptr
                + frame * self.rtv_size as usize,
        };
        let dsv = self.dsv_heap.GetCPUDescriptorHandleForHeapStart();
        let render_rtv = if self.sample_count > 1 {
            self.msaa_rtv_heap.GetCPUDescriptorHandleForHeapStart()
        } else {
            rtv
        };
        list.OMSetRenderTargets(1, Some(&render_rtv), false, Some(&dsv));
        list.ClearRenderTargetView(render_rtv, &settings.clear_color, None);
        list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, &[]);
        list.RSSetViewports(&[D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: self.width as f32,
            Height: self.height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        }]);
        list.RSSetScissorRects(&[RECT {
            left: 0,
            top: 0,
            right: self.width as i32,
            bottom: self.height as i32,
        }]);

        if let Some(scene) = &self.scene {
            list.SetGraphicsRootSignature(&self.root_sig);
            list.SetDescriptorHeaps(&[Some(scene.srv_heap.clone())]);
            list.SetGraphicsRootConstantBufferView(0, self.frame_cb.GetGPUVirtualAddress());
            list.SetGraphicsRootDescriptorTable(4, scene.environment_table);
            if scene.environment_enabled {
                list.SetPipelineState(&self.pso_sky);
                list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                list.DrawInstanced(3, 1, 0, 0);
            }
            list.SetGraphicsRootShaderResourceView(
                1,
                scene.tlas.as_ref().map_or(0, |t| t.GetGPUVirtualAddress()),
            );
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.SetPipelineState(&self.pso_cull);

            let mut double_sided = false;
            for (mesh_i, mat_i, transform) in &scene.instances {
                let mesh = &scene.meshes[*mesh_i];
                let mat = &scene.materials[*mat_i];
                if mat.double_sided != double_sided {
                    double_sided = mat.double_sided;
                    list.SetPipelineState(if double_sided {
                        &self.pso_no_cull
                    } else {
                        &self.pso_cull
                    });
                }
                let mut dc = mat.draw;
                dc.model = transform.to_cols_array();
                list.SetGraphicsRoot32BitConstants(2, 32, &dc as *const _ as *const c_void, 0);
                list.SetGraphicsRootDescriptorTable(3, mat.table);
                list.IASetVertexBuffers(
                    0,
                    Some(&[D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: mesh.vb.GetGPUVirtualAddress(),
                        SizeInBytes: mesh.vertex_bytes,
                        StrideInBytes: VERTEX_STRIDE,
                    }]),
                );
                list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: mesh.ib.GetGPUVirtualAddress(),
                    SizeInBytes: mesh.index_count * 4,
                    Format: DXGI_FORMAT_R32_UINT,
                }));
                list.DrawIndexedInstanced(mesh.index_count, 1, 0, 0, 0);
            }
        }

        if self.sample_count > 1 {
            transition(
                &list,
                &render_target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            );
            transition(
                &list,
                &target,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
            );
            list.ResolveSubresource(&target, 0, &render_target, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            transition(
                &list,
                &target,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
                D3D12_RESOURCE_STATE_PRESENT,
            );
        } else {
            transition(
                &list,
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            );
        }
        list.Close()?;
        self.execute_list()?;
        self.swapchain
            .Present(if settings.vsync { 1 } else { 0 }, DXGI_PRESENT(0))
            .ok()
            .context("Present failed (device lost?)")?;
        // One frame in flight: simple and safe for the beta.
        self.wait_idle()
    }
}

impl Renderer for D3d12Renderer {
    fn adapter_info(&self) -> &AdapterInfo {
        &self.info
    }

    fn load_scene(&mut self, scene: &Scene) -> Result<()> {
        unsafe {
            self.wait_idle()?;
            self.scene = None;
            self.load_scene_inner(scene)
        }
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            self.width = 0;
            self.height = 0;
            return Ok(());
        }
        unsafe {
            self.wait_idle()?;
            self.targets.clear();
            self.depth = None;
            self.width = width;
            self.height = height;
            self.swapchain.ResizeBuffers(
                FRAME_COUNT,
                width,
                height,
                DXGI_FORMAT_UNKNOWN,
                DXGI_SWAP_CHAIN_FLAG(0),
            )?;
            self.create_targets()
        }
    }

    fn render(&mut self, camera: &Camera, settings: &RenderSettings) -> Result<()> {
        unsafe { self.render_inner(camera, settings) }
    }
}

impl Drop for D3d12Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.wait_idle();
            let _ = CloseHandle(self.fence_event);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// setup helpers
// ---------------------------------------------------------------------------------------------

unsafe fn pick_adapter(
    factory: &IDXGIFactory6,
    pref: GpuPreference,
) -> Result<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> {
    let mut fallback: Option<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> = None;
    let mut i = 0;
    while let Ok(adapter) =
        factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
    {
        i += 1;
        let desc = adapter.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }
        if GpuVendor::from_pci_id(desc.VendorId).matches(pref) {
            return Ok((adapter, desc));
        }
        fallback.get_or_insert((adapter, desc));
    }
    match fallback {
        Some(f) => {
            log::warn!("requested GPU vendor {pref:?} not found, using the default adapter");
            Ok(f)
        }
        None => Err(anyhow!("no hardware D3D12 adapter found")),
    }
}

unsafe fn create_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature> {
    let material_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: SRV_PER_MATERIAL,
        BaseShaderRegister: 1,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let environment_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 6,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        // 0: frame constants (b0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // 1: top-level acceleration structure (t0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // 2: per-draw constants (b1)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                    Num32BitValues: 32,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // 3: material textures (t1..t5)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &material_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // 4: HDR environment map (t6)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &environment_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_ANISOTROPIC,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        MipLODBias: 0.0,
        MaxAnisotropy: 8,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };
    let env_sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MipLODBias: 0.0,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: 1,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };
    let samplers = [sampler, env_sampler];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    let mut blob: Option<ID3DBlob> = None;
    let mut err: Option<ID3DBlob> = None;
    if let Err(e) = D3D12SerializeRootSignature(
        &desc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &mut blob,
        Some(&mut err),
    ) {
        let msg = err.map(|b| blob_to_string(&b)).unwrap_or_default();
        bail!("root signature serialization failed: {e} {msg}");
    }
    let blob = blob.context("no root signature blob")?;
    let bytes =
        std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
    Ok(device.CreateRootSignature(0, bytes)?)
}

unsafe fn blob_to_string(b: &ID3DBlob) -> String {
    String::from_utf8_lossy(std::slice::from_raw_parts(
        b.GetBufferPointer() as *const u8,
        b.GetBufferSize(),
    ))
    .into_owned()
}

unsafe fn create_pso(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    cull: D3D12_CULL_MODE,
    sample_count: u32,
    sky: bool,
) -> Result<ID3D12PipelineState> {
    let elements = [
        input_element(s!("POSITION"), DXGI_FORMAT_R32G32B32_FLOAT, 0),
        input_element(s!("NORMAL"), DXGI_FORMAT_R32G32B32_FLOAT, 12),
        input_element(s!("TANGENT"), DXGI_FORMAT_R32G32B32A32_FLOAT, 24),
        input_element(s!("TEXCOORD"), DXGI_FORMAT_R32G32_FLOAT, 40),
    ];
    let rt_blend = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: false.into(),
        LogicOpEnable: false.into(),
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };
    let stencil_op = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as *const c_void,
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr() as *const c_void,
            BytecodeLength: ps.len(),
        },
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: false.into(),
            RenderTarget: [rt_blend; 8],
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: cull,
            FrontCounterClockwise: true.into(),
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
            DepthClipEnable: true.into(),
            MultisampleEnable: (sample_count > 1).into(),
            AntialiasedLineEnable: false.into(),
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: (!sky).into(),
            DepthWriteMask: if sky {
                D3D12_DEPTH_WRITE_MASK_ZERO
            } else {
                D3D12_DEPTH_WRITE_MASK_ALL
            },
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            StencilReadMask: 0xFF,
            StencilWriteMask: 0xFF,
            FrontFace: stencil_op,
            BackFace: stencil_op,
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: if sky {
                std::ptr::null()
            } else {
                elements.as_ptr()
            },
            NumElements: if sky { 0 } else { elements.len() as u32 },
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        ..Default::default()
    };
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    let result = device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc);
    ManuallyDrop::drop(&mut desc.pRootSignature);
    Ok(result?)
}

unsafe fn select_msaa_samples(device: &ID3D12Device, requested: u32) -> u32 {
    let requested = requested.clamp(1, 8);
    for count in [8, 4, 2] {
        if count > requested {
            continue;
        }
        let supports = |format| {
            let mut quality = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
                Format: format,
                SampleCount: count,
                Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
                NumQualityLevels: 0,
            };
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                    &mut quality as *mut _ as *mut c_void,
                    size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as u32,
                )
                .is_ok()
                && quality.NumQualityLevels > 0
        };
        if supports(DXGI_FORMAT_R8G8B8A8_UNORM) && supports(DXGI_FORMAT_D32_FLOAT) {
            return count;
        }
    }
    1
}

fn input_element(name: PCSTR, format: DXGI_FORMAT, offset: u32) -> D3D12_INPUT_ELEMENT_DESC {
    D3D12_INPUT_ELEMENT_DESC {
        SemanticName: name,
        SemanticIndex: 0,
        Format: format,
        InputSlot: 0,
        AlignedByteOffset: offset,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        InstanceDataStepRate: 0,
    }
}

// ---------------------------------------------------------------------------------------------
// DXR acceleration structures
// ---------------------------------------------------------------------------------------------

unsafe fn alloc_as_buffers(
    device5: &ID3D12Device5,
    inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    keep: &mut Vec<ID3D12Resource>,
) -> Result<(ID3D12Resource, u64)> {
    let device: ID3D12Device = device5.cast()?;
    let mut prebuild = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    device5.GetRaytracingAccelerationStructurePrebuildInfo(inputs, &mut prebuild);
    let result = create_buffer(
        &device,
        align_up(prebuild.ResultDataMaxSizeInBytes, 256),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    )?;
    let scratch = create_buffer(
        &device,
        align_up(prebuild.ScratchDataSizeInBytes, 256),
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    )?;
    let scratch_va = scratch.GetGPUVirtualAddress();
    keep.push(scratch);
    Ok((result, scratch_va))
}

unsafe fn build_blas(
    device5: &ID3D12Device5,
    list: &ID3D12GraphicsCommandList4,
    mesh: &GpuMesh,
    vertex_count: u32,
    keep: &mut Vec<ID3D12Resource>,
) -> Result<ID3D12Resource> {
    let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_R32_UINT,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: mesh.index_count,
                VertexCount: vertex_count,
                IndexBuffer: mesh.ib.GetGPUVirtualAddress(),
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: mesh.vb.GetGPUVirtualAddress(),
                    StrideInBytes: VERTEX_STRIDE as u64,
                },
            },
        },
    };
    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            pGeometryDescs: &geometry,
        },
    };
    let (result, scratch_va) = alloc_as_buffers(device5, &inputs, keep)?;
    let build = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        DestAccelerationStructureData: result.GetGPUVirtualAddress(),
        Inputs: inputs,
        SourceAccelerationStructureData: 0,
        ScratchAccelerationStructureData: scratch_va,
    };
    list.BuildRaytracingAccelerationStructure(&build, None);
    Ok(result)
}

unsafe fn build_tlas(
    device5: &ID3D12Device5,
    list: &ID3D12GraphicsCommandList4,
    meshes: &[GpuMesh],
    instances: &[(usize, usize, Mat4)],
    keep: &mut Vec<ID3D12Resource>,
) -> Result<ID3D12Resource> {
    let device: ID3D12Device = device5.cast()?;
    let mut descs: Vec<D3D12_RAYTRACING_INSTANCE_DESC> = Vec::with_capacity(instances.len());
    for (i, (mesh, _, m)) in instances.iter().enumerate() {
        let Some(blas) = &meshes[*mesh].blas else {
            continue;
        };
        // 3x4 row-major from a column-major matrix.
        let mut t = [0.0f32; 12];
        for row in 0..3 {
            for col in 0..4 {
                t[row * 4 + col] = m.col(col)[row];
            }
        }
        descs.push(D3D12_RAYTRACING_INSTANCE_DESC {
            Transform: t,
            _bitfield1: (i as u32 & 0x00FF_FFFF) | (0xFF << 24), // InstanceID | InstanceMask
            _bitfield2: 0,                                       // hit group offset | flags
            AccelerationStructure: blas.GetGPUVirtualAddress(),
        });
    }
    if descs.is_empty() {
        bail!("no instances to build a TLAS from");
    }
    let bytes = std::slice::from_raw_parts(
        descs.as_ptr() as *const u8,
        descs.len() * size_of::<D3D12_RAYTRACING_INSTANCE_DESC>(),
    );
    let inst_buf = create_buffer(
        &device,
        bytes.len() as u64,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_FLAG_NONE,
    )?;
    write_to_buffer(&inst_buf, 0, bytes)?;
    let inst_va = inst_buf.GetGPUVirtualAddress();
    keep.push(inst_buf);

    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: descs.len() as u32,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            InstanceDescs: inst_va,
        },
    };
    let (result, scratch_va) = alloc_as_buffers(device5, &inputs, keep)?;
    let build = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        DestAccelerationStructureData: result.GetGPUVirtualAddress(),
        Inputs: inputs,
        SourceAccelerationStructureData: 0,
        ScratchAccelerationStructureData: scratch_va,
    };
    list.BuildRaytracingAccelerationStructure(&build, None);
    Ok(result)
}

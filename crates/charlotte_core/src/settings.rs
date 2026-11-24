use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuPreference {
    /// Highest-performance adapter the OS reports.
    #[default]
    Auto,
    Nvidia,
    Amd,
    Intel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Microsoft,
    Other(u32),
}

impl GpuVendor {
    pub fn from_pci_id(id: u32) -> Self {
        match id {
            0x10DE => Self::Nvidia,
            0x1002 | 0x1022 => Self::Amd,
            0x8086 => Self::Intel,
            0x1414 => Self::Microsoft,
            other => Self::Other(other),
        }
    }

    pub fn matches(self, pref: GpuPreference) -> bool {
        matches!(
            (pref, self),
            (GpuPreference::Auto, _)
                | (GpuPreference::Nvidia, GpuVendor::Nvidia)
                | (GpuPreference::Amd, GpuVendor::Amd)
                | (GpuPreference::Intel, GpuVendor::Intel)
        )
    }
}

#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub backend: &'static str,
    pub name: String,
    pub vendor: GpuVendor,
    pub dedicated_vram_mb: u64,
    /// True when hardware ray tracing (DXR 1.1 / ray query) is usable.
    pub ray_tracing: bool,
}

#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub gpu: GpuPreference,
    pub vsync: bool,
    /// Enable the GPU validation / debug layers (slow, needs the Graphics Tools feature on Windows).
    pub debug_layer: bool,
    /// Master switch for ray tracing. When false no acceleration structures are built.
    pub ray_tracing: bool,
    pub rt_shadows: bool,
    /// Ray-traced ambient occlusion. Noisy: there is no denoiser yet.
    pub rt_ao: bool,
    pub rt_ao_radius: f32,
    pub rt_ao_strength: f32,
    /// Requested multisample anti-aliasing level. Backends select the highest supported
    /// level at or below this request; 1 disables MSAA.
    pub msaa_samples: u32,
    /// Number of cosine-weighted DXR AO rays per pixel (1..=32).
    pub rt_ao_samples: u32,
    /// Rays used to sample the angular size of punctual light sources (1..=16).
    pub rt_shadow_samples: u32,
    /// Angular radius in radians used to soften ray-traced shadows.
    pub rt_shadow_softness: f32,
    /// Image-based lighting contribution multiplier.
    pub ibl_strength: f32,
    /// Linear multiplier applied before tone mapping.
    pub exposure: f32,
    pub sky_color: [f32; 3],
    pub ground_color: [f32; 3],
    pub clear_color: [f32; 4],
    /// If the scene has no lights, add a default sun so it is not black.
    pub default_light_if_none: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            gpu: GpuPreference::Auto,
            vsync: true,
            debug_layer: cfg!(debug_assertions),
            ray_tracing: true,
            rt_shadows: true,
            rt_ao: true,
            rt_ao_radius: 0.9,
            rt_ao_strength: 1.2,
            msaa_samples: 4,
            rt_ao_samples: 16,
            rt_shadow_samples: 8,
            rt_shadow_softness: 0.008,
            ibl_strength: 1.0,
            exposure: 1.0,
            sky_color: [0.35, 0.45, 0.6],
            ground_color: [0.12, 0.11, 0.10],
            clear_color: [0.05, 0.06, 0.08, 1.0],
            default_light_if_none: true,
        }
    }
}

/// Everything a backend needs to attach to a window.
#[derive(Clone, Copy, Debug)]
pub struct WindowHandles {
    pub window: RawWindowHandle,
    pub display: RawDisplayHandle,
    pub width: u32,
    pub height: u32,
}

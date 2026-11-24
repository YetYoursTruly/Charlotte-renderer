//! Backend-independent types shared by every Charlotte renderer backend.

mod camera;
mod primitives;
mod scene;
mod settings;

pub use camera::Camera;
pub use primitives::{plane, uv_sphere};
pub use scene::{
    default_sun, AlphaMode, EnvironmentMap, ImageData, Light, LightKind, Material, MeshData,
    MeshInstance, Scene, Vertex,
};
pub use settings::{AdapterInfo, GpuPreference, GpuVendor, RenderSettings, WindowHandles};

pub use anyhow;
pub use glam;
pub use raw_window_handle;

/// Implemented by every backend (D3D12, Vulkan).
pub trait Renderer {
    fn adapter_info(&self) -> &AdapterInfo;
    /// Uploads a scene to the GPU, replacing any previously loaded scene.
    fn load_scene(&mut self, scene: &Scene) -> anyhow::Result<()>;
    fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()>;
    /// Renders and presents one frame.
    fn render(&mut self, camera: &Camera, settings: &RenderSettings) -> anyhow::Result<()>;
}

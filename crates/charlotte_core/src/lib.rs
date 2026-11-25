//! Backend-independent types shared by every Charlotte renderer backend.

mod camera;
mod primitives;
mod scene;
mod settings;

pub use camera::Camera;
pub use primitives::{plane, uv_sphere};
pub use scene::{
    default_sun, AlphaMode, DynamicDraw, EnvironmentMap, ImageData, Light, LightKind, Material,
    LodLevel, MeshData, MeshInstance, Scene, Vertex,
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
    /// Replaces the dynamic transparent instances rendered after the static scene.
    /// Instance mesh and material indices refer to the scene passed to `load_scene`.
    /// Dynamic instances do not participate in ray-tracing acceleration structures.
    /// Backends without a dynamic draw path return an error.
    fn set_dynamic_instances(&mut self, _instances: &[MeshInstance]) -> anyhow::Result<()> {
        anyhow::bail!("this renderer backend does not support dynamic instances")
    }
    /// Replaces transient transparent draws, optionally overriding each draw's
    /// material base color. The default preserves compatibility by ignoring
    /// overrides on backends that only implement `set_dynamic_instances`.
    fn set_dynamic_draws(&mut self, draws: &[DynamicDraw]) -> anyhow::Result<()> {
        let instances: Vec<_> = draws
            .iter()
            .map(|draw| MeshInstance {
                mesh: draw.mesh,
                material: draw.material,
                transform: draw.transform,
            })
            .collect();
        self.set_dynamic_instances(&instances)
    }
    /// Renders and presents one frame.
    fn render(&mut self, camera: &Camera, settings: &RenderSettings) -> anyhow::Result<()>;
}

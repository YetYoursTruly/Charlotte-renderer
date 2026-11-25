//! Charlotte facade crate. Add `charlotte-renderer = "0.1.0-beta.2"` to Cargo.toml.
//! and pick a backend at runtime.
//!
//! ```ignore
//! let scene = charlotte_renderer::gltf::load_gltf("model.glb")?;
//! let mut renderer = charlotte_renderer::create_renderer(Backend::Auto, &handles, &settings)?;
//! renderer.load_scene(&scene)?;
//! loop { renderer.render(&camera, &settings)?; }
//! ```

pub use charlotte_core::*;
#[cfg(feature = "gltf")]
pub use charlotte_gltf as gltf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Backend {
    /// Direct3D 12 on Windows, otherwise Vulkan. Never silently changes backend on failure.
    #[default]
    Auto,
    D3D12,
    /// Experimental: currently only presents a cleared frame.
    Vulkan,
}

pub fn create_renderer(
    backend: Backend,
    handles: &WindowHandles,
    settings: &RenderSettings,
) -> anyhow::Result<Box<dyn Renderer>> {
    match backend {
        Backend::D3D12 => d3d12(handles, settings),
        Backend::Vulkan => vulkan(handles, settings),
        Backend::Auto => {
            if cfg!(windows) {
                d3d12(handles, settings)
            } else {
                vulkan(handles, settings)
            }
        }
    }
}

#[cfg(all(feature = "d3d12", windows))]
fn d3d12(h: &WindowHandles, s: &RenderSettings) -> anyhow::Result<Box<dyn Renderer>> {
    Ok(Box::new(charlotte_d3d12::D3d12Renderer::new(h, s)?))
}

#[cfg(not(all(feature = "d3d12", windows)))]
fn d3d12(_: &WindowHandles, _: &RenderSettings) -> anyhow::Result<Box<dyn Renderer>> {
    anyhow::bail!("the Direct3D 12 backend needs Windows and the `d3d12` feature")
}

#[cfg(feature = "vulkan")]
fn vulkan(h: &WindowHandles, s: &RenderSettings) -> anyhow::Result<Box<dyn Renderer>> {
    Ok(Box::new(charlotte_vulkan::VulkanRenderer::new(h, s)?))
}

#[cfg(not(feature = "vulkan"))]
fn vulkan(_: &WindowHandles, _: &RenderSettings) -> anyhow::Result<Box<dyn Renderer>> {
    anyhow::bail!("the Vulkan backend needs the `vulkan` feature")
}

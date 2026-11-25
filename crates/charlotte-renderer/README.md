# charlotte-renderer

Charlotte is a Direct3D 12-first PBR renderer for Rust. It provides scene,
camera, material and renderer APIs, with optional glTF/GLB loading.

Add the published Charlotte beta to an application with:

```toml
[dependencies]
charlotte-renderer = "0.1.0-beta.2"
```

The facade crate re-exports the core types and offers the `gltf` module by
default. On Windows, `Backend::Auto` selects Direct3D 12. The Vulkan backend is
experimental and currently only clears and presents a frame.

The Windows D3D12 build provisions Microsoft's DXC runtime automatically and
copies its two DLLs next to the application executable. The first build needs
internet access; set `CHARLOTTE_DXC_DIR` to use an existing local runtime.

See the workspace README for the current renderer capabilities and limitations.

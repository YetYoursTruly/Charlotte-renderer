# charlotte_d3d12

Windows Direct3D 12 backend for Charlotte's PBR renderer. Most applications
should use the `charlotte` facade and select `Backend::Auto` or `Backend::D3D12`.

On Windows, the crate's build script provisions the pinned Microsoft DXC runtime
and copies `dxcompiler.dll` and `dxil.dll` next to the Cargo-built executable.
The first build needs network access to NuGet. To use an existing installation,
set `CHARLOTTE_DXC_DIR` to the directory containing both DLLs.

use anyhow::{anyhow, Result};

const SOURCE: &str = include_str!("../shaders/pbr.hlsl");

/// Compiles the embedded HLSL with DXC. DXC loads its runtime DLLs from the process
/// directory or PATH; shader-source errors are reported separately from loader errors.
pub fn compile(entry: &str, profile: &str, rt: bool) -> Result<Vec<u8>> {
    let defines = [("CHARLOTTE_RT", Some(if rt { "1" } else { "0" }))];
    hassle_rs::compile_hlsl("pbr.hlsl", SOURCE, entry, profile, &["-O3"], &defines)
        .map_err(|e| anyhow!("DXC failed to compile {entry} ({profile}): {e}"))
}

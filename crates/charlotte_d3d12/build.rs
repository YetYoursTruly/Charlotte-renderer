use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DXC_VERSION: &str = "1.9.2609.5";
const DXC_PACKAGE_URL: &str = "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.dxc/1.9.2609.5/microsoft.direct3d.dxc.1.9.2609.5.nupkg";

fn main() {
    println!("cargo:rerun-if-env-changed=CHARLOTTE_DXC_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("windows") {
        return;
    }

    if let Err(error) = provision_dxc() {
        panic!(
            "failed to provision the DirectX Shader Compiler runtime: {error}\n\
               Install DXC from Microsoft's Windows SDK, or set CHARLOTTE_DXC_DIR to a folder \
               containing dxcompiler.dll and dxil.dll. The runtime DLLs are copied next to the \
               Cargo-built application executable."
        );
    }
}

fn provision_dxc() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is not set")?);
    // OUT_DIR = target[/<triple>]/<profile>/build/<crate-hash>/out.
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .ok_or("could not determine Cargo's target profile directory")?;
    fs::create_dir_all(profile_dir)?;

    let target_runtime = has_runtime(profile_dir);
    if target_runtime {
        println!(
            "cargo:warning=Using DXC runtime already present in {}",
            profile_dir.display()
        );
        return Ok(());
    }

    let source_dir = match env::var_os("CHARLOTTE_DXC_DIR") {
        Some(path) => PathBuf::from(path),
        None => download_and_extract(&out_dir)?,
    };

    for dll in ["dxcompiler.dll", "dxil.dll"] {
        let source = source_dir.join(dll);
        if !source.is_file() {
            return Err(format!("{} is missing from {}", dll, source_dir.display()).into());
        }
        copy_if_different(&source, &profile_dir.join(dll))?;
    }

    // Preserve Microsoft's redistributable notice alongside the copied runtime.
    if let Some(license) = find_file(&out_dir.join("charlotte-dxc-package"), "LICENSE-LLVM.txt")? {
        copy_if_different(&license, &profile_dir.join("DXC-LICENSE-LLVM.txt"))?;
    }

    println!(
        "cargo:warning=DXC {} runtime copied to {}",
        DXC_VERSION,
        profile_dir.display()
    );
    Ok(())
}

fn has_runtime(dir: &Path) -> bool {
    dir.join("dxcompiler.dll").is_file() && dir.join("dxil.dll").is_file()
}

fn download_and_extract(out_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let archive = out_dir.join(format!("microsoft.direct3d.dxc-{DXC_VERSION}.zip"));
    let package_dir = out_dir.join("charlotte-dxc-package");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;
    let architecture = match target_arch.as_str() {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => return Err(format!("DXC runtime is not configured for Windows {other}").into()),
    };
    let dll_dir = package_dir
        .join("build")
        .join("native")
        .join("bin")
        .join(architecture);
    if has_runtime(&dll_dir) {
        return Ok(dll_dir);
    }

    if !archive.is_file() {
        println!(
            "cargo:warning=Downloading Microsoft's DirectX Shader Compiler {} for this build",
            DXC_VERSION
        );
        download_archive(&archive)?;
    }
    fs::create_dir_all(&package_dir)?;
    extract_archive(&archive, &package_dir)?;
    if !has_runtime(&dll_dir) {
        return Err(format!(
            "Microsoft DXC NuGet package did not contain both runtime DLLs under {}",
            dll_dir.display()
        )
        .into());
    }
    Ok(dll_dir)
}

fn download_archive(destination: &Path) -> Result<(), Box<dyn Error>> {
    let curl = Command::new("curl.exe")
        .args([
            "--fail",
            "--location",
            "--retry",
            "3",
            "--silent",
            "--show-error",
        ])
        .arg("--output")
        .arg(destination)
        .arg(DXC_PACKAGE_URL)
        .output();
    if let Ok(output) = curl {
        if output.status.success() {
            return Ok(());
        }
        eprintln!(
            "curl.exe could not download DXC: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let url = ps_quote(DXC_PACKAGE_URL);
    let path = ps_quote(&destination.to_string_lossy());
    let script = format!(
        "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '{url}' -OutFile '{path}'"
    );
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "DXC download failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

fn extract_archive(archive: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    let archive = ps_quote(&archive.to_string_lossy());
    let destination = ps_quote(&destination.to_string_lossy());
    let script = format!(
        "$ErrorActionPreference='Stop'; Expand-Archive -LiteralPath '{archive}' -DestinationPath '{destination}' -Force"
    );
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "could not extract the DXC package: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

fn ps_quote(value: &str) -> String {
    value.replace('\'', "''")
}

fn copy_if_different(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    if let (Ok(source), Ok(destination)) = (source.canonicalize(), destination.canonicalize()) {
        if source == destination {
            return Ok(());
        }
    }
    fs::copy(source, destination)?;
    Ok(())
}

fn find_file(root: &Path, filename: &str) -> Result<Option<PathBuf>, Box<dyn Error>> {
    if !root.is_dir() {
        return Ok(None);
    }
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_file() && path.file_name().is_some_and(|name| name == filename) {
            return Ok(Some(path));
        }
        if path.is_dir() {
            if let Some(found) = find_file(&path, filename)? {
                return Ok(Some(found));
            }
        }
    }
    Ok(None)
}

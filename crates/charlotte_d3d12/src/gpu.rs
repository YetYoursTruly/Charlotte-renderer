//! Small D3D12 helpers: buffers, barriers, texture upload.

use std::ffi::c_void;
use std::mem::ManuallyDrop;

use anyhow::{Context, Result};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) & !(a - 1)
}

pub fn heap_props(ty: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: ty,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    }
}

pub fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    }
}

pub unsafe fn create_buffer(
    device: &ID3D12Device,
    size: u64,
    heap: D3D12_HEAP_TYPE,
    state: D3D12_RESOURCE_STATES,
    flags: D3D12_RESOURCE_FLAGS,
) -> Result<ID3D12Resource> {
    let mut res: Option<ID3D12Resource> = None;
    device.CreateCommittedResource(
        &heap_props(heap),
        D3D12_HEAP_FLAG_NONE,
        &buffer_desc(size.max(256), flags),
        state,
        None,
        &mut res,
    )?;
    res.context("CreateCommittedResource returned no resource")
}

pub unsafe fn write_to_buffer(res: &ID3D12Resource, offset: usize, data: &[u8]) -> Result<()> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    res.Map(0, None, Some(&mut ptr))?;
    std::ptr::copy_nonoverlapping(data.as_ptr(), (ptr as *mut u8).add(offset), data.len());
    res.Unmap(0, None);
    Ok(())
}

pub unsafe fn transition(
    list: &ID3D12GraphicsCommandList4,
    res: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(res.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    list.ResourceBarrier(std::slice::from_ref(&barrier));
    // Release the COM reference we cloned into the barrier.
    let t = ManuallyDrop::into_inner(barrier.Anonymous.Transition);
    drop(ManuallyDrop::into_inner(t.pResource));
}

/// UAV barrier on all resources (used between acceleration structure builds).
pub unsafe fn uav_barrier_all(list: &ID3D12GraphicsCommandList4) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: ManuallyDrop::new(None),
            }),
        },
    };
    list.ResourceBarrier(std::slice::from_ref(&barrier));
}

/// Creates a GPU-local buffer filled with `data`. The staging buffer is pushed to `keep`
/// and must stay alive until the command list has finished executing.
pub unsafe fn upload_to_default(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList4,
    data: &[u8],
    keep: &mut Vec<ID3D12Resource>,
) -> Result<ID3D12Resource> {
    let size = data.len().max(16) as u64;
    let dst = create_buffer(
        device,
        size,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE,
    )?;
    let up = create_buffer(
        device,
        size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_FLAG_NONE,
    )?;
    write_to_buffer(&up, 0, data)?;
    list.CopyBufferRegion(&dst, 0, &up, 0, data.len() as u64);
    transition(
        list,
        &dst,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    keep.push(up);
    Ok(dst)
}

/// One mip level of tightly packed RGBA8 pixels.
pub struct Mip {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

struct HdrMip {
    width: u32,
    height: u32,
    data: Vec<[f32; 4]>,
}

fn build_hdr_mips(width: u32, height: u32, rgba: &[[f32; 4]]) -> Vec<HdrMip> {
    let mut mips = vec![HdrMip {
        width,
        height,
        data: rgba.to_vec(),
    }];
    while mips.last().is_some_and(|m| m.width > 1 || m.height > 1) {
        let prev = mips.last().unwrap();
        let (w, h) = ((prev.width / 2).max(1), (prev.height / 2).max(1));
        let mut data = vec![[0.0; 4]; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let x0 = x * 2;
                let x1 = (x0 + 1) % prev.width;
                let y0 = y * 2;
                let y1 = (y0 + 1).min(prev.height - 1);
                let coords = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
                for c in 0..4 {
                    data[(y * w + x) as usize][c] = coords
                        .iter()
                        .map(|&(sx, sy)| prev.data[(sy * prev.width + sx) as usize][c])
                        .sum::<f32>()
                        * 0.25;
                }
            }
        }
        mips.push(HdrMip {
            width: w,
            height: h,
            data,
        });
    }
    mips
}

/// Uploads a linear HDR equirectangular image with a filtered mip chain for rough reflections.
pub unsafe fn create_environment_texture(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList4,
    width: u32,
    height: u32,
    rgba: &[[f32; 4]],
    keep: &mut Vec<ID3D12Resource>,
) -> Result<(ID3D12Resource, u32)> {
    let mips = build_hdr_mips(width, height, rgba);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: mips.len() as u16,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut tex: Option<ID3D12Resource> = None;
    device.CreateCommittedResource(
        &heap_props(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        &mut tex,
    )?;
    let tex = tex.context("HDR environment texture creation failed")?;
    let n = mips.len();
    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut row_sizes = vec![0u64; n];
    let mut total = 0u64;
    device.GetCopyableFootprints(
        &desc,
        0,
        n as u32,
        0,
        Some(layouts.as_mut_ptr()),
        Some(rows.as_mut_ptr()),
        Some(row_sizes.as_mut_ptr()),
        Some(&mut total),
    );
    let upload = create_buffer(
        device,
        total,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_FLAG_NONE,
    )?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    upload.Map(0, None, Some(&mut ptr))?;
    for (i, mip) in mips.iter().enumerate() {
        let row_bytes = (mip.width * 16) as usize;
        for y in 0..rows[i] as usize {
            let dst = (ptr as *mut u8)
                .add(layouts[i].Offset as usize + y * layouts[i].Footprint.RowPitch as usize);
            std::ptr::copy_nonoverlapping(
                mip.data.as_ptr().cast::<u8>().add(y * row_bytes),
                dst,
                row_bytes,
            );
        }
    }
    upload.Unmap(0, None);
    for i in 0..n {
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(upload.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layouts[i],
            },
        };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        drop(ManuallyDrop::into_inner(dst.pResource));
        drop(ManuallyDrop::into_inner(src.pResource));
    }
    transition(
        list,
        &tex,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    );
    keep.push(upload);
    Ok((tex, n as u32))
}

fn srgb_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Box-filter mip chain. Colour data is averaged in linear space; RGB only (alpha is always linear).
pub fn build_mips(width: u32, height: u32, rgba: &[u8], srgb: bool) -> Vec<Mip> {
    let lut: Vec<f32> = (0..256).map(|i| srgb_to_linear(i as u8)).collect();
    let mut mips = vec![Mip {
        width,
        height,
        data: rgba.to_vec(),
    }];
    loop {
        let prev = mips.last().unwrap();
        if prev.width == 1 && prev.height == 1 {
            break;
        }
        let (w, h) = ((prev.width / 2).max(1), (prev.height / 2).max(1));
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let x0 = (x * 2).min(prev.width - 1);
                let x1 = (x * 2 + 1).min(prev.width - 1);
                let y0 = (y * 2).min(prev.height - 1);
                let y1 = (y * 2 + 1).min(prev.height - 1);
                for c in 0..4 {
                    let px =
                        |xx: u32, yy: u32| prev.data[((yy * prev.width + xx) * 4) as usize + c];
                    let samples = [px(x0, y0), px(x1, y0), px(x0, y1), px(x1, y1)];
                    let v = if srgb && c < 3 {
                        let avg = samples.iter().map(|&s| lut[s as usize]).sum::<f32>() * 0.25;
                        linear_to_srgb(avg)
                    } else {
                        ((samples.iter().map(|&s| s as u32).sum::<u32>() + 2) / 4) as u8
                    };
                    data[((y * w + x) * 4) as usize + c] = v;
                }
            }
        }
        mips.push(Mip {
            width: w,
            height: h,
            data,
        });
    }
    mips
}

/// Creates a shader-readable RGBA8 texture with a full mip chain and records the upload.
pub unsafe fn create_texture(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList4,
    width: u32,
    height: u32,
    rgba: &[u8],
    srgb: bool,
    keep: &mut Vec<ID3D12Resource>,
) -> Result<(ID3D12Resource, u32)> {
    let mips = build_mips(width, height, rgba, srgb);
    let format = if srgb {
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
    } else {
        DXGI_FORMAT_R8G8B8A8_UNORM
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: mips.len() as u16,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut tex: Option<ID3D12Resource> = None;
    device.CreateCommittedResource(
        &heap_props(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        &mut tex,
    )?;
    let tex = tex.context("texture creation failed")?;

    let n = mips.len();
    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut row_sizes = vec![0u64; n];
    let mut total = 0u64;
    device.GetCopyableFootprints(
        &desc,
        0,
        n as u32,
        0,
        Some(layouts.as_mut_ptr()),
        Some(rows.as_mut_ptr()),
        Some(row_sizes.as_mut_ptr()),
        Some(&mut total),
    );
    let upload = create_buffer(
        device,
        total,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_FLAG_NONE,
    )?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    upload.Map(0, None, Some(&mut ptr))?;
    for (i, mip) in mips.iter().enumerate() {
        let row_bytes = (mip.width * 4) as usize;
        for y in 0..rows[i] as usize {
            let dst = (ptr as *mut u8)
                .add(layouts[i].Offset as usize + y * layouts[i].Footprint.RowPitch as usize);
            std::ptr::copy_nonoverlapping(mip.data.as_ptr().add(y * row_bytes), dst, row_bytes);
        }
    }
    upload.Unmap(0, None);

    for i in 0..n {
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(upload.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layouts[i],
            },
        };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        drop(ManuallyDrop::into_inner(dst.pResource));
        drop(ManuallyDrop::into_inner(src.pResource));
    }
    transition(
        list,
        &tex,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    );
    keep.push(upload);
    Ok((tex, n as u32))
}

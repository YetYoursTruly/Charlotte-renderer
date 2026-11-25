//! EXPERIMENTAL Vulkan 1.3 backend.
//!
//! What works: instance + validation layer, NVIDIA / AMD / Intel device selection, surface,
//! swapchain (with resize), dynamic rendering, per-frame sync, presenting a cleared frame, and
//! detection of `VK_KHR_ray_query` + `VK_KHR_acceleration_structure` support.
//!
//! What does NOT work yet: drawing the scene (no pipelines, textures or ray tracing). `load_scene`
//! only records statistics. Use the D3D12 backend for actual rendering on Windows.

use std::ffi::{c_char, CStr};

use anyhow::{bail, Context, Result};
use ash::{khr, vk, Device, Entry, Instance};
use charlotte_core::{
    AdapterInfo, Camera, GpuPreference, GpuVendor, RenderSettings, Renderer, Scene, WindowHandles,
};

/// Parses and validates WGSL and emits Vulkan-ready SPIR-V. Keeping shader
/// source in WGSL gives the Vulkan backend a compiler path without requiring
/// an external GLSL compiler installation on end-user machines.
fn compile_wgsl(source: &str) -> Result<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| anyhow::anyhow!("WGSL parse failed: {}", error.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| anyhow::anyhow!("WGSL validation failed: {error}"))?;
    naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        None,
    )
    .map_err(|error| anyhow::anyhow!("SPIR-V generation failed: {error}"))
}

pub struct VulkanRenderer {
    info: AdapterInfo,
    vsync: bool,
    _entry: Entry,
    instance: Instance,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    phys: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    swapchain_loader: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    render_finished: Vec<vk::Semaphore>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    needs_recreate: bool,
}

struct Picked {
    phys: vk::PhysicalDevice,
    queue_family: u32,
    info: AdapterInfo,
}

impl VulkanRenderer {
    pub fn new(handles: &WindowHandles, settings: &RenderSettings) -> Result<Self> {
        unsafe { Self::new_inner(handles, settings) }
    }

    unsafe fn new_inner(handles: &WindowHandles, settings: &RenderSettings) -> Result<Self> {
        let entry = Entry::load()
            .context("could not load the Vulkan loader (is a Vulkan driver installed?)")?;

        let app_name = c"Charlotte";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .engine_name(app_name)
            .api_version(vk::API_VERSION_1_3);
        let extensions: Vec<*const c_char> =
            ash_window::enumerate_required_extensions(handles.display)?.to_vec();

        let validation = c"VK_LAYER_KHRONOS_validation";
        let mut layers: Vec<*const c_char> = Vec::new();
        if settings.debug_layer {
            let available = entry.enumerate_instance_layer_properties()?;
            if available
                .iter()
                .any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == validation)
            {
                layers.push(validation.as_ptr());
            } else {
                log::warn!("Vulkan validation layer requested but not installed");
            }
        }
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extensions)
                .enabled_layer_names(&layers),
            None,
        )?;
        let surface =
            ash_window::create_surface(&entry, &instance, handles.display, handles.window, None)?;
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let picked = pick_device(&instance, &surface_loader, surface, settings.gpu)?;
        log::info!(
            "Vulkan adapter: {} ({:?}, {} MB, ray query: {})",
            picked.info.name,
            picked.info.vendor,
            picked.info.dedicated_vram_mb,
            picked.info.ray_tracing
        );

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(picked.queue_family)
            .queue_priorities(&priorities)];
        let device_exts = [khr::swapchain::NAME.as_ptr()];
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let device = instance.create_device(
            picked.phys,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_exts)
                .push_next(&mut f13),
            None,
        )?;
        let queue = device.get_device_queue(picked.queue_family, 0);
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        let pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(picked.queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let cmd = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let image_available = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
        let in_flight = device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )?;

        let mut r = Self {
            info: picked.info,
            vsync: settings.vsync,
            _entry: entry,
            instance,
            surface_loader,
            surface,
            phys: picked.phys,
            device,
            queue,
            swapchain_loader,
            swapchain: vk::SwapchainKHR::null(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
            images: Vec::new(),
            views: Vec::new(),
            render_finished: Vec::new(),
            pool,
            cmd,
            image_available,
            in_flight,
            needs_recreate: false,
        };
        r.create_swapchain(handles.width, handles.height)?;
        Ok(r)
    }

    unsafe fn destroy_swapchain_resources(&mut self) {
        for &s in &self.render_finished {
            self.device.destroy_semaphore(s, None);
        }
        for &v in &self.views {
            self.device.destroy_image_view(v, None);
        }
        self.render_finished.clear();
        self.views.clear();
    }

    unsafe fn create_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        let caps = self
            .surface_loader
            .get_physical_device_surface_capabilities(self.phys, self.surface)?;
        let formats = self
            .surface_loader
            .get_physical_device_surface_formats(self.phys, self.surface)?;
        let fmt = formats
            .iter()
            .find(|f| {
                matches!(
                    f.format,
                    vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                ) && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .or_else(|| formats.first().copied())
            .context("surface reports no formats")?;
        let modes = self
            .surface_loader
            .get_physical_device_surface_present_modes(self.phys, self.surface)?;
        let present_mode = if !self.vsync && modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if !self.vsync && modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        };
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        let mut count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            count = count.min(caps.max_image_count);
        }

        let old = self.swapchain;
        let swapchain = self.swapchain_loader.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(count)
                .image_format(fmt.format)
                .image_color_space(fmt.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(old),
            None,
        )?;
        if old != vk::SwapchainKHR::null() {
            self.swapchain_loader.destroy_swapchain(old, None);
        }
        self.destroy_swapchain_resources();

        self.swapchain = swapchain;
        self.format = fmt.format;
        self.extent = extent;
        self.images = self.swapchain_loader.get_swapchain_images(swapchain)?;
        for &image in &self.images {
            let view = self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(fmt.format)
                    .subresource_range(color_range()),
                None,
            )?;
            self.views.push(view);
            self.render_finished.push(
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
            );
        }
        self.needs_recreate = false;
        Ok(())
    }

    unsafe fn render_inner(&mut self, settings: &RenderSettings) -> Result<()> {
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(());
        }
        let d = &self.device;
        d.wait_for_fences(&[self.in_flight], true, u64::MAX)?;
        if self.needs_recreate {
            d.device_wait_idle()?;
            self.create_swapchain(self.extent.width, self.extent.height)?;
        }
        let (index, suboptimal) = match self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available,
            vk::Fence::null(),
        ) {
            Ok(v) => v,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_recreate = true;
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };
        if suboptimal {
            self.needs_recreate = true;
        }
        let d = &self.device;
        d.reset_fences(&[self.in_flight])?;
        d.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
        d.begin_command_buffer(
            self.cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let image = self.images[index as usize];
        let to_attachment = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(color_range())];
        d.cmd_pipeline_barrier2(
            self.cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&to_attachment),
        );

        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.views[index as usize])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: settings.clear_color,
                },
            })];
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.extent,
            })
            .layer_count(1)
            .color_attachments(&attachments);
        d.cmd_begin_rendering(self.cmd, &rendering);
        // TODO: scene pass (pipelines, descriptors, PBR, ray query) goes here.
        d.cmd_end_rendering(self.cmd);

        let to_present = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::empty())
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .subresource_range(color_range())];
        d.cmd_pipeline_barrier2(
            self.cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&to_present),
        );
        d.end_command_buffer(self.cmd)?;

        let wait = [self.image_available];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmds = [self.cmd];
        let signal = [self.render_finished[index as usize]];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&cmds)
            .signal_semaphores(&signal);
        d.queue_submit(self.queue, &[submit], self.in_flight)?;

        let swapchains = [self.swapchain];
        let indices = [index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal)
            .swapchains(&swapchains)
            .image_indices(&indices);
        match self.swapchain_loader.queue_present(self.queue, &present) {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_recreate = true,
            Err(e) => return Err(e.into()),
        }
        Ok(())
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

unsafe fn pick_device(
    instance: &Instance,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
    pref: GpuPreference,
) -> Result<Picked> {
    let mut best: Option<(i32, Picked)> = None;
    for phys in instance.enumerate_physical_devices()? {
        let props = instance.get_physical_device_properties(phys);
        if props.api_version < vk::API_VERSION_1_3 {
            continue;
        }
        let exts = instance.enumerate_device_extension_properties(phys)?;
        let has = |name: &CStr| {
            exts.iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == name)
        };
        if !has(khr::swapchain::NAME) {
            continue;
        }
        let ok13 = {
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f13);
            instance.get_physical_device_features2(phys, &mut f2);
            f13.dynamic_rendering == vk::TRUE && f13.synchronization2 == vk::TRUE
        };
        if !ok13 {
            continue;
        }
        let families = instance.get_physical_device_queue_family_properties(phys);
        let Some(queue_family) = families.iter().enumerate().position(|(i, f)| {
            f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && surface_loader
                    .get_physical_device_surface_support(phys, i as u32, surface)
                    .unwrap_or(false)
        }) else {
            continue;
        };

        let vendor = GpuVendor::from_pci_id(props.vendor_id);
        let mem = instance.get_physical_device_memory_properties(phys);
        let vram: u64 = mem.memory_heaps[..mem.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum();
        let mut score = 0;
        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            score += 1000;
        }
        if pref != GpuPreference::Auto && vendor.matches(pref) {
            score += 10_000;
        }
        let info = AdapterInfo {
            backend: "Vulkan (experimental)",
            name: CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned(),
            vendor,
            dedicated_vram_mb: vram / (1024 * 1024),
            ray_tracing: has(khr::ray_query::NAME) && has(khr::acceleration_structure::NAME),
        };
        if best.as_ref().map_or(true, |(s, _)| score > *s) {
            best = Some((
                score,
                Picked {
                    phys,
                    queue_family: queue_family as u32,
                    info,
                },
            ));
        }
    }
    match best {
        Some((_, p)) => Ok(p),
        None => bail!("no Vulkan 1.3 device with dynamic rendering and presentation support found"),
    }
}

impl Renderer for VulkanRenderer {
    fn adapter_info(&self) -> &AdapterInfo {
        &self.info
    }

    fn load_scene(&mut self, scene: &Scene) -> Result<()> {
        log::warn!(
            "Vulkan backend is experimental and cannot draw scenes yet ({} meshes, {} materials ignored)",
            scene.meshes.len(),
            scene.materials.len()
        );
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            self.extent = vk::Extent2D {
                width: 0,
                height: 0,
            };
            return Ok(());
        }
        unsafe {
            self.device.device_wait_idle()?;
            self.create_swapchain(width, height)
        }
    }

    fn render(&mut self, _camera: &Camera, settings: &RenderSettings) -> Result<()> {
        unsafe { self.render_inner(settings) }
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.destroy_swapchain_resources();
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_fence(self.in_flight, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

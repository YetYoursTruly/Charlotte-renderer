//! Usage:
//!   viewer [model.glb|model.gltf] [options]
//!   viewer --pbr-dir path/to/texture_folder     (sphere showcasing a PBR texture set)
//!
//! Options: --backend d3d12|vulkan  --gpu nvidia|amd|intel  --no-rt  --no-ao
//!          --environment path.hdr|path.exr --ao-samples N --ao-radius F --ao-strength F
//!          --shadow-samples N --shadow-softness F --ibl-strength F --msaa 1|2|4|8 --exposure F --no-vsync --fps
//! Keys:    WASD/QE move (Shift = fast), hold right mouse to look, [ ] exposure,
//!          R toggle RT shadows, O toggle RT AO, Esc quit.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Result};
use charlotte::glam::{Mat4, Vec3};
use charlotte::{
    Backend, Camera, EnvironmentMap, GpuPreference, Light, LightKind, Material, RenderSettings,
    Renderer, Scene, WindowHandles,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::CursorGrabMode;
use winit::window::{Window, WindowId};

struct Args {
    model: Option<PathBuf>,
    pbr_dir: Option<PathBuf>,
    environment: Option<PathBuf>,
    show_fps: bool,
    backend: Backend,
    settings: RenderSettings,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: None,
        pbr_dir: None,
        environment: None,
        show_fps: false,
        backend: Backend::D3D12,
        settings: RenderSettings::default(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--backend" => {
                a.backend = match it.next().as_deref() {
                    Some("d3d12") | Some("dx12") => Backend::D3D12,
                    Some("vulkan") => Backend::Vulkan,
                    other => bail!("unknown backend {other:?}"),
                }
            }
            "--gpu" => {
                a.settings.gpu = match it.next().as_deref() {
                    Some("nvidia") => GpuPreference::Nvidia,
                    Some("amd") => GpuPreference::Amd,
                    Some("intel") => GpuPreference::Intel,
                    other => bail!("unknown gpu {other:?}"),
                }
            }
            "--no-rt" => a.settings.ray_tracing = false,
            "--ao" => a.settings.rt_ao = true,
            "--no-ao" => a.settings.rt_ao = false,
            "--environment" | "--env" => a.environment = it.next().map(PathBuf::from),
            "--msaa" => {
                a.settings.msaa_samples = it.next().and_then(|v| v.parse().ok()).unwrap_or(4);
            }
            "--ao-samples" => {
                a.settings.rt_ao_samples = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(8)
                    .clamp(1, 32);
            }
            "--shadow-samples" => {
                a.settings.rt_shadow_samples = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(4)
                    .clamp(1, 16);
            }
            "--shadow-softness" => {
                a.settings.rt_shadow_softness = it
                    .next()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.008)
                    .clamp(0.0, 0.25);
            }
            "--ao-radius" => {
                a.settings.rt_ao_radius = it
                    .next()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.9)
                    .max(0.01);
            }
            "--ao-strength" => {
                a.settings.rt_ao_strength = it
                    .next()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(1.2)
                    .clamp(0.0, 4.0);
            }
            "--ibl-strength" => {
                a.settings.ibl_strength = it
                    .next()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(1.0)
                    .max(0.0);
            }
            "--no-vsync" => a.settings.vsync = false,
            "--fps" => a.show_fps = true,
            "--exposure" => {
                a.settings.exposure = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0)
            }
            "--pbr-dir" => a.pbr_dir = it.next().map(PathBuf::from),
            s if s.starts_with("--") => bail!("unknown option {s}"),
            path => a.model = Some(PathBuf::from(path)),
        }
    }
    Ok(a)
}

fn demo_scene(pbr_dir: Option<&PathBuf>) -> Result<Scene> {
    let mut scene = Scene::default();
    let sphere = scene.add_mesh(charlotte::uv_sphere(0.45, 48, 32));
    let ground = scene.add_mesh(charlotte::plane(8.0, 8.0));
    let ground_mat = scene.add_material(Material::solid([0.5, 0.5, 0.5], 0.0, 0.8));
    scene.add_instance(
        ground,
        ground_mat,
        Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0)),
    );

    if let Some(dir) = pbr_dir {
        let mat = scene.add_pbr_material_from_dir(dir)?;
        scene.add_instance(
            sphere,
            mat,
            Mat4::from_scale(Vec3::splat(2.0)) * Mat4::from_translation(Vec3::ZERO),
        );
    } else {
        // 5x5 grid: metallic increases along x, roughness along z.
        for xi in 0..5 {
            for zi in 0..5 {
                let m = scene.add_material(Material::solid(
                    [0.9, 0.35, 0.2],
                    xi as f32 / 4.0,
                    (zi as f32 / 4.0).max(0.05),
                ));
                let pos = Vec3::new((xi as f32 - 2.0) * 1.1, 0.0, (zi as f32 - 2.0) * 1.1);
                scene.add_instance(sphere, m, Mat4::from_translation(pos));
            }
        }
    }
    scene.lights.push(Light {
        kind: LightKind::Directional,
        color: Vec3::new(1.0, 0.96, 0.9),
        intensity: 3.5,
        range: None,
        position: Vec3::ZERO,
        direction: Vec3::new(-0.5, -1.0, -0.35).normalize(),
    });
    scene.lights.push(Light {
        kind: LightKind::Point,
        color: Vec3::new(0.4, 0.6, 1.0),
        intensity: 25.0,
        range: None,
        position: Vec3::new(-3.0, 2.0, 2.0),
        direction: Vec3::NEG_Y,
    });
    Ok(scene)
}

struct App {
    args: Args,
    scene: Scene,
    camera: Camera,
    window: Option<Arc<Window>>,
    renderer: Option<Box<dyn Renderer>>,
    keys: HashSet<KeyCode>,
    looking: bool,
    last_frame: Instant,
    last_fps_update: Instant,
    fps_frames: u32,
    title_base: String,
    effective_ao_samples: u32,
    effective_shadow_samples: u32,
    slow_frame_count: u32,
    fast_frame_count: u32,
}

impl App {
    fn init(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Charlotte viewer")
                    .with_inner_size(LogicalSize::new(1600, 900)),
            )?,
        );

        let size = window.inner_size();

        let handles = WindowHandles {
            window: window.window_handle()?.as_raw(),
            display: window.display_handle()?.as_raw(),
            width: size.width,
            height: size.height,
        };

        let mut renderer =
            charlotte::create_renderer(self.args.backend, &handles, &self.args.settings)?;

        let info = renderer.adapter_info().clone();

        self.title_base = format!(
            "Charlotte viewer - {} on {} (RT: {})",
            info.backend, info.name, info.ray_tracing
        );
        window.set_title(&self.title_base);

        renderer.load_scene(&self.scene)?;

        self.renderer = Some(renderer);
        self.window = Some(window);
        self.last_frame = Instant::now();
        self.last_fps_update = Instant::now();
        self.fps_frames = 0;

        Ok(())
    }

    fn frame(&mut self) -> Result<()> {
        let now = Instant::now();
        let elapsed = now - self.last_frame;
        let dt = elapsed.as_secs_f32().min(0.1);
        self.last_frame = now;
        self.update_ray_quality(dt);

        let speed = if self.keys.contains(&KeyCode::ShiftLeft) {
            12.0
        } else {
            3.0
        } * dt;

        let fwd = self.camera.forward();
        let right = self.camera.right();

        for (key, dir) in [
            (KeyCode::KeyW, fwd),
            (KeyCode::KeyS, -fwd),
            (KeyCode::KeyD, right),
            (KeyCode::KeyA, -right),
            (KeyCode::KeyE, Vec3::Y),
            (KeyCode::KeyQ, -Vec3::Y),
        ] {
            if self.keys.contains(&key) {
                self.camera.position += dir * speed;
            }
        }

        {
            let mut frame_settings = self.args.settings.clone();
            frame_settings.rt_ao_samples = self.effective_ao_samples;
            frame_settings.rt_shadow_samples = self.effective_shadow_samples;
            if let Some(r) = self.renderer.as_mut() {
                r.render(&self.camera, &frame_settings)?;
            }
        }
        self.update_fps();

        Ok(())
    }

    fn update_ray_quality(&mut self, frame_seconds: f32) {
        if !self.args.settings.ray_tracing {
            return;
        }
        if frame_seconds > 0.0185 {
            self.slow_frame_count = self.slow_frame_count.saturating_add(1);
            self.fast_frame_count = 0;
        } else if frame_seconds < 0.0172 {
            self.fast_frame_count = self.fast_frame_count.saturating_add(1);
            self.slow_frame_count = 0;
        } else {
            self.slow_frame_count = 0;
            self.fast_frame_count = 0;
        }

        if self.slow_frame_count >= 6 {
            let old = (self.effective_ao_samples, self.effective_shadow_samples);
            if self.args.settings.rt_ao {
                let floor = self.args.settings.rt_ao_samples.clamp(1, 32).min(4);
                self.effective_ao_samples = (self.effective_ao_samples / 2).max(floor);
            }
            if self.args.settings.rt_shadows {
                let target = self.args.settings.rt_shadow_samples.clamp(1, 16);
                self.effective_shadow_samples =
                    (self.effective_shadow_samples / 2).max(1).min(target);
            }
            self.slow_frame_count = 0;
            if old != (self.effective_ao_samples, self.effective_shadow_samples) {
                log::warn!(
                    "sustained slow frames: reducing DXR samples to AO {} / shadows {}",
                    self.effective_ao_samples,
                    self.effective_shadow_samples
                );
            }
        } else if self.fast_frame_count >= 600 {
            let old = (self.effective_ao_samples, self.effective_shadow_samples);
            if self.args.settings.rt_ao {
                self.effective_ao_samples = (self.effective_ao_samples * 2)
                    .min(self.args.settings.rt_ao_samples.clamp(1, 32));
            }
            if self.args.settings.rt_shadows {
                self.effective_shadow_samples = (self.effective_shadow_samples * 2)
                    .min(self.args.settings.rt_shadow_samples.clamp(1, 16));
            }
            self.fast_frame_count = 0;
            if old != (self.effective_ao_samples, self.effective_shadow_samples) {
                log::info!(
                    "frame-time headroom restored: DXR samples AO {} / shadows {}",
                    self.effective_ao_samples,
                    self.effective_shadow_samples
                );
            }
        }
    }

    fn update_fps(&mut self) {
        if !self.args.show_fps {
            return;
        }
        self.fps_frames += 1;
        let elapsed = self.last_fps_update.elapsed();
        if elapsed < std::time::Duration::from_millis(500) {
            return;
        }
        let fps = self.fps_frames as f32 / elapsed.as_secs_f32();
        if let Some(window) = &self.window {
            window.set_title(&format!(
                "{} | {:.0} FPS | AO {} / shadows {} rays",
                self.title_base, fps, self.effective_ao_samples, self.effective_shadow_samples
            ));
        }
        self.fps_frames = 0;
        self.last_fps_update = Instant::now();
    }

    fn grab_cursor(&self) {
        if let Some(window) = &self.window {
            let result = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));

            if let Err(e) = result {
                eprintln!("Failed to grab cursor: {e}");
            }

            window.set_cursor_visible(false);
        }
    }

    fn ungrab_cursor(&self) {
        if let Some(window) = &self.window {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_none() {
            if let Err(e) = self.init(el) {
                eprintln!("failed to start renderer: {e:#}");
                el.exit();
            }
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        if !self.looking {
            return;
        }

        if let DeviceEvent::MouseMotion { delta } = event {
            let sensitivity = 0.002;

            self.camera.yaw += delta.0 as f32 * sensitivity;
            self.camera.pitch =
                (self.camera.pitch - delta.1 as f32 * sensitivity).clamp(-1.55, 1.55);
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),

            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    if let Err(e) = r.resize(size.width, size.height) {
                        eprintln!("resize failed: {e:#}");
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.keys.insert(code);

                        let s = &mut self.args.settings;

                        match code {
                            KeyCode::Escape => el.exit(),
                            KeyCode::BracketLeft => {
                                s.exposure = (s.exposure * 0.8).max(0.001);
                            }
                            KeyCode::BracketRight => s.exposure *= 1.25,
                            KeyCode::KeyR => s.rt_shadows = !s.rt_shadows,
                            KeyCode::KeyO => s.rt_ao = !s.rt_ao,
                            _ => {}
                        }
                    } else {
                        self.keys.remove(&code);
                    }
                }
            }

            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                let pressed = state == ElementState::Pressed;

                if pressed != self.looking {
                    self.looking = pressed;

                    if pressed {
                        self.grab_cursor();
                    } else {
                        self.ungrab_cursor();
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if let Err(e) = self.frame() {
                    eprintln!("render error: {e:#}");
                    el.exit();
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = parse_args()?;

    let mut scene = match &args.model {
        Some(path) => charlotte::gltf::load_gltf(path)?,
        None => demo_scene(args.pbr_dir.as_ref())?,
    };
    scene.environment = Some(match &args.environment {
        Some(path) => EnvironmentMap::from_path(path)?,
        None => EnvironmentMap::from_exr_bytes(
            include_bytes!(
                "../../../crates/charlotte_core/assets/environments/qwantani_dusk_2_puresky_4k.exr"
            ),
            "Qwantani dusk",
        )?,
    });
    if let Some(environment) = &scene.environment {
        log::info!(
            "environment: {} ({}x{})",
            environment.name,
            environment.width,
            environment.height
        );
    }

    let mut camera = Camera::default();

    if let Some((min, max)) = scene.bounds() {
        camera.frame_bounds(min, max);
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let effective_ao_samples = args.settings.rt_ao_samples.clamp(1, 32);
    let effective_shadow_samples = args.settings.rt_shadow_samples.clamp(1, 16);

    let mut app = App {
        args,
        scene,
        camera,
        window: None,
        renderer: None,
        keys: HashSet::new(),
        looking: false,
        last_frame: Instant::now(),
        last_fps_update: Instant::now(),
        fps_frames: 0,
        title_base: "Charlotte viewer".into(),
        effective_ao_samples,
        effective_shadow_samples,
        slow_frame_count: 0,
        fast_frame_count: 0,
    };

    event_loop.run_app(&mut app)?;

    Ok(())
}

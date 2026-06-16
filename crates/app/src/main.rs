//! Mining the Sky real-time client.
//!
//! A wgpu/WebGPU app (native + browser via wasm) that renders a live
//! procedural planet and flies an interactive launch-to-orbit. The planet is an
//! orthographic raymarch of the baked worldgen texture; on top of it we draw
//! the `sim` crate's staged ascent: drag to orbit the camera, scroll to zoom,
//! press Space to launch Pioneer I from the seed-47 spaceport and watch it fly
//! a gravity turn into a parking orbit. This is the start of the Caelum-style
//! renderer; the camera/overlay here is the seam the 3D LOD renderer grows from.

use std::sync::Arc;

use glam::{Mat3, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

mod mission;
use mission::Mission;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    scale: f32,
    time: f32,
    sun: [f32; 4],
    cx: [f32; 4],
    cy: [f32; 4],
    cz: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OverlayVertex {
    pos: [f32; 2],
    color: [f32; 3],
}

const OVERLAY_CAP: u64 = 8192;

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_buf: wgpu::Buffer,
    start: instant_now::Instant,
    last_t: f32,

    // mission + camera + flight state
    mission: Mission,
    az: f32,
    el: f32,
    scale: f32,
    launched: bool,
    clock: f32, // mission-elapsed seconds
    warp: f32,

    // input
    dragging: bool,
    last_cursor: (f64, f64),
}

impl State {
    async fn new(window: Arc<Window>) -> State {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        let surface = instance.create_surface(window.clone()).expect("surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no adapter");

        let limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits::downlevel_defaults()
        } else {
            wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet"),
            source: wgpu::ShaderSource::Wgsl(include_str!("planet.wgsl").into()),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Baked world texture (RGB albedo, A = city-light emission).
        let planet_img = image::load_from_memory(include_bytes!("../assets/planet.png"))
            .expect("decode planet.png")
            .to_rgba8();
        let (tw, th) = planet_img.dimensions();
        let planet_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("planet-tex"),
            size: wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &planet_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &planet_img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * tw),
                rows_per_image: Some(th),
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );
        let planet_view = planet_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let planet_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("planet-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&planet_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&planet_sampler),
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Overlay pipeline: pre-projected clip-space line list, no bind groups.
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
        });
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let overlay_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("overlay-pipeline"),
                layout: Some(&overlay_layout),
                vertex: wgpu::VertexState {
                    module: &overlay_shader,
                    entry_point: Some("vs"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<OverlayVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 8,
                                shader_location: 1,
                            },
                        ],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let overlay_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay-buf"),
            size: OVERLAY_CAP * std::mem::size_of::<OverlayVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mission = Mission::pioneer_from_spaceport();

        State {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buf,
            bind_group,
            overlay_pipeline,
            overlay_buf,
            start: instant_now::Instant::now(),
            last_t: 0.0,
            az: mission.spaceport_lon,
            el: mission.spaceport_lat,
            scale: 1.25,
            launched: false,
            clock: 0.0,
            warp: 8.0,
            mission,
            dragging: false,
            last_cursor: (0.0, 0.0),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// World-from-view rotation: column 2 is the world point facing the camera.
    fn camera_rot(&self) -> Mat3 {
        let d = Vec3::new(
            self.el.cos() * self.az.cos(),
            self.el.sin(),
            self.el.cos() * self.az.sin(),
        );
        let xc = Vec3::Y.cross(d).normalize();
        let yc = d.cross(xc).normalize();
        Mat3::from_cols(xc, yc, d)
    }

    fn toggle_launch(&mut self) {
        self.launched = !self.launched;
        self.clock = 0.0;
        if self.launched {
            log::info!("Liftoff: Pioneer I");
        }
    }

    fn render(&mut self) {
        // advance the mission clock by the real frame dt (clamped against hitches)
        let t = self.start.elapsed().as_secs_f32();
        let frame_dt = (t - self.last_t).clamp(0.0, 0.1);
        self.last_t = t;
        if self.launched {
            self.clock += frame_dt * self.warp;
        }

        let rot = self.camera_rot();

        // sun direction: world-space, slowly rotating, phased so the spaceport
        // starts near the day side.
        let st = t * 0.03 + self.mission.spaceport_lon;
        let sun = Vec3::new(st.cos() * 0.95, 0.28, st.sin() * 0.95).normalize();

        let uniforms = Uniforms {
            resolution: [self.config.width as f32, self.config.height as f32],
            scale: self.scale,
            time: t,
            sun: [sun.x, sun.y, sun.z, 0.0],
            cx: [rot.x_axis.x, rot.x_axis.y, rot.x_axis.z, 0.0],
            cy: [rot.y_axis.x, rot.y_axis.y, rot.y_axis.z, 0.0],
            cz: [rot.z_axis.x, rot.z_axis.y, rot.z_axis.z, 0.0],
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // build + upload overlay geometry for this frame
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let verts = self.build_overlay(rot, aspect);
        let n = verts.len().min(OVERLAY_CAP as usize);
        if n > 0 {
            self.queue
                .write_buffer(&self.overlay_buf, 0, bytemuck::cast_slice(&verts[..n]));
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);

            if n > 0 {
                pass.set_pipeline(&self.overlay_pipeline);
                pass.set_vertex_buffer(0, self.overlay_buf.slice(..));
                pass.draw(0..n as u32, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    /// Project a world unit-sphere point through the orthographic camera to
    /// clip space. Returns `None` when the point is hidden behind the planet.
    fn project(p: Vec3, rt: Mat3, aspect: f32, scale: f32) -> Option<[f32; 2]> {
        let v = rt * p;
        let occluded = v.z < 0.0 && (v.x * v.x + v.y * v.y) < 1.0;
        if occluded {
            None
        } else {
            Some([v.x / (aspect * scale), v.y / scale])
        }
    }

    fn build_overlay(&self, rot: Mat3, aspect: f32) -> Vec<OverlayVertex> {
        let rt = rot.transpose();
        let scale = self.scale;
        let mut out: Vec<OverlayVertex> = Vec::new();

        let polyline = |pts: &[Vec3], color: [f32; 3], out: &mut Vec<OverlayVertex>| {
            let mut prev: Option<[f32; 2]> = None;
            for &p in pts {
                let cur = Self::project(p, rt, aspect, scale);
                if let (Some(a), Some(b)) = (prev, cur) {
                    out.push(OverlayVertex { pos: a, color });
                    out.push(OverlayVertex { pos: b, color });
                }
                prev = cur;
            }
        };

        // predicted parking orbit ring (drawn first, dim)
        if self.mission.reached {
            polyline(&self.mission.ring, [0.25, 0.7, 0.45], &mut out);
        }

        // predicted full ascent (dim), then the flown portion (bright)
        let path_pts: Vec<Vec3> = self.mission.path.iter().map(|(_, p)| *p).collect();
        polyline(&path_pts, [0.20, 0.45, 0.55], &mut out);

        let flown: Vec<Vec3> = self
            .mission
            .path
            .iter()
            .filter(|(t, _)| *t <= self.clock)
            .map(|(_, p)| *p)
            .collect();
        polyline(&flown, [0.45, 0.9, 1.0], &mut out);

        // rocket marker: a small screen-space diamond at its current position
        let rp = self.mission.rocket_pos(if self.launched { self.clock } else { 0.0 });
        if let Some(c) = Self::project(rp, rt, aspect, scale) {
            let off = 0.022f32;
            let ox = off / aspect;
            let col = [1.0, 0.85, 0.25];
            let top = [c[0], c[1] + off];
            let right = [c[0] + ox, c[1]];
            let bot = [c[0], c[1] - off];
            let left = [c[0] - ox, c[1]];
            for (a, b) in [(top, right), (right, bot), (bot, left), (left, top)] {
                out.push(OverlayVertex { pos: a, color: col });
                out.push(OverlayVertex { pos: b, color: col });
            }
        }

        out
    }
}

enum UserEvent {
    Ready(State),
}

struct App {
    proxy: EventLoopProxy<UserEvent>,
    state: Option<State>,
}

impl App {
    fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        App { proxy, state: None }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("Mining the Sky");
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.body())
                .and_then(|body| {
                    let canvas = web_sys::Element::from(window.canvas()?);
                    body.append_child(&canvas).ok()
                })
                .expect("append canvas");
        }

        let proxy = self.proxy.clone();
        let win = window.clone();
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            let state = State::new(win).await;
            let _ = proxy.send_event(UserEvent::Ready(state));
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = pollster::block_on(State::new(win));
            let _ = proxy.send_event(UserEvent::Ready(state));
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        let UserEvent::Ready(state) = event;
        log::info!(
            "Controls: drag = orbit camera, scroll = zoom, Space = launch/reset, [ / ] = time warp"
        );
        state.window.request_redraw();
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
                state.window.request_redraw();
            }
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                if button == MouseButton::Left {
                    state.dragging = btn_state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                if state.dragging {
                    let dx = (x - state.last_cursor.0) as f32;
                    let dy = (y - state.last_cursor.1) as f32;
                    state.az -= dx * 0.005;
                    state.el = (state.el + dy * 0.005).clamp(-1.5, 1.5);
                }
                state.last_cursor = (x, y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 60.0,
                };
                state.scale = (state.scale * (1.0 - dy * 0.12)).clamp(0.12, 3.0);
            }
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state == ElementState::Pressed && !key_event.repeat {
                    match key_event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => state.toggle_launch(),
                        PhysicalKey::Code(KeyCode::BracketRight) => {
                            state.warp = (state.warp * 2.0).min(256.0);
                        }
                        PhysicalKey::Code(KeyCode::BracketLeft) => {
                            state.warp = (state.warp * 0.5).max(1.0);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("event loop");
    let app = App::new(event_loop.create_proxy());

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let _ = console_log::init_with_level(log::Level::Info);
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let mut app = app;
        event_loop.run_app(&mut app).expect("run");
    }
}

/// Small cross-target clock: `std::time::Instant` panics on wasm, so use the
/// browser performance clock there.
mod instant_now {
    #[derive(Copy, Clone)]
    pub struct Instant {
        #[cfg(not(target_arch = "wasm32"))]
        inner: std::time::Instant,
        #[cfg(target_arch = "wasm32")]
        start_ms: f64,
    }

    impl Instant {
        #[cfg(not(target_arch = "wasm32"))]
        pub fn now() -> Self {
            Instant { inner: std::time::Instant::now() }
        }
        #[cfg(target_arch = "wasm32")]
        pub fn now() -> Self {
            Instant { start_ms: now_ms() }
        }

        #[cfg(not(target_arch = "wasm32"))]
        pub fn elapsed(&self) -> std::time::Duration {
            self.inner.elapsed()
        }
        #[cfg(target_arch = "wasm32")]
        pub fn elapsed(&self) -> std::time::Duration {
            std::time::Duration::from_secs_f64((now_ms() - self.start_ms) / 1000.0)
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn now_ms() -> f64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0)
    }
}

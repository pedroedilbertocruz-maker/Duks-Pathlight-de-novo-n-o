// PathLight GI - shared definitions. Prepended to every pass at load time.
//
// Every pass declares (with the same binding numbers):
//   @binding(0) u            : Uniforms
//   @binding(1) depth_tex    : raw scene depth snapshot (R32Float, reversed-Z: 1 = near, 0 = far/sky)
//   @binding(2) normal_tex   : view-space normals, rgb = n * 0.5 + 0.5, a < 0.5 = "no surface"
// Passes that need it also declare @binding(3) color_tex (the opaque scene colour snapshot).
// All scene textures are addressed by UV so their resolution may differ from the GI buffers.

struct Uniforms {
    proj: mat4x4f,                 // view -> clip (WebGPU clip, reversed-Z)
    inv_proj: mat4x4f,             // clip -> view
    world_from_view: mat4x4f,      // current camera
    prev_proj_from_world: mat4x4f, // previous frame's world -> clip
    dims: vec4f,   // GI buffer size: w, h, 1/w, 1/h
    full: vec4f,   // scene snapshot size: w, h, (unused), (unused)
    a: vec4f,      // ray_length, thickness, intensity, bounce_gain
    b: vec4f,      // ao_radius, ao_strength, sky_gain, emissive_boost
    c: vec4f,      // color_bleed, max_history_frames, history_clamp_k, denoise_strength
    d: vec4f,      // near_plane, history_valid, denoise_step, (unused)
    i: vec4u,      // frame_index, rays_per_pixel, steps_per_ray, debug_view
}

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const SKY_DEPTH_EPS: f32 = 1e-6;

fn luma(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// PCG hash -> two uniform floats in [0, 1).
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn hash2(p: vec2u, seed: u32) -> vec2f {
    let h1 = pcg(p.x + pcg(p.y + pcg(seed)));
    let h2 = pcg(h1 ^ 0x9e3779b9u);
    return vec2f(f32(h1 >> 8u), f32(h2 >> 8u)) * (1.0 / 16777216.0);
}

fn tex_px(dim: vec2u, uv: vec2f) -> vec2i {
    return clamp(vec2i(uv * vec2f(dim)), vec2i(0), vec2i(dim) - 1);
}

fn depth_at_uv(uv: vec2f) -> f32 {
    return textureLoad(depth_tex, tex_px(textureDimensions(depth_tex), uv), 0).r;
}

fn normal_at_uv(uv: vec2f) -> vec4f {
    return textureLoad(normal_tex, tex_px(textureDimensions(normal_tex), uv), 0);
}

// View-space position from a raw depth sample at uv.
fn view_pos(uv: vec2f, depth: f32) -> vec3f {
    let clip = vec4f(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let t = u.inv_proj * clip;
    return t.xyz / t.w;
}

// Positive distance along the view axis (-z) for a depth sample.
fn view_w(uv: vec2f, depth: f32) -> f32 {
    return -view_pos(uv, depth).z;
}

fn project_uv(p: vec3f) -> vec2f {
    let clip = u.proj * vec4f(p, 1.0);
    let ndc = clip.xy / clip.w;
    return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// Rough albedo from a lit pixel colour. Dark (shadowed) pixels still receive bounce light.
fn albedo_proxy(c: vec3f) -> vec3f {
    let m = max(max(c.r, c.g), c.b);
    let hue = c / max(m, 1e-3);
    let brightness = 0.3 + 0.7 * sqrt(clamp(m, 0.0, 1.0));
    return mix(vec3f(brightness), hue * brightness, u.c.x);
}

fn is_sky(depth: f32, nrm: vec4f) -> bool {
    return depth <= SKY_DEPTH_EPS || nrm.a < 0.5;
}

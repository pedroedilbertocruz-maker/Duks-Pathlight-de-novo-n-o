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
    l0: vec4f,     // ambient_scale (1 = no darkening), light_gain, shadow_strength, shadow_steps
    l1: vec4f,     // light_count, restore (how much game lighting returns near lights), unused, time
    // Scene lights in VIEW space: [2k] = pos.xyz, radius ; [2k+1] = rgb (flicker applied), kind
    lights: array<vec4f, 64>,
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


// ---------------------------------------------------------------------------------------------
// Dynamic lights (torches, lantern, sword sparks). Darkness mode: the scene is scaled down by
// u.l0.x and brought back / lit up only where lights reach, with screen-space shadows.
// ---------------------------------------------------------------------------------------------

// Interleaved gradient noise: stable per-pixel dither (no temporal shimmer).
fn ign(p: vec2u) -> f32 {
    return fract(52.9829189 * fract(dot(vec2f(p), vec2f(0.06711056, 0.00583715))));
}

// Windowed, near-source-softened inverse square: 1 at the light, exactly 0 at radius r.
fn light_atten(d2: f32, r: f32) -> f32 {
    let x = d2 / (r * r);
    if x >= 1.0 {
        return 0.0;
    }
    let win = 1.0 - x;
    let core = r * 0.22;
    return win * win * (core * core) / (d2 + core * core);
}

// 0 = fully shadowed, 1 = lit. Marches the segment surface -> light through the depth buffer.
fn light_shadow(p: vec3f, n: vec3f, lp: vec3f, pix: vec2u) -> f32 {
    let steps = i32(u.l0.w);
    if steps <= 0 {
        return 1.0;
    }
    let origin = p + n * (0.004 * -p.z + 2.0);
    let to = lp - origin;
    let jitter = ign(pix);
    let thick = max(u.a.y, 40.0);
    var hits = 0.0;
    for (var s = 0; s < steps; s++) {
        let t = (f32(s) + jitter) / f32(steps) * 0.9;
        let q = origin + to * t;
        if q.z > -u.d.x {
            break;
        }
        let uv = project_uv(q);
        if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) {
            continue;
        }
        let d = depth_at_uv(uv);
        if d <= SKY_DEPTH_EPS {
            continue;
        }
        let diff = -q.z - view_w(uv, d);
        if diff > -q.z * 0.01 + 2.0 && diff < thick {
            hits += 1.0;
        }
    }
    let occluded = clamp(hits * 3.0 / f32(steps), 0.0, 1.0);
    return 1.0 - occluded * u.l0.z;
}

// Returns rgb = incoming light (before albedo) and a = coverage 0..1 (how much of the original
// game lighting should be restored here). `shadows` enables screen-space shadow rays for the
// strongest few lights.
fn light_field(p: vec3f, n: vec3f, pix: vec2u, shadows: bool) -> vec4f {
    var irr = vec3f(0.0);
    var cover = 0.0;
    let count = u32(u.l1.x);
    var budget = 4;
    for (var k = 0u; k < count; k++) {
        let lp = u.lights[2u * k];
        let lc = u.lights[2u * k + 1u];
        let to = lp.xyz - p;
        let d2 = dot(to, to);
        let r = lp.w;
        if d2 >= r * r {
            continue;
        }
        let att = light_atten(d2, r);
        let l = to * inverseSqrt(max(d2, 1e-4));
        let ndl = clamp(dot(n, l) * 0.8 + 0.2, 0.0, 1.0); // wrapped lambert: soft terminator
        var vis = 1.0;
        if shadows && budget > 0 && att * ndl * max(lc.r, max(lc.g, lc.b)) > 0.02 {
            vis = light_shadow(p, n, lp.xyz, pix);
            budget -= 1;
        }
        irr += lc.rgb * (att * ndl * vis);
        cover += att * ndl * vis * clamp(max(lc.r, max(lc.g, lc.b)), 0.0, 1.0);
    }
    return vec4f(irr, min(cover, 1.0));
}

// Scene colour after darkness + lights: what a surface of colour `c` looks like at point p.
fn scene_scale(cover: f32) -> f32 {
    return mix(u.l0.x, 1.0, clamp(cover * u.l1.y, 0.0, 1.0));
}

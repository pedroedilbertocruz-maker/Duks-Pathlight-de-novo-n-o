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
    fog0: vec4f,   // fog_start, fog_end, fog_enabled (0/1), range_center (NDC x)
    fog1: vec4f,   // fog colour rgb (0..1), fog_strength
    fogk: array<vec4f, 3>, // fog range-adjust table (10 values, already /64 and pair-swapped)
    sun0: vec4f,   // direction TO the sun/moon in view space (xyz), strength 0..1 (0 = off)
    sun1: vec4f,   // march length (world units), shadow floor (darkest lit fraction), steps, unused
    sun2: vec4f,   // sun/moon light colour rgb, warmth
    sun3: vec4f,   // god-ray strength, bloom strength, unused, unused
    link0: vec4f,  // Link's body segment start (view space), exclusion radius (0 = off)
    link1: vec4f,  // Link's body segment end (view space)
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

// Soft screen-space shadow. 0 = fully shadowed, 1 = lit.
// The light is treated as a small disc (flames have size): a few rays per pixel go to jittered
// points on that disc, each marching the segment surface -> light through the depth buffer. The
// fraction of blocked rays gives a penumbra instead of a hard polygon edge. Shadows never become
// fully black (light bounces around in real life) and fade out where the light is weak.
const SHADOW_RAYS: i32 = 4;
const SHADOW_FLOOR: f32 = 0.30;  // darkest a shadowed pixel can get, as a fraction of its lit value

fn light_shadow(p: vec3f, n: vec3f, lp: vec3f, pix: vec2u, fade: f32) -> f32 {
    let total_steps = i32(u.l0.w);
    let strength = u.l0.z * fade;
    if total_steps <= 0 || strength <= 0.001 {
        return 1.0;
    }
    let steps = max(total_steps / 2, 4);
    // Push the start off the surface (more at grazing angles) so flat ground never shadows itself.
    let dist_view = -p.z;
    let toward = normalize(lp - p);
    let graze = 1.0 - clamp(dot(n, toward), 0.0, 1.0);
    let origin = p + n * (0.006 * dist_view + 2.5 + 6.0 * graze);
    let full = lp - origin;
    let full_len = length(full);
    let dir = full / max(full_len, 1e-3);

    // Light "size" -> penumbra width. About 7% of the distance, between 8 and 70 world units.
    let size = clamp(full_len * 0.07, 8.0, 70.0);
    var up = vec3f(0.0, 1.0, 0.0);
    if abs(dir.y) > 0.9 {
        up = vec3f(1.0, 0.0, 0.0);
    }
    let t1 = normalize(cross(dir, up));
    let t2 = cross(dir, t1);

    let jitter = ign(pix);
    let jitter2 = ign(pix + vec2u(17u, 59u));
    let thick = max(u.a.y, 40.0);
    var blocked = 0.0;
    for (var r = 0; r < SHADOW_RAYS; r++) {
        let ang = (f32(r) + jitter2) * 2.39996323 + jitter * TAU;
        let rad = sqrt((f32(r) + 0.5) / f32(SHADOW_RAYS)) * size;
        let aim = lp + t1 * (cos(ang) * rad) + t2 * (sin(ang) * rad);
        let to = aim - origin;
        var hits = 0.0;
        for (var s = 0; s < steps; s++) {
            let t = (f32(s) + fract(jitter + f32(r) * 0.618034)) / f32(steps) * 0.92;
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
            if diff > -q.z * 0.012 + 3.0 && diff < thick {
                hits += 1.0;
            }
        }
        // A ray is blocked once a couple of its samples land behind geometry.
        blocked += clamp(hits * 0.7, 0.0, 1.0);
    }
    let occluded = blocked / f32(SHADOW_RAYS);
    return 1.0 - occluded * strength * (1.0 - SHADOW_FLOOR);
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
        // No shadows for huge room lights (they act as fill, and would cast enormous hard shadows)
        // or for sword-spark flashes (too brief to matter). Shadows fade out where the light is weak.
        if shadows && budget > 0 && r < 1100.0 && lc.w < 1.5 &&
            att * ndl * max(lc.r, max(lc.g, lc.b)) > 0.02
        {
            vis = light_shadow(p, n, lp.xyz, pix, clamp(att * 7.0, 0.0, 1.0));
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


// ---------------------------------------------------------------------------------------------
// Fog. The game's own fog is baked into the scene colour by the renderer (per draw, linear in view
// depth). Darkening a fogged pixel would remove its fog, so after darkness/lights we put back the
// fog that was lost. Same formula as Aurora's GX fog (PERSP_LIN + range adjust), so the result
// looks like fog applied *after* lighting ("deferred fog").
// ---------------------------------------------------------------------------------------------

fn fog_k(i: u32) -> f32 {
    return u.fogk[i >> 2u][i & 3u];
}

// 0 = no fog, 1 = fully fogged. `z` is the distance along the view axis.
fn fog_factor(uv_x: f32, z: f32) -> f32 {
    if u.fog0.z < 0.5 {
        return 0.0;
    }
    let off = (uv_x * 2.0 - 1.0) - u.fog0.w;
    let idx = clamp(9.0 - abs(off) * 9.0, 0.0, 9.0);
    let lo = u32(idx);
    let hi = min(lo + 1u, 9u);
    let fr = idx - f32(lo);
    let k = max(fog_k(lo) * (1.0 - fr) + fog_k(hi) * fr, 0.000001);
    let lut = sqrt(off * off + k * k) / k;
    return clamp((z * lut - u.fog0.x) / max(u.fog0.y - u.fog0.x, 1.0), 0.0, 1.0);
}


// ---------------------------------------------------------------------------------------------
// Sun / moon shadows. Same idea as the torch shadows: march from the surface toward the sun through
// the depth buffer; a few jittered rays across the sun's disc give soft edges. Only surfaces facing
// the sun are darkened (the others already are in shade). Returns 1 = lit .. sun1.y = shadowed.
// ---------------------------------------------------------------------------------------------

// True if a view-space point lies inside Link's body (the game draws his own shadow; we skip him).
fn in_link(q: vec3f) -> bool {
    let r = u.link0.w;
    if r <= 0.0 {
        return false;
    }
    let a = u.link0.xyz;
    let ab = u.link1.xyz - a;
    let t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-3), 0.0, 1.0);
    let c = a + ab * t;
    return dot(q - c, q - c) < r * r;
}

fn sun_shadow(p: vec3f, n: vec3f, pix: vec2u) -> f32 {
    let strength = u.sun0.w;
    if strength <= 0.001 {
        return 1.0;
    }
    if in_link(p) {
        return 1.0;  // Link himself: leave it to the game's own shadow
    }
    let sdir = normalize(u.sun0.xyz);
    let ndl = dot(n, sdir);
    if ndl <= 0.02 {
        return 1.0;  // facing away from the sun: already in shade
    }
    let dist_view = -p.z;
    let far_fade = 1.0 - smoothstep(3500.0, 9000.0, dist_view);
    if far_fade <= 0.001 {
        return 1.0;
    }
    let steps = i32(u.sun1.z);
    let len = u.sun1.x;
    let origin = p + n * (0.008 * dist_view + 3.0) + sdir * (0.004 * dist_view);
    var up = vec3f(0.0, 1.0, 0.0);
    if abs(sdir.y) > 0.9 {
        up = vec3f(1.0, 0.0, 0.0);
    }
    let t1 = normalize(cross(sdir, up));
    let t2 = cross(sdir, t1);
    let jitter = ign(pix);
    let jitter2 = ign(pix + vec2u(31u, 7u));
    let thick = max(u.a.y * 3.0, 260.0);
    var blocked = 0.0;
    const RAYS: i32 = 3;
    for (var r = 0; r < RAYS; r++) {
        let ang = (f32(r) + jitter2) * 2.39996323 + jitter * TAU;
        let rad = sqrt((f32(r) + 0.5) / f32(RAYS)) * 0.03;  // angular radius of the sun's disc
        let dir = normalize(sdir + t1 * (cos(ang) * rad) + t2 * (sin(ang) * rad));
        var hits = 0.0;
        for (var s = 0; s < steps; s++) {
            let f = (f32(s) + fract(jitter + f32(r) * 0.618034)) / f32(steps);
            let q = origin + dir * (f * f * len + 4.0);
            if q.z > -u.d.x {
                break;
            }
            let uv = project_uv(q);
            if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) {
                break;  // left the screen: nothing more we can know
            }
            let d = depth_at_uv(uv);
            if d <= SKY_DEPTH_EPS {
                continue;
            }
            let diff = -q.z - view_w(uv, d);
            if diff > -q.z * 0.015 + 4.0 && diff < thick && !in_link(view_pos(uv, d)) {
                hits += 1.0;
            }
        }
        blocked += clamp(hits * 0.7, 0.0, 1.0);
    }
    let occluded = blocked / f32(RAYS);
    let face = smoothstep(0.02, 0.3, ndl);
    return 1.0 - occluded * strength * far_fade * face * (1.0 - u.sun1.y);
}


// ---------------------------------------------------------------------------------------------
// Sun rays (crepuscular "god rays") and bloom. Radial march from the pixel toward the sun's screen
// position: sky samples carry light, geometry blocks it, so trees and roofs cut the glow into shafts.
// Returns additive rgb. Works for sky pixels too (that is the bloom around the sun).
// ---------------------------------------------------------------------------------------------

fn sun_screen() -> vec3f {
    // xy = uv of the sun, z = 1 if it is in front of the camera.
    let d = normalize(u.sun0.xyz);
    if d.z > -0.05 {
        return vec3f(0.5, 0.5, 0.0);
    }
    let uv = project_uv(d * 10000.0);
    return vec3f(uv, 1.0);
}

fn sun_rays(uv: vec2f, pix: vec2u, here_is_sky: bool) -> vec3f {
    if u.sun0.w <= 0.001 || (u.sun3.x <= 0.001 && u.sun3.y <= 0.001) {
        return vec3f(0.0);
    }
    let s = sun_screen();
    if s.z < 0.5 {
        return vec3f(0.0);
    }
    let aspect = u.full.x / max(u.full.y, 1.0);
    let to_sun = s.xy - uv;
    let dist = length(vec2f(to_sun.x * aspect, to_sun.y));
    let col = u.sun2.rgb;

    // Bloom: tight core + wide halo, strongest on sky, faint over geometry.
    let glow = 0.55 * exp(-dist * 22.0) + 0.22 * exp(-dist * 5.0);
    let bloom = col * glow * u.sun3.y * u.sun0.w * select(0.0, 0.5, here_is_sky);

    var rays = vec3f(0.0);
    if u.sun3.x > 0.001 {
        const N: i32 = 12;
        let jitter = ign(pix);
        var sum = 0.0;
        var decay = 1.0;
        for (var i = 0; i < N; i++) {
            let f = (f32(i) + jitter) / f32(N);
            let q = uv + to_sun * f;
            if any(q < vec2f(0.0)) || any(q > vec2f(1.0)) {
                decay *= 0.94;
                continue;
            }
            let dq = depth_at_uv(q);
            let nq = normal_at_uv(q);
            if is_sky(dq, nq) {
                let dd = length(vec2f((s.xy.x - q.x) * aspect, s.xy.y - q.y));
                sum += decay * (0.25 + 1.2 * exp(-dd * 6.0));
            }
            decay *= 0.94;
        }
        // Fade by how far the pixel is from the sun, so far-away pixels don't get a flat veil.
        let reach = exp(-dist * 1.6);
        // Over geometry the shafts are only a faint haze; the strong part belongs to the sky.
        let on_geo = select(0.18, 1.0, here_is_sky);
        rays = mix(col, vec3f(1.0), 0.35) * (sum / f32(N)) * 0.45 * reach * u.sun3.x * u.sun0.w * on_geo;
    }
    let total = bloom + rays;
    return min(total, select(vec3f(0.12), vec3f(0.8), here_is_sky));
}

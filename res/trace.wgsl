// Pass 1 - Monte Carlo path tracing in screen space.
//
// For every GI pixel, cast `rays_per_pixel` cosine-weighted rays from the surface and march them
// through the depth buffer. A hit returns the radiance of the surface it landed on (the lit scene
// colour) plus that surface's own indirect light from the previous frame (infinite bounces through
// temporal feedback). Rays that escape to visible sky gather sky light. Output:
//   rgb = incoming indirect radiance estimate, a = short-range occlusion (contact shadowing).

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var color_tex: texture_2d<f32>;
@group(0) @binding(4) var prev_gi: texture_2d<f32>;
@group(0) @binding(5) var raw_out: texture_storage_2d<rgba16float, write>;

struct Hit {
    found: bool,
    sky: bool,
    uv: vec2f,
    dist: f32,
}

fn color_at_uv(uv: vec2f) -> vec3f {
    return textureLoad(color_tex, tex_px(textureDimensions(color_tex), uv), 0).rgb;
}

fn prev_gi_at_uv(uv: vec2f) -> vec3f {
    return textureLoad(prev_gi, tex_px(textureDimensions(prev_gi), uv), 0).rgb;
}

fn scene_w_at(uv: vec2f) -> f32 {
    let d = depth_at_uv(uv);
    return select(view_w(uv, d), 1e30, d <= SKY_DEPTH_EPS);
}

fn march(origin: vec3f, dir: vec3f, jitter: f32) -> Hit {
    var hit: Hit;
    hit.found = false;
    hit.sky = false;
    hit.uv = vec2f(0.0);
    hit.dist = 0.0;

    let near = u.d.x;
    var len = u.a.x;
    if origin.z + dir.z * len > -near {
        len = (-near - origin.z) / dir.z; // only reachable when dir.z > 0
    }
    if len <= 1.0 {
        return hit;
    }
    let end = origin + dir * len;
    let uv0 = project_uv(origin);
    let uv1 = project_uv(end);
    let w0 = -origin.z;
    let w1 = -end.z;
    let k0 = 1.0 / w0;
    let k1 = 1.0 / w1;

    let steps = i32(u.i.z);
    let thickness = u.a.y;
    var prev_f = 0.0;
    var last_uv = uv0;
    var last_sky = false;

    for (var s = 0; s < steps; s++) {
        // Quadratic spacing: fine steps close to the surface, coarse far away.
        let f = (f32(s) + jitter) / f32(steps);
        let t = f * f;
        let uv = mix(uv0, uv1, t);
        if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) {
            break;
        }
        last_uv = uv;
        let ray_w = 1.0 / mix(k0, k1, t);
        let d = depth_at_uv(uv);
        if d <= SKY_DEPTH_EPS {
            last_sky = true;
            prev_f = f;
            continue;
        }
        last_sky = false;

        let diff = ray_w - view_w(uv, d);
        let t_prev = prev_f * prev_f;
        let span = abs(ray_w - 1.0 / mix(k0, k1, t_prev));
        if diff > ray_w * 0.002 && diff < max(thickness, span * 1.5) {
            let n = normal_at_uv(uv);
            if n.a >= 0.5 && dot(normalize(n.xyz * 2.0 - 1.0), dir) > 0.25 {
                // Back face of something: let the ray continue.
                prev_f = f;
                continue;
            }
            // Binary-search refinement between the last miss and this hit.
            var lo = prev_f;
            var hi = f;
            for (var j = 0; j < 5; j++) {
                let m = 0.5 * (lo + hi);
                let tm = m * m;
                let uvm = mix(uv0, uv1, tm);
                let wm = 1.0 / mix(k0, k1, tm);
                if wm > scene_w_at(uvm) {
                    hi = m;
                } else {
                    lo = m;
                }
            }
            let th = hi * hi;
            let hit_w = 1.0 / mix(k0, k1, th);
            let dw = w1 - w0;
            let lambda = select(th, (hit_w - w0) / dw, abs(dw) > 1e-3);
            hit.found = true;
            hit.uv = mix(uv0, uv1, th);
            hit.dist = lambda * len;
            return hit;
        }
        prev_f = f;
    }

    if last_sky {
        hit.sky = true;
        hit.uv = last_uv;
    }
    return hit;
}

fn orthonormal_basis(n: vec3f) -> mat3x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3f(b, s + n.y * n.y * a, -n.y);
    return mat3x3f(t, bt, n);
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) gid: vec3u) {
    let q = vec2i(gid.xy);
    let size = vec2i(u.dims.xy);
    if q.x >= size.x || q.y >= size.y {
        return;
    }
    let uv = (vec2f(q) + 0.5) * u.dims.zw;
    let depth = depth_at_uv(uv);
    let nrm = normal_at_uv(uv);
    if is_sky(depth, nrm) {
        textureStore(raw_out, q, vec4f(0.0));
        return;
    }

    let p = view_pos(uv, depth);
    var n = normalize(nrm.xyz * 2.0 - 1.0);
    if dot(n, p) > 0.0 {
        n = -n; // face the camera
    }
    let basis = orthonormal_basis(n);
    let origin = p + n * (0.004 * -p.z + 1.0);

    let rays = u.i.y;
    let frame = f32(u.i.x);
    let base_dir = hash2(vec2u(q), 0xA511E9B3u);
    let base_jit = hash2(vec2u(q), 0x63D83595u);

    var radiance = vec3f(0.0);
    var occlusion = 0.0;
    for (var r = 0u; r < rays; r++) {
        let seq = frame * f32(rays) + f32(r);
        // R2 low-discrepancy sequence with a per-pixel random offset: converges over frames.
        let xi = fract(base_dir + seq * vec2f(0.7548776662, 0.5698402909));
        let jitter = 0.05 + 0.95 * fract(base_jit.x + seq * 0.6180339887);

        let sin_t = sqrt(xi.x);
        let cos_t = sqrt(1.0 - xi.x);
        let phi = TAU * xi.y;
        let local = vec3f(sin_t * cos(phi), sin_t * sin(phi), cos_t);
        let dir = normalize(basis * local);

        let hit = march(origin, dir, jitter);
        if hit.found {
            let c = color_at_uv(hit.uv);
            // What the hit surface looks like after darkness + lights (no shadow rays here).
            let hd = depth_at_uv(hit.uv);
            let hp = view_pos(hit.uv, hd);
            var hn = normalize(normal_at_uv(hit.uv).xyz * 2.0 - 1.0);
            if dot(hn, hp) > 0.0 {
                hn = -hn;
            }
            let lfh = light_field(hp, hn, vec2u(0u), false);
            var l = c * scene_scale(lfh.a) + c * (u.b.w * smoothstep(0.6, 1.0, luma(c)) * 2.0);
            l += albedo_proxy(c) * lfh.rgb * u.l0.y;                // torch light bouncing on
            l += albedo_proxy(c) * prev_gi_at_uv(hit.uv) * u.a.w; // additional bounces
            l = min(l, vec3f(4.0));
            let fade = 1.0 - smoothstep(0.7, 1.0, hit.dist / u.a.x);
            radiance += l * fade;
            occlusion += 1.0 - clamp(hit.dist / u.b.x, 0.0, 1.0);
        } else if hit.sky {
            radiance += min(color_at_uv(hit.uv), vec3f(4.0)) * u.b.z;
        }
    }
    let inv = 1.0 / f32(rays);
    textureStore(raw_out, q, vec4f(radiance * inv, occlusion * inv));
}

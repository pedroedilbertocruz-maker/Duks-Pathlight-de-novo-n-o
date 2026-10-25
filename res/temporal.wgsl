// Pass 2 - temporal accumulation (the "progressive" part of path tracing).
//
// Reprojects the previous accumulated result into the current frame, rejects it where the surface
// changed (depth mismatch / off-screen), clamps it to the local colour neighbourhood to limit
// ghosting, and blends it with this frame's noisy samples using an exponential history that
// grows up to `max_history_frames`.
//   accum_out = rgba16f (rgb radiance, a occlusion)
//   meta_out  = rg32f   (r = view-space w of this pixel, g = history length)

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var raw_tex: texture_2d<f32>;
@group(0) @binding(5) var hist_tex: texture_2d<f32>;
@group(0) @binding(6) var hist_meta: texture_2d<f32>;
@group(0) @binding(7) var accum_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var meta_out: texture_storage_2d<rg32float, write>;

fn raw_at(p: vec2i, size: vec2i) -> vec4f {
    return textureLoad(raw_tex, clamp(p, vec2i(0), size - 1), 0);
}

@compute @workgroup_size(8, 8, 1)
fn temporal(@builtin(global_invocation_id) gid: vec3u) {
    let q = vec2i(gid.xy);
    let size = vec2i(u.dims.xy);
    if q.x >= size.x || q.y >= size.y {
        return;
    }
    let uv = (vec2f(q) + 0.5) * u.dims.zw;
    let depth = depth_at_uv(uv);
    let nrm = normal_at_uv(uv);
    if is_sky(depth, nrm) {
        textureStore(accum_out, q, vec4f(0.0));
        textureStore(meta_out, q, vec4f(0.0));
        return;
    }
    let p = view_pos(uv, depth);
    let w = -p.z;

    // Neighbourhood statistics of this frame's samples.
    let raw = raw_at(q, size);
    var m1 = vec3f(0.0);
    var m2 = vec3f(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let s = raw_at(q + vec2i(dx, dy), size).rgb;
            m1 += s;
            m2 += s * s;
        }
    }
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3f(0.0)));

    // Reproject.
    var hist = vec4f(0.0);
    var hist_len = 0.0;
    var have_hist = false;
    let prev_clip = u.prev_proj_from_world * (u.world_from_view * vec4f(p, 1.0));
    if u.d.y > 0.5 && prev_clip.w > 0.0 {
        let ndc = prev_clip.xy / prev_clip.w;
        let puv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if all(puv >= vec2f(0.0)) && all(puv <= vec2f(1.0)) {
            let pp = puv * u.dims.xy - 0.5;
            let base = floor(pp);
            let fr = pp - base;
            var wsum = 0.0;
            var hacc = vec4f(0.0);
            var lacc = 0.0;
            for (var j = 0; j < 2; j++) {
                for (var i = 0; i < 2; i++) {
                    let tp = vec2i(base) + vec2i(i, j);
                    if tp.x < 0 || tp.y < 0 || tp.x >= size.x || tp.y >= size.y {
                        continue;
                    }
                    let hm = textureLoad(hist_meta, tp, 0).rg;
                    if hm.g < 0.5 {
                        continue;
                    }
                    if abs(hm.r - prev_clip.w) > 0.04 * prev_clip.w + 1.0 {
                        continue; // different surface (disocclusion)
                    }
                    let bw = select(1.0 - fr.x, fr.x, i == 1) * select(1.0 - fr.y, fr.y, j == 1);
                    wsum += bw;
                    hacc += textureLoad(hist_tex, tp, 0) * bw;
                    lacc += hm.g * bw;
                }
            }
            if wsum > 0.05 {
                hist = hacc / wsum;
                hist_len = lacc / wsum;
                have_hist = true;
            }
        }
    }

    var result = raw;
    var new_len = 1.0;
    if have_hist {
        let k = u.c.z;
        let clamped = clamp(hist.rgb, mean - sigma * k, mean + sigma * k);
        new_len = min(hist_len + 1.0, u.c.y);
        result = mix(vec4f(clamped, hist.a), raw, 1.0 / new_len);
    }
    textureStore(accum_out, q, result);
    textureStore(meta_out, q, vec4f(w, new_len, 0.0, 0.0));
}

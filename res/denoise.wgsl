// Pass 3 - edge-aware a-trous wavelet denoiser (SVGF-style). Run three times with step 1, 2, 4.
// Weights: view-space plane distance, normal agreement and luminance similarity. Luminance
// tolerance widens where the temporal history is still short (noisier pixels).

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var src_tex: texture_2d<f32>;
@group(0) @binding(6) var meta_tex: texture_2d<f32>;
@group(0) @binding(7) var dst_tex: texture_storage_2d<rgba16float, write>;

const KERNEL = array<f32, 3>(0.375, 0.25, 0.0625);

@compute @workgroup_size(8, 8, 1)
fn denoise(@builtin(global_invocation_id) gid: vec3u) {
    let q = vec2i(gid.xy);
    let size = vec2i(u.dims.xy);
    if q.x >= size.x || q.y >= size.y {
        return;
    }
    let uv = (vec2f(q) + 0.5) * u.dims.zw;
    let depth = depth_at_uv(uv);
    let nrm = normal_at_uv(uv);
    if is_sky(depth, nrm) {
        textureStore(dst_tex, q, vec4f(0.0));
        return;
    }
    let p = view_pos(uv, depth);
    var n = normalize(nrm.xyz * 2.0 - 1.0);
    if dot(n, p) > 0.0 {
        n = -n;
    }
    let center = textureLoad(src_tex, q, 0);
    let center_luma = luma(center.rgb);
    let hist_len = max(textureLoad(meta_tex, q, 0).g, 1.0);
    let strength = u.c.w;
    let sigma_l = mix(0.04, 0.8, strength) * (1.0 + 6.0 / hist_len);
    let plane_tol = 0.02 * -p.z + 0.5;
    let stride = i32(u.d.z);

    var sum = vec4f(0.0);
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let tq = q + vec2i(dx, dy) * stride;
            if tq.x < 0 || tq.y < 0 || tq.x >= size.x || tq.y >= size.y {
                continue;
            }
            let tuv = (vec2f(tq) + 0.5) * u.dims.zw;
            let td = depth_at_uv(tuv);
            let tn = normal_at_uv(tuv);
            if is_sky(td, tn) {
                continue;
            }
            let tp = view_pos(tuv, td);
            let tnn = normalize(tn.xyz * 2.0 - 1.0);
            let s = textureLoad(src_tex, tq, 0);

            let w_depth = exp(-abs(dot(n, tp - p)) / plane_tol);
            let w_normal = pow(max(abs(dot(n, tnn)), 0.0), 16.0);
            let w_luma = exp(-abs(luma(s.rgb) - center_luma) / sigma_l);
            let weight = KERNEL[abs(dx)] * KERNEL[abs(dy)] * w_depth * w_normal * w_luma;
            sum += s * weight;
            wsum += weight;
        }
    }
    textureStore(dst_tex, q, select(center, sum / wsum, wsum > 1e-5));
}

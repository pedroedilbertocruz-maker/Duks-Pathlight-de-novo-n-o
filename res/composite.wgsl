// Pass 4 - composite. Fullscreen triangle drawn into the scene colour target.
//
// Normal mode uses premultiplied blending (out = src.rgb + dst * (1 - src.a)):
//   src.rgb = albedo * indirect light * (1 - colour)   (colour bleeding, shadow fill, sky light)
//   src.a   = contact occlusion           (darkens crevices)
// GI is upsampled with depth-aware bilinear weights so half-resolution GI keeps clean edges.
//
// Debug views (debug_view): 1 indirect light only, 2 occlusion, 3 normals, 4 raw GI radiance,
// 5 dynamic light only, 6 darkness scale.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var color_tex: texture_2d<f32>;
@group(0) @binding(4) var gi_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    out.position = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

fn upsample_gi(uv: vec2f, w_here: f32) -> vec4f {
    let size = vec2i(u.dims.xy);
    let pp = uv * u.dims.xy - 0.5;
    let base = floor(pp);
    let fr = pp - base;
    var sum = vec4f(0.0);
    var wsum = 0.0;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let tp = clamp(vec2i(base) + vec2i(i, j), vec2i(0), size - 1);
            let tuv = (vec2f(tp) + 0.5) * u.dims.zw;
            let td = depth_at_uv(tuv);
            let tn = normal_at_uv(tuv);
            if is_sky(td, tn) {
                continue;
            }
            let bw = select(1.0 - fr.x, fr.x, i == 1) * select(1.0 - fr.y, fr.y, j == 1);
            let dw = abs(view_w(tuv, td) - w_here) / (0.02 * w_here + 0.5);
            let weight = max(bw, 0.001) * exp(-dw);
            sum += textureLoad(gi_tex, tp, 0) * weight;
            wsum += weight;
        }
    }
    return select(vec4f(0.0), sum / wsum, wsum > 1e-5);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let depth = depth_at_uv(in.uv);
    let nrm = normal_at_uv(in.uv);
    if u.i.w == 3u {
        return vec4f(select(vec3f(0.0), nrm.xyz, nrm.a >= 0.5), 1.0);
    }
    if is_sky(depth, nrm) {
        if u.i.w != 0u {
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
        return vec4f(sun_rays(in.uv, vec2u(in.position.xy), true), 0.0);  // dst + glow
    }
    let color = textureLoad(color_tex, tex_px(textureDimensions(color_tex), in.uv), 0).rgb;
    let gi = upsample_gi(in.uv, view_w(in.uv, depth));

    // Darkness + dynamic lights. With no darkening and no lights this is the identity.
    let p = view_pos(in.uv, depth);
    var n = normalize(nrm.xyz * 2.0 - 1.0);
    if dot(n, p) > 0.0 {
        n = -n;
    }
    let lf = light_field(p, n, vec2u(in.position.xy), true);
    let scale = scene_scale(lf.a);
    let albedo = albedo_proxy(color);
    let base = color * scale;
    // Light reaching the surface is tinted by its albedo; "screen"-style so it never blows out.
    let direct = albedo * lf.rgb * u.l0.y * (vec3f(1.0) - clamp(base, vec3f(0.0), vec3f(0.95)));
    // Same energy-conserving add for the bounced light, relative to the darkened surface.
    let indirect = albedo * gi.rgb * u.a.z * (vec3f(1.0) - clamp(base, vec3f(0.0), vec3f(0.95)));
    let occlusion = clamp(gi.a * u.b.y, 0.0, 0.95);

    switch u.i.w {
        case 1u: { return vec4f(indirect, 1.0); }
        case 2u: { return vec4f(vec3f(1.0 - occlusion), 1.0); }
        case 4u: { return vec4f(gi.rgb, 1.0); }
        case 5u: { return vec4f(direct, 1.0); }
        case 6u: { return vec4f(vec3f(scale), 1.0); }
        default: {}
    }
    // premultiplied: out = rgb + dst * (1 - a)  =>  dst * m + light, with m = scale * (1 - occlusion).
    // dst already contains the game's fog. Darkening scaled that fog by m, so add back the lost part
    // (1 - m) * fog and hide the lights behind fog: the haze stays as bright as in the original.
    let ss = sun_shadow(p, n, vec2u(in.position.xy));
    let m = scale * (1.0 - occlusion) * ss;

    // Sun colour: warm light on surfaces that face the sun and are not in shadow, cool skylight in
    // cast shadows, plus the rays / bloom. All "screen"-style so nothing blows out.
    var sunlight = vec3f(0.0);
    if u.sun0.w > 0.001 {
        let ndl = clamp(dot(n, normalize(u.sun0.xyz)), 0.0, 1.0);
        let lit = clamp((ss - u.sun1.y) / max(1.0 - u.sun1.y, 1e-3), 0.0, 1.0);
        let room = vec3f(1.0) - clamp(base, vec3f(0.0), vec3f(0.95));
        let warm = albedo * u.sun2.rgb * (ndl * lit * 0.22 * u.sun2.w * u.sun0.w);
        let cool = albedo * vec3f(0.32, 0.5, 0.95) * ((1.0 - lit) * 0.10 * u.sun2.w * u.sun0.w);
        sunlight = (warm + cool) * room;
    }
    let rays = sun_rays(in.uv, vec2u(in.position.xy), false);
    let fog_f = fog_factor(in.uv.x, -p.z);
    let fog_hide = clamp(fog_f * u.fog1.w, 0.0, 1.0);
    let fog_back = (1.0 - m) * fog_f * u.fog1.w * u.fog1.rgb;
    return vec4f((indirect + direct + sunlight) * (1.0 - fog_hide) + fog_back + rays, 1.0 - m);
}

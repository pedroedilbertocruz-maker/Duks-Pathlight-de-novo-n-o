// PathLight GI - real-time path-traced-style global illumination for Dusklight.
//
// This mod approximates what a hardware path tracer produces (colour bleeding, soft indirect
// light, sky light, contact shadowing, multi-bounce light) using only the data the Dusklight
// GfxService exposes after the opaque pass: scene colour, depth and view-space normals.
//
// Per frame (all on the GPU, after opaque scene draws and before translucent lists):
//   1. trace    - Monte Carlo cosine-weighted rays marched through the depth buffer; hits return the
//                 lit colour of what they hit plus that surface's previous-frame indirect light
//                 (infinite bounces through temporal feedback).
//   2. temporal - reprojected history accumulation with neighbourhood clamping.
//   3. denoise  - 3x edge-aware a-trous wavelet filter (depth / normal / luminance).
//   4. composite- depth-aware upsample and energy-conserving blend over the scene.
//
// The WGSL lives in res/ and is also exercised by tools/harness against a synthetic scene.

#include "mods/service.hpp"
#include "mods/svc/camera.h"
#include "mods/svc/config.h"
#include "mods/svc/gfx.h"
#include "mods/svc/log.h"
#include "mods/svc/resource.h"
#include "mods/svc/ui.h"

#include "lights.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <vector>
#include <webgpu/webgpu.h>

DEFINE_MOD();
IMPORT_SERVICE(LogService, svc_log);
IMPORT_SERVICE(ConfigService, svc_config);
IMPORT_SERVICE(ResourceService, svc_resource);
IMPORT_SERVICE(UiService, svc_ui);
IMPORT_SERVICE(GfxService, svc_gfx);
IMPORT_SERVICE(CameraService, svc_camera);

namespace {

// ---------------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------------

struct Option {
    const char* name;
    ConfigVarHandle handle;
};

ConfigVarHandle g_cvarEnabled = 0;
ConfigVarHandle g_cvarQuality = 0;
ConfigVarHandle g_cvarHalfRes = 0;
ConfigVarHandle g_cvarIntensity = 0;
ConfigVarHandle g_cvarBounces = 0;
ConfigVarHandle g_cvarDistance = 0;
ConfigVarHandle g_cvarAo = 0;
ConfigVarHandle g_cvarBleed = 0;
ConfigVarHandle g_cvarSky = 0;
ConfigVarHandle g_cvarEmissive = 0;
ConfigVarHandle g_cvarDenoise = 0;
ConfigVarHandle g_cvarHistory = 0;
ConfigVarHandle g_cvarDebug = 0;
// Dark lighting (Dark Souls 2 style)
ConfigVarHandle g_cvarDarkMode = 0;
ConfigVarHandle g_cvarDarkness = 0;
ConfigVarHandle g_cvarLightGain = 0;
ConfigVarHandle g_cvarLightRadius = 0;
ConfigVarHandle g_cvarShadows = 0;
ConfigVarHandle g_cvarFlicker = 0;
ConfigVarHandle g_cvarSparks = 0;
ConfigVarHandle g_cvarKeepFog = 0;
ConfigVarHandle g_cvarFogStrength = 0;
ConfigVarHandle g_cvarSunShadows = 0;
ConfigVarHandle g_cvarSunStrength = 0;
ConfigVarHandle g_cvarSunLength = 0;
ConfigVarHandle g_cvarSunRays = 0;
ConfigVarHandle g_cvarSunWarmth = 0;
ConfigVarHandle g_cvarSunBloom = 0;
ConfigVarHandle g_cvarSunTod = 0;

GfxComputeTypeHandle g_computeType = 0;
GfxDrawTypeHandle g_drawType = 0;
GfxStageHookHandle g_afterOpaqueHook = 0;
UiWindowHandle g_controlsWindow = 0;

ResourceBuffer g_commonSource = RESOURCE_BUFFER_INIT;
ResourceBuffer g_traceSource = RESOURCE_BUFFER_INIT;
ResourceBuffer g_temporalSource = RESOURCE_BUFFER_INIT;
ResourceBuffer g_denoiseSource = RESOURCE_BUFFER_INIT;
ResourceBuffer g_compositeSource = RESOURCE_BUFFER_INIT;

GfxDeviceInfo g_deviceInfo = GFX_DEVICE_INFO_INIT;
GfxRenderTargetLayout g_sceneTargetLayout = GFX_RENDER_TARGET_LAYOUT_INIT;

WGPUComputePipeline g_tracePipeline = nullptr;
WGPUComputePipeline g_temporalPipeline = nullptr;
WGPUComputePipeline g_denoisePipeline = nullptr;
WGPUBindGroupLayout g_traceLayout = nullptr;
WGPUBindGroupLayout g_temporalLayout = nullptr;
WGPUBindGroupLayout g_denoiseLayout = nullptr;
WGPURenderPipeline g_compositePipeline = nullptr;
WGPURenderPipeline g_compositeDebugPipeline = nullptr;
WGPUBindGroupLayout g_compositeLayout = nullptr;
WGPUBindGroupLayout g_compositeDebugLayout = nullptr;

// ---------------------------------------------------------------------------------------------
// GI render targets (resolution of the GI buffers, possibly half the scene resolution)
// ---------------------------------------------------------------------------------------------

struct GiTargets {
    uint32_t width = 0;
    uint32_t height = 0;
    WGPUTexture textures[10] = {};
    WGPUTextureView views[10] = {};
    // Indices into textures/views.
    enum : int { Raw, AccumA, AccumB, MetaA, MetaB, TmpA, TmpB, Final, Count };
};

// The render worker reads a GiTargets by pointer from the payload. Replaced targets are kept alive
// for a few frames so in-flight payloads never see freed GPU objects.
GiTargets* g_targets = nullptr;
struct Retired {
    GiTargets* targets;
    int framesLeft;
};
std::vector<Retired> g_retired;

uint32_t g_pingPong = 0;
bool g_havePrevCamera = false;
float g_prevProjFromWorld[16] = {};
float g_prevEye[3] = {};
uint32_t g_frameIndex = 0;
bool g_warnedNoDepth = false;
std::atomic<bool> g_chainExecuted{false};
bool g_loggedChain = false;

// Mirror of the WGSL `Uniforms` struct (res/common.wgsl). Keep in sync.
struct GiUniforms {
    float proj[16];
    float inv_proj[16];
    float world_from_view[16];
    float prev_proj_from_world[16];
    float dims[4];
    float full[4];
    float a[4];  // ray_length, thickness, intensity, bounce_gain
    float b[4];  // ao_radius, ao_strength, sky_gain, emissive_boost
    float c[4];  // color_bleed, max_history_frames, history_clamp_k, denoise_strength
    float d[4];  // near_plane, history_valid, denoise_step, unused
    uint32_t i[4];  // frame_index, rays_per_pixel, steps_per_ray, debug_view
    float l0[4];    // ambient_scale, light_gain, shadow_strength, shadow_steps
    float l1[4];    // light_count, restore, unused, time
    float fog0[4];  // fog_start, fog_end, fog_enabled, range_center
    float fog1[4];  // fog rgb, fog_strength
    float fogk[12]; // fog range-adjust table (10 used)
    float sun0[4];  // view-space direction to the sun/moon, strength
    float sun1[4];  // march length, shadow floor, steps, unused
    float sun2[4];  // sun/moon light colour rgb (time of day), warmth
    float sun3[4];  // god-ray strength, bloom strength, unused, unused
    float link0[4]; // Link's feet-segment start (view space) + exclusion radius (0 = none)
    float link1[4]; // Link's segment end (view space)
    float lights[64 * 4];  // view-space pos + radius, then rgb + kind, per light
};
static_assert(sizeof(GiUniforms) == 368 + 32 + 80 + 32 + 32 + 32 + 1024, "GiUniforms must match res/common.wgsl");
static_assert(sizeof(GiUniforms) % 16 == 0);

struct ComputePayload {
    const GiTargets* targets;
    WGPUTextureView depth;
    WGPUTextureView normal;
    WGPUTextureView color;
    uint32_t uniformOffset[4];  // trace/temporal, denoise step 1, 2, 4
    uint32_t uniformSize;
    uint32_t width;
    uint32_t height;
    uint32_t pingPong;
};
static_assert(sizeof(ComputePayload) <= GFX_INLINE_DRAW_PAYLOAD_SIZE);
static_assert(std::is_trivially_copyable_v<ComputePayload>);

struct CompositePayload {
    const GiTargets* targets;
    WGPUTextureView depth;
    WGPUTextureView normal;
    WGPUTextureView color;
    uint32_t uniformOffset;
    uint32_t uniformSize;
    uint32_t debugView;
    uint32_t pad;
};
static_assert(sizeof(CompositePayload) <= GFX_INLINE_DRAW_PAYLOAD_SIZE);
static_assert(std::is_trivially_copyable_v<CompositePayload>);

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

int64_t get_int_option(ConfigVarHandle handle, int64_t fallback) {
    int64_t value = fallback;
    if (handle == 0 || svc_config->get_int(mod_ctx, handle, &value) != MOD_OK) {
        return fallback;
    }
    return value;
}

bool get_bool_option(ConfigVarHandle handle, bool fallback) {
    bool value = fallback;
    if (handle == 0 || svc_config->get_bool(mod_ctx, handle, &value) != MOD_OK) {
        return fallback;
    }
    return value;
}

float percent(ConfigVarHandle handle, int64_t fallback, int64_t lo, int64_t hi) {
    return static_cast<float>(std::clamp<int64_t>(get_int_option(handle, fallback), lo, hi)) /
           100.0f;
}

void quality_counts(int64_t quality, uint32_t& rays, uint32_t& steps) {
    switch (std::clamp<int64_t>(quality, 0, 3)) {
    case 0:
        rays = 1;
        steps = 12;
        break;
    case 1:
        rays = 2;
        steps = 16;
        break;
    default:
    case 2:
        rays = 2;
        steps = 24;
        break;
    case 3:
        rays = 4;
        steps = 32;
        break;
    }
}

WGPUShaderModule create_shader_module(const char* label, const ResourceBuffer& pass) {
    // Every pass is prefixed with res/common.wgsl.
    std::string code(static_cast<const char*>(g_commonSource.data), g_commonSource.size);
    code += '\n';
    code.append(static_cast<const char*>(pass.data), pass.size);
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = {code.data(), code.size()};
    WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    moduleDesc.nextInChain = &wgsl.chain;
    moduleDesc.label = {label, WGPU_STRLEN};
    return wgpuDeviceCreateShaderModule(g_deviceInfo.device, &moduleDesc);
}

bool build_compute_pipeline(const char* label, const ResourceBuffer& source, const char* entry,
    WGPUComputePipeline& outPipeline, WGPUBindGroupLayout& outLayout) {
    WGPUShaderModule module = create_shader_module(label, source);
    if (module == nullptr) {
        return false;
    }
    WGPUComputePipelineDescriptor pipelineDesc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipelineDesc.label = {label, WGPU_STRLEN};
    pipelineDesc.compute.module = module;
    pipelineDesc.compute.entryPoint = {entry, WGPU_STRLEN};
    outPipeline = wgpuDeviceCreateComputePipeline(g_deviceInfo.device, &pipelineDesc);
    wgpuShaderModuleRelease(module);
    if (outPipeline == nullptr) {
        return false;
    }
    outLayout = wgpuComputePipelineGetBindGroupLayout(outPipeline, 0);
    return outLayout != nullptr;
}

bool build_composite_pipeline(const GfxRenderTargetLayout& targetLayout, bool blend,
    WGPURenderPipeline& outPipeline, WGPUBindGroupLayout& outLayout) {
    WGPUShaderModule module = create_shader_module("PathLight composite", g_compositeSource);
    if (module == nullptr) {
        return false;
    }

    // Premultiplied blend: out = src.rgb + dst * (1 - src.a); destination alpha is preserved.
    WGPUBlendState blendState{
        .color =
            {
                .operation = WGPUBlendOperation_Add,
                .srcFactor = WGPUBlendFactor_One,
                .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
            },
        .alpha =
            {
                .operation = WGPUBlendOperation_Add,
                .srcFactor = WGPUBlendFactor_Zero,
                .dstFactor = WGPUBlendFactor_One,
            },
    };
    WGPUColorTargetState colorTargets[GFX_MAX_COLOR_ATTACHMENTS];
    const uint32_t colorTargetCount = gfx_init_color_target_states(
        &targetLayout, colorTargets, blend ? &blendState : nullptr, WGPUColorWriteMask_All);
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = {"fs_main", WGPU_STRLEN};
    fragment.targetCount = colorTargetCount;
    fragment.targets = colorTargets;
    // Depth state must match the scene pass even though it is never touched.
    WGPUDepthStencilState depthStencil = WGPU_DEPTH_STENCIL_STATE_INIT;
    depthStencil.format = targetLayout.depth_stencil_format;
    depthStencil.depthWriteEnabled = WGPUOptionalBool_False;
    depthStencil.depthCompare = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor pipelineDesc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pipelineDesc.label = {blend ? "PathLight composite" : "PathLight composite (debug)",
        WGPU_STRLEN};
    pipelineDesc.vertex.module = module;
    pipelineDesc.vertex.entryPoint = {"vs_main", WGPU_STRLEN};
    pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipelineDesc.depthStencil = &depthStencil;
    pipelineDesc.multisample.count = targetLayout.sample_count;
    pipelineDesc.fragment = &fragment;
    outPipeline = wgpuDeviceCreateRenderPipeline(g_deviceInfo.device, &pipelineDesc);
    wgpuShaderModuleRelease(module);
    if (outPipeline == nullptr) {
        return false;
    }
    outLayout = wgpuRenderPipelineGetBindGroupLayout(outPipeline, 0);
    return outLayout != nullptr;
}

void release_composite() {
    for (auto* pipeline : {&g_compositePipeline, &g_compositeDebugPipeline}) {
        if (*pipeline != nullptr) {
            wgpuRenderPipelineRelease(*pipeline);
            *pipeline = nullptr;
        }
    }
    for (auto* layout : {&g_compositeLayout, &g_compositeDebugLayout}) {
        if (*layout != nullptr) {
            wgpuBindGroupLayoutRelease(*layout);
            *layout = nullptr;
        }
    }
    g_sceneTargetLayout = GFX_RENDER_TARGET_LAYOUT_INIT;
}

bool ensure_composite(const GfxRenderTargetLayout& layout) {
    if (g_compositePipeline != nullptr && g_compositeDebugPipeline != nullptr &&
        g_sceneTargetLayout.key == layout.key)
    {
        return true;
    }
    release_composite();
    if (!build_composite_pipeline(layout, true, g_compositePipeline, g_compositeLayout) ||
        !build_composite_pipeline(layout, false, g_compositeDebugPipeline, g_compositeDebugLayout))
    {
        release_composite();
        return false;
    }
    g_sceneTargetLayout = layout;
    return true;
}

void destroy_targets(GiTargets* targets) {
    if (targets == nullptr) {
        return;
    }
    for (int i = 0; i < GiTargets::Count; ++i) {
        if (targets->views[i] != nullptr) {
            wgpuTextureViewRelease(targets->views[i]);
        }
        if (targets->textures[i] != nullptr) {
            wgpuTextureRelease(targets->textures[i]);
        }
    }
    delete targets;
}

void retire_targets() {
    if (g_targets != nullptr) {
        g_retired.push_back({g_targets, 4});
        g_targets = nullptr;
    }
    g_havePrevCamera = false;
}

void tick_retired() {
    for (auto it = g_retired.begin(); it != g_retired.end();) {
        if (--it->framesLeft <= 0) {
            destroy_targets(it->targets);
            it = g_retired.erase(it);
        } else {
            ++it;
        }
    }
}

bool ensure_targets(uint32_t width, uint32_t height) {
    if (g_targets != nullptr && g_targets->width == width && g_targets->height == height) {
        return true;
    }
    retire_targets();

    auto* targets = new GiTargets();
    targets->width = width;
    targets->height = height;
    const WGPUTextureUsage usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
    bool ok = true;
    for (int i = 0; i < GiTargets::Count && ok; ++i) {
        const bool isMeta = i == GiTargets::MetaA || i == GiTargets::MetaB;
        WGPUTextureDescriptor texDesc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        texDesc.label = {"PathLight GI", WGPU_STRLEN};
        texDesc.usage = usage;
        texDesc.size = {width, height, 1};
        texDesc.format = isMeta ? WGPUTextureFormat_RG32Float : WGPUTextureFormat_RGBA16Float;
        targets->textures[i] = wgpuDeviceCreateTexture(g_deviceInfo.device, &texDesc);
        ok = targets->textures[i] != nullptr;
        if (ok) {
            targets->views[i] = wgpuTextureCreateView(targets->textures[i], nullptr);
            ok = targets->views[i] != nullptr;
        }
    }
    if (!ok) {
        destroy_targets(targets);
        return false;
    }
    g_targets = targets;
    return true;
}

constexpr uint32_t div_ceil(uint32_t numerator, uint32_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

// ---------------------------------------------------------------------------------------------
// Render worker thread callbacks
// ---------------------------------------------------------------------------------------------

void on_compute(
    ModContext*, const GfxComputeContext* ctx, const void* payload, size_t payloadSize, void*) {
    if (payloadSize != sizeof(ComputePayload) || g_tracePipeline == nullptr) {
        return;
    }
    ComputePayload data;
    std::memcpy(&data, payload, sizeof(data));
    if (data.targets == nullptr || data.depth == nullptr || data.normal == nullptr ||
        data.color == nullptr)
    {
        return;
    }
    const GiTargets& t = *data.targets;
    const uint32_t cur = data.pingPong & 1u;
    const uint32_t prev = cur ^ 1u;
    const WGPUTextureView accumPrev = t.views[prev == 0 ? GiTargets::AccumA : GiTargets::AccumB];
    const WGPUTextureView accumCur = t.views[cur == 0 ? GiTargets::AccumA : GiTargets::AccumB];
    const WGPUTextureView metaPrev = t.views[prev == 0 ? GiTargets::MetaA : GiTargets::MetaB];
    const WGPUTextureView metaCur = t.views[cur == 0 ? GiTargets::MetaA : GiTargets::MetaB];

    struct Entry {
        uint32_t binding;
        WGPUTextureView view;       // texture entry if non-null
        uint32_t uniformOffset;     // otherwise uniform entry
    };
    const auto makeBindGroup = [&](WGPUBindGroupLayout layout, std::initializer_list<Entry> list) {
        WGPUBindGroupEntry entries[12];
        size_t count = 0;
        for (const Entry& e : list) {
            WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
            entry.binding = e.binding;
            if (e.view != nullptr) {
                entry.textureView = e.view;
            } else {
                entry.buffer = ctx->uniform_buffer;
                entry.offset = e.uniformOffset;
                entry.size = data.uniformSize;
            }
            entries[count++] = entry;
        }
        WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        desc.layout = layout;
        desc.entryCount = count;
        desc.entries = entries;
        return wgpuDeviceCreateBindGroup(ctx->device, &desc);
    };
    const auto tex = [](uint32_t binding, WGPUTextureView view) { return Entry{binding, view, 0}; };
    const auto ubo = [](uint32_t offset) { return Entry{0, nullptr, offset}; };

    WGPUBindGroup groups[5] = {};
    groups[0] = makeBindGroup(g_traceLayout,
        {ubo(data.uniformOffset[0]), tex(1, data.depth), tex(2, data.normal), tex(3, data.color),
            tex(4, t.views[GiTargets::Final]), tex(5, t.views[GiTargets::Raw])});
    groups[1] = makeBindGroup(g_temporalLayout,
        {ubo(data.uniformOffset[0]), tex(1, data.depth), tex(2, data.normal),
            tex(4, t.views[GiTargets::Raw]), tex(5, accumPrev), tex(6, metaPrev),
            tex(7, accumCur), tex(8, metaCur)});
    groups[2] = makeBindGroup(g_denoiseLayout,
        {ubo(data.uniformOffset[1]), tex(1, data.depth), tex(2, data.normal), tex(4, accumCur),
            tex(6, metaCur), tex(7, t.views[GiTargets::TmpA])});
    groups[3] = makeBindGroup(g_denoiseLayout,
        {ubo(data.uniformOffset[2]), tex(1, data.depth), tex(2, data.normal),
            tex(4, t.views[GiTargets::TmpA]), tex(6, metaCur), tex(7, t.views[GiTargets::TmpB])});
    groups[4] = makeBindGroup(g_denoiseLayout,
        {ubo(data.uniformOffset[3]), tex(1, data.depth), tex(2, data.normal),
            tex(4, t.views[GiTargets::TmpB]), tex(6, metaCur), tex(7, t.views[GiTargets::Final])});

    bool valid = true;
    for (WGPUBindGroup group : groups) {
        valid = valid && group != nullptr;
    }
    if (valid) {
        WGPUComputePassDescriptor passDesc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        passDesc.label = {"PathLight GI", WGPU_STRLEN};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(ctx->encoder, &passDesc);
        const uint32_t gx = div_ceil(data.width, 8);
        const uint32_t gy = div_ceil(data.height, 8);
        const auto dispatch = [&](WGPUComputePipeline pipeline, WGPUBindGroup group) {
            wgpuComputePassEncoderSetPipeline(pass, pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
        };
        dispatch(g_tracePipeline, groups[0]);
        dispatch(g_temporalPipeline, groups[1]);
        dispatch(g_denoisePipeline, groups[2]);
        dispatch(g_denoisePipeline, groups[3]);
        dispatch(g_denoisePipeline, groups[4]);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        g_chainExecuted.store(true, std::memory_order_release);
    }
    for (WGPUBindGroup group : groups) {
        if (group != nullptr) {
            wgpuBindGroupRelease(group);
        }
    }
}

void on_draw(
    ModContext*, const GfxDrawContext* ctx, const void* payload, size_t payloadSize, void*) {
    if (payloadSize != sizeof(CompositePayload) || !ensure_composite(ctx->layout)) {
        return;
    }
    CompositePayload data;
    std::memcpy(&data, payload, sizeof(data));
    const bool debug = data.debugView != 0;
    WGPURenderPipeline pipeline = debug ? g_compositeDebugPipeline : g_compositePipeline;
    WGPUBindGroupLayout layout = debug ? g_compositeDebugLayout : g_compositeLayout;
    if (data.targets == nullptr || data.depth == nullptr || data.normal == nullptr ||
        data.color == nullptr || pipeline == nullptr)
    {
        return;
    }

    WGPUBindGroupEntry entries[5];
    for (auto& entry : entries) {
        entry = WGPU_BIND_GROUP_ENTRY_INIT;
    }
    entries[0].binding = 0;
    entries[0].buffer = ctx->uniform_buffer;
    entries[0].offset = data.uniformOffset;
    entries[0].size = data.uniformSize;
    entries[1].binding = 1;
    entries[1].textureView = data.depth;
    entries[2].binding = 2;
    entries[2].textureView = data.normal;
    entries[3].binding = 3;
    entries[3].textureView = data.color;
    entries[4].binding = 4;
    entries[4].textureView = data.targets->views[GiTargets::Final];
    WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    desc.layout = layout;
    desc.entryCount = 5;
    desc.entries = entries;
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(ctx->device, &desc);
    if (bindGroup == nullptr) {
        return;
    }
    wgpuRenderPassEncoderSetPipeline(ctx->pass, pipeline);
    wgpuRenderPassEncoderSetBindGroup(ctx->pass, 0, bindGroup, 0, nullptr);
    wgpuRenderPassEncoderDraw(ctx->pass, 3, 1, 0, 0);
    wgpuBindGroupRelease(bindGroup);
}

// ---------------------------------------------------------------------------------------------
// Game thread: stage hook
// ---------------------------------------------------------------------------------------------

// Per-frame dynamic-light state, built on the game thread right before the uniforms are filled.
struct FrameLights {
    float ambientScale = 1.0f;
    float gain = 0.0f;
    float shadowStrength = 0.0f;
    float shadowSteps = 0.0f;
    int count = 0;
    float lights[64 * 4] = {};
};
FrameLights g_frameLights;
FogSnapshot g_frameFog;
SunSnapshot g_frameSun;
float g_darkAmount = 0.0f;  // smoothed 0..1 "how dark is this place"
bool g_darkInit = false;
std::chrono::steady_clock::time_point g_darkTime;
const std::chrono::steady_clock::time_point g_startTime = std::chrono::steady_clock::now();

float smooth_flicker(const SceneLight& l, float t, float flickerOption) {
    const float phase = l.pos[0] * 0.013f + l.pos[2] * 0.021f + l.pos[1] * 0.007f;
    const float n = 0.5f * std::sin(t * 11.3f + phase) + 0.3f * std::sin(t * 17.9f + phase * 1.7f) +
                    0.2f * std::sin(t * 29.3f + phase * 0.6f);
    const float amount = (l.flicker * 0.45f + 0.03f) * flickerOption;
    return 1.0f + n * amount;
}

void prepare_frame_lights(const CameraInfo& camera, const void* gameViewport) {
    FrameLights& f = g_frameLights;
    f = FrameLights{};
    collect_fog(gameViewport, g_frameFog);
    if (!get_bool_option(g_cvarKeepFog, true)) {
        g_frameFog.valid = false;
    }
    collect_sun(camera.eye, g_frameSun);
    if (!get_bool_option(g_cvarSunShadows, true)) {
        g_frameSun.valid = false;
    }

    const auto now = std::chrono::steady_clock::now();
    float dt = 1.0f / 60.0f;
    if (g_darkInit) {
        dt = std::clamp(std::chrono::duration<float>(now - g_darkTime).count(), 0.0f, 0.25f);
    }
    g_darkTime = now;
    const float t = std::chrono::duration<float>(now - g_startTime).count();

    LightSnapshot snap;
    collect_scene_lights(camera.eye, snap);

    const int64_t mode = std::clamp<int64_t>(get_int_option(g_cvarDarkMode, 1), 0, 3);
    const float slider = percent(g_cvarDarkness, 80, 0, 100);
    float target = 0.0f;
    switch (snap.stageClass) {
    case STAGE_DUNGEON:
        target = mode >= 1 ? slider : 0.0f;
        break;
    case STAGE_INTERIOR:
        target = mode >= 2 ? slider : 0.0f;
        break;
    default:  // outdoors: only in "everywhere" mode, and mostly at night
        target = mode >= 3 ? slider * (0.4f + 0.6f * snap.night) : 0.0f;
        break;
    }
    if (!g_darkInit) {
        g_darkAmount = target;  // no fade on the very first frame
        g_darkInit = true;
    } else {
        const float step = dt / 1.2f;  // ~1.2 s fade between lit and dark places
        g_darkAmount += std::clamp(target - g_darkAmount, -step, step);
    }

    f.ambientScale = 1.0f - 0.93f * g_darkAmount;
    if (g_darkAmount < 0.01f) {
        return;  // identical to plain GI: no extra lights
    }

    f.gain = percent(g_cvarLightGain, 100, 0, 400) * g_darkAmount;
    const float shadow = percent(g_cvarShadows, 100, 0, 100);
    static const float kSteps[4] = {6.0f, 10.0f, 14.0f, 20.0f};
    f.shadowStrength = shadow;
    f.shadowSteps = shadow > 0.0f
                        ? kSteps[std::clamp<int64_t>(get_int_option(g_cvarQuality, 2), 0, 3)]
                        : 0.0f;

    const float radiusScale = percent(g_cvarLightRadius, 100, 30, 300);
    const float flickerOption = percent(g_cvarFlicker, 100, 0, 200);
    const bool sparks = get_bool_option(g_cvarSparks, true);
    const float* m = camera.view_from_world;  // column-major
    for (int i = 0; i < snap.count && f.count < 32; ++i) {
        const SceneLight& l = snap.lights[i];
        if (l.kind == LIGHT_KIND_SPARK && !sparks) {
            continue;
        }
        const float flick = l.kind == LIGHT_KIND_SPARK ? 1.0f : smooth_flicker(l, t, flickerOption);
        float* out = f.lights + f.count * 8;
        for (int r = 0; r < 3; ++r) {
            out[r] = m[r] * l.pos[0] + m[4 + r] * l.pos[1] + m[8 + r] * l.pos[2] + m[12 + r];
        }
        out[3] = l.radius * radiusScale * (1.0f + (flick - 1.0f) * 0.5f);
        out[4] = l.color[0] * flick;
        out[5] = l.color[1] * flick;
        out[6] = l.color[2] * flick;
        out[7] = static_cast<float>(l.kind);
        ++f.count;
    }
}

void fill_uniforms(GiUniforms& u, const CameraInfo& camera, uint32_t giWidth, uint32_t giHeight,
    uint32_t sceneWidth, uint32_t sceneHeight, bool historyValid, uint32_t debugView) {
    std::memcpy(u.proj, camera.proj_from_view, sizeof(u.proj));
    std::memcpy(u.inv_proj, camera.view_from_proj, sizeof(u.inv_proj));
    std::memcpy(u.world_from_view, camera.world_from_view, sizeof(u.world_from_view));
    std::memcpy(u.prev_proj_from_world, historyValid ? g_prevProjFromWorld : camera.proj_from_world,
        sizeof(u.prev_proj_from_world));

    u.dims[0] = static_cast<float>(giWidth);
    u.dims[1] = static_cast<float>(giHeight);
    u.dims[2] = 1.0f / u.dims[0];
    u.dims[3] = 1.0f / u.dims[1];
    u.full[0] = static_cast<float>(sceneWidth);
    u.full[1] = static_cast<float>(sceneHeight);
    u.full[2] = u.full[0] / u.dims[0];
    u.full[3] = u.full[1] / u.dims[1];

    const float distance =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarDistance, 600), 50, 3000));
    u.a[0] = distance;
    u.a[1] = std::clamp(distance * 0.08f, 8.0f, 150.0f);  // surface thickness
    u.a[2] = percent(g_cvarIntensity, 110, 0, 400);
    u.a[3] = percent(g_cvarBounces, 60, 0, 90);
    u.b[0] = std::max(distance * 0.2f, 10.0f);  // contact-occlusion radius
    u.b[1] = percent(g_cvarAo, 60, 0, 100);
    u.b[2] = percent(g_cvarSky, 50, 0, 300);
    u.b[3] = percent(g_cvarEmissive, 50, 0, 300);
    u.c[0] = percent(g_cvarBleed, 100, 0, 100);
    u.c[1] = static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarHistory, 24), 1, 96));
    u.c[2] = 3.0f;
    u.c[3] = percent(g_cvarDenoise, 50, 0, 100);
    u.d[0] = std::max(camera.near_plane, 0.01f);
    u.d[1] = historyValid ? 1.0f : 0.0f;
    u.d[2] = 1.0f;
    u.d[3] = 0.0f;

    uint32_t rays = 2, steps = 24;
    quality_counts(get_int_option(g_cvarQuality, 2), rays, steps);
    u.i[0] = g_frameIndex;
    u.i[1] = rays;
    u.i[2] = steps;
    u.i[3] = debugView;

    const FrameLights& f = g_frameLights;
    u.l0[0] = f.ambientScale;
    u.l0[1] = f.gain;
    u.l0[2] = f.shadowStrength;
    u.l0[3] = f.shadowSteps;
    u.l1[0] = static_cast<float>(f.count);
    u.l1[1] = 0.7f;  // how much of the game's own lighting returns near a light
    u.l1[2] = 0.0f;
    u.l1[3] = 0.0f;
    const FogSnapshot& fog = g_frameFog;
    u.fog0[0] = fog.start;
    u.fog0[1] = fog.end;
    u.fog0[2] = fog.valid ? 1.0f : 0.0f;
    u.fog0[3] = fog.centerNdc;
    u.fog1[0] = fog.color[0];
    u.fog1[1] = fog.color[1];
    u.fog1[2] = fog.color[2];
    u.fog1[3] = percent(g_cvarFogStrength, 100, 0, 200);
    std::memcpy(u.fogk, fog.k, sizeof(fog.k));
    const SunSnapshot& sun = g_frameSun;
    std::memset(u.sun0, 0, sizeof(u.sun0));
    std::memset(u.sun1, 0, sizeof(u.sun1));
    std::memset(u.sun2, 0, sizeof(u.sun2));
    std::memset(u.sun3, 0, sizeof(u.sun3));
    std::memset(u.link0, 0, sizeof(u.link0));
    std::memset(u.link1, 0, sizeof(u.link1));
    if (sun.valid) {
        const float* m = camera.view_from_world;  // column-major; direction: w = 0
        for (int r = 0; r < 3; ++r) {
            u.sun0[r] = m[r] * sun.dir[0] + m[4 + r] * sun.dir[1] + m[8 + r] * sun.dir[2];
        }
        u.sun0[3] = sun.strength * percent(g_cvarSunStrength, 100, 0, 100);
        // Low sun -> long shadows: an object of height H needs a ray of H / sin(elevation).
        const float elev = std::max(sun.dir[1], 0.25f);
        u.sun1[0] = 1000.0f * percent(g_cvarSunLength, 100, 30, 300) / elev;
        u.sun1[1] = 0.45f;  // darkest a sun-shadowed surface gets
        static const float kSunSteps[4] = {8.0f, 10.0f, 14.0f, 20.0f};
        u.sun1[2] = kSunSteps[std::clamp<int64_t>(get_int_option(g_cvarQuality, 2), 0, 3)];
        {
            // Light colour from the sun's height: gold at dawn/dusk, warm white at noon, blue moon.
            const float h = std::clamp(sun.dir[1], 0.0f, 1.0f);
            const float k = std::clamp((h - 0.05f) / 0.5f, 0.0f, 1.0f);
            float c[3] = {1.0f, 0.93f, 0.82f};
            if (sun.isMoon) {
                c[0] = 0.45f; c[1] = 0.58f; c[2] = 1.0f;
            } else if (get_bool_option(g_cvarSunTod, true)) {
                c[0] = 1.0f;
                c[1] = 0.72f + 0.24f * k;
                c[2] = 0.50f + 0.40f * k;
            }
            for (int r = 0; r < 3; ++r) u.sun2[r] = c[r];
            u.sun2[3] = percent(g_cvarSunWarmth, 100, 0, 200) * (sun.isMoon ? 0.35f : 1.0f);
            u.sun3[0] = percent(g_cvarSunRays, 100, 0, 300) * (sun.isMoon ? 0.3f : 1.0f);
            u.sun3[1] = percent(g_cvarSunBloom, 100, 0, 200) * (sun.isMoon ? 0.5f : 1.0f);
        }
        if (sun.linkValid) {
            // Link's body as a vertical segment (feet+20 .. head+10); the game already draws his shadow.
            const float a[3] = {sun.linkPos[0], sun.linkPos[1] + 20.0f, sun.linkPos[2]};
            const float b[3] = {sun.linkPos[0], sun.linkPos[1] + 170.0f, sun.linkPos[2]};
            for (int r = 0; r < 3; ++r) {
                u.link0[r] = m[r] * a[0] + m[4 + r] * a[1] + m[8 + r] * a[2] + m[12 + r];
                u.link1[r] = m[r] * b[0] + m[4 + r] * b[1] + m[8 + r] * b[2] + m[12 + r];
            }
            u.link0[3] = 75.0f;
        }
    }
    std::memcpy(u.lights, f.lights, sizeof(u.lights));
}

void on_scene_after_opaque(ModContext*, const GfxStageContext* stageCtx, void*) {
    tick_retired();
    if (!get_bool_option(g_cvarEnabled, true)) {
        if (g_targets != nullptr) {
            retire_targets();  // drop stale history while disabled
        }
        return;
    }
    if (stageCtx == nullptr || stageCtx->struct_size < sizeof(GfxStageContext) ||
        stageCtx->game_view == nullptr)
    {
        return;
    }

    CameraInfo camera = CAMERA_INFO_INIT;
    if (svc_camera->get_camera(mod_ctx, stageCtx->game_view, &camera) != MOD_OK) {
        return;
    }

    GfxResolveDesc resolveDesc = GFX_RESOLVE_DESC_INIT;
    resolveDesc.color = true;
    resolveDesc.depth = true;
    resolveDesc.normal = true;
    GfxResolvedTargets resolved = GFX_RESOLVED_TARGETS_INIT;
    if (svc_gfx->resolve_pass(mod_ctx, &resolveDesc, &resolved) != MOD_OK ||
        resolved.depth == nullptr || resolved.color == nullptr)
    {
        if (!g_warnedNoDepth) {
            g_warnedNoDepth = true;
            svc_log->warn(mod_ctx, "scene colour/depth snapshots unavailable; PathLight GI idle");
        }
        return;
    }
    // The first request enables normals from the next frame on.
    if (resolved.normal == nullptr) {
        return;
    }

    const uint32_t divisor = get_bool_option(g_cvarHalfRes, true) ? 2 : 1;
    const uint32_t width = resolved.width / divisor;
    const uint32_t height = resolved.height / divisor;
    if (width < 32 || height < 32) {
        return;
    }
    const bool hadTargets = g_targets != nullptr;
    if (!ensure_targets(width, height)) {
        return;
    }

    // History is only valid for a continuous camera on unchanged targets.
    bool historyValid = hadTargets && g_havePrevCamera && g_targets != nullptr;
    if (historyValid) {
        const float dx = camera.eye[0] - g_prevEye[0];
        const float dy = camera.eye[1] - g_prevEye[1];
        const float dz = camera.eye[2] - g_prevEye[2];
        if (std::sqrt(dx * dx + dy * dy + dz * dz) > 1500.0f) {
            historyValid = false;  // teleport / scene cut
        }
    }

    const uint32_t debugView =
        static_cast<uint32_t>(std::clamp<int64_t>(get_int_option(g_cvarDebug, 0), 0, 6));

    prepare_frame_lights(camera, stageCtx->game_viewport);

    // One uniform block per dispatch group (trace/temporal, then three denoise strides).
    GiUniforms uniforms{};
    fill_uniforms(uniforms, camera, width, height, resolved.width, resolved.height, historyValid,
        debugView);
    GfxRange ranges[4] = {};
    const float strides[4] = {1.0f, 1.0f, 2.0f, 4.0f};
    for (int i = 0; i < 4; ++i) {
        uniforms.d[2] = strides[i];
        if (svc_gfx->push_uniform(mod_ctx, &uniforms, sizeof(uniforms), &ranges[i]) != MOD_OK) {
            return;
        }
    }

    ComputePayload computePayload{};
    computePayload.targets = g_targets;
    computePayload.depth = resolved.depth;
    computePayload.normal = resolved.normal;
    computePayload.color = resolved.color;
    for (int i = 0; i < 4; ++i) {
        computePayload.uniformOffset[i] = ranges[i].offset;
    }
    computePayload.uniformSize = ranges[0].size;
    computePayload.width = width;
    computePayload.height = height;
    computePayload.pingPong = g_pingPong;
    if (svc_gfx->push_compute(mod_ctx, g_computeType, &computePayload, sizeof(computePayload)) !=
        MOD_OK)
    {
        return;
    }

    CompositePayload drawPayload{};
    drawPayload.targets = g_targets;
    drawPayload.depth = resolved.depth;
    drawPayload.normal = resolved.normal;
    drawPayload.color = resolved.color;
    drawPayload.uniformOffset = ranges[0].offset;
    drawPayload.uniformSize = ranges[0].size;
    drawPayload.debugView = debugView;
    svc_gfx->push_draw(mod_ctx, g_drawType, &drawPayload, sizeof(drawPayload));

    std::memcpy(g_prevProjFromWorld, camera.proj_from_world, sizeof(g_prevProjFromWorld));
    std::memcpy(g_prevEye, camera.eye, sizeof(g_prevEye));
    g_havePrevCamera = true;
    g_pingPong ^= 1u;
    g_frameIndex = (g_frameIndex + 1u) & 4095u;
}

// ---------------------------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------------------------

void add_control(UiElementHandle pane, const UiControlDesc& desc) {
    svc_ui->pane_add_control(mod_ctx, pane, &desc, nullptr);
}

void add_toggle(UiElementHandle pane, const char* label, ConfigVarHandle cvar, const char* help) {
    UiControlDesc control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_TOGGLE;
    control.label = label;
    control.help_rml = help;
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = cvar;
    add_control(pane, control);
}

void add_number(UiElementHandle pane, const char* label, ConfigVarHandle cvar, int min, int max,
    int step, const char* suffix, const char* help) {
    UiControlDesc control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_NUMBER;
    control.label = label;
    control.help_rml = help;
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = cvar;
    control.min = min;
    control.max = max;
    control.step = step;
    control.suffix = suffix;
    add_control(pane, control);
}

void add_select(UiElementHandle pane, const char* label, ConfigVarHandle cvar,
    const char* const* options, int count, const char* help) {
    UiControlDesc control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_SELECT;
    control.label = label;
    control.help_rml = help;
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = cvar;
    control.options = options;
    control.option_count = count;
    add_control(pane, control);
}

ModResult build_controls_tab(
    ModContext*, UiWindowHandle, UiElementHandle left, UiElementHandle right, void*, ModError*) {
    static const char* kQuality[] = {"Low (1 ray)", "Medium (2 rays)", "High (2 rays, long)",
        "Ultra (4 rays)"};
    static const char* kDebug[] = {"Off", "Indirect light only", "Occlusion", "Normals",
        "GI radiance", "Dynamic light only", "Darkness scale"};
    static const char* kDarkMode[] = {"Off (GI only)", "Dungeons and caves", "Dungeons, caves and interiors",
        "Everywhere (extra dark at night)"};

    svc_ui->pane_add_section(mod_ctx, left, "Path-traced GI");
    add_toggle(left, "Enabled", g_cvarEnabled, "Turns the global illumination pass on or off.");
    add_select(left, "Quality", g_cvarQuality, kQuality, 4,
        "Rays per pixel per frame and march steps per ray. Light accumulates over time, so even "
        "Low converges to a smooth result when the camera is still.");
    add_toggle(left, "Half Resolution", g_cvarHalfRes,
        "Traces at half resolution and upsamples with depth-aware filtering. Recommended; "
        "roughly 3-4x faster.");
    add_number(left, "Intensity", g_cvarIntensity, 0, 400, 5, "%",
        "Strength of the indirect light added to the scene.");
    add_number(left, "Bounces", g_cvarBounces, 0, 90, 5, "%",
        "How much light is carried over from surface to surface across frames (multi-bounce). "
        "0 = single bounce.");
    add_number(left, "Ray Distance", g_cvarDistance, 50, 3000, 50, "",
        "How far (in game units) a light ray may travel. Larger = more distant color bleeding "
        "but a bit more noise.");

    svc_ui->pane_add_section(mod_ctx, right, "Look");
    add_number(right, "Contact Shadows", g_cvarAo, 0, 100, 5, "%",
        "Darkening of creases and contact areas from nearby geometry.");
    add_number(right, "Color Bleeding", g_cvarBleed, 0, 100, 5, "%",
        "0 = bounce light is neutral, 100 = tinted by the color of nearby surfaces.");
    add_number(right, "Sky Light", g_cvarSky, 0, 300, 10, "%",
        "How much visible sky lights the scene.");
    add_number(right, "Emissive Boost", g_cvarEmissive, 0, 300, 10, "%",
        "Extra bounce light from very bright pixels such as lamps, fire and magic.");

    svc_ui->pane_add_section(mod_ctx, left, "Dark lighting (Dark Souls 2 style)");
    add_select(left, "Where", g_cvarDarkMode, kDarkMode, 4,
        "Where the world is darkened. Light then comes only from torches, candles, your lantern, "
        "fire and sword sparks.");
    add_number(left, "Darkness", g_cvarDarkness, 0, 100, 5, "%",
        "How dark unlit areas are. 100 = almost pitch black.");
    add_number(left, "Light Strength", g_cvarLightGain, 0, 400, 10, "%",
        "Brightness of torches, lantern and sparks.");
    add_number(left, "Light Radius", g_cvarLightRadius, 30, 300, 10, "%",
        "How far each light reaches.");
    add_number(left, "Light Shadows", g_cvarShadows, 0, 100, 10, "%",
        "Objects block light and cast shadows (screen-space).");
    add_number(left, "Flame Flicker", g_cvarFlicker, 0, 200, 10, "%",
        "How much torch and lantern light flickers.");
    add_toggle(left, "Sword Sparks", g_cvarSparks,
        "A brief flash of light where your sword hits something.");
    add_toggle(left, "Keep Fog", g_cvarKeepFog,
        "Puts the game's fog back after darkening, so haze stays visible in dark places. Same idea "
        "as the Deferred Fog mod: turn that mod OFF while this is on, or fog is applied twice.");
    add_number(left, "Fog Strength", g_cvarFogStrength, 0, 200, 10, "%",
        "How strongly the restored fog shows.");

    add_toggle(left, "Sun Shadows", g_cvarSunShadows,
        "Buildings, trees and rocks cast shadows from the sun (and a fainter moon). They move with "
        "the in-game clock. Outdoors only.");
    add_number(left, "Sun Shadow Strength", g_cvarSunStrength, 0, 100, 5, "%",
        "How dark sun shadows are.");
    add_number(left, "Sun Rays", g_cvarSunRays, 0, 300, 10, "%",
        "Visible light shafts from the sun through trees and buildings (needs the sun in front of "
        "the camera).");
    add_number(left, "Sun Warmth", g_cvarSunWarmth, 0, 200, 10, "%",
        "Warm sunlight on lit surfaces and cool skylight in shadow.");
    add_number(left, "Sun Bloom", g_cvarSunBloom, 0, 200, 10, "%",
        "Glow around the sun.");
    add_toggle(left, "Time-of-Day Colour", g_cvarSunTod,
        "Sunlight turns golden at dawn and dusk. The moon is blue.");
    add_number(left, "Sun Shadow Length", g_cvarSunLength, 30, 300, 10, "%",
        "How far shadows are searched for. Raise it for tall objects.");

    svc_ui->pane_add_section(mod_ctx, right, "Denoiser");
    add_number(right, "Denoise Strength", g_cvarDenoise, 0, 100, 5, "%",
        "Spatial edge-aware filtering. Higher is smoother but softer.");
    add_number(right, "Temporal Frames", g_cvarHistory, 1, 96, 1, "",
        "How many frames of light are accumulated. Higher is smoother when still; lower reacts "
        "faster to moving lights and objects.");
    add_select(right, "Debug View", g_cvarDebug, kDebug, 7,
        "Visualize individual parts of the effect.");
    return MOD_OK;
}

void on_controls_window_closed(ModContext*, UiWindowHandle, void*) {
    g_controlsWindow = 0;
}

void on_open_controls(ModContext*, void*) {
    if (g_controlsWindow != 0) {
        return;
    }
    UiTabDesc tabs[1] = {UI_TAB_DESC_INIT};
    tabs[0].title = "PathLight GI";
    tabs[0].build = build_controls_tab;
    UiWindowDesc desc = UI_WINDOW_DESC_INIT;
    desc.tabs = tabs;
    desc.tab_count = 1;
    desc.on_closed = on_controls_window_closed;
    if (svc_ui->window_push(mod_ctx, &desc, &g_controlsWindow) != MOD_OK) {
        svc_log->error(mod_ctx, "failed to open PathLight GI controls window");
    }
}

ModResult build_panel(ModContext*, UiElementHandle panel, void*, ModError*) {
    UiControlDesc control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_TOGGLE;
    control.label = "Enabled";
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = g_cvarEnabled;
    add_control(panel, control);

    control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_BUTTON;
    control.label = "Open Controls";
    control.on_pressed = on_open_controls;
    add_control(panel, control);
    return MOD_OK;
}

ModResult register_bool_option(
    const char* name, bool defaultValue, ConfigVarHandle& outHandle, ModError* error) {
    ConfigVarDesc cvarDesc = CONFIG_VAR_DESC_INIT;
    cvarDesc.name = name;
    cvarDesc.type = CONFIG_VAR_BOOL;
    cvarDesc.default_bool = defaultValue;
    if (svc_config->register_var(mod_ctx, &cvarDesc, &outHandle) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register option");
    }
    return MOD_OK;
}

ModResult register_int_option(
    const char* name, int64_t defaultValue, ConfigVarHandle& outHandle, ModError* error) {
    ConfigVarDesc cvarDesc = CONFIG_VAR_DESC_INIT;
    cvarDesc.name = name;
    cvarDesc.type = CONFIG_VAR_INT;
    cvarDesc.default_int = defaultValue;
    if (svc_config->register_var(mod_ctx, &cvarDesc, &outHandle) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register option");
    }
    return MOD_OK;
}

ModResult load_shaders(ModError* error) {
    struct Shader {
        const char* file;
        ResourceBuffer* buffer;
    };
    for (const Shader& shader : {Shader{"common.wgsl", &g_commonSource},
             Shader{"trace.wgsl", &g_traceSource}, Shader{"temporal.wgsl", &g_temporalSource},
             Shader{"denoise.wgsl", &g_denoiseSource},
             Shader{"composite.wgsl", &g_compositeSource}})
    {
        const ModResult result = svc_resource->load(mod_ctx, shader.file, shader.buffer);
        if (result != MOD_OK) {
            return mods::set_error(error, result, "failed to load PathLight GI shaders");
        }
    }
    return MOD_OK;
}

}  // namespace

extern "C" {

MOD_EXPORT ModResult mod_initialize(ModError* error) {
    ModResult result = load_shaders(error);
    if (result != MOD_OK) {
        return result;
    }

    result = register_bool_option("giEnabled", true, g_cvarEnabled, error);
    if (result == MOD_OK) result = register_int_option("quality", 2, g_cvarQuality, error);
    if (result == MOD_OK) result = register_bool_option("halfRes", true, g_cvarHalfRes, error);
    if (result == MOD_OK) result = register_int_option("intensity", 110, g_cvarIntensity, error);
    if (result == MOD_OK) result = register_int_option("bounces", 60, g_cvarBounces, error);
    if (result == MOD_OK) result = register_int_option("rayDistance", 600, g_cvarDistance, error);
    if (result == MOD_OK) result = register_int_option("contactShadows", 60, g_cvarAo, error);
    if (result == MOD_OK) result = register_int_option("colorBleeding", 100, g_cvarBleed, error);
    if (result == MOD_OK) result = register_int_option("skyLight", 50, g_cvarSky, error);
    if (result == MOD_OK) result = register_int_option("emissiveBoost", 50, g_cvarEmissive, error);
    if (result == MOD_OK) result = register_int_option("denoise", 50, g_cvarDenoise, error);
    if (result == MOD_OK) result = register_int_option("temporalFrames", 24, g_cvarHistory, error);
    if (result == MOD_OK) result = register_int_option("debugView", 0, g_cvarDebug, error);
    if (result == MOD_OK) result = register_int_option("darkMode", 1, g_cvarDarkMode, error);
    if (result == MOD_OK) result = register_int_option("darkness", 80, g_cvarDarkness, error);
    if (result == MOD_OK) result = register_int_option("lightStrength", 100, g_cvarLightGain, error);
    if (result == MOD_OK) result = register_int_option("lightRadius", 100, g_cvarLightRadius, error);
    if (result == MOD_OK) result = register_int_option("lightShadows", 100, g_cvarShadows, error);
    if (result == MOD_OK) result = register_int_option("flameFlicker", 100, g_cvarFlicker, error);
    if (result == MOD_OK) result = register_bool_option("swordSparks", true, g_cvarSparks, error);
    if (result == MOD_OK) result = register_bool_option("keepFog", true, g_cvarKeepFog, error);
    if (result == MOD_OK) result = register_int_option("fogStrength", 100, g_cvarFogStrength, error);
    if (result == MOD_OK) result = register_bool_option("sunShadows", true, g_cvarSunShadows, error);
    if (result == MOD_OK) result = register_int_option("sunStrength", 85, g_cvarSunStrength, error);
    if (result == MOD_OK) result = register_int_option("sunLength", 100, g_cvarSunLength, error);
    if (result == MOD_OK) result = register_int_option("sunRays", 50, g_cvarSunRays, error);
    if (result == MOD_OK) result = register_int_option("sunWarmth", 70, g_cvarSunWarmth, error);
    if (result == MOD_OK) result = register_int_option("sunBloom", 60, g_cvarSunBloom, error);
    if (result == MOD_OK) result = register_bool_option("sunTod", true, g_cvarSunTod, error);
    if (result != MOD_OK) {
        return result;
    }

    if (svc_gfx->get_device_info(mod_ctx, &g_deviceInfo) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to query device info");
    }
    if (!build_compute_pipeline(
            "PathLight trace", g_traceSource, "trace", g_tracePipeline, g_traceLayout) ||
        !build_compute_pipeline("PathLight temporal", g_temporalSource, "temporal",
            g_temporalPipeline, g_temporalLayout) ||
        !build_compute_pipeline(
            "PathLight denoise", g_denoiseSource, "denoise", g_denoisePipeline, g_denoiseLayout))
    {
        return mods::set_error(error, MOD_ERROR, "failed to create PathLight compute pipelines");
    }

    GfxComputeTypeDesc computeDesc = GFX_COMPUTE_TYPE_DESC_INIT;
    computeDesc.label = "PathLight GI";
    computeDesc.callback = on_compute;
    if (svc_gfx->register_compute_type(mod_ctx, &computeDesc, &g_computeType) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register compute type");
    }
    GfxDrawTypeDesc drawDesc = GFX_DRAW_TYPE_DESC_INIT;
    drawDesc.label = "PathLight composite";
    drawDesc.draw = on_draw;
    if (svc_gfx->register_draw_type(mod_ctx, &drawDesc, &g_drawType) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register draw type");
    }
    GfxStageHookDesc stageDesc = GFX_STAGE_HOOK_DESC_INIT;
    stageDesc.callback = on_scene_after_opaque;
    if (svc_gfx->register_stage_hook(
            mod_ctx, GFX_STAGE_SCENE_AFTER_OPAQUE, &stageDesc, &g_afterOpaqueHook) != MOD_OK)
    {
        return mods::set_error(error, MOD_ERROR, "failed to register stage hook");
    }

    UiModsPanelDesc panelDesc = UI_MODS_PANEL_DESC_INIT;
    panelDesc.build = build_panel;
    svc_ui->register_mods_panel(mod_ctx, &panelDesc);

    svc_log->info(mod_ctx, "PathLight GI ready");
    return MOD_OK;
}

MOD_EXPORT ModResult mod_update(ModError*) {
    if (!g_loggedChain && g_chainExecuted.load(std::memory_order_acquire)) {
        g_loggedChain = true;
        svc_log->info(mod_ctx, "PathLight GI chain executed OK");
    }
    return MOD_OK;
}

MOD_EXPORT ModResult mod_shutdown(ModError*) {
    svc_resource->free(mod_ctx, &g_commonSource);
    svc_resource->free(mod_ctx, &g_traceSource);
    svc_resource->free(mod_ctx, &g_temporalSource);
    svc_resource->free(mod_ctx, &g_denoiseSource);
    svc_resource->free(mod_ctx, &g_compositeSource);

    destroy_targets(g_targets);
    g_targets = nullptr;
    for (auto& retired : g_retired) {
        destroy_targets(retired.targets);
    }
    g_retired.clear();

    const auto releasePipeline = [](WGPUComputePipeline& pipeline) {
        if (pipeline != nullptr) {
            wgpuComputePipelineRelease(pipeline);
            pipeline = nullptr;
        }
    };
    const auto releaseLayout = [](WGPUBindGroupLayout& layout) {
        if (layout != nullptr) {
            wgpuBindGroupLayoutRelease(layout);
            layout = nullptr;
        }
    };
    releasePipeline(g_tracePipeline);
    releasePipeline(g_temporalPipeline);
    releasePipeline(g_denoisePipeline);
    releaseLayout(g_traceLayout);
    releaseLayout(g_temporalLayout);
    releaseLayout(g_denoiseLayout);
    release_composite();

    g_cvarEnabled = g_cvarQuality = g_cvarHalfRes = g_cvarIntensity = g_cvarBounces = 0;
    g_cvarDistance = g_cvarAo = g_cvarBleed = g_cvarSky = g_cvarEmissive = 0;
    g_cvarDenoise = g_cvarHistory = g_cvarDebug = 0;
    g_cvarDarkMode = g_cvarDarkness = g_cvarLightGain = g_cvarLightRadius = 0;
    g_cvarShadows = g_cvarFlicker = g_cvarSparks = 0;
    g_cvarKeepFog = g_cvarFogStrength = 0;
    g_cvarSunShadows = g_cvarSunStrength = g_cvarSunLength = 0;
    g_cvarSunRays = g_cvarSunWarmth = g_cvarSunBloom = g_cvarSunTod = 0;
    g_darkInit = false;
    g_darkAmount = 0.0f;
    g_computeType = g_drawType = 0;
    g_afterOpaqueHook = 0;
    g_controlsWindow = 0;
    g_havePrevCamera = false;
    g_pingPong = 0;
    g_frameIndex = 0;
    return MOD_OK;
}
}

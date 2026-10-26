// PathLight GI - game-side light gathering (see lights.h).

// IMPORTANT: standard headers must come BEFORE any game header. On MSVC the game's types.h defines
// `nullptr` as `NULL` (MSVC reports __cplusplus < 201103 by default), which breaks every standard
// header that is included afterwards.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "global.h"

#include "d/actor/d_a_alink.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo.h"

// Restore the real keyword for the rest of this file.
#ifdef nullptr
#undef nullptr
#endif

#include "lights.h"

namespace {

float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

struct Spark {
    float pos[3] = {};
    float energy = 0.0f;
};
Spark g_spark;
std::chrono::steady_clock::time_point g_lastTime;
bool g_haveTime = false;

float frame_dt() {
    const auto now = std::chrono::steady_clock::now();
    float dt = 1.0f / 60.0f;
    if (g_haveTime) {
        dt = std::chrono::duration<float>(now - g_lastTime).count();
    }
    g_lastTime = now;
    g_haveTime = true;
    return std::clamp(dt, 0.0f, 0.25f);
}

int classify_stage(const char* name) {
    if (name == nullptr || name[0] == '\0' || name[1] != '_') {
        return STAGE_OUTDOORS;
    }
    switch (name[0]) {
    case 'D': return STAGE_DUNGEON;
    case 'R': return STAGE_INTERIOR;
    default: return STAGE_OUTDOORS;
    }
}

// 0 = full day, 1 = night. Smooth ramps around dusk (18-21) and dawn (4-7).
float night_factor() {
    const float hour = static_cast<float>(dKy_getdaytime_hour()) +
                       static_cast<float>(dKy_getdaytime_minute()) / 60.0f;
    const auto smooth = [](float a, float b, float x) {
        const float t = clamp01((x - a) / (b - a));
        return t * t * (3.0f - 2.0f * t);
    };
    if (hour >= 12.0f) {
        return smooth(18.0f, 21.0f, hour);
    }
    return 1.0f - smooth(4.0f, 7.0f, hour);
}

}  // namespace

void collect_scene_lights(const float eye[3], LightSnapshot& out) {
    out.count = 0;
    out.stageClass = classify_stage(dComIfGp_getStartStageName());
    out.night = night_factor();
    out.twilight = dKy_darkworld_check() != 0;
    const float dt = frame_dt();

    struct Candidate {
        SceneLight light;
        float score;
    };
    Candidate candidates[128];
    int candidateCount = 0;
    const auto distance_to_eye = [&](const float p[3]) {
        const float dx = p[0] - eye[0], dy = p[1] - eye[1], dz = p[2] - eye[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    // 1) Lights registered by the game: torches, candles, bombs, fire arrows, magnet boots, ...
    dScnKy_env_light_c* env = dKy_getEnvlight();
    if (env != nullptr) {
        for (int i = 0; i < 100 && candidateCount < 120; ++i) {
            const LIGHT_INFLUENCE* li = env->pointlight[i];
            if (li == nullptr || !(li->mPow > 1.0f) || li->mPow > 20000.0f) {
                continue;  // unused slot, or the huge "global ambient" pseudo light
            }
            const float r = std::max<float>(li->mColor.r, 0) / 255.0f;
            const float g = std::max<float>(li->mColor.g, 0) / 255.0f;
            const float b = std::max<float>(li->mColor.b, 0) / 255.0f;
            if (std::max({r, g, b}) < 0.01f) {
                continue;
            }
            Candidate& c = candidates[candidateCount++];
            c.light.pos[0] = li->mPosition.x;
            c.light.pos[1] = li->mPosition.y;
            c.light.pos[2] = li->mPosition.z;
            c.light.color[0] = r;
            c.light.color[1] = g;
            c.light.color[2] = b;
            c.light.radius = li->mPow;
            c.light.flicker = clamp01(li->mFluctuation / 255.0f);
            c.light.kind = LIGHT_KIND_GAME;
            c.score = distance_to_eye(c.light.pos) - li->mPow;
        }
    }

    // 2) Link: lantern / glowing sword / wolf glow, and sword-hit sparks.
    daAlink_c* link = static_cast<daAlink_c*>(dComIfGp_getLinkPlayer());
    out.linkAvailable = link != nullptr && env != nullptr;
    if (out.linkAvailable) {
        const float maxPower = std::max(daAlinkHIO_huLight_c0::m.mPower, 0.01f);
        const float lit = clamp01(link->field_0x33fc / maxPower);
        if (lit > 0.02f) {
            const BOSS_LIGHT& bl = env->field_0x0c18[0];
            Candidate& c = candidates[candidateCount++];
            c.light.pos[0] = bl.mPos.x;
            c.light.pos[1] = bl.mPos.y;
            c.light.pos[2] = bl.mPos.z;
            c.light.color[0] = bl.mColor.r / 255.0f * lit;
            c.light.color[1] = bl.mColor.g / 255.0f * lit;
            c.light.color[2] = bl.mColor.b / 255.0f * lit;
            c.light.radius = 700.0f;
            c.light.flicker = 0.35f;
            c.light.kind = LIGHT_KIND_LANTERN;
            c.score = -1e9f;  // never dropped
        }

        // Sword hit: the attack capsules report a hit position for one game tick.
        for (int i = 0; i < 3; ++i) {
            if (link->mAtCps[i].ChkAtHit()) {
                const cXyz* hit = link->mAtCps[i].GetAtHitPosP();
                if (hit != nullptr) {
                    g_spark.pos[0] = hit->x;
                    g_spark.pos[1] = hit->y;
                    g_spark.pos[2] = hit->z;
                    g_spark.energy = 1.0f;
                }
            }
        }
    }
    if (g_spark.energy > 0.0f) {
        Candidate& c = candidates[candidateCount++];
        std::memcpy(c.light.pos, g_spark.pos, sizeof(g_spark.pos));
        const float e = g_spark.energy * g_spark.energy;  // fast, snappy decay
        c.light.color[0] = 2.4f * e;
        c.light.color[1] = 1.7f * e;
        c.light.color[2] = 0.9f * e;
        c.light.radius = 320.0f + 180.0f * g_spark.energy;
        c.light.flicker = 0.0f;
        c.light.kind = LIGHT_KIND_SPARK;
        c.score = -1e9f;
        g_spark.energy = std::max(0.0f, g_spark.energy - dt / 0.22f);
    }

    // Keep the closest/most relevant lights.
    if (candidateCount > kMaxSceneLights) {
        std::nth_element(candidates, candidates + kMaxSceneLights, candidates + candidateCount,
            [](const Candidate& a, const Candidate& b) { return a.score < b.score; });
        candidateCount = kMaxSceneLights;
    }
    for (int i = 0; i < candidateCount; ++i) {
        out.lights[i] = candidates[i].light;
    }
    out.count = candidateCount;
}

void collect_fog(const void* gameViewport, FogSnapshot& out) {
    out = FogSnapshot{};
    dScnKy_env_light_c* env = dKy_getEnvlight();
    if (env == nullptr) {
        return;
    }
    const float start = env->mFogNear;
    const float end = env->mFogFar;
    if (!(end > start) || start < 0.0f) {
        return;
    }
    out.start = start;
    out.end = end;
    out.color[0] = clamp01(static_cast<float>(env->fog_col.r) / 255.0f);
    out.color[1] = clamp01(static_cast<float>(env->fog_col.g) / 255.0f);
    out.color[2] = clamp01(static_cast<float>(env->fog_col.b) / 255.0f);

    // Range adjust: Aurora scales fog by a per-screen-column factor built from this table.
    if (env->mFogAdjEnable) {
        float cx = 0.0f;
        if (gameViewport != nullptr) {
            const view_port_class* vp = static_cast<const view_port_class*>(gameViewport);
            if (vp->width > 1.0f) {
                cx = (static_cast<float>(env->mFogAdjCenter) - vp->x_orig) / vp->width * 2.0f - 1.0f;
            }
        }
        out.centerNdc = cx;
        for (int i = 0; i < 10; ++i) {
            out.k[i] = static_cast<float>(env->mXFogTbl.r[i ^ 1]) / 64.0f;
        }
    } else {
        for (float& k : out.k) {
            k = 1.0e6f;  // factor -> 1: no range adjustment
        }
    }
    out.valid = true;
}

void collect_sun(const float eye[3], SunSnapshot& out) {
    out = SunSnapshot{};
    dScnKy_env_light_c* env = dKy_getEnvlight();
    if (env == nullptr || classify_stage(dComIfGp_getStartStageName()) != STAGE_OUTDOORS) {
        return;
    }
    const auto smooth = [](float a, float b, float x) {
        const float t = clamp01((x - a) / (b - a));
        return t * t * (3.0f - 2.0f * t);
    };
    const auto unit = [](float x, float y, float z, float v[3]) {
        const float len = std::sqrt(x * x + y * y + z * z);
        if (!(len > 1.0f)) {
            return false;
        }
        v[0] = x / len;
        v[1] = y / len;
        v[2] = z / len;
        return true;
    };
    // The game places the sun at camera + offset and the moon as a pure offset.
    float sun[3] = {}, moon[3] = {};
    const bool haveSun = unit(env->sun_pos.x - eye[0], env->sun_pos.y - eye[1],
        env->sun_pos.z - eye[2], sun);
    const bool haveMoon = unit(env->moon_pos.x, env->moon_pos.y, env->moon_pos.z, moon);
    const float sunK = haveSun ? smooth(0.03f, 0.28f, sun[1]) : 0.0f;
    const float moonK = haveMoon ? 0.4f * smooth(0.05f, 0.3f, moon[1]) * (1.0f - sunK) : 0.0f;
    const float* src = sunK >= moonK ? sun : moon;
    out.strength = std::max(sunK, moonK);
    if (out.strength < 0.01f) {
        return;
    }
    std::memcpy(out.dir, src, sizeof(out.dir));
    out.isMoon = src == moon;
    out.valid = true;
    const daAlink_c* link = static_cast<daAlink_c*>(dComIfGp_getLinkPlayer());
    if (link != nullptr) {
        out.linkValid = true;
        out.linkPos[0] = link->current.pos.x;
        out.linkPos[1] = link->current.pos.y;
        out.linkPos[2] = link->current.pos.z;
    }
}

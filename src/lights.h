// PathLight GI - game-side light gathering.
//
// Reads the lights the game itself keeps for the current frame (torches, candles, bombs, fire
// arrows, ... via g_env_light.pointlight[]), Link's lantern / glowing sword, short sword-hit
// sparks, and classifies the current stage (dungeon / cave / interior / outdoors). Everything is
// returned in plain structs so the renderer side needs no game headers.
#pragma once

#include <cstdint>

constexpr int kMaxSceneLights = 32;

enum SceneLightKind : int {
    LIGHT_KIND_GAME = 0,    // torches, candles, bombs, fire arrows, ...
    LIGHT_KIND_LANTERN = 1, // Link's lantern / glowing sword / wolf glow
    LIGHT_KIND_SPARK = 2,   // short flash where the sword hit something
};

struct SceneLight {
    float pos[3];    // world space
    float color[3];  // 0..1 (may exceed 1 for very hot lights)
    float radius;    // world units; light reaches zero at this distance
    float flicker;   // 0..1 flame flicker amount
    int kind;
};

enum StageClass : int {
    STAGE_OUTDOORS = 0,  // fields, towns, title screen, unknown
    STAGE_DUNGEON = 1,   // D_* : dungeons, bosses, caves and grottos
    STAGE_INTERIOR = 2,  // R_* : houses, shops, rooms
};

struct LightSnapshot {
    int count = 0;
    SceneLight lights[kMaxSceneLights];
    int stageClass = STAGE_OUTDOORS;
    float night = 0.0f;       // 0 = day, 1 = deep night (from the in-game clock)
    bool twilight = false;    // twilight realm
    bool linkAvailable = false;
};

// Must be called on the game thread. `eye` (world space) is used to keep the closest lights when
// more than kMaxSceneLights are active.
void collect_scene_lights(const float eye[3], LightSnapshot& out);

// The game's current scene fog (what the renderer bakes into every fogged draw).
struct FogSnapshot {
    bool valid = false;
    float start = 0.0f;       // view-space distance where fog begins
    float end = 1.0f;         // ... and where it is complete
    float color[3] = {};      // 0..1
    float centerNdc = 0.0f;   // range-adjust centre in NDC x (-1..1)
    float k[10] = {};         // range-adjust table, already divided by 64 and pair-swapped
};

// Must be called on the game thread. `gameViewport` is GfxStageContext::game_viewport (may be null).
void collect_fog(const void* gameViewport, FogSnapshot& out);

// Sun (day) or moon (night) as a shadow-casting directional light, from the game's clock.
struct SunSnapshot {
    bool valid = false;       // false indoors / in caves / when the sun and moon are both down
    float dir[3] = {};        // world-space unit vector pointing TOWARD the light
    float strength = 0.0f;    // 0..1, fades near the horizon and for the weaker moon
    bool isMoon = false;      // the light is the moon, not the sun
    bool linkValid = false;   // Link (or the wolf) is present: exclude him, the game draws his shadow
    float linkPos[3] = {};    // world-space position of his feet
};

// Must be called on the game thread. `eye` is the world-space camera position.
void collect_sun(const float eye[3], SunSnapshot& out);

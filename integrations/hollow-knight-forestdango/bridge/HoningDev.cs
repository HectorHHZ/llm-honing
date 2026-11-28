// Added by the Honing devmode study (2026-11-25). NOT part of the author's project.
//
// The knobs of the devmode bridge (experiments/hk-forestdango-demo/20261125_1515_devmode_speedup/README.md has the
// measurements). Every knob is OFF unless it is set, and none does anything unless HONING_SESSION_DIR is set (the
// bridge itself is inert without it), so a game started without them plays as with bridge e.
//
// A. The deterministic game: its results do not depend on how fast the machine runs or on whether frames are drawn
//   HONING_FAST=1               all of the knobs of this block at once (this is "the deterministic game")
//   HONING_CLOCK=virtual        the game's scripts see a wall clock that advances 1/HONING_FRAME_HZ s per frame
//                               (HoningClock.cs): hit-stop, room-change waits, menu fades and the state machines'
//                               real-time waits take the same number of frames at any speed
//   HONING_LOADS=sync           scene loads are synchronous: always exactly one frame (overlay of SceneLoad.cs,
//                               SceneAdditiveLoadConditional.cs, ScenePreloader.cs, StartManager.cs)
//   HONING_BOOT=direct          the new-game opening films are not played (overlay of OpeningSequence.cs); the start
//                               sequence then always takes the same number of frames
//   HONING_AUDIO=virtual        a state machine that waits for a sound to end waits for the clip's length on the virtual
//                               wall clock instead of asking the sound system (overlay of the four AudioPlay actions)
//   HONING_RENDER_RNG=isolate   the random numbers that drawing a frame takes (the camera's film grain takes two per
//                               tile, a number that depends on the window size) are given back
//   HONING_VISIBILITY=geometry  "is this on screen" (camera shake on an enemy's death) from the camera's rectangle,
//                               not from the renderer (overlay of EnemyDeathEffects.cs)
//   HONING_IDS=stable           ids of objects made at run time by order of first sight, lists in the engine's object
//                               order (HoningVisible.cs): the engine's numbers differ between drawn and undrawn games
//   HONING_STEP_AT=loop         the bridge's own start sequence runs from a hook at the end of the frame loop
//   HONING_CAMERA_LOCK_FIX=2    the camera lock fix only for scenes without a tile map (see HoningBridge.cs)
//   and, without a window (-batchmode -nographics): HONING_ASPECT=16:9 (default there) gives every camera the aspect
//   of the game's window, because the engine's screen is 640 x 480 then.
// Speed, with the deterministic game: start the player with -batchmode -nographics (nothing is drawn, no display is
// needed). Other knobs that only concern speed or measuring:
//   HONING_POLL=fast            the bridge looks for the next command without sleeping at first
//   HONING_LOG_BATCH=1          observations.jsonl / visible_observations.jsonl are written in batches (not for tools
//                               that follow these files while the game runs)
//   "observe" in a command      "full" (default) | "debug" | "none": what is observed after that window
//   HONING_VISIBLE=0            no visible observation at all (timing studies)
//   HONING_TARGET_FPS=30        a cap on frames per wall second (30 = the game at its real speed; default none)
//   HONING_TIMING=1|2, HONING_FRAME_LOG=1|2, HONING_FRAME_LOG_BOOT=1, HONING_INTROSPECT=1   probes (HoningProbe.cs)
//   HONING_RENDER=skip          kept as a record of what does NOT work (see HoningBridge.cs)
// B. Developer controls (they change the game on purpose; each is logged in <session>/dev_events.jsonl when it acts,
//    and the debug row gets a "dev" object while one is in use)
//   HONING_DEV_REVIVE=off|safe|spot|marker   what a death does: off = the game's death (default); safe = the knight
//                               is put back on the last ground it safely stood on (at least 1 s before the death), full
//                               health; spot = stays where it died; marker = the game's own hazard-respawn point. No
//                               death animation, no reload, no loss of geo; the death is counted and logged.
//                               (HeroController.Die asks HoningDev.InterceptDeath first: overlay of HeroController.cs)
//   HONING_DEV_DAMAGE=on|nohealthloss|ghost  nohealthloss = hits knock back but take no health (the game's own
//                               takeNoDamage flag); ghost = nothing can hit the knight (damage mode NO_DAMAGE)
//   HONING_DEV_DEATH_WAIT=<s>   length of the pause between the death and the fade-out (game: 2.85 s)
//   HONING_DEV_SKIP_FADE=1      the 0.5 s black-screen wait of a room change is cut (needs the virtual clock)
//   commands at any time, {"id":<observation id>,"action":"dev","op":...}; the answer is a new observation of the same
//   game state with the result in the debug row under dev.result:
//     state | heal | set_health value | set_soul value | revive mode | damage mode | death_wait seconds |
//     skip_fade flag | teleport x y [place] | load scene gate | set name flag/value (dash, wall_jump, double_jump,
//     super_dash, spell, nail, geo, max_health) | cheat_p (what the author's P key does) | lookahead plan ...
// C. Lookahead (op "lookahead"): HoningLookahead.cs.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using GlobalEnums;
using UnityEngine;

public static class HoningDev
{
    // ---- knobs read by the overlay files (static: they may be asked before the bridge exists)
    public static readonly bool Active;
    public static readonly bool SyncLoads, DirectBoot, GeometricVisibility, IsolateRenderRandom, VirtualAudio;
    public static bool RenderSkip;                  // set by the bridge (HONING_RENDER=skip and no pictures)
    public static Func<bool> InterceptDeath;        // null unless a revive mode is on (HeroController.Die asks it)

    private static HoningBridge bridge;
    private static string session;
    private static readonly WaitForEndOfFrame endOfFrame = new WaitForEndOfFrame();
    private static bool renderRequested;
    public static int EndOfFrameRequests, RenderedOnRequest;

    // ---- developer controls
    private static string reviveMode = "off", damage = "on";
    private static float deathWait = -1f;
    private static bool skipFade;
    private static int deathsIntercepted, revives, commands, teleports, loads;
    private static string lastEvent = "";
    private static FieldInfo deathWaitField;
    private static float deathWaitOriginal = -1f;
    private static MethodInfo invulnerableMethod;

    private struct Safe
    {
        public int frame;
        public string scene;
        public Vector3 position;
        public bool facingRight;
    }

    private static readonly List<Safe> safe = new List<Safe>();
    private static int frameCounter;

    [Serializable]
    private class Window
    {
        public int frames;
        public string action;
        public string[] hold;
        public string[] press;
        public string[] first;
    }

    [Serializable]
    private class Cmd
    {
        public int id;
        public string op;
        public string scene, gate, name, mode, text, path;
        public float x, y, seconds;
        public int value, frames;
        public bool flag, place;
        public Window[] plan;
    }

    private static string Env(string name, string fallback)
    {
        string v = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrEmpty(v) ? fallback : v;
    }

    static HoningDev()
    {
        Active = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("HONING_SESSION_DIR"));
        bool fast = Active && Env("HONING_FAST", "0") == "1";
        SyncLoads = Active && (fast || Env("HONING_LOADS", "async") == "sync");
        DirectBoot = Active && (fast || Env("HONING_BOOT", "opening") == "direct");
        GeometricVisibility = Active && (fast || Env("HONING_VISIBILITY", "renderer") == "geometry");
        IsolateRenderRandom = Active && (fast || Env("HONING_RENDER_RNG", "shared") == "isolate");
        VirtualAudio = Active && (fast || Env("HONING_AUDIO", "real") == "virtual");
    }

    public static bool Fast
    {
        get { return Active && Env("HONING_FAST", "0") == "1"; }
    }

    public static void Start(HoningBridge owner, string sessionDir)
    {
        bridge = owner;
        session = sessionDir;
        reviveMode = Env("HONING_DEV_REVIVE", "off");
        damage = Env("HONING_DEV_DAMAGE", "on");
        deathWait = float.Parse(Env("HONING_DEV_DEATH_WAIT", "-1"), CultureInfo.InvariantCulture);
        skipFade = Env("HONING_DEV_SKIP_FADE", "0") == "1";
        ApplyReviveMode();
    }

    private static void ApplyReviveMode()
    {
        if (reviveMode != "off")
        {
            InterceptDeath = OnDeath;
        }
        else
        {
            InterceptDeath = null;
        }
    }

    public static void SetSession(string sessionDir)
    {
        session = sessionDir;
    }

    public static string Describe()
    {
        return "loads=" + (SyncLoads ? "sync" : "async") + " boot=" + (DirectBoot ? "direct" : "opening") + " visibility=" + (GeometricVisibility ? "geometry" : "renderer")
               + " render_rng=" + (IsolateRenderRandom ? "isolate" : "shared") + " audio=" + (VirtualAudio ? "virtual" : "real")
               + " | dev: revive=" + reviveMode + " damage=" + damage + " death_wait=" + (deathWait < 0f ? "game" : deathWait.ToString(CultureInfo.InvariantCulture))
               + " skip_fade=" + (skipFade ? 1 : 0);
    }

    public static string Summary()
    {
        return "eof_requests=" + EndOfFrameRequests + " audio_waits_virtual=" + AudioWaitsVirtual + " dev_deaths=" + deathsIntercepted + " revives=" + revives
               + " dev_commands=" + commands;
    }

    private static void Event(string text)
    {
        lastEvent = text;
        if (HoningLookahead.InChild)
        {
            return;     // a lookahead copy keeps the original's event log and the engine log untouched
        }
        try
        {
            File.AppendAllText(Path.Combine(session, "dev_events.jsonl"),
                "{\"frame\":" + frameCounter + ",\"engine_frame\":" + HoningClock.EngineFrameCount + ",\"event\":" + Quote(text) + "}\n");
        }
        catch (Exception)
        {
        }
        Debug.Log("[HoningDev] " + text);
    }

    private static string Quote(string s)
    {
        var sb = new StringBuilder();
        HoningVisible.Str(sb, s);
        return sb.ToString();
    }

    private static string Num(double v)
    {
        return v.ToString("0.####", CultureInfo.InvariantCulture);
    }

    // ------------------------------------------------------------------------------------------ end of frame / drawing
    // Used by the game's scripts in place of "new WaitForEndOfFrame()" (overlay). The value is the engine's own
    // WaitForEndOfFrame; with HONING_RENDER=skip the frame in which it is asked for is drawn, because the engine only
    // ends such a wait on a frame that it draws.
    public static object EndOfFrame()
    {
        EndOfFrameRequests++;
        if (RenderSkip)
        {
            renderRequested = true;
            UnityEngine.Rendering.OnDemandRendering.renderFrameInterval = 1;
        }
        return endOfFrame;
    }

    // called by the bridge at the very end of every frame
    public static void EndOfLoop()
    {
        if (RenderSkip)
        {
            if (renderRequested)
            {
                RenderedOnRequest++;
                renderRequested = false;
            }
            UnityEngine.Rendering.OnDemandRendering.renderFrameInterval = 1000000;
        }
    }

    // ------------------------------------------------------------------------------------------ sounds that are waited for
    private static readonly Dictionary<int, double> audioEnd = new Dictionary<int, double>();
    public static int AudioWaitsVirtual;

    private static void NoteSound(AudioSource a, AudioClip clip, bool loop)
    {
        double length = clip != null ? clip.length / Math.Max(0.01, Math.Abs(a.pitch)) : 0.0;
        double end = loop ? double.PositiveInfinity : HoningClock.Now + length;
        double old;
        int id = a.GetInstanceID();
        if (audioEnd.TryGetValue(id, out old) && old > end && HoningClock.Now < old)
        {
            end = old;      // an earlier sound of this source is still running
        }
        audioEnd[id] = end;
        if (audioEnd.Count > 4096)
        {
            audioEnd.Clear();   // sources come and go; forgetting only makes IsPlaying ask the engine once more
        }
    }

    public static void Play(AudioSource a)
    {
        a.Play();
        if (VirtualAudio)
        {
            NoteSound(a, a.clip, a.loop);
        }
    }

    public static void PlayOneShot(AudioSource a, AudioClip clip)
    {
        a.PlayOneShot(clip);
        if (VirtualAudio)
        {
            NoteSound(a, clip, false);
        }
    }

    public static void PlayOneShot(AudioSource a, AudioClip clip, float volume)
    {
        a.PlayOneShot(clip, volume);
        if (VirtualAudio)
        {
            NoteSound(a, clip, false);
        }
    }

    public static void Stop(AudioSource a)
    {
        a.Stop();
        if (VirtualAudio)
        {
            audioEnd.Remove(a.GetInstanceID());
        }
    }

    public static bool IsPlaying(AudioSource a)
    {
        if (!VirtualAudio)
        {
            return a.isPlaying;
        }
        double end;
        if (audioEnd.TryGetValue(a.GetInstanceID(), out end))
        {
            AudioWaitsVirtual++;
            return HoningClock.Now < end;
        }
        // not started through the calls above: take the engine's word once and count the clip's length from now
        if (!a.isPlaying)
        {
            return false;
        }
        NoteSound(a, a.clip, a.loop);
        return true;
    }

    // "is this renderer on screen", from the cameras' rectangles (HONING_VISIBILITY=geometry)
    public static bool OnScreen(Renderer r)
    {
        if (r == null || !r.enabled || !r.gameObject.activeInHierarchy)
        {
            return false;
        }
        Bounds b = r.bounds;
        int layer = 1 << r.gameObject.layer;
        foreach (Camera cam in Camera.allCameras)
        {
            if ((cam.cullingMask & layer) == 0)
            {
                continue;
            }
            if (GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(cam), b))
            {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------------------------------ per frame
    // after the physics steps of every played frame
    public static void AfterFrame()
    {
        frameCounter++;
        HeroController hero = HeroController.SilentInstance;
        GameManager gm = GameManager.UnsafeInstance;
        if (hero == null || gm == null)
        {
            return;
        }
        if (damage == "nohealthloss")
        {
            hero.takeNoDamage = true;
        }
        else if (damage == "ghost")
        {
            hero.damageMode = DamageMode.NO_DAMAGE;
        }
        if (deathWait >= 0f)
        {
            SetDeathWait(hero, deathWait);
        }
        if (skipFade && HoningClock.Virtual)
        {
            // the wait of GameManager.BeginSceneTransitionRoutine counts wall time: let it see 10 s per frame
            HoningClock.StepOverride = gm.IsInSceneTransition && gm.gameState == GameState.EXITING_LEVEL ? 10.0 : 0.0;
        }
        if (reviveMode == "safe")
        {
            HeroControllerStates c = hero.cState;
            if (c != null && c.onGround && hero.acceptingInput && !c.recoiling && !c.dead && !c.hazardDeath && !c.hazardRespawning && !c.transitioning
                && !c.invulnerable && gm.gameState == GameState.PLAYING)
            {
                if (safe.Count >= 900)
                {
                    safe.RemoveRange(0, 300);
                }
                safe.Add(new Safe { frame = frameCounter, scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
                                    position = hero.transform.position, facingRight = c.facingRight });
            }
        }
    }

    private static void SetDeathWait(HeroController hero, float seconds)
    {
        if (deathWaitField == null)
        {
            deathWaitField = typeof(HeroController).GetField("DEATH_WAIT", BindingFlags.NonPublic | BindingFlags.Instance);
        }
        if (deathWaitField == null)
        {
            return;
        }
        float now = (float)deathWaitField.GetValue(hero);
        if (deathWaitOriginal < 0f)
        {
            deathWaitOriginal = now;
        }
        if (now != seconds)
        {
            deathWaitField.SetValue(hero, seconds);
            Event("death wait " + Num(now) + " s -> " + Num(seconds) + " s");
        }
    }

    // ------------------------------------------------------------------------------------------ revive
    private static void Place(HeroController hero, Vector3 where, bool snapCamera)
    {
        Vector3 p = hero.transform.position;
        hero.transform.position = new Vector3(where.x, where.y, p.z);
        Rigidbody2D body = hero.GetComponent<Rigidbody2D>();
        if (body != null)
        {
            body.position = new Vector2(where.x, where.y);
            body.velocity = Vector2.zero;
        }
        GameManager gm = GameManager.UnsafeInstance;
        if (snapCamera && gm != null && gm.cameraCtrl != null)
        {
            gm.cameraCtrl.PositionToHero(false);
        }
    }

    private static void StartInvulnerability(HeroController hero, float seconds)
    {
        if (invulnerableMethod == null)
        {
            invulnerableMethod = typeof(HeroController).GetMethod("Invulnerable", BindingFlags.NonPublic | BindingFlags.Instance);
        }
        if (invulnerableMethod != null)
        {
            hero.StartCoroutine((IEnumerator)invulnerableMethod.Invoke(hero, new object[] { seconds }));
        }
    }

    // Asked by HeroController.Die before anything of the death has happened (health is 0). true = the death is taken
    // over: the knight lives on with full health.
    private static bool OnDeath()
    {
        HeroController hero = HeroController.SilentInstance;
        GameManager gm = GameManager.UnsafeInstance;
        if (hero == null || gm == null || reviveMode == "off")
        {
            return false;
        }
        deathsIntercepted++;
        Vector3 at = hero.transform.position;
        string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        Vector3 target = at;
        string how = "stays where it died";
        if (reviveMode == "safe")
        {
            int found = -1;
            for (int i = safe.Count - 1; i >= 0; i--)
            {
                if (safe[i].scene == scene && safe[i].frame <= frameCounter - 30)
                {
                    found = i;
                    break;
                }
            }
            if (found < 0)
            {
                for (int i = safe.Count - 1; i >= 0; i--)
                {
                    if (safe[i].scene == scene)
                    {
                        found = i;
                        break;
                    }
                }
            }
            if (found >= 0)
            {
                target = safe[found].position;
                how = "put back on the ground it stood on " + (frameCounter - safe[found].frame) + " frames earlier";
                if (safe[found].facingRight)
                {
                    hero.FaceRight();
                }
                else
                {
                    hero.FaceLeft();
                }
                safe.RemoveRange(found + 1, safe.Count - found - 1);
            }
            else
            {
                target = hero.FindGroundPoint(gm.playerData.hazardRespawnLocation, true);
                how = "no safe ground recorded in this room: put at the game's hazard-respawn point";
            }
        }
        else if (reviveMode == "marker")
        {
            target = hero.FindGroundPoint(gm.playerData.hazardRespawnLocation, true);
            how = "put at the game's hazard-respawn point";
        }
        hero.MaxHealth();
        if (reviveMode != "spot")
        {
            Place(hero, target, true);
        }
        StartInvulnerability(hero, hero.INVUL_TIME);
        revives++;
        Event("DEATH " + deathsIntercepted + " in " + scene + " at (" + Num(at.x) + ", " + Num(at.y) + "): revived in place (" + reviveMode + "), " + how
              + ", now at (" + Num(target.x) + ", " + Num(target.y) + "), health " + gm.playerData.health + "/" + gm.playerData.maxHealth);
        return true;
    }

    // ------------------------------------------------------------------------------------------ commands
    private static string Ok(string what)
    {
        return "{\"ok\":true,\"did\":" + Quote(what) + "}";
    }

    private static string Fail(string why)
    {
        return "{\"ok\":false,\"error\":" + Quote(why) + "}";
    }

    public static string Command(string json)
    {
        commands++;
        Cmd c;
        try
        {
            c = JsonUtility.FromJson<Cmd>(json);
        }
        catch (Exception e)
        {
            return Fail("unreadable command: " + e.Message);
        }
        HeroController hero = HeroController.SilentInstance;
        GameManager gm = GameManager.UnsafeInstance;
        if (c == null || string.IsNullOrEmpty(c.op))
        {
            return Fail("no op");
        }
        if (hero == null || gm == null || gm.playerData == null)
        {
            return Fail("no game in progress");
        }
        PlayerData pd = gm.playerData;
        string result;
        switch (c.op)
        {
            case "state":
                result = Ok("state");
                break;
            case "heal":
                hero.MaxHealth();
                result = Ok("health " + pd.health + "/" + pd.maxHealth);
                break;
            case "set_health":
                if (c.value < 1 || c.value > pd.maxHealth)
                {
                    return Fail("health must be within 1.." + pd.maxHealth);
                }
                if (c.value > pd.health)
                {
                    hero.AddHealth(c.value - pd.health);
                }
                else if (c.value < pd.health)
                {
                    hero.TakeHealth(pd.health - c.value);
                    hero.proxyFSM.SendEvent("HeroCtrl-HeroDamaged");
                }
                result = Ok("health " + pd.health + "/" + pd.maxHealth);
                break;
            case "set_soul":
                hero.SetMPCharge(Mathf.Clamp(c.value, 0, pd.maxMP));
                result = Ok("soul " + pd.MPCharge + "/" + pd.maxMP);
                break;
            case "revive":
                if (c.mode != "off" && c.mode != "safe" && c.mode != "spot" && c.mode != "marker")
                {
                    return Fail("mode must be off, safe, spot or marker");
                }
                reviveMode = c.mode;
                ApplyReviveMode();
                result = Ok("revive mode " + reviveMode);
                break;
            case "damage":
                if (c.mode != "on" && c.mode != "nohealthloss" && c.mode != "ghost")
                {
                    return Fail("mode must be on, nohealthloss or ghost");
                }
                damage = c.mode;
                hero.takeNoDamage = damage == "nohealthloss";
                hero.damageMode = damage == "ghost" ? DamageMode.NO_DAMAGE : DamageMode.FULL_DAMAGE;
                result = Ok("damage " + damage);
                break;
            case "death_wait":
                deathWait = c.seconds;
                if (deathWait < 0f && deathWaitOriginal >= 0f)
                {
                    SetDeathWait(hero, deathWaitOriginal);
                }
                else if (deathWait >= 0f)
                {
                    SetDeathWait(hero, deathWait);
                }
                result = Ok("death wait " + (deathWait < 0f ? "game" : Num(deathWait)));
                break;
            case "skip_fade":
                skipFade = c.flag;
                if (!skipFade)
                {
                    HoningClock.StepOverride = 0.0;
                }
                result = HoningClock.Virtual ? Ok("skip_fade " + skipFade) : Fail("skip_fade needs HONING_CLOCK=virtual");
                break;
            case "teleport":
            {
                Vector3 to = new Vector3(c.x, c.y, 0f);
                if (c.place)
                {
                    to = hero.FindGroundPoint(new Vector2(c.x, c.y), true);
                }
                Place(hero, to, true);
                teleports++;
                result = Ok("knight put at (" + Num(to.x) + ", " + Num(to.y) + ")" + (c.place ? " (on the ground below the given point)" : ""));
                break;
            }
            case "load":
                if (gm.IsInSceneTransition)
                {
                    return Fail("a scene transition is in progress");
                }
                if (gm.gameState != GameState.PLAYING)
                {
                    return Fail("not in gameplay (state " + gm.gameState + ")");
                }
                gm.BeginSceneTransition(new GameManager.SceneLoadInfo
                {
                    SceneName = c.scene,
                    EntryGateName = string.IsNullOrEmpty(c.gate) ? "left1" : c.gate,
                    HeroLeaveDirection = null,
                    EntryDelay = 0f,
                    WaitForSceneTransitionCameraFade = true,
                    PreventCameraFadeOut = false,
                    Visualization = GameManager.SceneLoadVisualizations.Default,
                    AlwaysUnloadUnusedAssets = false
                });
                loads++;
                result = Ok("scene transition to " + c.scene + " gate " + (string.IsNullOrEmpty(c.gate) ? "left1" : c.gate) + " started (the game's own transition)");
                break;
            case "set":
                switch (c.name)
                {
                    case "dash": pd.hasDash = c.flag; pd.canDash = c.flag; break;
                    case "wall_jump": pd.hasWalljump = c.flag; pd.canWallJump = c.flag; break;
                    case "double_jump": pd.hasDoubleJump = c.flag; break;
                    case "super_dash": pd.hasSuperDash = c.flag; break;
                    case "spell": pd.hasSpell = c.flag; pd.fireballLevel = c.flag ? Mathf.Max(1, pd.fireballLevel) : 0; break;
                    case "nail": pd.nailDamage = c.value; PlayMakerFSM.BroadcastEvent("UPDATE NAIL DAMAGE"); break;
                    case "geo": if (c.value > pd.geo) { hero.AddGeo(c.value - pd.geo); } else { pd.geo = c.value; } break;
                    case "max_health": pd.maxHealth = c.value; pd.maxHealthBase = c.value; hero.MaxHealth(); break;
                    default: return Fail("unknown name '" + c.name + "' (dash, wall_jump, double_jump, super_dash, spell, nail, geo, max_health)");
                }
                result = Ok("set " + c.name + " flag=" + c.flag + " value=" + c.value);
                break;
            case "cheat_p":
                // what the author's P key does (CheatManager.Update), value for value
                pd.nailDamage = 100;
                pd.hasDash = true;
                pd.hasSpell = true;
                pd.hasWalljump = true;
                pd.hasDoubleJump = true;
                pd.hasSuperDash = true;
                hero.AddHealth(pd.maxHealth - pd.health);
                hero.AddMPCharge(pd.maxMP - pd.MPCharge);
                pd.fireballLevel = 2;
                pd.quakeLevel = 1;
                pd.AddGeo(500);
                pd.hasCityKey = true;
                pd.hasWhiteKey = true;
                pd.hasLoveKey = true;
                result = Ok("the author's P cheat: nail 100, dash, spell (fireball 2, quake 1), wall jump, double jump, super dash, full health and soul, +500 geo, three keys");
                break;
            case "lookahead":
                result = HoningLookahead.Run(json, bridge, session);
                if (result == null)
                {
                    return null;        // this process is the copy
                }
                break;
            default:
                return Fail("unknown op '" + c.op + "'");
        }
        Event("command " + c.op + ": " + result);
        return result;
    }

    // what the debug observation says about the developer controls (only when one is in use)
    public static void AppendDebug(StringBuilder sb, string result)
    {
        bool inUse = reviveMode != "off" || damage != "on" || deathWait >= 0f || skipFade || commands > 0;
        if (!inUse && result == null)
        {
            return;
        }
        sb.Append(",\"dev\":{\"revive\":\"").Append(reviveMode).Append("\",\"damage\":\"").Append(damage).Append("\",\"deaths\":").Append(deathsIntercepted)
          .Append(",\"revives\":").Append(revives).Append(",\"commands\":").Append(commands).Append(",\"last_event\":").Append(Quote(lastEvent));
        if (result != null)
        {
            sb.Append(",\"result\":").Append(result);
        }
        sb.Append('}');
    }
}

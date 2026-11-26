// Added by the Honing task (2026-11-18, phase 2). NOT part of the author's project.
//
// DEVMODE COPY (2026-11-25, experiments/hk-forestdango-demo/20261125_1515_devmode_speedup) of
// code/forestdango-hk-demo-work/Assets/Scripts/Honing/HoningBridge.cs, version forestdango-bridge-2026-11-19e (kept
// unchanged in code/hk-devmode/bridge_original/). Everything added here is behind a knob whose default is the
// behaviour of bridge e; the knobs are listed in HoningDev.cs. Additions are marked "devmode".
//
// A file bridge between this game and an external harness, with the same protocol as the Hollow Knight bridge of
// the Honing project (HoningBridge.cs / VisibleObservation.cs of code/work/viewport-hollow-knight) and its second
// implementation in mega-dean/hallowdale (honing.ml). It is inert unless the environment variable
// HONING_SESSION_DIR is set when the player starts. Gameplay code is not changed: the bridge only
//   (1) supplies key states through InControl's keyboard provider (the game reads its keys there),
//   (2) fixes the time step (HONING_FRAME_HZ frames per second of game time, default 30) and the random seed,
//   (3) stops the game between action windows by blocking the main thread at the end of a frame,
//   (4) calls the game's own "new game" and "skip cutscene" functions during the start sequence,
//   (5) reads game state to write observations.
//
//   <session>/observation.json        latest human-visible observation, schema honing.visible.v1 (HoningVisible.cs)
//   <session>/debug_observation.json  latest global/debug state (plumbing and evaluation only)
//   <session>/observations.jsonl      every debug observation, one per line
//   <session>/visible_observations.jsonl
//   <session>/action.json             written by the harness: one native action window
//                                     {"id": <id of the latest observation>, "action": "...", "frames": N,
//                                      optional "hold": [...], "press": [...], "first": [...]}
//   <session>/frames/%06d.png         one picture per frame of an action window (HONING_CAPTURE=1)
//   <session>/bridge.log
//
// A window of N frames plays exactly N frames with a fixed step of 1/HONING_FRAME_HZ s; then the next observation
// is written and nothing moves until the next action arrives.
// What one bridge frame is. The game reads its input in Update and moves its bodies in the physics steps
// (FixedUpdate), and Unity runs the physics steps of a frame BEFORE that frame's Update. So a key that goes down in
// a frame moves the knight only in the physics steps of the following frame: one whole frame of lag at 30 Hz, where
// a display at 60 Hz has half of it. The bridge therefore cuts its frames between the physics steps and Update:
//     bridge frame = [ Update: the keys are read ] [ the picture is drawn ] [ the physics steps that use those keys ]
// The observation is written after the last physics steps of a window and before the next Update (a hook at the
// start of the engine's Update phase). The game itself runs exactly as it always does; only the place where the
// bridge stops it is chosen so that the effect of a window's keys is inside that window.
//   hold   keys that are down in every frame of the window
//   press  a fresh press in the first frame (a key that is still down from before is let go for the first frame
//          and pressed in the second), held for one frame. JUMP is held to the end of the window; with
//          HONING_JUMP_MODE=sticky (default) it then stays down until the knight stops rising, so that every jump
//          is a full jump (the Hollow Knight bridge never released the jump button); with "window" it is let go
//          when the window ends, which cuts the jump short.
//   first  direction keys that are down in the first frame INSTEAD of the directions of "hold"
//   Turn before a sideways strike. In this game a strike goes the way the knight faces, and the knight turns in
//   the physics steps after the frame in which the direction key is read. Left/right + NAIL pressed in one frame
//   while facing the other way would therefore strike backwards. When a window asks for a strike with a sideways
//   direction and the knight faces the other way, the bridge gives the direction in the press frame and the NAIL
//   press one frame later (in the next window's first frame if this window has no frame left): "attack_left" and
//   "attack_right" always strike the side they name, one frame later when a turn is needed.
// Named actions: wait left right up down jump jump_left jump_right attack attack_left attack_right attack_up
// attack_down dash dash_left dash_right focus cast cast_up cast_down quick_cast super_dash inventory, and "keys"
// (only hold/press/first), and "stop".
//
// Start (HONING_SCENE):
//   new                     the title menu's own new-game call, the opening sequences skipped the way the game
//                           lets a player skip them, the fall into King's Pass; the first observation is written
//                           when the knight stands and accepts input.
//   jump:<Scene>:<gate>     the same, then a DEBUG START: the game's scene transition to <gate> of <Scene>
//   jump:<Scene>:<gate>@x,y ... and the knight is then put at (x, y) of that room.
//                           Nothing else is set (abilities, health, nail are those of a new game). A text on screen
//                           says so for the first seconds of the recording.
//   boss:<Name>             (2026-11-19, phase 3) a LABELLED BOSS START from the table BossStarts below. Only
//                           "FalseKnight": the new-game sequence above, the game's scene transition to Crossroads_10
//                           gate left1 (the room's only gates lie below the arena), then the knight is put on the
//                           arena floor at its left entrance, (9.6, feet 26.07), facing right, at the left end of the
//                           floor, left of the (open) battle gate and 3.6 units before the battle trigger (the "Battle
//                           Scene" trigger box, x 13.5..14.5), so the fight starts the way it starts for a player who
//                           walks in. The knight is put there only at a moment when no enemy stands within 5 units
//                           (the husks walk about the arena before the fight; a Zombie Barger attacks anything within
//                           4.4 units: its "Attack Range" box): the start waits at the arrival spot for that, at most
//                           10 s of game time, and logs how long. Health is set to the maximum (5 of 5, via
//                           HeroController.MaxHealth, which also updates the HUD) and soul to 0 (SetMPCharge(0));
//                           nail damage 5 and no abilities stay those of a new game. Handover when the knight has
//                           stood there for 2 frames. A text on screen says so for the first seconds.
//
// Camera lock fix (HONING_CAMERA_LOCK_FIX, default 1; 0 = the build as it is). A CameraLockArea fills in its
// unspecified bounds (-1) with the room's defaults only when the camera's tile map belongs to its own scene; the
// lock areas of an additively loaded scene without a tile map (Crossroads_10_boss: "CameraLockArea B", "B2") keep
// -1. CameraController.LockToArea reads -1 as "default", but ReleaseLock copies the raw values of the remaining
// area, so in the False Knight arena the camera flies to the room's bottom-left corner the first time the knight
// leaves one of the two areas while inside the other (a jump above y 29.7 and back), for any player. With the fix
// the bridge writes into such areas, once per area, exactly the values their own ValidateBounds would have written
// (x 14.6 .. xLimit, y 8.3 .. yLimit of the room) and logs it. Nothing else of the camera is changed.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading;
using GlobalEnums;
using InControl;
using UnityEngine;
using UnityEngine.LowLevel;

public class HoningKeyboard : IKeyboardProvider
{
    private readonly IKeyboardProvider names = new UnityKeyboardProvider();
    private readonly HashSet<Key> down = new HashSet<Key>();

    public void Setup()
    {
    }

    public void Reset()
    {
    }

    public void Update()
    {
    }

    public bool AnyKeyIsPressed()
    {
        return down.Count > 0;
    }

    public bool GetKeyIsPressed(Key control)
    {
        return down.Contains(control);
    }

    public string GetNameForKey(Key control)
    {
        return names.GetNameForKey(control);
    }

    public void Set(List<Key> keys)
    {
        down.Clear();
        foreach (Key k in keys)
        {
            down.Add(k);
        }
    }

    public bool IsDown(Key k)
    {
        return down.Contains(k);
    }
}

public class HoningBridge : MonoBehaviour
{
    [Serializable]
    public class Command        // devmode: public, a lookahead plan is a list of these (HoningLookahead.cs)
    {
        public int id;
        public string action;
        public int frames;
        public string model;
        public float probability;
        public string[] hold;
        public string[] press;
        public string[] first;
        public string observe;      // devmode: "full" (default) | "debug" | "none": what is observed after this window
    }

    private static readonly Dictionary<string, Key> KeyOf = new Dictionary<string, Key>
    {
        { "LEFT", Key.LeftArrow }, { "RIGHT", Key.RightArrow }, { "UP", Key.UpArrow }, { "DOWN", Key.DownArrow },
        { "JUMP", Key.X }, { "NAIL", Key.Z }, { "DASH", Key.D }, { "SUPER_DASH", Key.W }, { "CAST", Key.F },
        { "QUICK_CAST", Key.Q }, { "INVENTORY", Key.I }, { "PAUSE", Key.Escape }, { "SUBMIT", Key.Return },
    };

    // b: a bridge frame is [Update, picture, physics steps] (a: [physics steps, Update, picture], input one frame late)
    // c: turn before a sideways strike
    // d (2026-11-19): boss start "boss:FalseKnight", camera lock fix (knob), camera lock state in the debug row
    // e (2026-11-19): no internal names in the visible observation (look = kind word, opaque room code in ids), door list
    // devmode-a (2026-11-25): bridge e + knobs for exact speed-up, developer controls and lookahead (HoningDev.cs)
    // devmode-c: b + pictures in HONING_STEP_AT=loop mode (CaptureLoop); nothing else changed
    public const string Version = "forestdango-bridge-devmode-2026-11-25c";
    private static readonly string[] Directions = { "LEFT", "RIGHT", "UP", "DOWN" };
    private static bool booted;

    // labelled boss starts: scene, entry gate, where the knight is put (x, feet y), facing right, label text
    private class BossStart
    {
        public string scene, gate, text;
        public float x, feet;
        public bool facingRight;
    }

    private static readonly Dictionary<string, BossStart> BossStarts = new Dictionary<string, BossStart>
    {
        { "FalseKnight", new BossStart { scene = "Crossroads_10", gate = "left1", x = 9.6f, feet = 26.07f, facingRight = true,
            text = "False Knight arena, left end of the floor, 3.6 units before the battle trigger" } },
    };

    // configuration
    private string session;
    private bool capture, captureBoot, jpg, stickyJump = true, debugProfile, logVisible = true;
    private int frameHz = 30, seed = 42;
    private double modelTimeout = 180;
    private string startSpec = "new", jumpScene, jumpGate;
    private bool hasPlace;
    private float placeX, placeY;
    private BossStart boss;                 // set for a boss:<Name> start
    private string bossName;
    private bool cameraLockFix = true;
    private bool cameraLockFixOwnTileMapOnly;
    private readonly HashSet<int> fixedLocks = new HashSet<int>();
    private int lockFixCheckFrame = -1000;

    // state
    private HoningKeyboard keyboard;
    private HoningVisible visible;
    private bool handedOver, quitting;
    private int bootPhase, bootFrames, menuFrames, settleFrames, skips;
    private int sequence, framesPlayed, remaining, windowFrames, windowIndex;
    private readonly List<string> winHold = new List<string>(), winPress = new List<string>(), winFirst = new List<string>();
    private readonly Dictionary<string, int> pressStart = new Dictionary<string, int>();
    private bool jumpLatched;
    private bool carryNail;          // a strike that was put off by a turn and did not fit into its window
    private int turnsBeforeStrike;
    private int jumpPressedAtFrame = -1000;
    private int deaths, hazardRespawns, sceneChanges, fixedSteps;
    private bool wasDead, wasHazard;
    private string lastScene = "";
    private int lastHealth = -1;
    private string label, modelLabel = "";
    private int labelUntilFrame;
    private GUIStyle labelStyle;
    private FieldInfo dialogueHidden;
    private string lastWindowAction = "";
    private string devResult;               // devmode: result of the dev command that is being answered
    private bool visibleOff;                // devmode: HONING_VISIBLE=0, no visible observation is built (timing studies)
    private bool frameLogBoot;              // devmode: HONING_FRAME_LOG_BOOT=1
    private UnityEngine.Random.State renderRandom;
    private float forcedAspect;
    private bool fastPoll;                  // devmode: HONING_POLL=fast
    private bool logBatch;                  // devmode: HONING_LOG_BATCH=1
    private string observeLevel = "full";   // devmode: per window, from the command's "observe"
    private readonly StringBuilder visibleBuf = new StringBuilder(), debugBuf = new StringBuilder();
    private int aspectsForced;

    // ---- devmode: knobs (defaults = bridge e) and probe state
    private string stepAt = "eof";          // HONING_STEP_AT: eof (the start sequence and the pictures run from a WaitForEndOfFrame
                                            // coroutine, as in bridge e) | loop (from a hook at the end of the engine's frame loop)
    private string renderMode = "all";      // HONING_RENDER: all | skip (OnDemandRendering: the engine draws no frame)
    private int frameLog, timingLevel;      // HONING_FRAME_LOG, HONING_TIMING
    private bool introspect;                // HONING_INTROSPECT
    private int fixedInFrame, fixedAll, eofResumes, loopEnds, eofProbeAsked, eofProbeDone;
    private readonly StringBuilder frameBuf = new StringBuilder();
    private readonly StringBuilder timingBuf = new StringBuilder();
    private long observeTicks, blockTicks, frameObserveTicks, frameBlockTicks;
    private readonly List<double> winFrameMs = new List<double>();
    private double[] winSeg;
    private double winObserveMs, winBlockMs, lastVisibleMs, lastDebugMs, lastWriteMs;
    private bool frameInWindow;
    private int timedWindow;

    public static void Boot()
    {
        if (booted)
        {
            return;
        }
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("HONING_SESSION_DIR")))
        {
            return;
        }
        booted = true;
        GameObject go = new GameObject("Honing model bridge");
        DontDestroyOnLoad(go);
        go.AddComponent<HoningBridge>();
    }

    private static string Env(string name, string fallback)
    {
        string v = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrEmpty(v) ? fallback : v;
    }

    private static float Num(string text)
    {
        return float.Parse(text, CultureInfo.InvariantCulture);
    }

    private void Log(string text)
    {
        try
        {
            File.AppendAllText(Path.Combine(session, "bridge.log"),
                "[" + Time.frameCount + " t=" + Time.time.ToString("0.000", CultureInfo.InvariantCulture) + "] " + text + "\n");
        }
        catch (Exception)
        {
        }
        if (!HoningLookahead.InChild)
        {
            Debug.Log("[HoningBridge] " + text);
        }
    }

    private void Start()
    {
        session = Environment.GetEnvironmentVariable("HONING_SESSION_DIR");
        Directory.CreateDirectory(session);
        capture = Env("HONING_CAPTURE", "0") == "1";
        captureBoot = Env("HONING_CAPTURE_BOOT", "0") == "1";
        jpg = Env("HONING_CAPTURE_FORMAT", "png") == "jpg";
        frameHz = int.Parse(Env("HONING_FRAME_HZ", "30"));
        seed = int.Parse(Env("HONING_SEED", "42"));
        modelTimeout = double.Parse(Env("HONING_MODEL_TIMEOUT_SECONDS", "180"), CultureInfo.InvariantCulture);
        stickyJump = Env("HONING_JUMP_MODE", "sticky") != "window";
        debugProfile = Env("HONING_OBSERVATION_PROFILE", "visible") == "global_debug";
        logVisible = Env("HONING_LOG_VISIBLE", "1") == "1";
        startSpec = Env("HONING_SCENE", "new");
        cameraLockFix = Env("HONING_CAMERA_LOCK_FIX", "1") != "0";
        // devmode: "2" (the default with HONING_FAST=1) = the same fix, but only for lock areas of a scene that has no
        // tile map of its own (the boss scenes it was written for). With "1" (bridge e) the fix also touches the lock
        // areas of the NEXT ROOM during a room change, in the frames in which both rooms are loaded and the camera still
        // holds the old room's tile map: it then writes the OLD room's limits into them, and the game's own
        // ValidateBounds, which runs a moment later, keeps them (seen: Dirtmouth's areas got Crossroads_10's 81.4
        // instead of 273.4).
        cameraLockFixOwnTileMapOnly = Env("HONING_CAMERA_LOCK_FIX", HoningDev.Fast ? "2" : "1") == "2";
        if (startSpec.StartsWith("boss:"))
        {
            bossName = startSpec.Substring(5);
            if (!BossStarts.TryGetValue(bossName, out boss))
            {
                Log("unknown boss start '" + bossName + "' (known: " + string.Join(", ", new List<string>(BossStarts.Keys).ToArray()) + ")");
                Quit();
                return;
            }
            jumpScene = boss.scene;
            jumpGate = boss.gate;
            hasPlace = true;
            placeX = boss.x;
            placeY = boss.feet;     // feet height; converted to the transform's height when the knight is put there
        }
        else if (startSpec.StartsWith("jump:"))
        {
            string rest = startSpec.Substring(5);
            int at = rest.IndexOf('@');
            if (at >= 0)
            {
                string[] xy = rest.Substring(at + 1).Split(',');
                placeX = Num(xy[0]);
                placeY = Num(xy[1]);
                hasPlace = true;
                rest = rest.Substring(0, at);
            }
            string[] parts = rest.Split(':');
            jumpScene = parts[0];
            jumpGate = parts.Length > 1 ? parts[1] : "left1";
        }
        if (capture)
        {
            Directory.CreateDirectory(Path.Combine(session, "frames"));
        }
        if (capture && captureBoot)
        {
            Directory.CreateDirectory(Path.Combine(session, "boot_frames"));
        }
        Application.runInBackground = true;
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
        Time.captureFramerate = frameHz;
        UnityEngine.Random.InitState(seed);
        keyboard = new HoningKeyboard();
        visible = new HoningVisible();
        visible.cell = int.Parse(Env("HONING_VISIBLE_CELL_PIXELS", "20"));
        visible.step = Num(Env("HONING_GEOM_STEP", "0.5"));
        string anchor = Env("HONING_VIEW_ANCHOR", "");
        if (anchor != "")
        {
            if (anchor.StartsWith("feet:"))
            {
                anchor = anchor.Substring(5);
            }
            string[] xy = anchor.Split(',');
            visible.anchored = true;
            visible.anchorX = Num(xy[0]);
            visible.anchorY = Num(xy[1]);
        }
        visible.zoom = Num(Env("HONING_VIEW_ZOOM", "1"));
        visible.stableIds = Env("HONING_IDS", HoningDev.Fast ? "stable" : "instance") == "stable";
        if (Env("HONING_UI_DUMP", "0") == "1")
        {
            visible.UiDump = "";     // debugging: list the screen-space UI renderers with every observation
        }
        string size = Env("HONING_VIEW_SIZE", "");
        if (size != "")
        {
            string[] wh = size.Split('x');
            visible.viewW = int.Parse(wh[0]);
            visible.viewH = int.Parse(wh[1]);
        }
        Log("bridge start: version=" + Version + " scene=" + startSpec + " frame_hz=" + frameHz + " jump_mode=" + (stickyJump ? "sticky" : "window") + " capture=" + capture
            + " seed=" + seed + " obs_mode=geometry step=" + visible.step + " view=" + visible.viewW + "x" + visible.viewH
            + (visible.anchored ? " anchor=" + visible.anchorX + "," + visible.anchorY + " zoom=" + visible.zoom : " (game window)")
            + " screen=" + Screen.width + "x" + Screen.height + " camera_lock_fix=" + (!cameraLockFix ? "0" : cameraLockFixOwnTileMapOnly ? "2" : "1"));
        // ---- devmode knobs
        stepAt = Env("HONING_STEP_AT", HoningDev.Fast ? "loop" : "eof");
        // "skip" (OnDemandRendering) is kept only as a record of what does not work: a wait for the end of the frame
        // never ends on a frame that is not drawn, and the engine decides at the start of a frame whether it draws, so
        // the game's own coroutines (a crawler's turn, entering a room) stall. Use -batchmode -nographics instead.
        renderMode = Env("HONING_RENDER", "all");
        frameLog = int.Parse(Env("HONING_FRAME_LOG", "0"));
        timingLevel = int.Parse(Env("HONING_TIMING", "0"));
        introspect = Env("HONING_INTROSPECT", "0") == "1";
        visibleOff = Env("HONING_VISIBLE", "1") == "0";
        frameLogBoot = Env("HONING_FRAME_LOG_BOOT", "0") == "1";
        fastPoll = Env("HONING_POLL", "slow") == "fast";
        // devmode (HONING_LOG_BATCH=1): observations.jsonl and visible_observations.jsonl are written in batches and at
        // the end instead of after every window (do not use with a tool that follows these files while the game runs)
        logBatch = Env("HONING_LOG_BATCH", "0") == "1";
        // devmode (HONING_TARGET_FPS, default -1 = as fast as the machine goes, as in bridge e): a cap on drawn frames per
        // wall second; 30 with HONING_FRAME_HZ=30 plays the game at its real speed (for "normal speed" reference runs)
        Application.targetFrameRate = int.Parse(Env("HONING_TARGET_FPS", "-1"));
        if (capture)
        {
            renderMode = "all";         // pictures need every frame drawn
        }
        if (renderMode == "skip")
        {
            stepAt = "loop";            // a WaitForEndOfFrame coroutine does not resume on a frame that is not drawn
            HoningDev.RenderSkip = true;
            UnityEngine.Rendering.OnDemandRendering.renderFrameInterval = 1000000;
        }
        string screen = Env("HONING_SCREEN", "");
        if (screen != "")
        {
            // devmode: without a window (-batchmode) the engine's screen is 640 x 480; the game's camera takes its
            // aspect from the screen, so the size has to be asked for
            string[] wh = screen.Split('x');
            Screen.SetResolution(int.Parse(wh[0]), int.Parse(wh[1]), false);
        }
        string aspect = Env("HONING_ASPECT", Application.isBatchMode ? "16:9" : "");
        if (aspect != "")
        {
            string[] ab = aspect.Split(':');
            forcedAspect = Num(ab[0]) / Num(ab[1]);
        }
        HoningDev.Start(this, session);
        Log("devmode knobs: ids=" + (visible.stableIds ? "stable" : "instance") + " poll=" + (fastPoll ? "fast" : "slow") + " log_batch=" + (logBatch ? 1 : 0) + " clock=" + (HoningClock.Virtual ? "virtual" : "real") + " step_at=" + stepAt + " render=" + renderMode + " frame_log=" + frameLog
            + " timing=" + timingLevel + " introspect=" + (introspect ? 1 : 0) + " batchmode=" + (Application.isBatchMode ? 1 : 0)
            + " graphics=" + SystemInfo.graphicsDeviceType + " " + HoningDev.Describe());
        InstallLoopHook();
        HoningLoop.Install(timingLevel > 0, timingLevel > 1);
        HoningLoop.EndOfLoop = OnEndOfLoop;
        HoningLoop.FrameDone = OnFrameDone;
        HoningLookahead.endCopy = note => { FlushAll(); HoningLookahead.Finish(note, framesPlayed); };
        if (HoningDev.Fast)
        {
            HoningLookahead.Prepare();
        }
        if (HoningDev.IsolateRenderRandom)
        {
            // devmode (HONING_RENDER_RNG=isolate): whatever random numbers the drawing of a frame takes (the camera's film
            // grain takes two per tile, i.e. a number that depends on the window size) are given back, so that the game's
            // random sequence is the same whether, and at what size, frames are drawn
            HoningLoop.BeforeRender = () => { renderRandom = UnityEngine.Random.state; };
            HoningLoop.AfterRender = () => { UnityEngine.Random.state = renderRandom; };
            HoningLoop.BeforeInvisible = () => { renderRandom = UnityEngine.Random.state; };
            HoningLoop.AfterInvisible = () => { UnityEngine.Random.state = renderRandom; };
        }
        if (timingLevel > 0)
        {
            File.WriteAllText(Path.Combine(session, "playerloop_markers.txt"), string.Join("\n", HoningLoop.Names) + "\n");
        }
        if (stepAt == "eof")
        {
            StartCoroutine(Loop());
        }
        else if (capture)
        {
            StartCoroutine(CaptureLoop());
        }
        StartCoroutine(EofProbe());
    }

    // devmode: the end of the engine's frame loop (after rendering and everything else of the frame)
    // devmode (HONING_ASPECT, e.g. "16:9"; default: on without a window, off otherwise): without a window the engine's
    // screen is 640 x 480 and every camera would take 4:3 from it. The game's own camera limits are written for 16:9
    // (14.6 x 8.3 half-extents), so each camera is given the aspect of the game's window.
    private void ForceAspect()
    {
        if (forcedAspect <= 0f)
        {
            return;
        }
        foreach (Camera cam in Camera.allCameras)
        {
            if (cam.targetTexture == null && Mathf.Abs(cam.aspect - forcedAspect) > 1e-6f)
            {
                cam.aspect = forcedAspect;
                aspectsForced++;
            }
        }
    }

    private void OnEndOfLoop()
    {
        loopEnds++;
        ForceAspect();
        HoningDev.EndOfLoop();
        if (quitting || stepAt != "loop")
        {
            return;
        }
        try
        {
            Step();
        }
        catch (Exception e)
        {
            Log("ERROR in the bridge step: " + e);
            Quit();
        }
    }

    // devmode: PlayerLoop timing of the frame that has just ended (HONING_TIMING)
    private void OnFrameDone(long[] seg, long total)
    {
        if (timingLevel > 0)
        {
            if (winSeg == null)
            {
                winSeg = new double[seg.Length];
            }
            double frameMs = HoningLoop.Ms(total - frameObserveTicks - frameBlockTicks);
            if (frameInWindow)
            {
                winFrameMs.Add(Math.Round(frameMs, 4));
                for (int i = 0; i < seg.Length; i++)
                {
                    winSeg[i] += HoningLoop.Ms(seg[i]);
                }
            }
            winObserveMs += HoningLoop.Ms(frameObserveTicks);
            winBlockMs += HoningLoop.Ms(frameBlockTicks);
        }
        frameInWindow = false;
        frameObserveTicks = 0;
        frameBlockTicks = 0;
    }

    // devmode: one timing row per action window, written when the next window starts
    private void FlushTiming(bool final)
    {
        if (timingLevel <= 0)
        {
            return;
        }
        if (winSeg != null && (winFrameMs.Count > 0 || final))
        {
            StringBuilder b = timingBuf;
            b.Append("{\"w\":").Append(timedWindow).Append(",\"frames\":").Append(winFrameMs.Count).Append(",\"frameMs\":[");
            for (int i = 0; i < winFrameMs.Count; i++)
            {
                b.Append(i > 0 ? "," : "").Append(HoningProbe.R(winFrameMs[i]));
            }
            b.Append("],\"observeMs\":").Append(HoningProbe.R(winObserveMs)).Append(",\"blockMs\":").Append(HoningProbe.R(winBlockMs))
             .Append(",\"visibleMs\":").Append(HoningProbe.R(lastVisibleMs)).Append(",\"debugMs\":").Append(HoningProbe.R(lastDebugMs))
             .Append(",\"writeMs\":").Append(HoningProbe.R(lastWriteMs)).Append(",\"seg\":{");
            bool first = true;
            for (int i = 0; i < winSeg.Length; i++)
            {
                if (winSeg[i] == 0.0)
                {
                    continue;
                }
                b.Append(first ? "\"" : ",\"").Append(HoningLoop.Names[i]).Append("\":").Append(HoningProbe.R(winSeg[i]));
                first = false;
            }
            b.Append("}}\n");
            winFrameMs.Clear();
            winObserveMs = 0;
            winBlockMs = 0;
            Array.Clear(winSeg, 0, winSeg.Length);
        }
        timedWindow = sequence;
        if (final || timingBuf.Length > 100000)
        {
            File.AppendAllText(Path.Combine(session, "timing.jsonl"), timingBuf.ToString());
            timingBuf.Length = 0;
        }
    }

    public void FlushAll()
    {
        FlushTiming(true);
        FlushFrameLog(true);
        FlushObservationLogs();
    }

    private void FlushObservationLogs()
    {
        if (visibleBuf.Length > 0)
        {
            File.AppendAllText(Path.Combine(session, "visible_observations.jsonl"), visibleBuf.ToString());
            visibleBuf.Length = 0;
        }
        if (debugBuf.Length > 0)
        {
            File.AppendAllText(Path.Combine(session, "observations.jsonl"), debugBuf.ToString());
            debugBuf.Length = 0;
        }
    }

    private void FlushFrameLog(bool final)
    {
        if (frameLog > 0 && (final || frameBuf.Length > 200000))
        {
            File.AppendAllText(Path.Combine(session, "frames.jsonl"), frameBuf.ToString());
            frameBuf.Length = 0;
        }
    }

    // Called once per frame after the frame's physics steps and before any script's Update.
    private void InstallLoopHook()
    {
        PlayerLoopSystem loop = PlayerLoop.GetCurrentPlayerLoop();
        bool done = false;
        for (int i = 0; i < loop.subSystemList.Length; i++)
        {
            if (loop.subSystemList[i].type == typeof(UnityEngine.PlayerLoop.Update))
            {
                var subs = new List<PlayerLoopSystem>(loop.subSystemList[i].subSystemList);
                subs.Insert(0, new PlayerLoopSystem { type = typeof(HoningBridge), updateDelegate = BeforeUpdate });
                loop.subSystemList[i].subSystemList = subs.ToArray();
                done = true;
            }
        }
        if (!done)
        {
            throw new InvalidOperationException("the engine's Update phase was not found in the player loop");
        }
        PlayerLoop.SetPlayerLoop(loop);
    }

    private void BeforeUpdate()
    {
        if (quitting || !handedOver)
        {
            return;
        }
        try
        {
            if (remaining > 0)
            {
                // the physics steps of one frame of the action window have just run
                framesPlayed++;
                remaining--;
                windowIndex++;
                Track();
                HoningDev.AfterFrame();     // devmode: developer controls that act between frames (all off by default)
                if (frameLog > 0)
                {
                    HoningProbe.FrameLine(frameBuf, frameLog, "f", framesPlayed, sequence, fixedInFrame, fixedAll);
                }
            }
            fixedInFrame = 0;
            if (remaining == 0)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                Observe();
                long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
                FlushFrameLog(false);
                WaitForAction();
                long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
                frameObserveTicks += t1 - t0;
                frameBlockTicks += t2 - t1;
                FlushTiming(false);
            }
            if (remaining > 0)
            {
                frameInWindow = true;
                ApplyKeys(windowIndex);     // the keys this frame's Update will read
            }
        }
        catch (Exception e)
        {
            Log("ERROR in the bridge step: " + e);
            if (HoningLookahead.InChild)
            {
                HoningLookahead.Finish("error in the bridge step: " + e.Message, framesPlayed);
            }
            Quit();
        }
    }

    private void FixedUpdate()
    {
        fixedInFrame++;
        fixedAll++;
        if (handedOver)
        {
            fixedSteps++;
        }
    }

    private IEnumerator Loop()
    {
        var endOfFrame = new WaitForEndOfFrame();
        while (!quitting)
        {
            yield return endOfFrame;
            eofResumes++;
            if (stepAt != "eof")
            {
                continue;       // devmode: the step runs from the end-of-loop hook instead
            }
            try
            {
                Step();
            }
            catch (Exception e)
            {
                Log("ERROR in the bridge step: " + e);
                Quit();
            }
        }
    }

    // devmode: counts how often a WaitForEndOfFrame of the engine really ends (once a second of game time is asked)
    private IEnumerator EofProbe()
    {
        var wait = new WaitForSeconds(1f);
        while (!quitting)
        {
            yield return wait;
            eofProbeAsked++;
            yield return new WaitForEndOfFrame();
            eofProbeDone++;
        }
    }

    private void Quit()
    {
        try
        {
            FlushTiming(true);
            FlushFrameLog(true);
            FlushObservationLogs();
        }
        catch (Exception)
        {
        }
        quitting = true;
        Application.Quit();
    }

    private void InstallKeyboard()
    {
        if (!InputManager.IsSetup || ReferenceEquals(InputManager.KeyboardProvider, keyboard))
        {
            return;
        }
        PropertyInfo property = typeof(InputManager).GetProperty("KeyboardProvider", BindingFlags.Public | BindingFlags.Static);
        MethodInfo setter = property != null ? property.GetSetMethod(true) : null;
        if (setter == null)
        {
            throw new InvalidOperationException("InputManager.KeyboardProvider cannot be set");
        }
        setter.Invoke(null, new object[] { keyboard });
        Log("keyboard provider installed (InControl reads the bridge's key states from now on)");
    }

    // ------------------------------------------------------------------------------------------------ flow
    public string Segment(out bool canAct)
    {
        GameManager gm = GameManager.UnsafeInstance;
        HeroController hero = HeroController.SilentInstance;
        canAct = false;
        if (gm == null || hero == null || hero.cState == null)
        {
            return "loading";
        }
        HeroControllerStates c = hero.cState;
        if (c.dead)
        {
            return "dead";
        }
        if (c.hazardDeath || c.hazardRespawning)
        {
            return "hazard_respawn";
        }
        if (gm.gameState == GameState.CUTSCENE)
        {
            return "cutscene";
        }
        if (gm.gameState == GameState.PAUSED || gm.isPaused)
        {
            return "paused";
        }
        if (gm.IsInSceneTransition || c.transitioning || hero.transitionState != HeroTransitionState.WAITING_TO_TRANSITION
            || gm.gameState != GameState.PLAYING)
        {
            return "transition";
        }
        if (DialogueText() != null)
        {
            return "dialogue";
        }
        if (hero.controlReqlinquished)
        {
            return "scripted";
        }
        if (!hero.acceptingInput)
        {
            return "recoil";
        }
        canAct = true;
        return "play";
    }

    // The text of a dialogue box that is on screen, or null.
    private string DialogueText()
    {
        foreach (DialogueBox box in FindObjectsOfType<DialogueBox>())
        {
            if (dialogueHidden == null)
            {
                dialogueHidden = typeof(DialogueBox).GetField("hidden", BindingFlags.NonPublic | BindingFlags.Instance);
            }
            bool hidden = dialogueHidden != null && (bool)dialogueHidden.GetValue(box);
            if (hidden)
            {
                continue;
            }
            TMPro.TextMeshPro mesh = box.GetComponent<TMPro.TextMeshPro>();
            Renderer rend = box.GetComponent<Renderer>();
            if (mesh != null && !string.IsNullOrEmpty(mesh.text) && (rend == null || rend.enabled))
            {
                return mesh.text;
            }
        }
        return null;
    }

    // ------------------------------------------------------------------------------------------------ start
    private void BootStep()
    {
        bootFrames++;
        if (frameLogBoot)
        {
            // devmode: the state of every frame of the start sequence (HONING_FRAME_LOG_BOOT=1), to find what differs
            HoningProbe.FrameLine(frameBuf, Math.Max(1, frameLog), "boot", -bootFrames, 0, fixedInFrame, fixedAll);
            fixedInFrame = 0;
            if (frameBuf.Length > 200000)
            {
                File.AppendAllText(Path.Combine(session, "frames.jsonl"), frameBuf.ToString());
                frameBuf.Length = 0;
            }
        }
        if (capture && captureBoot && bootFrames % 2 == 0)
        {
            Capture(Path.Combine(session, "boot_frames", (bootFrames / 2).ToString("D06")));
        }
        if (bootFrames > frameHz * 600)
        {
            Log("boot: no control after 600 s of game time, giving up (phase " + bootPhase + ")");
            Quit();
            return;
        }
        GameManager gm = GameManager.UnsafeInstance;
        if (gm == null)
        {
            return;
        }
        string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        HeroController hero = HeroController.SilentInstance;
        bool canAct;
        Segment(out canAct);
        bool standing = canAct && hero != null && hero.cState.onGround;
        if (bootPhase == 0)
        {
            // the title menu: the function the save-slot button calls
            if (scene == "Menu_Title" && gm.gameState == GameState.MAIN_MENU && gm.ui != null && gm.ui.menuState == MainMenuState.MAIN_MENU)
            {
                menuFrames++;
                if (menuFrames >= 2 * frameHz)
                {
                    Log("boot: title menu is up; calling UIManager.UIStartNewGame()");
                    gm.ui.UIStartNewGame();
                    bootPhase = 1;
                }
            }
        }
        else if (bootPhase == 1)
        {
            // the opening sequences: skip each one as soon as the game offers the skip prompt (what a second key
            // press does in InputHandler.CutSceneInput)
            InputHandler ih = gm.inputHandler;
            // devmode: with HONING_BOOT=direct there is no film to skip (the opening scene loads King's Pass at once)
            if (!HoningDev.DirectBoot && gm.gameState == GameState.CUTSCENE && ih != null && ih.skipMode == SkipPromptMode.SKIP_PROMPT && !ih.skippingCutscene)
            {
                ih.skippingCutscene = true;
                gm.SkipCutscene();
                skips++;
                Log("boot: skipped an opening sequence (" + skips + ")");
            }
            if (scene == "Tutorial_01" && standing)
            {
                Log("boot: the knight stands in King's Pass after " + bootFrames + " frames (" + skips + " sequences skipped)");
                bootPhase = jumpScene != null ? 2 : 9;
            }
        }
        else if (bootPhase == 2)
        {
            Log("boot: DEBUG START, scene transition to " + jumpScene + " gate " + jumpGate);
            gm.BeginSceneTransition(new GameManager.SceneLoadInfo
            {
                SceneName = jumpScene,
                EntryGateName = jumpGate,
                HeroLeaveDirection = null,
                EntryDelay = 0f,
                WaitForSceneTransitionCameraFade = true,
                PreventCameraFadeOut = false,
                Visualization = GameManager.SceneLoadVisualizations.Default,
                AlwaysUnloadUnusedAssets = false
            });
            bootPhase = 3;
            settleFrames = 0;
        }
        else if (bootPhase == 3)
        {
            if (scene == jumpScene && standing)
            {
                settleFrames++;
                if (settleFrames == 1)
                {
                    Vector3 arrived = hero.transform.position;
                    Log("boot: DEBUG START, arrived in " + scene + " at (" + arrived.x.ToString("0.00", CultureInfo.InvariantCulture) + ", "
                        + arrived.y.ToString("0.00", CultureInfo.InvariantCulture) + ") (transform position)");
                }
                if (settleFrames >= frameHz / 2 && boss != null && settleFrames < frameHz / 2 + 10 * frameHz)
                {
                    // a boss start: wait at the arrival spot until no enemy stands within 5 units of the place
                    string near = EnemyNear(placeX, placeY, 5f, 3f);
                    if (near != null)
                    {
                        if (settleFrames == frameHz / 2)
                        {
                            Log("boot: BOSS START " + bossName + ": waiting for a moment with no enemy within 5 units of the place (" + near + ")");
                        }
                        return;
                    }
                    Log("boot: BOSS START " + bossName + ": place clear after " + (settleFrames - frameHz / 2) + " frames of waiting");
                }
                else if (boss != null && settleFrames == frameHz / 2 + 10 * frameHz)
                {
                    Log("boot: BOSS START " + bossName + ": no clear moment within 10 s; the knight is put there anyway ("
                        + EnemyNear(placeX, placeY, 5f, 3f) + ")");
                }
                if (settleFrames >= frameHz / 2)
                {
                    if (hasPlace)
                    {
                        Vector3 p = hero.transform.position;
                        float y = placeY;
                        if (boss != null)
                        {
                            // a boss start gives the height of the feet: the transform sits above the bottom of the body box
                            BoxCollider2D box = hero.GetComponent<BoxCollider2D>();
                            y = placeY + (box != null ? p.y - box.bounds.min.y : 1.39f) + 0.02f;
                        }
                        hero.transform.position = new Vector3(placeX, y, p.z);
                        Rigidbody2D body = hero.GetComponent<Rigidbody2D>();
                        if (body != null)
                        {
                            body.velocity = Vector2.zero;
                        }
                        if (boss != null)
                        {
                            if (boss.facingRight)
                            {
                                hero.FaceRight();
                            }
                            else
                            {
                                hero.FaceLeft();
                            }
                        }
                        if (gm.cameraCtrl != null)
                        {
                            gm.cameraCtrl.PositionToHero(false);
                        }
                        Log("boot: " + (boss != null ? "BOSS START " + bossName : "DEBUG START") + ", knight put at (" + placeX + ", "
                            + y.ToString("0.00", CultureInfo.InvariantCulture) + ") (transform position)");
                        bootPhase = 4;
                        settleFrames = 0;
                    }
                    else
                    {
                        bootPhase = 9;
                    }
                }
            }
        }
        else if (bootPhase == 4)
        {
            // hand over as soon as the knight stands at the place (waiting longer would let the room act on a
            // knight nobody controls yet)
            settleFrames++;
            if (standing && settleFrames >= 2)
            {
                if (boss != null)
                {
                    // what a player who arrives here rested would have: full health, no soul (new-game nail and abilities)
                    PlayerData pd0 = gm.playerData;
                    int before = pd0.health, soulBefore = pd0.MPCharge;
                    hero.MaxHealth();
                    hero.SetMPCharge(0);
                    Log("boot: BOSS START " + bossName + ": health " + before + " -> " + pd0.health + "/" + pd0.maxHealth + ", soul " + soulBefore
                        + " -> " + pd0.MPCharge + ", nail " + pd0.nailDamage + ", dash=" + pd0.hasDash + " walljump=" + pd0.hasWalljump
                        + " doublejump=" + pd0.hasDoubleJump + " spell=" + pd0.hasSpell + " (not changed)");
                }
                bootPhase = 9;
            }
        }
        if (bootPhase == 9)
        {
            UnityEngine.Random.InitState(seed);
            handedOver = true;
            remaining = 0;
            lastScene = scene;
            if (boss != null)
            {
                PlayerData pd = gm.playerData;
                label = "HONING BOSS START (debug, not normal travel): " + boss.text + "; knight put at (" + placeX + ", " + placeY
                        + "); health set to " + pd.health + "/" + pd.maxHealth + ", soul 0; nail " + pd.nailDamage + ", no abilities (new game)";
                labelUntilFrame = 6 * frameHz;
            }
            else if (jumpScene != null)
            {
                PlayerData pd = gm.playerData;
                label = "HONING DEBUG START: " + jumpScene + " via gate " + jumpGate + (hasPlace ? ", knight put at (" + placeX + ", " + placeY + ")" : "")
                        + "; new-game state: " + pd.health + "/" + pd.maxHealth + " masks, nail " + pd.nailDamage + ", no abilities";
                labelUntilFrame = 6 * frameHz;
            }
            Vector3 at = hero.transform.position;
            Log("handover: scene=" + scene + " x=" + at.x.ToString("0.00", CultureInfo.InvariantCulture) + " y=" + at.y.ToString("0.00", CultureInfo.InvariantCulture)
                + " boot_frames=" + bootFrames + " health=" + gm.playerData.health + "/" + gm.playerData.maxHealth + " nail=" + gm.playerData.nailDamage
                + " dash=" + gm.playerData.hasDash + " walljump=" + gm.playerData.hasWalljump + " doublejump=" + gm.playerData.hasDoubleJump
                + " spell=" + gm.playerData.hasSpell);
            Log("devmode at handover: engine_frames=" + HoningClock.EngineFrameCount + " eof_resumes=" + eofResumes + " loop_ends=" + loopEnds + " eof_probe=" + eofProbeDone + "/" + eofProbeAsked
                + " time=" + HoningProbe.D(HoningClock.EngineTimeAsDouble) + " fixed_time=" + HoningProbe.D(HoningClock.EngineFixedTimeAsDouble)
                + " fixed_steps_all=" + fixedAll + " rng=" + JsonUtility.ToJson(UnityEngine.Random.state)
                + " knight_renderer_visible=" + (hero.GetComponent<Renderer>() != null && hero.GetComponent<Renderer>().isVisible ? 1 : 0)
                + " renderers_visible=" + CountVisible()
                + " screen=" + Screen.width + "x" + Screen.height + " aspects_forced=" + aspectsForced + " camera_aspect=" + (HoningVisible.GameCamera() != null ? HoningProbe.F(HoningVisible.GameCamera().aspect) : "-"));
            if (introspect)
            {
                File.WriteAllText(Path.Combine(session, "introspection_handover_" + scene + ".json"), HoningProbe.Introspect());
            }
            if (frameLog > 0)
            {
                HoningProbe.FrameLine(frameBuf, frameLog, "handover", framesPlayed, sequence, fixedInFrame, fixedAll);
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ the loop
    // End of a frame (the picture is drawn): the start sequence, and the pictures of the action windows.
    private void Step()
    {
        InstallKeyboard();
        if (!handedOver)
        {
            FixCameraLocks(false);
            BootStep();
            return;
        }
        if (remaining > 0 && capture && stepAt == "eof")
        {
            Capture(Path.Combine(session, "frames", framesPlayed.ToString("D06")));
        }
    }

    // devmode: with HONING_STEP_AT=loop the pictures are still taken where a picture can be taken, at the engine's end
    // of frame (after the frame is drawn, before it is shown), by a coroutine that does nothing else
    private IEnumerator CaptureLoop()
    {
        var endOfFrame = new WaitForEndOfFrame();
        while (!quitting)
        {
            yield return endOfFrame;
            if (handedOver && remaining > 0 && capture)
            {
                try
                {
                    Capture(Path.Combine(session, "frames", framesPlayed.ToString("D06")));
                }
                catch (Exception e)
                {
                    Log("ERROR while taking a picture: " + e);
                }
            }
        }
    }

    private void Capture(string pathWithoutExtension)
    {
        Texture2D picture = ScreenCapture.CaptureScreenshotAsTexture();
        try
        {
            if (jpg)
            {
                File.WriteAllBytes(pathWithoutExtension + ".jpg", picture.EncodeToJPG(92));
            }
            else
            {
                File.WriteAllBytes(pathWithoutExtension + ".png", picture.EncodeToPNG());
            }
        }
        finally
        {
            Destroy(picture);
        }
    }

    // An enemy (alive, with a HealthManager) within dx / dy of a place, as "name at (x, y)", or null.
    private static string EnemyNear(float x, float y, float dx, float dy)
    {
        foreach (HealthManager hm in FindObjectsOfType<HealthManager>())
        {
            if (hm.isDead || !hm.gameObject.activeInHierarchy)
            {
                continue;
            }
            Collider2D col = hm.GetComponent<Collider2D>();
            Vector3 p = col != null ? col.bounds.center : hm.transform.position;
            float bottom = col != null ? col.bounds.min.y : p.y;
            if (Mathf.Abs(p.x - x) < dx && Mathf.Abs(bottom - y) < dy)
            {
                return hm.gameObject.name + " at (" + p.x.ToString("0.0", CultureInfo.InvariantCulture) + ", " + bottom.ToString("0.0", CultureInfo.InvariantCulture) + ")";
            }
        }
        return null;
    }

    // The camera lock fix (see the text at the top). Cheap: a handful of lock areas per room; checked every 10 frames,
    // at every observation and every frame of the start sequence, so an area is fixed long before anyone can reach it.
    private void FixCameraLocks(bool force)
    {
        if (!cameraLockFix || (!force && Time.frameCount - lockFixCheckFrame < 10))
        {
            return;
        }
        lockFixCheckFrame = Time.frameCount;
        GameManager gm = GameManager.UnsafeInstance;
        CameraController cc = gm != null ? gm.cameraCtrl : null;
        if (cc == null || cc.tilemap == null)
        {
            return;
        }
        UnityEngine.SceneManagement.Scene room = cc.tilemap.gameObject.scene;
        if (room != UnityEngine.SceneManagement.SceneManager.GetActiveScene())
        {
            return;             // in the middle of a room change: xLimit / yLimit may still be the old room's
        }
        foreach (CameraLockArea area in FindObjectsOfType<CameraLockArea>())
        {
            if (area.gameObject.scene == room || fixedLocks.Contains(area.GetInstanceID()))
            {
                continue;       // an area of the room's own scene validates itself (CameraLockArea.Start)
            }
            if (cameraLockFixOwnTileMapOnly && SceneHasTileMap(area.gameObject.scene))
            {
                continue;       // devmode: a room that is being loaded; its areas validate themselves once it is the room
            }
            if (area.cameraXMin != -1f && area.cameraXMax != -1f && area.cameraYMin != -1f && area.cameraYMax != -1f)
            {
                continue;
            }
            string before = F(area.cameraXMin) + "," + F(area.cameraXMax) + "," + F(area.cameraYMin) + "," + F(area.cameraYMax);
            if (area.cameraXMin == -1f) area.cameraXMin = 14.6f;
            if (area.cameraXMax == -1f) area.cameraXMax = cc.xLimit;
            if (area.cameraYMin == -1f) area.cameraYMin = 8.3f;
            if (area.cameraYMax == -1f) area.cameraYMax = cc.yLimit;
            fixedLocks.Add(area.GetInstanceID());
            Log("camera lock fix: '" + area.name + "' of scene " + area.gameObject.scene.name + " (room " + room.name + "): bounds " + before
                + " -> " + F(area.cameraXMin) + "," + F(area.cameraXMax) + "," + F(area.cameraYMin) + "," + F(area.cameraYMax)
                + " (the defaults its ValidateBounds would set; it never ran because the camera's tile map is not in its scene)");
        }
    }

    // devmode probe: how many renderers the engine counts as visible (does culling happen when nothing is drawn?)
    private static string CountVisible()
    {
        int visible = 0, all = 0;
        foreach (Renderer r in FindObjectsOfType<Renderer>())
        {
            all++;
            if (r.isVisible)
            {
                visible++;
            }
        }
        return visible + "/" + all;
    }

    private readonly Dictionary<int, bool> sceneTileMaps = new Dictionary<int, bool>();

    private bool SceneHasTileMap(UnityEngine.SceneManagement.Scene scene)
    {
        bool has;
        if (sceneTileMaps.TryGetValue(scene.handle, out has))
        {
            return has;
        }
        has = false;
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            if (root.GetComponentInChildren<tk2dTileMap>(true) != null)
            {
                has = true;
                break;
            }
        }
        sceneTileMaps[scene.handle] = has;
        return has;
    }

    private static string F(float v)
    {
        return v.ToString("0.###", CultureInfo.InvariantCulture);
    }

    // bookkeeping after every played frame
    private void Track()
    {
        GameManager gm = GameManager.UnsafeInstance;
        HeroController hero = HeroController.SilentInstance;
        string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        if (scene != lastScene)
        {
            sceneChanges++;
            Log("scene " + lastScene + " -> " + scene + " (frame " + framesPlayed + ")");
            lastScene = scene;
            if (introspect)
            {
                File.WriteAllText(Path.Combine(session, "introspection_f" + framesPlayed + "_" + scene + ".json"), HoningProbe.Introspect());
            }
            FixCameraLocks(true);
        }
        FixCameraLocks(false);
        if (hero != null && hero.cState != null)
        {
            if (hero.cState.dead && !wasDead)
            {
                deaths++;
                Log("death " + deaths + " in " + scene + " (frame " + framesPlayed + ")");
            }
            wasDead = hero.cState.dead;
            bool hazard = hero.cState.hazardDeath;
            if (hazard && !wasHazard)
            {
                hazardRespawns++;
            }
            wasHazard = hazard;
            if (jumpLatched && framesPlayed - jumpPressedAtFrame >= 3)
            {
                Rigidbody2D body = hero.GetComponent<Rigidbody2D>();
                if (body == null || body.velocity.y <= 0.01f)
                {
                    jumpLatched = false;       // the knight is not rising any more: holding on changes nothing
                }
            }
        }
    }

    private void ApplyKeys(int index)
    {
        var names = new List<string>();
        if (index == 0 && winFirst.Count > 0)
        {
            names.AddRange(winFirst);
            foreach (string k in winHold)
            {
                if (Array.IndexOf(Directions, k) < 0)
                {
                    names.Add(k);
                }
            }
        }
        else
        {
            names.AddRange(winHold);
        }
        foreach (string k in winPress)
        {
            int start = pressStart[k];
            int stop = k == "JUMP" ? windowFrames : start + 1;
            if (index >= start && index < stop)
            {
                names.Add(k);
            }
        }
        if (jumpLatched && !winPress.Contains("JUMP"))
        {
            names.Add("JUMP");
        }
        var keys = new List<Key>();
        foreach (string k in names)
        {
            Key key;
            if (KeyOf.TryGetValue(k, out key) && !keys.Contains(key))
            {
                keys.Add(key);
            }
        }
        keyboard.Set(keys);
    }

    private static bool ActionKeys(string action, List<string> hold, List<string> press)
    {
        switch (action)
        {
            case "wait":
            case "keys":
            case "":
                return true;
            case "left": hold.Add("LEFT"); return true;
            case "right": hold.Add("RIGHT"); return true;
            case "up": hold.Add("UP"); return true;
            case "down": hold.Add("DOWN"); return true;
            case "jump": press.Add("JUMP"); return true;
            case "jump_left": hold.Add("LEFT"); press.Add("JUMP"); return true;
            case "jump_right": hold.Add("RIGHT"); press.Add("JUMP"); return true;
            case "attack": press.Add("NAIL"); return true;
            case "attack_left": hold.Add("LEFT"); press.Add("NAIL"); return true;
            case "attack_right": hold.Add("RIGHT"); press.Add("NAIL"); return true;
            case "attack_up": hold.Add("UP"); press.Add("NAIL"); return true;
            case "attack_down": hold.Add("DOWN"); press.Add("NAIL"); return true;
            case "dash": press.Add("DASH"); return true;
            case "dash_left": hold.Add("LEFT"); press.Add("DASH"); return true;
            case "dash_right": hold.Add("RIGHT"); press.Add("DASH"); return true;
            case "focus": hold.Add("CAST"); return true;
            case "cast": press.Add("CAST"); return true;
            case "cast_up": hold.Add("UP"); press.Add("CAST"); return true;
            case "cast_down": hold.Add("DOWN"); press.Add("CAST"); return true;
            case "quick_cast": press.Add("QUICK_CAST"); return true;
            case "super_dash": hold.Add("SUPER_DASH"); return true;
            case "interact": press.Add("UP"); return true;
            case "inventory": press.Add("INVENTORY"); return true;
            default: return false;
        }
    }

    private void WaitForAction()
    {
        string path = Path.Combine(session, "action.json");
        float start = HoningClock.EngineRealtimeSinceStartup;
        int refused = -1;
        while (true)
        {
            Command c = null;
            if (HoningLookahead.InChild)
            {
                // devmode: this process is a lookahead copy; it plays its plan and ends (HoningLookahead.cs)
                c = HoningLookahead.NextWindow(sequence);
                if (c == null)
                {
                    FlushAll();
                    HoningLookahead.Finish(null, framesPlayed);
                }
            }
            else
            {
            try
            {
                if (File.Exists(path))
                {
                    c = JsonUtility.FromJson<Command>(File.ReadAllText(path));
                }
            }
            catch (Exception)
            {
                c = null;    // half-written file
            }
            }
            if (c != null && c.id == sequence)
            {
                if (c.action == "stop")
                {
                    Log("bridge stop: frames=" + framesPlayed + " windows=" + (sequence - 1) + " deaths=" + deaths
                        + " | devmode: engine_frames=" + HoningClock.EngineFrameCount + " eof_resumes=" + eofResumes + " loop_ends=" + loopEnds + " eof_probe=" + eofProbeDone + "/" + eofProbeAsked + " " + HoningDev.Summary());
                    FlushTiming(true);
                    FlushFrameLog(true);
                    FlushObservationLogs();
                    Quit();
                    return;
                }
                if (c.action == "dev")
                {
                    // devmode: a developer command (HoningDev.cs). It plays no frame; the answer is the next observation
                    // (same game state, new id) with the command's result in debug_observation.json under "dev".
                    devResult = HoningDev.Command(File.ReadAllText(path));
                    if (HoningLookahead.InChild)
                    {
                        // this process is the copy that a lookahead command has just made: from here on it writes into its
                        // own directory and takes its windows from the plan
                        session = HoningLookahead.Dir;
                        HoningDev.SetSession(session);
                        devResult = null;
                        continue;
                    }
                    Log("dev command: " + devResult);
                    observeLevel = "full";
                    Observe();
                    continue;
                }
                var hold = new List<string>();
                var press = new List<string>();
                if (ActionKeys(c.action ?? "", hold, press) && c.frames >= 1 && c.frames <= 600)
                {
                    if (c.hold != null)
                    {
                        hold.AddRange(c.hold);
                    }
                    if (c.press != null)
                    {
                        press.AddRange(c.press);
                    }
                    winHold.Clear();
                    winHold.AddRange(hold);
                    winPress.Clear();
                    foreach (string k in press)
                    {
                        if (!winPress.Contains(k))
                        {
                            winPress.Add(k);
                        }
                    }
                    winFirst.Clear();
                    if (c.first != null)
                    {
                        winFirst.AddRange(c.first);
                    }
                    bool carried = carryNail && !winPress.Contains("NAIL");
                    if (carried)
                    {
                        // the strike of the previous window, after its turn: it goes out in this window's first
                        // frame, the way the knight faces now, whatever direction this window holds (the knight
                        // does not turn while it strikes)
                        winPress.Add("NAIL");
                    }
                    carryNail = false;
                    pressStart.Clear();
                    foreach (string k in winPress)
                    {
                        Key key;
                        pressStart[k] = KeyOf.TryGetValue(k, out key) && keyboard.IsDown(key) ? 1 : 0;
                    }
                    if (winPress.Contains("NAIL") && !carried)
                    {
                        // turn before a sideways strike (see the text at the top)
                        List<string> with = winFirst.Count > 0 ? winFirst : winHold;
                        bool left = with.Contains("LEFT"), right = with.Contains("RIGHT");
                        HeroController hero = HeroController.SilentInstance;
                        if (left != right && !with.Contains("UP") && !with.Contains("DOWN") && hero != null && hero.cState != null
                            && hero.cState.facingRight != right)
                        {
                            pressStart["NAIL"] += 1;
                            turnsBeforeStrike++;
                            if (pressStart["NAIL"] >= c.frames)
                            {
                                winPress.Remove("NAIL");
                                pressStart.Remove("NAIL");
                                carryNail = true;
                            }
                        }
                    }
                    if (winPress.Contains("JUMP"))
                    {
                        jumpLatched = stickyJump;
                        jumpPressedAtFrame = framesPlayed + pressStart["JUMP"];
                    }
                    observeLevel = c.observe == "debug" || c.observe == "none" ? c.observe : "full";
                    windowFrames = c.frames;
                    remaining = c.frames;
                    windowIndex = 0;
                    lastWindowAction = c.action ?? "";
                    modelLabel = (c.model ?? "") + " | " + c.action + " | p=" + c.probability.ToString("F3", CultureInfo.InvariantCulture);
                    return;
                }
                if (refused != c.id)
                {
                    refused = c.id;
                    Log("refused action '" + c.action + "' frames=" + c.frames + " (not executed)");
                }
                if (HoningLookahead.InChild)
                {
                    FlushAll();
                    HoningLookahead.Finish("a planned window was refused: " + c.action + " frames=" + c.frames, framesPlayed);
                }
            }
            if (HoningClock.EngineRealtimeSinceStartup - start > modelTimeout)
            {
                Log("timed out waiting for an action");
                Quit();
                return;
            }
            // devmode (HONING_POLL=fast): look again at once for the first milliseconds, then ever more slowly, so that a
            // fast driver is answered within ~0.05 ms and a slow model does not keep a core busy. Default: every 2 ms.
            if (!fastPoll)
            {
                Thread.Sleep(2);
            }
            else
            {
                float waited = HoningClock.EngineRealtimeSinceStartup - start;
                if (waited < 0.004f)
                {
                    Thread.SpinWait(200);
                }
                else if (waited < 0.05f)
                {
                    Thread.Sleep(0);
                }
                else if (waited < 1f)
                {
                    Thread.Sleep(1);
                }
                else
                {
                    Thread.Sleep(2);
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ observations
    private void WriteAtomic(string name, string text)
    {
        string path = Path.Combine(session, name), tmp = path + ".tmp";
        File.WriteAllText(tmp, text);
        if (File.Exists(path))
        {
            File.Delete(path);
        }
        File.Move(tmp, path);
    }

    private void Observe()
    {
        ForceAspect();
        FixCameraLocks(true);
        sequence++;
        bool canAct;
        string segment = Segment(out canAct);
        string dialogue = segment == "dialogue" ? DialogueText() : null;
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        bool light = visibleOff || observeLevel != "full";
        string vis = light ? "{\"schema\":\"honing.visible.v1\",\"profile\":\"" + (visibleOff ? "off" : observeLevel) + "\",\"frame\":{\"id\":" + sequence + ",\"tick\":" + framesPlayed
                             + ",\"simulation_time\":" + HoningVisible.F(Time.time) + "},\"flow\":{\"can_act\":" + (canAct ? "true" : "false") + ",\"segment\":\"" + segment + "\"}}"
                                : visible.Build(sequence, framesPlayed, Time.time, canAct, segment, dialogue != null, dialogue);
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        foreach (string line in visible.NameLog)
        {
            Log("visible: " + line);      // e: what the door list turned into a plain solid (bridge.log only)
        }
        visible.NameLog.Clear();
        string debug = observeLevel == "none" && !visibleOff ? "{\"id\":" + sequence + ",\"frame\":" + framesPlayed + ",\"observe\":\"none\"}" : DebugJson(canAct, segment);
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        if (visible.UiDump != null)
        {
            File.WriteAllText(Path.Combine(session, "ui_dump_" + sequence.ToString("D05") + ".txt"), visible.UiDump);
        }
        WriteAtomic("debug_observation.json", debug);
        if (!logBatch)
        {
            File.AppendAllText(Path.Combine(session, "observations.jsonl"), debug + "\n");
        }
        else
        {
            debugBuf.Append(debug).Append('\n');
        }
        if (logVisible && !logBatch)
        {
            File.AppendAllText(Path.Combine(session, "visible_observations.jsonl"), vis + "\n");
        }
        else if (logVisible)
        {
            visibleBuf.Append(vis).Append('\n');     // devmode: with HONING_LOG_BATCH=1 the logs are written in batches
            if (visibleBuf.Length > 400000)
            {
                FlushObservationLogs();
            }
        }
        WriteAtomic("observation.json", debugProfile ? debug : vis);   // last: the harness waits for this file
        long t3 = System.Diagnostics.Stopwatch.GetTimestamp();
        if (HoningLookahead.InChild)
        {
            HoningLookahead.Progress(framesPlayed);
        }
        lastVisibleMs = HoningLoop.Ms(t1 - t0);
        lastDebugMs = HoningLoop.Ms(t2 - t1);
        lastWriteMs = HoningLoop.Ms(t3 - t2);
        devResult = null;
    }

    private string DebugJson(bool canAct, string segment)
    {
        var sb = new StringBuilder(4096);
        GameManager gm = GameManager.UnsafeInstance;
        HeroController hero = HeroController.SilentInstance;
        string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        float x = 0f, y = 0f, vx = 0f, vy = 0f, feet = 0f, bodyW = 0f, bodyH = 0f;
        bool grounded = false, facingRight = true;
        HeroControllerStates c = hero != null ? hero.cState : null;
        if (hero != null)
        {
            Vector3 p = hero.transform.position;
            x = p.x;
            y = p.y;
            Rigidbody2D body = hero.GetComponent<Rigidbody2D>();
            if (body != null)
            {
                vx = body.velocity.x;
                vy = body.velocity.y;
            }
            BoxCollider2D box = hero.GetComponent<BoxCollider2D>();
            if (box != null)
            {
                feet = box.bounds.min.y;
                bodyW = box.bounds.size.x;
                bodyH = box.bounds.size.y;
                x = box.bounds.center.x;
            }
            if (c != null)
            {
                grounded = c.onGround;
                facingRight = c.facingRight;
            }
        }
        PlayerData pd = gm != null ? gm.playerData : null;
        Func<double, string> f = HoningVisible.F;
        sb.Append("{\"id\":").Append(sequence).Append(",\"frame\":").Append(framesPlayed);
        if (sequence == 1)
        {
            sb.Append(",\"bridge\":\"").Append(Version).Append('"');
        }
        sb.Append(",\"health\":").Append(pd != null ? Mathf.Max(0, pd.health) : 0);
        sb.Append(",\"jumps\":").Append(grounded ? 1 : 0).Append(",\"fixedSteps\":").Append(fixedSteps);
        sb.Append(",\"scene\":");
        HoningVisible.Str(sb, scene);
        // x = middle of the body, y = the feet (bottom of the body collider); "pos" is the object's own position
        sb.Append(",\"x\":").Append(f(Math.Round(x, 4))).Append(",\"y\":").Append(f(Math.Round(feet, 4)));
        sb.Append(",\"vx\":").Append(f(Math.Round(vx, 4))).Append(",\"vy\":").Append(f(Math.Round(vy, 4)));
        sb.Append(",\"simulationTime\":").Append(f(Math.Round(Time.time, 4)));
        sb.Append(",\"grounded\":").Append(grounded ? "true" : "false");
        sb.Append(",\"facing\":").Append(facingRight ? "\"right\"" : "\"left\"");
        sb.Append(",\"pos\":[").Append(f(Math.Round(hero != null ? hero.transform.position.x : 0f, 4))).Append(',').Append(f(Math.Round(y, 4))).Append(']');
        sb.Append(",\"body\":[").Append(f(Math.Round(bodyW, 4))).Append(',').Append(f(Math.Round(bodyH, 4))).Append(']');
        sb.Append(",\"healthMax\":").Append(pd != null ? pd.maxHealth : 0);
        sb.Append(",\"soul\":").Append(pd != null ? pd.MPCharge : 0);
        sb.Append(",\"geo\":").Append(pd != null ? pd.geo : 0);
        sb.Append(",\"nailDamage\":").Append(pd != null ? pd.nailDamage : 0);
        sb.Append(",\"deaths\":").Append(deaths).Append(",\"hazardRespawns\":").Append(hazardRespawns).Append(",\"sceneChanges\":").Append(sceneChanges);
        sb.Append(",\"turnsBeforeStrike\":").Append(turnsBeforeStrike);
        sb.Append(",\"canAct\":").Append(canAct ? "true" : "false").Append(",\"segment\":");
        HoningVisible.Str(sb, segment);
        sb.Append(",\"gameState\":");
        HoningVisible.Str(sb, gm != null ? gm.gameState.ToString() : "");
        sb.Append(",\"lastAction\":");
        HoningVisible.Str(sb, lastWindowAction);
        if (c != null)
        {
            sb.Append(",\"hero\":{\"acceptingInput\":").Append(hero.acceptingInput ? "true" : "false");
            sb.Append(",\"controlRelinquished\":").Append(hero.controlReqlinquished ? "true" : "false");
            sb.Append(",\"attacking\":").Append(c.attacking ? "true" : "false");
            sb.Append(",\"jumping\":").Append(c.jumping ? "true" : "false");
            sb.Append(",\"falling\":").Append(c.falling ? "true" : "false");
            sb.Append(",\"dashing\":").Append(c.dashing ? "true" : "false");
            sb.Append(",\"recoiling\":").Append(c.recoiling ? "true" : "false");
            sb.Append(",\"touchingWall\":").Append(c.touchingWall ? "true" : "false");
            sb.Append(",\"wallSliding\":").Append(c.wallSliding ? "true" : "false");
            sb.Append(",\"invulnerable\":").Append(c.invulnerable ? "true" : "false");
            sb.Append(",\"dead\":").Append(c.dead ? "true" : "false");
            sb.Append(",\"transition\":");
            HoningVisible.Str(sb, hero.transitionState.ToString());
            sb.Append('}');
        }
        sb.Append(",\"abilities\":[");
        if (pd != null)
        {
            var have = new List<string>();
            if (pd.hasDash) have.Add("dash");
            if (pd.hasWalljump) have.Add("wall_jump");
            if (pd.hasDoubleJump) have.Add("double_jump");
            if (pd.hasSuperDash) have.Add("super_dash");
            if (pd.hasSpell) have.Add("spell");
            for (int k = 0; k < have.Count; k++)
            {
                sb.Append(k > 0 ? "," : "").Append('"').Append(have[k]).Append('"');
            }
        }
        sb.Append("],\"camera\":[");
        if (visible.HasView)
        {
            sb.Append(f(Math.Round(visible.ViewLeft, 3))).Append(',').Append(f(Math.Round(visible.ViewBottom, 3))).Append(',')
              .Append(f(Math.Round(visible.ViewRight, 3))).Append(',').Append(f(Math.Round(visible.ViewTop, 3)));
        }
        sb.Append("],\"geometry\":{\"rects\":").Append(visible.RectCount).Append(",\"rebuilds\":").Append(visible.Rebuilds).Append(",\"objects\":").Append(visible.ObjectCount).Append('}');
        // d: the camera's lock state (evaluation and plumbing only)
        CameraController cc = gm != null ? gm.cameraCtrl : null;
        if (cc != null)
        {
            sb.Append(",\"cameraLock\":{\"mode\":");
            HoningVisible.Str(sb, cc.mode.ToString());
            sb.Append(",\"lock\":[").Append(f(Math.Round(cc.xLockMin, 3))).Append(',').Append(f(Math.Round(cc.xLockMax, 3))).Append(',')
              .Append(f(Math.Round(cc.yLockMin, 3))).Append(',').Append(f(Math.Round(cc.yLockMax, 3))).Append("],\"zones\":[");
            if (cc.lockZoneList != null)
            {
                for (int k = 0; k < cc.lockZoneList.Count; k++)
                {
                    CameraLockArea z = cc.lockZoneList[k];
                    if (k > 0)
                    {
                        sb.Append(',');
                    }
                    HoningVisible.Str(sb, z == null ? "null" : z.name + "@" + z.gameObject.scene.name);
                }
            }
            sb.Append("],\"target\":");
            HoningVisible.Str(sb, cc.camTarget != null ? cc.camTarget.mode.ToString() : "");
            sb.Append(",\"fixed\":").Append(fixedLocks.Count).Append('}');
        }
        sb.Append(",\"enemies\":[");
        bool firstEnemy = true;
        Vector2 centre = new Vector2(x, y);
        foreach (HealthManager hm in visible.Ordered(FindObjectsOfType<HealthManager>()))
        {
            Vector3 p = hm.transform.position;
            if (((Vector2)p - centre).sqrMagnitude > 60f * 60f)
            {
                continue;
            }
            if (!firstEnemy)
            {
                sb.Append(',');
            }
            firstEnemy = false;
            sb.Append("{\"name\":");
            HoningVisible.Str(sb, hm.gameObject.name);
            sb.Append(",\"x\":").Append(f(Math.Round(p.x, 3))).Append(",\"y\":").Append(f(Math.Round(p.y, 3)));
            sb.Append(",\"health\":").Append(hm.hp).Append(",\"dead\":").Append(hm.isDead ? "true" : "false").Append(",\"fsm\":");
            var states = new StringBuilder();
            foreach (PlayMakerFSM fsm in hm.GetComponents<PlayMakerFSM>())
            {
                if (states.Length > 0)
                {
                    states.Append(';');
                }
                states.Append(fsm.FsmName).Append('=').Append(fsm.ActiveStateName);
            }
            HoningVisible.Str(sb, states.ToString());
            sb.Append('}');
        }
        sb.Append("],\"exits\":[]");
        // devmode: only when a developer control is in use or a dev command was just answered
        HoningDev.AppendDebug(sb, devResult);
        sb.Append('}');
        return sb.ToString();
    }

    private void OnGUI()
    {
        if (!handedOver || label == null || framesPlayed >= labelUntilFrame)
        {
            return;
        }
        if (labelStyle == null)
        {
            labelStyle = new GUIStyle(GUI.skin.box);
            labelStyle.fontSize = Mathf.Max(11, Screen.height / 45);
            labelStyle.alignment = TextAnchor.MiddleCenter;
            labelStyle.wordWrap = true;
            labelStyle.normal.textColor = Color.white;
        }
        float w = Screen.width * 0.62f;
        GUI.Box(new Rect(Screen.width - w - 8f, 8f, w, Screen.height / 9f), label, labelStyle);
    }
}

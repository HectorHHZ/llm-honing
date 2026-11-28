// DEVMODE COPY (2026-11-25): the only change to this file is that its own timeouts and timings read the engine's real
// clock through HoningClock (the game's scripts may get a virtual clock, see HoningClock.cs).
// Added by the Honing first-look task (2026-11-18). NOT part of the author's project.
//
// A small debug helper for recording demos on a headless machine. It does nothing unless the environment
// variable HONING_DEBUG_DIR is set when the player starts. When it is set, the helper
//   * writes <dir>/state.json every HONING_DEBUG_EVERY frames (scene, game state, hero position/health/flags, the
//     camera position, and the enemies and damaging objects near the hero with their positions), so that a script
//     outside the game can tell what is going on without looking at pixels;
//   * appends <dir>/events.jsonl when the active scene changes;
//   * executes commands written to <dir>/cmd.json  ({"id": N, "op": ...}; a command runs once per new id):
//       {"op":"load","scene":"Crossroads_10","gate":"left1"}  go to a room through the game's own scene-transition
//                                                            code, exactly as a door would (labelled on screen)
//       {"op":"label","text":"...","seconds":4}               show a line of text on screen (for honest recordings)
//       {"op":"screenshot","path":"/abs/file.png"}
//       {"op":"quit"}
//   * optionally fixes the time step: HONING_FIXED_FPS=60 makes every rendered frame advance the game by 1/60 s
//     (the step of a 60 Hz display), however long the frame took to render. Software rendering on a busy server is
//     far slower than real time; with a fixed step the game still plays at its true speed frame by frame.
//   * optionally saves frames for a video: with HONING_CAPTURE_DIR set it fixes the step to 1/HONING_CAPTURE_FPS
//     (default 60) and writes every HONING_CAPTURE_SKIP-th frame (default 2, i.e. a 30 fps video) as
//     <capture dir>/000000.png, ...
//   * optionally waits for the controlling script: while the file <dir>/gate exists, the game stops at the end of
//     every frame's Update, after writing state.json, until <dir>/ack.txt holds that frame number (or 10 s pass).
//     Keys the script sends while the game waits arrive at the start of the next frame, so a scripted player gets
//     the same reaction time on every run, whatever the load of the machine.
// It never changes health, damage, physics, enemies or input. A debug jump is shown on screen for a few seconds
// ("HONING DEBUG JUMP ...") so that a recording cannot be mistaken for normal door travel.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using GlobalEnums;
using UnityEngine;

public class HoningDebug : MonoBehaviour
{
    [Serializable]
    private class Command
    {
        public int id;
        public string op;
        public string scene;
        public string gate;
        public string path;
        public string text;
        public float seconds;
    }

    [Serializable]
    private class EnemyInfo
    {
        public string name;
        public float x, y;          // transform position
        public float bx, by, bw, bh; // collider bounds (centre, size); zero when there is no enabled collider
        public float scaleX;
        public int hp;
        public bool dead;
        public string fsm;          // "FsmName=ActiveState;..." of the PlayMaker state machines on the object
    }

    [Serializable]
    private class HazardInfo
    {
        public string name;
        public float x, y, w, h;    // collider bounds (centre, size)
        public int damage;
    }

    [Serializable]
    private class State
    {
        public int frame;
        public float time;
        public float realtime;
        public float timeScale;
        public int fixedFps;
        public bool gated;
        public int gateTimeouts;
        public string scene;
        public string gameState;
        public bool paused;
        public bool inSceneTransition;
        public bool finishedEnteringScene;
        public int captureIndex;
        public int lastCommandId;
        public string lastCommandResult;
        public bool hero;
        public float x, y, vx, vy;
        public bool facingRight, onGround, acceptingInput, controlRelinquished, dead, transitioning, invulnerable;
        public bool attacking, upAttacking, downAttacking, jumping, falling, dashing, recoiling, touchingWall, wallSliding;
        public bool focusing, casting, hazardDeath, hazardRespawning, dashCooldown;
        public string heroTransitionState;
        public int health, maxHealth, mp, maxMp, geo;
        public int nailDamage;
        public bool hasDash, hasWalljump, hasDoubleJump, hasSpell, hasSuperDash;
        public bool camera;
        public float camX, camY;
        public EnemyInfo[] enemies;
        public HazardInfo[] hazards;
    }

    [Serializable]
    private class SceneEvent
    {
        public int frame;
        public float time;
        public float realtime;
        public string from;
        public string to;
    }

    private const float ListRadius = 45f;
    private const int MaxListed = 24;

    private static bool booted;
    private string dir;
    private int lastId;
    private string lastResult = "";
    private int every = 6;
    private string label;
    private float labelUntil;
    private int labelUntilFrame;
    private GUIStyle labelStyle;
    private string captureDir;
    private int captureSkip = 2;
    private int captureIndex;
    private int fixedFps;
    private bool gated;
    private int gateTimeouts;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Boot()
    {
        HoningBridge.Boot();     // the model bridge (HoningBridge.cs); it does nothing unless HONING_SESSION_DIR is set
        if (booted)
        {
            return;
        }
        string d = Environment.GetEnvironmentVariable("HONING_DEBUG_DIR");
        if (string.IsNullOrEmpty(d))
        {
            return;
        }
        booted = true;
        GameObject go = new GameObject("Honing debug helper");
        DontDestroyOnLoad(go);
        go.AddComponent<HoningDebug>().dir = d;
    }

    private static int EnvInt(string name, int fallback, int min, int max)
    {
        string v = Environment.GetEnvironmentVariable(name);
        int parsed;
        if (!string.IsNullOrEmpty(v) && int.TryParse(v, out parsed) && parsed >= min && parsed <= max)
        {
            return parsed;
        }
        return fallback;
    }

    private void Start()
    {
        try
        {
            Directory.CreateDirectory(dir);
            every = EnvInt("HONING_DEBUG_EVERY", 6, 1, 600);
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged += OnSceneChanged;
            Application.runInBackground = true;
            captureDir = Environment.GetEnvironmentVariable("HONING_CAPTURE_DIR");
            if (!string.IsNullOrEmpty(captureDir))
            {
                Directory.CreateDirectory(captureDir);
                fixedFps = EnvInt("HONING_CAPTURE_FPS", 60, 10, 120);
                captureSkip = EnvInt("HONING_CAPTURE_SKIP", 2, 1, 8);
            }
            else
            {
                captureDir = null;
                fixedFps = EnvInt("HONING_FIXED_FPS", 0, 10, 120);
            }
            if (fixedFps > 0)
            {
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = -1;
                Time.captureFramerate = fixedFps;
            }
        }
        catch (Exception e)
        {
            Debug.LogError("[HoningDebug] start failed: " + e);
        }
    }

    private void LateUpdate()
    {
        if (captureDir == null)
        {
            return;
        }
        try
        {
            if (Time.frameCount % captureSkip == 0)
            {
                ScreenCapture.CaptureScreenshot(Path.Combine(captureDir, captureIndex.ToString("D06") + ".png"));
                captureIndex++;
            }
        }
        catch (Exception e)
        {
            Debug.LogError("[HoningDebug] capture failed: " + e);
        }
    }

    private void OnDestroy()
    {
        UnityEngine.SceneManagement.SceneManager.activeSceneChanged -= OnSceneChanged;
    }

    private void OnSceneChanged(UnityEngine.SceneManagement.Scene from, UnityEngine.SceneManagement.Scene to)
    {
        try
        {
            SceneEvent ev = new SceneEvent
            {
                frame = Time.frameCount, time = Time.time, realtime = HoningClock.EngineRealtimeSinceStartup,
                from = from.name, to = to.name
            };
            File.AppendAllText(Path.Combine(dir, "events.jsonl"), JsonUtility.ToJson(ev) + "\n");
        }
        catch (Exception e)
        {
            Debug.LogError("[HoningDebug] event failed: " + e);
        }
    }

    private void Update()
    {
        try
        {
            ReadCommand();
            gated = File.Exists(Path.Combine(dir, "gate"));
            if (gated || Time.frameCount % every == 0)
            {
                WriteState();
            }
            if (gated)
            {
                WaitForAck();
            }
        }
        catch (Exception e)
        {
            Debug.LogError("[HoningDebug] update failed: " + e);
        }
    }

    // Stop here until the controlling script has seen this frame's state and answered (or gave up).
    private void WaitForAck()
    {
        string ack = Path.Combine(dir, "ack.txt");
        string gate = Path.Combine(dir, "gate");
        int frame = Time.frameCount;
        float start = HoningClock.EngineRealtimeSinceStartup;
        while (true)
        {
            try
            {
                if (File.Exists(ack))
                {
                    int seen;
                    if (int.TryParse(File.ReadAllText(ack).Trim(), out seen) && seen >= frame)
                    {
                        return;
                    }
                }
            }
            catch (IOException)
            {
                // the script is writing the file right now: look again
            }
            if (!File.Exists(gate))
            {
                return;
            }
            if (HoningClock.EngineRealtimeSinceStartup - start > 10f)
            {
                gateTimeouts++;
                return;
            }
            Thread.Sleep(1);
        }
    }

    private void ShowLabel(string text, float seconds)
    {
        label = text;
        if (fixedFps > 0)
        {
            // fixed step: count in game frames, so that the text stays for `seconds` of the recorded video
            labelUntilFrame = Time.frameCount + Mathf.CeilToInt(seconds * fixedFps);
            labelUntil = float.MaxValue;
        }
        else
        {
            labelUntilFrame = int.MaxValue;
            labelUntil = HoningClock.EngineRealtimeSinceStartup + seconds;
        }
    }

    private void ReadCommand()
    {
        string path = Path.Combine(dir, "cmd.json");
        if (!File.Exists(path))
        {
            return;
        }
        Command c;
        try
        {
            c = JsonUtility.FromJson<Command>(File.ReadAllText(path));
        }
        catch (Exception)
        {
            return;   // half-written file: try again next frame
        }
        if (c == null || c.id <= lastId || string.IsNullOrEmpty(c.op))
        {
            return;
        }
        lastId = c.id;
        if (c.op == "quit")
        {
            lastResult = "quit";
            WriteState();
            Application.Quit();
        }
        else if (c.op == "screenshot")
        {
            ScreenCapture.CaptureScreenshot(c.path);
            lastResult = "screenshot " + c.path;
        }
        else if (c.op == "label")
        {
            ShowLabel(c.text ?? "", c.seconds > 0f ? c.seconds : 4f);
            lastResult = "label";
        }
        else if (c.op == "load")
        {
            GameManager gm = GameManager.UnsafeInstance;
            if (gm == null)
            {
                lastResult = "load refused: no GameManager";
            }
            else if (gm.IsInSceneTransition)
            {
                lastResult = "load refused: a scene transition is in progress";
            }
            else if (HeroController.SilentInstance == null || gm.gameState != GameState.PLAYING)
            {
                lastResult = "load refused: not in gameplay (state " + gm.gameState + ")";
            }
            else
            {
                ShowLabel("HONING DEBUG JUMP -> " + c.scene + " [" + c.gate + "]  (not a normal door)", c.seconds > 0f ? c.seconds : 5f);
                gm.BeginSceneTransition(new GameManager.SceneLoadInfo
                {
                    SceneName = c.scene,
                    EntryGateName = c.gate,
                    HeroLeaveDirection = null,
                    EntryDelay = 0f,
                    WaitForSceneTransitionCameraFade = true,
                    PreventCameraFadeOut = false,
                    Visualization = GameManager.SceneLoadVisualizations.Default,
                    AlwaysUnloadUnusedAssets = false
                });
                lastResult = "load started: " + c.scene + " " + c.gate;
            }
        }
        else
        {
            lastResult = "unknown op " + c.op;
        }
    }

    private void WriteState()
    {
        State s = new State
        {
            frame = Time.frameCount, time = Time.time, realtime = HoningClock.EngineRealtimeSinceStartup, timeScale = Time.timeScale,
            fixedFps = fixedFps, gated = gated, gateTimeouts = gateTimeouts,
            scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
            lastCommandId = lastId, lastCommandResult = lastResult, captureIndex = captureIndex
        };
        GameManager gm = GameManager.UnsafeInstance;
        if (gm != null)
        {
            s.gameState = gm.gameState.ToString();
            s.paused = gm.isPaused;
            s.inSceneTransition = gm.IsInSceneTransition;
            s.finishedEnteringScene = gm.HasFinishedEnteringScene;
            if (gm.playerData != null)
            {
                s.health = gm.playerData.health;
                s.maxHealth = gm.playerData.maxHealth;
                s.mp = gm.playerData.MPCharge;
                s.maxMp = gm.playerData.maxMP;
                s.geo = gm.playerData.geo;
                s.nailDamage = gm.playerData.nailDamage;
                s.hasDash = gm.playerData.hasDash;
                s.hasWalljump = gm.playerData.hasWalljump;
                s.hasDoubleJump = gm.playerData.hasDoubleJump;
                s.hasSpell = gm.playerData.hasSpell;
                s.hasSuperDash = gm.playerData.hasSuperDash;
            }
        }
        Camera cam = Camera.main;
        if (cam != null)
        {
            s.camera = true;
            s.camX = cam.transform.position.x;
            s.camY = cam.transform.position.y;
        }
        HeroController hero = HeroController.SilentInstance;
        Vector2 centre = cam != null ? (Vector2)cam.transform.position : Vector2.zero;
        if (hero != null)
        {
            s.hero = true;
            Vector3 p = hero.transform.position;
            centre = p;
            s.x = p.x;
            s.y = p.y;
            Rigidbody2D body = hero.GetComponent<Rigidbody2D>();
            if (body != null)
            {
                s.vx = body.velocity.x;
                s.vy = body.velocity.y;
            }
            s.acceptingInput = hero.acceptingInput;
            s.controlRelinquished = hero.controlReqlinquished;
            s.heroTransitionState = hero.transitionState.ToString();
            HeroControllerStates c = hero.cState;
            if (c != null)
            {
                s.facingRight = c.facingRight;
                s.onGround = c.onGround;
                s.dead = c.dead;
                s.transitioning = c.transitioning;
                s.invulnerable = c.invulnerable;
                s.attacking = c.attacking;
                s.upAttacking = c.upAttacking;
                s.downAttacking = c.downAttacking;
                s.jumping = c.jumping;
                s.falling = c.falling;
                s.dashing = c.dashing;
                s.recoiling = c.recoiling;
                s.touchingWall = c.touchingWall;
                s.wallSliding = c.wallSliding;
                s.focusing = c.focusing;
                s.casting = c.casting;
                s.hazardDeath = c.hazardDeath;
                s.hazardRespawning = c.hazardRespawning;
                s.dashCooldown = c.dashCooldown;
            }
        }
        if (hero != null || cam != null)
        {
            ListNearby(s, centre);
        }
        string path = Path.Combine(dir, "state.json");
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonUtility.ToJson(s));
        if (File.Exists(path))
        {
            File.Delete(path);
        }
        File.Move(tmp, path);
    }

    // Enemies (objects with a HealthManager) and damaging objects (DamageHero with an enabled collider) near the hero.
    private void ListNearby(State s, Vector2 centre)
    {
        var enemies = new List<EnemyInfo>();
        foreach (HealthManager hm in FindObjectsOfType<HealthManager>())
        {
            Vector3 p = hm.transform.position;
            if (((Vector2)p - centre).sqrMagnitude > ListRadius * ListRadius)
            {
                continue;
            }
            var info = new EnemyInfo
            {
                name = hm.gameObject.name, x = p.x, y = p.y, scaleX = hm.transform.lossyScale.x, hp = hm.hp, dead = hm.isDead
            };
            Collider2D col = hm.GetComponent<Collider2D>();
            if (col != null && col.enabled)
            {
                Bounds b = col.bounds;
                info.bx = b.center.x;
                info.by = b.center.y;
                info.bw = b.size.x;
                info.bh = b.size.y;
            }
            var sb = new StringBuilder();
            foreach (PlayMakerFSM fsm in hm.GetComponents<PlayMakerFSM>())
            {
                if (sb.Length > 0)
                {
                    sb.Append(';');
                }
                sb.Append(fsm.FsmName).Append('=').Append(fsm.ActiveStateName);
            }
            info.fsm = sb.ToString();
            enemies.Add(info);
            if (enemies.Count >= MaxListed)
            {
                break;
            }
        }
        s.enemies = enemies.ToArray();

        var hazards = new List<KeyValuePair<float, HazardInfo>>();
        foreach (DamageHero dh in FindObjectsOfType<DamageHero>())
        {
            if (dh.damageDealt <= 0)
            {
                continue;
            }
            Collider2D col = dh.GetComponent<Collider2D>();
            if (col == null || !col.enabled)
            {
                continue;
            }
            Bounds b = col.bounds;
            float d = ((Vector2)b.ClosestPoint(centre) - centre).sqrMagnitude;
            if (d > ListRadius * ListRadius)
            {
                continue;
            }
            hazards.Add(new KeyValuePair<float, HazardInfo>(d, new HazardInfo
            {
                name = dh.gameObject.name, x = b.center.x, y = b.center.y, w = b.size.x, h = b.size.y, damage = dh.damageDealt
            }));
        }
        hazards.Sort((a, b) => a.Key.CompareTo(b.Key));
        int n = Mathf.Min(hazards.Count, MaxListed);
        s.hazards = new HazardInfo[n];
        for (int i = 0; i < n; i++)
        {
            s.hazards[i] = hazards[i].Value;
        }
    }

    private void OnGUI()
    {
        if (label == null || Time.frameCount > labelUntilFrame || HoningClock.EngineRealtimeSinceStartup > labelUntil)
        {
            return;
        }
        if (labelStyle == null)
        {
            labelStyle = new GUIStyle(GUI.skin.box);
            labelStyle.fontSize = 17;
            labelStyle.alignment = TextAnchor.MiddleCenter;
            labelStyle.wordWrap = false;
            labelStyle.normal.textColor = Color.white;
        }
        float w = Mathf.Min(Screen.width - 16f, labelStyle.CalcSize(new GUIContent(label)).x + 28f);
        GUI.Box(new Rect(Screen.width - w - 8f, 8f, w, 32f), label, labelStyle);
    }
}

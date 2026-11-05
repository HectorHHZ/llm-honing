using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

// Private smoke-test transport. The original Unity gameplay owns all rules.
[DefaultExecutionOrder(-32000)]
public class HoningBridge : MonoBehaviour
{
    public static bool Running = false;
    private static HoningBridge instance;
    public static ActionData Action = new ActionData();
    public static bool FirstFrame = false;
    private string directory;
    private int remaining;
    private int bridgeFrame;
    private Request pending;
    private bool loading;
    private bool firstUpdate = true;
    private string framesDirectory;
    private RenderTexture renderTarget;
    private Texture2D readback;

    [Serializable] public class ActionData {
        public float[] move = new float[] { 0, 0 };
        public float[] aim = new float[] { 1, 0 };
        public bool attack, interact, skill, switch_weapon;
    }
    [Serializable] public class Request {
        public string id, op;
        public int frames = 10, seed = 7;
        public ActionData action;
    }
    [Serializable] public class SceneData { public string name; public int index; }
    [Serializable] public class Actor {
        public int id, hp, max_hp, defence, energy;
        public string name, tag, weapon;
        public float x, y, vx, vy, speed;
        public bool dead, active, elite;
    }
    [Serializable] public class Projectile {
        public int id;
        public string name, tag;
        public float x, y, vx, vy;
    }
    [Serializable] public class Landmark {
        public string name, tag;
        public float x, y;
        public bool active;
    }
    [Serializable] public class Progress {
        public bool first_stage_end, game_over, player_in_main_room, player_in_boss_room;
        public int enemies_one, enemies_two;
    }
    [Serializable] public class Snapshot {
        public string schema = "honing.soul-knight.native.v1";
        public string native_frame_path;
        public int bridge_frame, unity_frame, advanced_frames;
        public float sim_time, fixed_delta_time, delta_time;
        public bool paused, terminal, won;
        public SceneData scene;
        public Actor player;
        public Actor[] enemies;
        public Projectile[] projectiles;
        public Landmark[] landmarks;
        public Progress progress;
        public ActionData executed_action;
    }
    [Serializable] public class Response {
        public string id;
        public bool ok;
        public string error;
        public Snapshot state;
    }

    public static void Bootstrap() {
        if (instance != null) return;
        GameObject obj = new GameObject("HoningSoulKnightBridge");
        UnityEngine.Object.DontDestroyOnLoad(obj);
        instance = obj.AddComponent<HoningBridge>();
    }

    void Awake() {
        directory = Environment.GetEnvironmentVariable("HONING_SOUL_KNIGHT_IPC");
        if (String.IsNullOrEmpty(directory)) throw new InvalidOperationException("HONING_SOUL_KNIGHT_IPC is required");
        Directory.CreateDirectory(directory);
        framesDirectory = Path.Combine(directory, "frames");
        Directory.CreateDirectory(framesDirectory);
        Application.runInBackground = true;
        Application.targetFrameRate = 50;
        QualitySettings.vSyncCount = 0;
        // Fixed rendered-frame duration is a smoke default, not a replay guarantee.
        Time.captureFramerate = 50;
        // Let the engine finish its native splash before the first Update.
        Time.timeScale = 1;
        StartCoroutine(EndFrames());
        Debug.Log("HONING native bridge ready: " + directory);
    }

    void Update() {
        if (firstUpdate) { Debug.Log("HONING first Update frame=" + Time.frameCount); firstUpdate = false; Time.timeScale = 0; }
        if (loading) return;
        if (pending != null) return;
        string path = Path.Combine(directory, "request.json");
        if (!File.Exists(path)) return;
        Request req = null;
        try {
            req = JsonUtility.FromJson<Request>(File.ReadAllText(path));
            File.Delete(path);
            if (req == null || String.IsNullOrEmpty(req.id)) throw new Exception("Invalid request");
            if (req.op == "state") { Reply(req, 0); return; }
            if (req.op == "reset") {
                pending = req;
                loading = true;
                Running = false;
                Time.timeScale = 0;
                UnityEngine.Random.InitState(req.seed);
                StartCoroutine(ResetScene());
                return;
            }
            if (req.op != "step") throw new Exception("Unknown operation");
            if (req.frames < 1 || req.frames > 250) throw new Exception("frames must be 1..250");
            ActionData action = req.action ?? new ActionData();
            ValidateAction(action);
            Action = action;
            pending = req;
            remaining = req.frames;
            FirstFrame = true;
            Running = true;
            Time.timeScale = 1;
        } catch (Exception ex) {
            if (File.Exists(path)) File.Delete(path);
            WriteResponse(new Response { id = req == null ? "invalid" : req.id, ok = false, error = ex.ToString() });
        }
    }

    private static void ValidateAction(ActionData action) {
        if (action.move == null || action.move.Length != 2 || action.aim == null || action.aim.Length != 2)
            throw new Exception("move and aim must contain two numbers");
        foreach (float value in action.move)
            if (value != -1 && value != 0 && value != 1) throw new Exception("Invalid move component");
        foreach (float value in action.aim)
            if (Single.IsNaN(value) || Single.IsInfinity(value) || Math.Abs(value) > 1000000)
                throw new Exception("Aim must be finite and within[-1000000,1000000] world units");
    }

    IEnumerator ResetScene() {
        foreach (string png in Directory.GetFiles(framesDirectory, "frame_*.png")) File.Delete(png);
        AsyncOperation op = SceneManager.LoadSceneAsync(1);
        while (!op.isDone) yield return null;
        bridgeFrame = 0;
        // Awake/Start setup is allowed; normal Update/FixedUpdate remains guarded.
        yield return null;
        yield return new WaitForEndOfFrame();
        loading = false;
        Running = false;
        Time.timeScale = 0;
        Action = new ActionData();
        CaptureFrame();
        Request req = pending;
        pending = null;
        Reply(req, 0);
    }

    IEnumerator EndFrames() {
        while (true) {
            yield return new WaitForEndOfFrame();
            if (!Running || pending == null || loading) continue;
            bridgeFrame++;
            CaptureFrame();
            remaining--;
            FirstFrame = false;
            if (remaining <= 0) {
                Running = false;
                Time.timeScale = 0;
                Request req = pending;
                pending = null;
                Reply(req, req.frames);
            }
        }
    }

    private Actor ReadActor(CharacterStats actor) {
        Rigidbody2D rb = actor.GetComponent<Rigidbody2D>();
        WeaponController weapon = actor.GetWeapon();
        EnemyStats enemy = actor as EnemyStats;
        PlayerController controller = actor.GetComponent<PlayerController>();
        return new Actor {
            id = actor.GetInstanceID(), name = actor.name, tag = actor.tag,
            x = actor.transform.position.x, y = actor.transform.position.y,
            vx = rb == null ? 0 : rb.velocity.x, vy = rb == null ? 0 : rb.velocity.y,
            hp = actor.CurrentHealth, max_hp = actor.MaxHealth, defence = actor.CurrentDefence,
            energy = actor.CurrentEnergy, dead = actor.isDead,
            active = actor.gameObject.activeInHierarchy, elite = enemy != null && enemy.isElite,
            speed = controller == null ? 0 : controller.speed,
            weapon = weapon == null ? "" : weapon.weaponData.name
        };
    }

    private Snapshot ReadState(int advanced) {
        PlayerStats player = UnityEngine.Object.FindObjectOfType<PlayerStats>();
        List<Actor> actors = new List<Actor>();
        foreach (EnemyStats enemy in Resources.FindObjectsOfTypeAll<EnemyStats>())
            if (enemy.gameObject.scene.IsValid()) actors.Add(ReadActor(enemy));
        foreach (BossStats boss in Resources.FindObjectsOfTypeAll<BossStats>())
            if (boss.gameObject.scene.IsValid()) actors.Add(ReadActor(boss));
        actors.Sort((a, b) => a.id.CompareTo(b.id));
        List<Projectile> projectiles = new List<Projectile>();
        foreach (BulletController bullet in UnityEngine.Object.FindObjectsOfType<BulletController>()) {
            Rigidbody2D rb = bullet.rb;
            projectiles.Add(new Projectile { id = bullet.GetInstanceID(), name = bullet.name, tag = bullet.tag,
                x = bullet.transform.position.x, y = bullet.transform.position.y,
                vx = rb == null ? 0 : rb.velocity.x, vy = rb == null ? 0 : rb.velocity.y });
        }
        List<Landmark> landmarks = new List<Landmark>();
        foreach (Transform tr in Resources.FindObjectsOfTypeAll<Transform>()) {
            if (!tr.gameObject.scene.IsValid()) continue;
            string lower = tr.name.ToLowerInvariant();
            if (lower.Contains("door") || lower.Contains("water") || lower.Contains("rewardbox"))
                landmarks.Add(new Landmark { name = tr.name, tag = tr.tag, x = tr.position.x, y = tr.position.y,
                    active = tr.gameObject.activeInHierarchy });
        }
        GameManager manager = UnityEngine.Object.FindObjectOfType<GameManager>();
        Progress progress = manager == null ? new Progress() : new Progress {
            first_stage_end = manager.isFirstStageEnd, game_over = manager.isGameOver,
            player_in_main_room = manager.isPlayerInMainRoom, player_in_boss_room = manager.isPlayerInBossRoom,
            enemies_one = manager.enemiesOne.Count, enemies_two = manager.enemiesTwo.Count };
        Scene scene = SceneManager.GetActiveScene();
        return new Snapshot {
            native_frame_path = Path.Combine(framesDirectory, "frame_" + bridgeFrame.ToString("D6") + ".png"),
            bridge_frame = bridgeFrame, unity_frame = Time.frameCount, advanced_frames = advanced,
            sim_time = Time.time, fixed_delta_time = Time.fixedDeltaTime, delta_time = Time.deltaTime,
            paused = !Running, scene = new SceneData { name = scene.name, index = scene.buildIndex },
            player = player == null ? null : ReadActor(player), enemies = actors.ToArray(),
            projectiles = projectiles.ToArray(), landmarks = landmarks.ToArray(), progress = progress,
            executed_action = Action,
            terminal = scene.buildIndex == 2 || (player != null && (player.isDead || progress.game_over)),
            won = progress.game_over || scene.buildIndex == 2
        };
    }

    private void CaptureFrame() {
        // Real original-engine offscreen rendering, including original HUD.
        Camera camera = Camera.main;
        if (camera == null) return;
        if (renderTarget == null) {
            renderTarget = new RenderTexture(960, 640, 24, RenderTextureFormat.ARGB32);
            readback = new Texture2D(960, 640, TextureFormat.RGB24, false);
        }
        foreach (Canvas canvas in UnityEngine.Object.FindObjectsOfType<Canvas>()) {
            if (canvas.renderMode == RenderMode.ScreenSpaceOverlay) {
                canvas.renderMode = RenderMode.ScreenSpaceCamera;
                canvas.worldCamera = camera;
                canvas.planeDistance = 1;
            }
        }
        Canvas.ForceUpdateCanvases();
        RenderTexture previous = camera.targetTexture;
        RenderTexture active = RenderTexture.active;
        camera.targetTexture = renderTarget;
        camera.Render();
        RenderTexture.active = renderTarget;
        readback.ReadPixels(new Rect(0, 0, 960, 640), 0, 0);
        readback.Apply();
        byte[] png = readback.EncodeToPNG();
        string path = Path.Combine(framesDirectory, "frame_" + bridgeFrame.ToString("D6") + ".png");
        File.WriteAllBytes(path + ".tmp", png);
        if (File.Exists(path)) File.Delete(path);
        File.Move(path + ".tmp", path);
        camera.targetTexture = previous;
        RenderTexture.active = active;
    }

    private void Reply(Request request, int advanced) {
        try { WriteResponse(new Response { id = request.id, ok = true, state = ReadState(advanced) }); }
        catch (Exception ex) { WriteResponse(new Response { id = request.id, ok = false, error = ex.ToString() }); }
    }
    private void WriteResponse(Response response) {
        string path = Path.Combine(directory, "response.json");
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonUtility.ToJson(response));
        if (File.Exists(path)) File.Delete(path);
        File.Move(tmp, path);
    }
}

public static class HoningInput {
    public static float GetAxisRaw(string name) {
        if (!HoningBridge.Running) return 0;
        if (name == "Horizontal") return HoningBridge.Action.move[0];
        if (name == "Vertical") return HoningBridge.Action.move[1];
        return GetAxis(name);
    }
    public static float GetAxis(string name) {
        return HoningBridge.Running && HoningBridge.FirstFrame && name == "Mouse ScrollWheel" && HoningBridge.Action.switch_weapon ? 1 : 0;
    }
    public static bool GetMouseButtonDown(int button) {
        return HoningBridge.Running && HoningBridge.FirstFrame && (button == 0 ? HoningBridge.Action.attack : button == 1 && HoningBridge.Action.skill);
    }
    public static bool GetKeyDown(KeyCode code) {
        return HoningBridge.Running && HoningBridge.FirstFrame && code == KeyCode.Space && HoningBridge.Action.interact;
    }
    public static Vector3 get_mousePosition() {
        Camera camera = Camera.main;
        if (camera == null) return Vector3.zero;
        return camera.WorldToScreenPoint(new Vector3(HoningBridge.Action.aim[0], HoningBridge.Action.aim[1], 0));
    }
}

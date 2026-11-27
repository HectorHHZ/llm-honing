using System;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

// A demo input adapter. The original controller still executes every movement,
// attack, cooldown and collision; the model only selects a discrete input.
public static class HoningInput
{
    public static bool Active;
    public static string Action = "wait";
    static bool edge;
    public static void Set(string action) { Action = action; edge = true; }
    public static void ConsumeEdges() { edge = false; }
    public static float GetAxis(string axis)
    {
        if (!Active) return Input.GetAxis(axis);
        if (axis == "Horizontal")
            return Action.EndsWith("left") ? -1 : Action.EndsWith("right") ? 1 : 0;
        if (axis == "Vertical")
            return Action == "attack_up" ? 1 : Action == "attack_down" ? -1 : 0;
        return 0;
    }
    public static bool GetButtonDown(string button)
    { return Active ? edge && button == "Jump" && Action.StartsWith("jump_") : Input.GetButtonDown(button); }
    public static bool GetButtonUp(string button)
    { return Active ? false : Input.GetButtonUp(button); }
    public static bool GetKeyDown(KeyCode key)
    {
        if (!Active) return Input.GetKeyDown(key);
        return edge && ((key == KeyCode.J && Action.StartsWith("attack_")) ||
                        (key == KeyCode.K && Action.StartsWith("dash_")));
    }
}

[DefaultExecutionOrder(-200)]
public class HoningBridge : MonoBehaviour
{
    [Serializable] public class Actor { public string name; public float x, y; public int health; }
    [Serializable] public class Exit { public float x, y, width, height; }
    [Serializable] public class Observation
    {
        public int id, frame, health, jumps, fixedSteps;
        public string scene;
        public float x, y, vx, vy, wallLeft, wallRight, simulationTime;
        public bool grounded, floorLeft, floorRight;
        public Actor[] enemies;
        public Exit[] exits;
    }
    [Serializable] public class Command
    { public int id; public string action; public int frames; public string model; public float probability; }
    static readonly string[] Actions = { "wait", "left", "right", "jump_left", "jump_right",
        "dash_left", "dash_right", "attack_left", "attack_right", "attack_up", "attack_down" };
    static bool booted;
    string session, modelLabel = "Waiting for a model";
    PlayerController player;
    int sequence, frame, remaining, fixedSteps;
    bool capture, ready, needsObservation, awaitingRenderedFrame, visibleProfile;
    float waitStart;
    double modelTimeoutSeconds = 180;

    public static void Boot()
    {
        if (booted || String.IsNullOrEmpty(Environment.GetEnvironmentVariable("HONING_SESSION_DIR"))) return;
        booted = true;
        var obj = new GameObject("Honing model bridge");
        DontDestroyOnLoad(obj);
        obj.AddComponent<HoningBridge>();
    }

    System.Collections.IEnumerator Start()
    {
        session = Environment.GetEnvironmentVariable("HONING_SESSION_DIR");
        modelTimeoutSeconds = Double.Parse(Environment.GetEnvironmentVariable("HONING_MODEL_TIMEOUT_SECONDS") ?? "180",
                                          CultureInfo.InvariantCulture);
        Directory.CreateDirectory(session);
        capture = Environment.GetEnvironmentVariable("HONING_CAPTURE") == "1";
        // Only the exact value "visible" selects the screen-only observation; anything else keeps the scene state.
        visibleProfile = Environment.GetEnvironmentVariable("HONING_OBSERVATION_PROFILE") == "visible";
        if (capture) Directory.CreateDirectory(Path.Combine(session, "frames"));
        Application.runInBackground = true;
        Application.targetFrameRate = 30;
        QualitySettings.vSyncCount = 0;
        Time.captureFramerate = 30;
        UnityEngine.Random.InitState(42);
        HoningInput.Active = true;
        string scene = Environment.GetEnvironmentVariable("HONING_SCENE") ?? "Spawn";
        if (SceneManager.GetActiveScene().name != scene) SceneManager.LoadScene(scene);
        yield return null;
        yield return null;
        if (visibleProfile) yield return new WaitForEndOfFrame();
        ready = true;
        Observe();
    }

    void WriteJson(string name, object value)
    {
        string path = Path.Combine(session, name), temp = path + ".tmp";
        File.WriteAllText(temp, JsonUtility.ToJson(value));
        if (File.Exists(path)) File.Delete(path);
        File.Move(temp, path);
    }

    bool BindPlayer()
    {
        if (player == null) player = FindObjectOfType<PlayerController>();
        return player != null;
    }

    void Observe()
    {
        Time.timeScale = 0;
        HoningInput.Set("wait");
        HoningInput.ConsumeEdges();
        waitStart = Time.realtimeSinceStartup;
        needsObservation = true;
        if (!BindPlayer()) return;
        needsObservation = false;
        var position = player.transform.position;
        var body = player.GetComponent<Rigidbody2D>();
        int terrain = LayerMask.GetMask("Platform", "Wall", "AirWall");
        var obs = new Observation {
            id = ++sequence, frame = frame, scene = SceneManager.GetActiveScene().name,
            simulationTime = Time.time, fixedSteps = fixedSteps,
            health = player.health, jumps = player.jumpLeft,
            x = position.x, y = position.y, vx = body.velocity.x, vy = body.velocity.y,
            grounded = Physics2D.CircleCast(position, .2f, Vector2.down, .5f, LayerMask.GetMask("Platform")).collider != null,
            floorLeft = Physics2D.Raycast(position + Vector3.left * 1.2f, Vector2.down, 2f, terrain).collider != null,
            floorRight = Physics2D.Raycast(position + Vector3.right * 1.2f, Vector2.down, 2f, terrain).collider != null,
            wallLeft = WallDistance(position, Vector2.left, terrain),
            wallRight = WallDistance(position, Vector2.right, terrain),
            enemies = FindObjectsOfType<EnemyController>()
                .OrderBy(e => Vector2.Distance(e.transform.position, position)).Take(6)
                .Select(e => new Actor { name = e.name, x = e.transform.position.x, y = e.transform.position.y, health = e.health }).ToArray(),
            exits = FindObjectsOfType<NextLevel>().Select(e => e.GetComponent<Collider2D>().bounds)
                .Select(b => new Exit { x = b.center.x, y = b.center.y, width = b.size.x, height = b.size.y }).ToArray()
        };
        if (visibleProfile) {
            var visible = VisibleObservation.Capture(player, sequence, frame, capture, session);
            WriteJson("observation.json", visible);
            WriteJson("debug_observation.json", obs);
            File.AppendAllText(Path.Combine(session, "visible_observations.jsonl"), JsonUtility.ToJson(visible) + "\n");
            if (sequence == 1) File.WriteAllText(Path.Combine(session, "scene-inspection.txt"), InspectScene());
        } else {
            WriteJson("observation.json", obs);
        }
        File.AppendAllText(Path.Combine(session, "observations.jsonl"), JsonUtility.ToJson(obs) + "\n");
    }

    // Diagnostic inventory of the first observed scene; it is written next to the trace and never sent to a model.
    static string InspectScene()
    {
        var b = new System.Text.StringBuilder();
        foreach (var c in FindObjectsOfType<Camera>()) b.AppendLine("CAM " + c.name + " pos=" + c.transform.position + " ortho=" + c.orthographic + " size=" + c.orthographicSize + " rect=" + c.pixelRect + " depth=" + c.depth + " mask=" + c.cullingMask);
        foreach (var r in FindObjectsOfType<Renderer>()) {
            b.AppendLine("RENDER " + r.GetType().Name + " name=" + r.name + " layer=" + LayerMask.LayerToName(r.gameObject.layer) + " enabled=" + r.enabled + " bounds=" + r.bounds + " sorting=" + r.sortingLayerName + "/" + r.sortingOrder + " material=" + (r.sharedMaterial == null ? "null" : r.sharedMaterial.shader.name));
            var sr = r as SpriteRenderer;
            if(sr != null) b.AppendLine(" SPRITE " + (sr.sprite == null ? "null" : sr.sprite.name + " read=" + sr.sprite.texture.isReadable + " tex=" + sr.sprite.texture.name + " rect=" + sr.sprite.rect + " size=" + sr.sprite.bounds + " mode=" + sr.drawMode) + " color=" + sr.color + " flip=" + sr.flipX + "," + sr.flipY);
        }
        foreach(var c in FindObjectsOfType<Collider2D>()) b.AppendLine("COL " + c.GetType().Name + " " + c.name + " layer=" + LayerMask.LayerToName(c.gameObject.layer) + " tag=" + c.tag + " trigger=" + c.isTrigger + " bounds=" + c.bounds + " render=" + (c.GetComponent<Renderer>() != null));
        return b.ToString();
    }

    static float WallDistance(Vector2 origin, Vector2 direction, int mask)
    { var hit = Physics2D.Raycast(origin, direction, 2f, mask); return hit.collider == null ? 2f : hit.distance; }

    void FixedUpdate() { if (ready && remaining > 0) fixedSteps++; }

    void Update()
    {
        if (!ready || remaining > 0 || awaitingRenderedFrame) return;
        if (needsObservation) { if (BindPlayer()) Observe(); return; }
        string path = Path.Combine(session, "action.json");
        if (File.Exists(path)) {
            Command command;
            try { command = JsonUtility.FromJson<Command>(File.ReadAllText(path)); }
            catch (Exception) { command = null; }
            if (command != null && command.id == sequence) {
                if (command.action == "stop") { Application.Quit(); ready = false; return; }
                if (Actions.Contains(command.action) && command.frames >= 1 && command.frames <= 120) {
                    modelLabel = command.model + " | " + command.action + " | p=" + command.probability.ToString("F3");
                    remaining = command.frames;
                    HoningInput.Set(command.action);
                    Time.timeScale = 1;
                }
            }
        }
        if (Time.realtimeSinceStartup - waitStart > modelTimeoutSeconds) {
            Debug.LogError("Honing bridge timed out waiting for a model response.");
            Application.Quit();
        }
    }

    void LateUpdate()
    {
        if (!ready || remaining <= 0) return;
        HoningInput.ConsumeEdges();
        if (capture) ScreenCapture.CaptureScreenshot(Path.Combine(session, "frames", frame.ToString("D06") + ".png"));
        frame++;
        remaining--;
        if (remaining == 0) {
            if (visibleProfile) { Time.timeScale = 0; awaitingRenderedFrame = true; StartCoroutine(ObserveAfterRender()); }
            else Observe();
        }
    }

    // The visible profile labels what the camera drew, so it observes after this frame has been rendered.
    System.Collections.IEnumerator ObserveAfterRender()
    {
        yield return new WaitForEndOfFrame();
        Observe();
        awaitingRenderedFrame = false;
    }

    void OnGUI()
    {
        if (!capture) return;
        GUI.Box(new Rect(8, 8, 660, 48), modelLabel + "\nModel input demo | Unity runtime | frame " + frame);
    }
}

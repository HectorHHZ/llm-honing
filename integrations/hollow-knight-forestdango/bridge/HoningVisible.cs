// DEVMODE COPY (2026-11-25): the only change to this file is that its own timeouts and timings read the engine's real
// clock through HoningClock (the game's scripts may get a virtual clock, see HoningClock.cs).
// Added by the Honing task (2026-11-18, phase 2). NOT part of the author's project. Used only by HoningBridge.
//
// The human-visible observation (schema honing.visible.v1) of the Honing bridge, observation mode "geometry":
// the game's own collision shapes that lie inside the camera's view are projected onto a screen grid. No pixels.
//
// What is reported, and from what:
//   terrain      colliders on the layer "Terrain" that are not triggers: the tile map's outline loops and the
//                polygon colliders are rasterised once per room (HONING_GEOM_STEP world units per raster cell) and
//                cut into rectangles whose top is a real surface and whose bottom is a real ceiling; box colliders
//                (floating platforms, doors, gates, lifts) are taken with their current bounds at every observation;
//   interactable a solid box that belongs to a Breakable on the door list (DoorNames; e: only objects that look like
//                doors, on the route King's Pass's "Tute Door"s): look "door";
//   hazards      enabled colliders of DamageHero components that do not belong to an enemy's body (spikes, shots,
//                shock waves, an enemy's weapon);
//   enemies      objects with a HealthManager that are alive, with their collider's bounds;
//   player       the knight's body collider;
//   HUD          health, maximum health, soul, geo (shown on the game's HUD).
// Names (e, 2026-11-19, the user's decision on leaks A1/A2): nothing in the observation carries a game-internal object
// or room name. `look` is a kind word: "terrain" (raster rectangles), "wall" (every other solid box, including a
// Breakable that is not on the door list, whatever it is called in the game), "enemy" (every enemy, the boss too),
// "hazard" (damaging things), "spikes" / "acid" (the game's own hazard types 2 and 3, which a player recognises on
// sight), "door" (door list only), "knight" (the player). Object ids are "<room code>/<letter><number>"; the room code
// is an opaque hash of the scene name (RoomToken), the same on every visit. debug_observation.json keeps every name.
// Bridge d and earlier used the objects' names ("collapser tute", "break floor", "false knight new", "falling
// barrel", ...) and called every Breakable solid "door".
// Everything is clipped to the rectangle of the game plane (z = 0) that the real camera shows; cells of the
// observation's screen outside that rectangle are "unknown" and carry no objects.
// Screen-space UI: every renderer the HUD camera draws (health masks, soul vessel, geo counter, area title cards,
// dialogue boxes, inventory) and every canvas graphic (menus) with a visible alpha is projected onto the screen as
// its bounding rectangle. Cells it touches get the unknown bit; cells wholly inside one such rectangle are unknown
// only; an actor, hazard or door wholly inside one is not reported. The rectangles include the sprites' transparent
// padding, so a little more than what is really covered is hidden (the conservative side).
//
// Where this mode can say MORE than a player sees (kept in the envelope under disclosure.can_exceed_human_view):
//   * collision that has no art (invisible walls), or art that hides an opening (secret rooms behind foreground
//     rock: the collision outline is reported as it is, so a hidden passage shows as open space) - none found on the
//     route (King's Pass, Dirtmouth, Forgotten Crossroads, the False Knight room);
//   * darkness: rooms that are dark without the lantern are reported like lit rooms - none on the route;
//   * a collider that is active while its sprite is not yet drawn (the sprite renderer's "enabled" flag is checked;
//     fades and off-screen sprite parts are not) - none seen on the route;
//   * ACCEPTED by the user on 2026-11-19 ("what a human player can see and understand is fair to give"), kept as is:
//     - no occlusion: things covered by foreground scenery, fog or particles are reported (route: the knight walking
//       behind the foreground rock at the King's Pass start; fog in Dirtmouth);
//     - decoration that looks dangerous is not reported as a hazard (route: the white grass in the King's Pass pit,
//       the iron spears at the Crossroads_01 entrance);
//     - exact geometry: raster rectangles and collider bounds, more precise than the picture (everywhere).
//   (The disclosure in the envelope lists these properties without the route places.)
// Where it says LESS: scenery, background, the look of things beyond a kind word, particles, map and menus.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using GlobalEnums;
using UnityEngine;

public class HoningVisible
{
    public const int Bg = 1, Wall = 2, Plat = 4, Haz = 8, Actor = 16, Inter = 32, Unk = 64, Scen = 128;
    private const int NominalW = 960, NominalH = 540;   // the game window as the observation sees it (16:9)
    private const int TerrainLayer = 8;

    // knobs
    public int cell = 20;
    public bool anchored;
    public float anchorX, anchorY, zoom = 1f;
    public int viewW = NominalW, viewH = NominalH;
    public float step = 0.5f;

    // static geometry of the current room
    private string geomKey = "";
    private float ox, oy;
    private int nx, ny;
    private bool[] solid = new bool[0];
    private readonly List<float[]> rects = new List<float[]>();
    private readonly HashSet<int> staticIds = new HashSet<int>();
    public int Rebuilds;
    public double LastBuildMs;

    // the last view (for the debug observation and tools)
    public float ViewLeft, ViewRight, ViewBottom, ViewTop, Ppu, OriginX, OriginY, FeetX, FeetY;
    public bool HasView;
    private float lastFeetX, lastFeetY;
    private bool haveFeet;

    private class Obj
    {
        public string id, kind, look;
        public int x, y, w, h;
    }

    public static string F(double v)
    {
        if (double.IsNaN(v) || double.IsInfinity(v))
        {
            return "0";
        }
        return v.ToString("0.#######", CultureInfo.InvariantCulture);
    }

    public static void Str(StringBuilder sb, string s)
    {
        sb.Append('"');
        if (s != null)
        {
            foreach (char c in s)
            {
                if (c == '"')
                {
                    sb.Append("\\\"");
                }
                else if (c == '\\')
                {
                    sb.Append("\\\\");
                }
                else if (c == '\n')
                {
                    sb.Append("\\n");
                }
                else if (c < 32)
                {
                    sb.Append(' ');
                }
                else
                {
                    sb.Append(c);
                }
            }
        }
        sb.Append('"');
    }

    public static string CleanName(string name)
    {
        if (name == null)
        {
            return "";
        }
        string n = name.Replace("(Clone)", "");
        int cut = n.IndexOf('(');
        if (cut > 0)
        {
            n = n.Substring(0, cut);
        }
        n = n.Trim();
        while (n.Length > 0 && (char.IsDigit(n[n.Length - 1]) || n[n.Length - 1] == ' ' || n[n.Length - 1] == '_'))
        {
            n = n.Substring(0, n.Length - 1);
        }
        return n.ToLowerInvariant();
    }

    // e (2026-11-19): an object id is "<room token>/<letter><number>". The room token is an opaque code of the room's
    // scene name ("room-" + 8 hex digits of its FNV-1a hash), the same code on every visit, so ids stay one-to-one with
    // bridge d's "<scene>/..." ids while the observation no longer carries the room's internal name.
    private static readonly Dictionary<string, string> roomTokens = new Dictionary<string, string>();

    public static string RoomToken(string scene)
    {
        string token;
        if (scene == null)
        {
            scene = "";
        }
        if (!roomTokens.TryGetValue(scene, out token))
        {
            uint h = 2166136261;
            foreach (char ch in scene)
            {
                h = (h ^ ch) * 16777619;
            }
            token = "room-" + h.ToString("x8");
            roomTokens[scene] = token;
        }
        return token;
    }

    // e (2026-11-19, the user's decision on leak A2): the objects that look like doors on screen. A solid that belongs to
    // a Breakable is reported as "interactable" / "door" only when the Breakable's (cleaned) name is in this list; any
    // other Breakable solid is reported as the plain solid it looks like ("wall" / "wall"), and nothing says it breaks.
    // On the route the list matches King's Pass's seven "Tute Door" instances (see LEAK_LIST.md for their places).
    public static readonly HashSet<string> DoorNames = new HashSet<string> { "tute door" };

    private readonly HashSet<int> loggedBreakables = new HashSet<int>();
    public readonly List<string> NameLog = new List<string>();     // drained by the bridge into bridge.log

    private static GameCameras cameras;

    public static Camera GameCamera()
    {
        if (cameras == null)
        {
            cameras = UnityEngine.Object.FindObjectOfType<GameCameras>();
        }
        if (cameras != null && cameras.mainCamera != null)
        {
            return cameras.mainCamera;
        }
        return Camera.main;
    }

    // ---- devmode (HONING_IDS=stable; default "instance" = bridge e): ids and order that do not depend on the engine's
    // numbering of objects made at run time. That numbering differs by a constant between a game that draws its
    // frames and one that does not (drawing itself makes a few engine objects), so an object spawned during play (a
    // boss's shock wave) would get another id. With "stable" an object made at run time is named by the order in
    // which this observer first saw it in the room ("s1", "s2", ...), objects of the room's own file keep their
    // number, and lists are put in the order of the engine's numbers (which is the same order in both kinds of game).
    public bool stableIds;
    private readonly Dictionary<int, int> runtimeIds = new Dictionary<int, int>();
    private string runtimeIdRoom = "";

    private string IdOf(int instanceId, string room)
    {
        if (!stableIds || instanceId >= 0)
        {
            return Math.Abs(instanceId).ToString();
        }
        if (room != runtimeIdRoom)
        {
            runtimeIds.Clear();
            runtimeIdRoom = room;
        }
        int k;
        if (!runtimeIds.TryGetValue(instanceId, out k))
        {
            k = runtimeIds.Count + 1;
            runtimeIds[instanceId] = k;
        }
        return "s" + k;
    }

    public T[] Ordered<T>(T[] items) where T : Component
    {
        if (!stableIds || items.Length < 2)
        {
            return items;
        }
        var keys = new int[items.Length];
        for (int i = 0; i < items.Length; i++)
        {
            keys[i] = items[i].gameObject.GetInstanceID();
        }
        Array.Sort(keys, items);
        return items;
    }

    private static bool IsTerrain(Collider2D c)
    {
        return c.gameObject.layer == TerrainLayer && !c.isTrigger && c.enabled;
    }

    // ------------------------------------------------------------------------------------------------ room geometry
    private void EnsureGeometry(Collider2D[] all, string scene)
    {
        long sum = 0;
        int count = 0;
        foreach (Collider2D c in all)
        {
            if (!(c is EdgeCollider2D) && !(c is PolygonCollider2D))
            {
                continue;
            }
            if (!IsTerrain(c) || c.attachedRigidbody != null && c.attachedRigidbody.bodyType == RigidbodyType2D.Dynamic)
            {
                continue;
            }
            Vector3 p = c.transform.position;
            sum += c.GetInstanceID() * 31L + (long)Mathf.Round(p.x * 64f) * 7L + (long)Mathf.Round(p.y * 64f) * 13L;
            count++;
        }
        string key = scene + ":" + count + ":" + sum;
        if (key == geomKey)
        {
            return;
        }
        float t0 = HoningClock.EngineRealtimeSinceStartup;
        geomKey = key;
        staticIds.Clear();
        var loops = new List<Vector2[]>();
        var polys = new List<List<Vector2[]>>();
        var chains = new List<Vector2[]>();
        float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
        foreach (Collider2D c in all)
        {
            EdgeCollider2D e = c as EdgeCollider2D;
            PolygonCollider2D pc = c as PolygonCollider2D;
            if (e == null && pc == null)
            {
                continue;
            }
            if (!IsTerrain(c) || c.attachedRigidbody != null && c.attachedRigidbody.bodyType == RigidbodyType2D.Dynamic)
            {
                continue;
            }
            staticIds.Add(c.GetInstanceID());
            var paths = new List<Vector2[]>();
            if (e != null)
            {
                Vector2[] local = e.points;
                var w = new Vector2[local.Length];
                for (int i = 0; i < local.Length; i++)
                {
                    w[i] = e.transform.TransformPoint(local[i] + e.offset);
                }
                if (w.Length >= 3 && (w[0] - w[w.Length - 1]).sqrMagnitude < 1e-4f)
                {
                    loops.Add(w);
                }
                else
                {
                    chains.Add(w);
                }
                paths.Add(w);
            }
            else
            {
                for (int k = 0; k < pc.pathCount; k++)
                {
                    Vector2[] local = pc.GetPath(k);
                    var w = new Vector2[local.Length];
                    for (int i = 0; i < local.Length; i++)
                    {
                        w[i] = pc.transform.TransformPoint(local[i] + pc.offset);
                    }
                    paths.Add(w);
                }
                polys.Add(paths);
            }
            foreach (Vector2[] w in paths)
            {
                foreach (Vector2 p in w)
                {
                    minX = Mathf.Min(minX, p.x);
                    maxX = Mathf.Max(maxX, p.x);
                    minY = Mathf.Min(minY, p.y);
                    maxY = Mathf.Max(maxY, p.y);
                }
            }
        }
        rects.Clear();
        if (count == 0 || maxX - minX > 2000f || maxY - minY > 2000f)
        {
            nx = ny = 0;
            solid = new bool[0];
            return;
        }
        ox = Mathf.Floor(minX) - 1f;
        oy = Mathf.Floor(minY) - 1f;
        nx = Mathf.CeilToInt((maxX - ox + 1f) / step);
        ny = Mathf.CeilToInt((maxY - oy + 1f) / step);
        solid = new bool[nx * ny];
        Fill(loops);                         // tile-map outlines: even-odd over all loops (holes are loops too)
        foreach (List<Vector2[]> paths in polys)
        {
            Fill(paths);                     // each polygon collider on its own
        }
        foreach (Vector2[] chain in chains)  // an open chain is a thin barrier
        {
            for (int k = 0; k + 1 < chain.Length; k++)
            {
                Vector2 a = chain[k], b = chain[k + 1];
                int n = Mathf.Max(1, Mathf.CeilToInt((b - a).magnitude / (step * 0.5f)));
                for (int s = 0; s <= n; s++)
                {
                    Vector2 p = Vector2.Lerp(a, b, s / (float)n);
                    int i = Mathf.FloorToInt((p.x - ox) / step), j = Mathf.FloorToInt((p.y - oy) / step);
                    if (i >= 0 && i < nx && j >= 0 && j < ny)
                    {
                        solid[j * nx + i] = true;
                    }
                }
            }
        }
        // rectangles: runs of solid cells in a column, joined with the same run in the next columns. The cell
        // above the top and below the bottom of every rectangle is free.
        var active = new Dictionary<long, int>();
        for (int i = 0; i < nx; i++)
        {
            var next = new Dictionary<long, int>();
            int j = 0;
            while (j < ny)
            {
                if (!solid[j * nx + i])
                {
                    j++;
                    continue;
                }
                int j0 = j;
                while (j < ny && solid[j * nx + i])
                {
                    j++;
                }
                long run = j0 * 100000L + (j - 1);
                int index;
                if (active.TryGetValue(run, out index))
                {
                    rects[index][2] = ox + (i + 1) * step;
                }
                else
                {
                    index = rects.Count;
                    rects.Add(new[] { ox + i * step, oy + j0 * step, ox + (i + 1) * step, oy + j * step });
                }
                next[run] = index;
            }
            active = next;
        }
        Rebuilds++;
        LastBuildMs = (HoningClock.EngineRealtimeSinceStartup - t0) * 1000.0;
    }

    private void Fill(List<Vector2[]> paths)
    {
        if (paths.Count == 0)
        {
            return;
        }
        float lo = float.MaxValue, hi = float.MinValue;
        foreach (Vector2[] p in paths)
        {
            foreach (Vector2 v in p)
            {
                lo = Mathf.Min(lo, v.y);
                hi = Mathf.Max(hi, v.y);
            }
        }
        int j0 = Mathf.Max(0, Mathf.FloorToInt((lo - oy) / step)), j1 = Mathf.Min(ny - 1, Mathf.CeilToInt((hi - oy) / step));
        var xs = new List<float>();
        for (int j = j0; j <= j1; j++)
        {
            float yc = oy + (j + 0.5f) * step;
            xs.Clear();
            foreach (Vector2[] p in paths)
            {
                for (int k = 0; k < p.Length; k++)
                {
                    Vector2 a = p[k], b = p[(k + 1) % p.Length];
                    if ((a.y <= yc && b.y > yc) || (b.y <= yc && a.y > yc))
                    {
                        xs.Add(a.x + (yc - a.y) / (b.y - a.y) * (b.x - a.x));
                    }
                }
            }
            xs.Sort();
            for (int k = 0; k + 1 < xs.Count; k += 2)
            {
                int i0 = Mathf.Max(0, Mathf.CeilToInt((xs[k] - ox) / step - 0.5f));
                int i1 = Mathf.Min(nx - 1, Mathf.CeilToInt((xs[k + 1] - ox) / step - 0.5f) - 1);
                for (int i = i0; i <= i1; i++)
                {
                    solid[j * nx + i] = true;
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ the envelope
    private int cols, rows;
    private int[] masks;
    private readonly List<Obj> objects = new List<Obj>();
    private float winX0, winY0, winX1, winY1;   // the real window on the observation's screen (px)
    private readonly List<float[]> uiRects = new List<float[]>();   // screen-space UI on the observation's screen (px)
    public string UiDump;                       // set to "" to get a list of the UI renderers found (debugging)

    private float SX(float wx)
    {
        return (wx - OriginX) * Ppu;
    }

    private float SY(float wy)
    {
        return (OriginY - wy) * Ppu;
    }

    private void Add(string id, string kind, string look, float x0, float y0, float x1, float y1, int bit)
    {
        // world rectangle -> what of it lies in the camera's view -> screen pixels whose centre it covers
        x0 = Mathf.Max(x0, ViewLeft);
        x1 = Mathf.Min(x1, ViewRight);
        y0 = Mathf.Max(y0, ViewBottom);
        y1 = Mathf.Min(y1, ViewTop);
        if (x1 <= x0 || y1 <= y0)
        {
            return;
        }
        float sx0 = Mathf.Max(0f, SX(x0)), sx1 = Mathf.Min(viewW, SX(x1)), sy0 = Mathf.Max(0f, SY(y1)), sy1 = Mathf.Min(viewH, SY(y0));
        int px0 = Mathf.CeilToInt(sx0 - 0.5f), px1 = Mathf.CeilToInt(sx1 - 0.5f) - 1;
        int py0 = Mathf.CeilToInt(sy0 - 0.5f), py1 = Mathf.CeilToInt(sy1 - 0.5f) - 1;
        if (px1 < px0 || py1 < py0)
        {
            return;
        }
        objects.Add(new Obj { id = id, kind = kind, look = look, x = px0, y = py0, w = px1 - px0 + 1, h = py1 - py0 + 1 });
        if (bit != 0)
        {
            for (int r = py0 / cell; r <= py1 / cell && r < rows; r++)
            {
                for (int c = px0 / cell; c <= px1 / cell && c < cols; c++)
                {
                    masks[r * cols + c] |= bit;
                }
            }
        }
    }

    private static float Opacity(Renderer r)
    {
        tk2dBaseSprite sprite = r.GetComponent<tk2dBaseSprite>();
        if (sprite != null)
        {
            return sprite.color.a;
        }
        SpriteRenderer sr = r as SpriteRenderer;
        if (sr != null)
        {
            return sr.sprite == null ? 0f : sr.color.a;
        }
        TMPro.TMP_Text text = r.GetComponent<TMPro.TMP_Text>();
        if (text != null)
        {
            return string.IsNullOrEmpty(text.text) ? 0f : text.color.a;
        }
        Material m = r.sharedMaterial;
        if (m != null && m.HasProperty("_Color"))
        {
            return m.color.a;
        }
        return 1f;
    }

    // Rectangles of the screen-space UI, on the observation's screen.
    private void CollectUi()
    {
        uiRects.Clear();
        var dump = UiDump != null ? new StringBuilder() : null;
        Camera hud = cameras != null ? cameras.hudCamera : null;
        if (hud != null && hud.isActiveAndEnabled)
        {
            foreach (Renderer r in UnityEngine.Object.FindObjectsOfType<Renderer>())
            {
                if (!r.enabled || (hud.cullingMask & (1 << r.gameObject.layer)) == 0 || r is ParticleSystemRenderer)
                {
                    continue;
                }
                float alpha = Opacity(r);
                Bounds b = r.bounds;
                float x0 = 2f, y0 = 2f, x1 = -1f, y1 = -1f;
                for (int k = 0; k < 4; k++)
                {
                    Vector3 corner = new Vector3((k & 1) == 0 ? b.min.x : b.max.x, (k & 2) == 0 ? b.min.y : b.max.y, b.center.z);
                    Vector3 v = hud.WorldToViewportPoint(corner);
                    x0 = Mathf.Min(x0, v.x);
                    x1 = Mathf.Max(x1, v.x);
                    y0 = Mathf.Min(y0, v.y);
                    y1 = Mathf.Max(y1, v.y);
                }
                bool counted = alpha >= 0.05f && x1 > 0f && x0 < 1f && y1 > 0f && y0 < 1f && x1 - x0 > 0.002f && y1 - y0 > 0.002f;
                if (dump != null)
                {
                    dump.Append(counted ? "UI  " : "skip").Append(' ').Append(r.GetType().Name).Append(' ').Append(r.gameObject.name)
                        .Append(" parent=").Append(r.transform.parent != null ? r.transform.parent.name : "-")
                        .Append(" alpha=").Append(F(alpha)).Append(" viewport=[").Append(F(x0)).Append(',').Append(F(y0)).Append(',').Append(F(x1)).Append(',').Append(F(y1)).Append("]\n");
                }
                if (counted)
                {
                    AddUi(Mathf.Clamp01(x0), Mathf.Clamp01(1f - y1), Mathf.Clamp01(x1), Mathf.Clamp01(1f - y0));
                }
            }
        }
        foreach (UnityEngine.UI.Graphic g in UnityEngine.Object.FindObjectsOfType<UnityEngine.UI.Graphic>())
        {
            if (!g.isActiveAndEnabled || g.canvas == null || !g.canvas.enabled || g.color.a < 0.05f || g.canvasRenderer.cull
                || g.canvasRenderer.GetAlpha() < 0.05f)
            {
                continue;
            }
            UnityEngine.UI.Text text = g as UnityEngine.UI.Text;
            if (text != null && string.IsNullOrEmpty(text.text))
            {
                continue;
            }
            bool hidden = false;
            foreach (CanvasGroup group in g.GetComponentsInParent<CanvasGroup>())
            {
                if (group.alpha < 0.05f)
                {
                    hidden = true;
                }
            }
            if (hidden)
            {
                continue;
            }
            var corners = new Vector3[4];
            g.rectTransform.GetWorldCorners(corners);
            Camera canvasCamera = g.canvas.renderMode == RenderMode.ScreenSpaceOverlay ? null : g.canvas.worldCamera;
            float x0 = float.MaxValue, y0 = float.MaxValue, x1 = float.MinValue, y1 = float.MinValue;
            foreach (Vector3 corner in corners)
            {
                Vector2 p = RectTransformUtility.WorldToScreenPoint(canvasCamera, corner);
                x0 = Mathf.Min(x0, p.x / Screen.width);
                x1 = Mathf.Max(x1, p.x / Screen.width);
                y0 = Mathf.Min(y0, p.y / Screen.height);
                y1 = Mathf.Max(y1, p.y / Screen.height);
            }
            bool counted = x1 > 0f && x0 < 1f && y1 > 0f && y0 < 1f && x1 - x0 > 0.002f && y1 - y0 > 0.002f;
            if (dump != null)
            {
                dump.Append(counted ? "UI  " : "skip").Append(" Graphic ").Append(g.GetType().Name).Append(' ').Append(g.gameObject.name)
                    .Append(" viewport=[").Append(F(x0)).Append(',').Append(F(y0)).Append(',').Append(F(x1)).Append(',').Append(F(y1)).Append("]\n");
            }
            if (counted)
            {
                AddUi(Mathf.Clamp01(x0), Mathf.Clamp01(1f - y1), Mathf.Clamp01(x1), Mathf.Clamp01(1f - y0));
            }
        }
        if (dump != null)
        {
            UiDump = dump.ToString();
        }
    }

    // a rectangle given in fractions of the game window (x right, y down) -> the observation's screen
    private void AddUi(float fx0, float fy0, float fx1, float fy1)
    {
        float w = winX1 - winX0, h = winY1 - winY0;
        float x0 = Mathf.Max(0f, winX0 + fx0 * w), x1 = Mathf.Min(viewW, winX0 + fx1 * w);
        float y0 = Mathf.Max(0f, winY0 + fy0 * h), y1 = Mathf.Min(viewH, winY0 + fy1 * h);
        if (x1 > x0 && y1 > y0)
        {
            uiRects.Add(new[] { x0, y0, x1, y1 });
        }
    }

    // Take out what the screen-space UI covers: objects wholly under one rectangle, and the cells it touches.
    private void ApplyUi()
    {
        if (uiRects.Count == 0)
        {
            return;
        }
        for (int k = objects.Count - 1; k >= 0; k--)
        {
            Obj o = objects[k];
            if (o.id.Contains("/t"))
            {
                continue;               // a terrain rectangle keeps its bounds; its covered cells are marked below
            }
            foreach (float[] u in uiRects)
            {
                if (o.x >= u[0] && o.y >= u[1] && o.x + o.w <= u[2] && o.y + o.h <= u[3])
                {
                    objects.RemoveAt(k);
                    break;
                }
            }
        }
        foreach (float[] u in uiRects)
        {
            int c0 = Mathf.FloorToInt(u[0] / cell), c1 = Mathf.Min(cols - 1, Mathf.CeilToInt(u[2] / cell) - 1);
            int r0 = Mathf.FloorToInt(u[1] / cell), r1 = Mathf.Min(rows - 1, Mathf.CeilToInt(u[3] / cell) - 1);
            for (int r = Mathf.Max(0, r0); r <= r1; r++)
            {
                for (int c = Mathf.Max(0, c0); c <= c1; c++)
                {
                    bool inside = c * cell >= u[0] && (c + 1) * cell <= u[2] && r * cell >= u[1] && (r + 1) * cell <= u[3];
                    masks[r * cols + c] = inside ? Unk : (masks[r * cols + c] | Unk);
                }
            }
        }
    }

    public string Build(int id, int tick, float simTime, bool canAct, string segment, bool dialogue, string dialogueText)
    {
        var sb = new StringBuilder(32768);
        GameManager gm = GameManager.UnsafeInstance;
        HeroController hero = HeroController.SilentInstance;
        Camera cam = GameCamera();
        string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        cols = viewW / cell;
        rows = viewH / cell;
        if (masks == null || masks.Length != cols * rows)
        {
            masks = new int[cols * rows];
        }
        objects.Clear();
        HasView = false;
        bool playerSeen = false;
        float boxX = 0f, boxY = 0f, boxW = 0f, boxH = 0f;
        bool facingRight = true;
        string room = RoomToken(scene);                 // e: ids carry an opaque room code, not the scene name
        string playerId = room + "/player";

        Bounds heroBounds = new Bounds();
        bool heroBox = false;
        if (hero != null)
        {
            BoxCollider2D body = hero.GetComponent<BoxCollider2D>();
            if (body != null)
            {
                heroBounds = body.bounds;
                heroBox = true;
                lastFeetX = heroBounds.center.x;
                lastFeetY = heroBounds.min.y;
                haveFeet = true;
            }
            if (hero.cState != null)
            {
                facingRight = hero.cState.facingRight;
            }
        }

        if (cam != null)
        {
            float dist = -cam.transform.position.z;
            Vector3 bl = cam.ViewportToWorldPoint(new Vector3(0f, 0f, dist));
            Vector3 tr = cam.ViewportToWorldPoint(new Vector3(1f, 1f, dist));
            if (tr.x - bl.x > 0.5f && tr.y - bl.y > 0.5f)
            {
                ViewLeft = bl.x;
                ViewBottom = bl.y;
                ViewRight = tr.x;
                ViewTop = tr.y;
                float ppu0 = NominalH / (ViewTop - ViewBottom);
                if (anchored && haveFeet)
                {
                    Ppu = ppu0 * zoom;
                    OriginX = lastFeetX - anchorX / Ppu;
                    OriginY = lastFeetY + anchorY / Ppu;
                }
                else
                {
                    Ppu = ppu0 * (anchored ? zoom : 1f);
                    OriginX = ViewLeft;
                    OriginY = ViewTop;
                }
                FeetX = lastFeetX;
                FeetY = lastFeetY;
                HasView = true;
            }
        }

        if (HasView)
        {
            winX0 = SX(ViewLeft);
            winY0 = SY(ViewTop);
            winX1 = SX(ViewRight);
            winY1 = SY(ViewBottom);
            Collider2D[] all = UnityEngine.Object.FindObjectsOfType<Collider2D>();
            EnsureGeometry(all, scene);
            // the grid: unknown outside the real window; inside it, what the room's raster says
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    float cx0 = c * cell, cy0 = r * cell, cx1 = cx0 + cell, cy1 = cy0 + cell;
                    int m = 0;
                    if (cx0 < winX0 || cy0 < winY0 || cx1 > winX1 || cy1 > winY1)
                    {
                        m |= Unk;
                    }
                    float ix0 = Mathf.Max(cx0, winX0), ix1 = Mathf.Min(cx1, winX1), iy0 = Mathf.Max(cy0, winY0), iy1 = Mathf.Min(cy1, winY1);
                    if (ix1 > ix0 && iy1 > iy0)
                    {
                        float wx0 = OriginX + ix0 / Ppu, wx1 = OriginX + ix1 / Ppu, wy1 = OriginY - iy0 / Ppu, wy0 = OriginY - iy1 / Ppu;
                        int i0 = Mathf.FloorToInt((wx0 - ox) / step + 1e-4f), i1 = Mathf.CeilToInt((wx1 - ox) / step - 1e-4f) - 1;
                        int j0 = Mathf.FloorToInt((wy0 - oy) / step + 1e-4f), j1 = Mathf.CeilToInt((wy1 - oy) / step - 1e-4f) - 1;
                        bool anySolid = false, anyFree = false;
                        for (int j = j0; j <= j1 && !(anySolid && anyFree); j++)
                        {
                            for (int i = i0; i <= i1; i++)
                            {
                                if (i >= 0 && i < nx && j >= 0 && j < ny && solid[j * nx + i])
                                {
                                    anySolid = true;
                                }
                                else
                                {
                                    anyFree = true;
                                }
                            }
                        }
                        if (anySolid)
                        {
                            m |= Wall;
                        }
                        if (anyFree || !anySolid)
                        {
                            m |= Bg;
                        }
                    }
                    masks[r * cols + c] = m;
                }
            }
            // the player
            Renderer heroRenderer = hero != null ? hero.GetComponent<Renderer>() : null;
            if (heroBox && (heroRenderer == null || heroRenderer.enabled))
            {
                int before = objects.Count;
                Add(playerId, "player", "knight", heroBounds.min.x, heroBounds.min.y, heroBounds.max.x, heroBounds.max.y, Actor);
                playerSeen = objects.Count > before;
                boxX = SX(heroBounds.min.x);
                boxY = SY(heroBounds.max.y);
                boxW = heroBounds.size.x * Ppu;
                boxH = heroBounds.size.y * Ppu;
            }
            // enemies
            foreach (HealthManager hm in Ordered(UnityEngine.Object.FindObjectsOfType<HealthManager>()))
            {
                if (hm.isDead)
                {
                    continue;
                }
                Collider2D col = hm.GetComponent<Collider2D>();
                if (col == null || !col.enabled)
                {
                    continue;
                }
                Renderer rend = hm.GetComponent<Renderer>();
                if (rend != null && !rend.enabled)
                {
                    continue;
                }
                Bounds b = col.bounds;
                // e: the look is the kind word for every enemy, the boss included (d: the object's name)
                Add(room + "/e" + IdOf(hm.gameObject.GetInstanceID(), room), "enemy", "enemy", b.min.x, b.min.y, b.max.x, b.max.y, Actor);
            }
            // hazards
            foreach (DamageHero dh in Ordered(UnityEngine.Object.FindObjectsOfType<DamageHero>()))
            {
                if (dh.damageDealt <= 0 || dh.GetComponent<HealthManager>() != null)
                {
                    continue;
                }
                Collider2D col = dh.GetComponent<Collider2D>();
                if (col == null || !col.enabled)
                {
                    continue;
                }
                Renderer rend = dh.GetComponent<Renderer>();
                if (rend != null && !rend.enabled)
                {
                    continue;
                }
                Bounds b = col.bounds;
                // e: the game's hazard types 2 and 3 are spikes and acid, which a player recognises on sight; anything else
                // that damages is "hazard" (d: the object's name, e.g. "falling barrel")
                string look = dh.hazardType == 2 ? "spikes" : dh.hazardType == 3 ? "acid" : "hazard";
                Add(room + "/h" + IdOf(dh.gameObject.GetInstanceID(), room), "visible_hazard", look, b.min.x, b.min.y, b.max.x, b.max.y, Haz);
            }
            // solid things that are not part of the room's raster: boxes (platforms, doors, gates), circles, ...
            foreach (Collider2D c in Ordered(all))
            {
                if (!IsTerrain(c) || staticIds.Contains(c.GetInstanceID()))
                {
                    continue;
                }
                Bounds b = c.bounds;
                Breakable breakable = c.GetComponentInParent<Breakable>();
                bool door = breakable != null && DoorNames.Contains(CleanName(breakable.gameObject.name));
                if (breakable != null && !door && loggedBreakables.Add(breakable.GetInstanceID()))
                {
                    NameLog.Add("breakable '" + breakable.gameObject.name + "' in " + scene + " at (" + F(Math.Round(b.center.x, 2)) + ", "
                                + F(Math.Round(b.center.y, 2)) + ") is not on the door list: reported as a plain solid (wall / wall)");
                }
                if (door)
                {
                    Add(room + "/d" + IdOf(c.gameObject.GetInstanceID(), room), "interactable", "door", b.min.x, b.min.y, b.max.x, b.max.y, Inter);
                }
                else
                {
                    // e: a solid box is "wall" / "wall" like the kind, whatever it is called in the game (d: the object's name,
                    // e.g. "collapser tute", "break floor", "battle gate", "plat_float")
                    Add(room + "/b" + IdOf(c.gameObject.GetInstanceID(), room), "wall", "wall", b.min.x, b.min.y, b.max.x, b.max.y, Wall);
                }
            }
            // the room's terrain rectangles
            for (int k = 0; k < rects.Count; k++)
            {
                float[] q = rects[k];
                if (q[2] > ViewLeft && q[0] < ViewRight && q[3] > ViewBottom && q[1] < ViewTop)
                {
                    Add(room + "/t" + k, "wall", "terrain", q[0], q[1], q[2], q[3], 0);
                }
            }
            CollectUi();
            ApplyUi();
            playerSeen = playerSeen && objects.Exists(o => o.id == playerId);
        }
        else
        {
            for (int k = 0; k < masks.Length; k++)
            {
                masks[k] = Unk;
            }
            winX0 = winY0 = winX1 = winY1 = 0f;
        }

        // ---- JSON
        sb.Append("{\"schema\":\"honing.visible.v1\",\"adapter\":\"forestdango-hk-visible-v2\",\"profile\":\"visible\",\"game\":\"hollow-knight-forestdango-demo\",\"obs_mode\":\"geometry\",");
        sb.Append("\"frame\":{\"id\":").Append(id).Append(",\"tick\":").Append(tick).Append(",\"simulation_time\":").Append(F(simTime)).Append("},");
        sb.Append("\"flow\":{\"can_act\":").Append(canAct ? "true" : "false").Append(",\"segment\":");
        Str(sb, segment);
        sb.Append("},");
        sb.Append("\"view\":{\"screen_rect\":[0,0,").Append(viewW).Append(',').Append(viewH).Append("],\"world_from_screen\":[");
        if (HasView)
        {
            sb.Append(F(1.0 / Ppu)).Append(",0,0,").Append(F(-1.0 / Ppu)).Append(',').Append(F(OriginX)).Append(',').Append(F(OriginY));
        }
        else
        {
            sb.Append("1,0,0,-1,0,0");
        }
        sb.Append("],\"window\":{\"size\":[").Append(NominalW).Append(',').Append(NominalH).Append("],\"rect\":[");
        sb.Append(F(Math.Round(winX0, 2))).Append(',').Append(F(Math.Round(winY0, 2))).Append(',').Append(F(Math.Round(winX1 - winX0, 2))).Append(',').Append(F(Math.Round(winY1 - winY0, 2)));
        sb.Append("],\"ui_rects\":[");
        for (int k = 0; HasView && k < uiRects.Count; k++)
        {
            float[] u = uiRects[k];
            sb.Append(k > 0 ? "," : "").Append('[').Append(F(Math.Round(u[0], 1))).Append(',').Append(F(Math.Round(u[1], 1))).Append(',')
              .Append(F(Math.Round(u[2] - u[0], 1))).Append(',').Append(F(Math.Round(u[3] - u[1], 1))).Append(']');
        }
        sb.Append("],\"player_centred\":");
        if (anchored)
        {
            sb.Append("{\"feet_at\":[").Append(F(anchorX)).Append(',').Append(F(anchorY)).Append("],\"zoom\":").Append(F(zoom)).Append('}');
        }
        else
        {
            sb.Append("false");
        }
        sb.Append("},\"world_units\":\"1 unit = 1 Unity world unit of the game (the knight's body is about 0.5 x 1.3 units); x right, y up; origin per room; screen x right/y down\"},");
        sb.Append("\"terrain\":{\"status\":").Append(HasView ? "\"available\"" : "\"unavailable\"").Append(",\"coverage\":\"full\",\"origin\":[0,0],\"cell_size\":[").Append(cell).Append(',').Append(cell).Append("],\"rows\":[");
        for (int r = 0; r < rows; r++)
        {
            if (r > 0)
            {
                sb.Append(',');
            }
            sb.Append('"');
            for (int c = 0; c < cols; c++)
            {
                sb.Append((char)(0x100 + masks[r * cols + c]));
            }
            sb.Append('"');
        }
        sb.Append("],\"legend\":{\"unknown\":\"? outside the game window, or covered by the HUD / a text box; never safe\",\"background\":\". no solid terrain here\",\"solid\":\"# solid terrain (wall, floor, platform)\",\"platform\":\"= (not used: all solid terrain is #)\",\"hazard\":\"! spikes, shots, an enemy's weapon\",\"actor\":\"@ the knight or an enemy\",\"interactable\":\"+ a door\"}},");
        sb.Append("\"objects\":[");
        for (int k = 0; k < objects.Count; k++)
        {
            Obj o = objects[k];
            if (k > 0)
            {
                sb.Append(',');
            }
            sb.Append("{\"id\":");
            Str(sb, o.id);
            sb.Append(",\"kind\":");
            Str(sb, o.kind);
            sb.Append(",\"bounds\":[").Append(o.x).Append(',').Append(o.y).Append(',').Append(o.w).Append(',').Append(o.h).Append("],\"look\":");
            Str(sb, o.look);
            sb.Append('}');
        }
        sb.Append("],\"self\":{\"object_id\":");
        if (playerSeen)
        {
            Str(sb, playerId);
            sb.Append(",\"screen_box\":[").Append(F(Math.Round(boxX, 2))).Append(',').Append(F(Math.Round(boxY, 2))).Append(',').Append(F(Math.Round(boxW, 2))).Append(',').Append(F(Math.Round(boxH, 2))).Append("],\"facing\":").Append(facingRight ? "\"right\"" : "\"left\"");
        }
        else
        {
            sb.Append("null");
        }
        sb.Append("},\"hud\":{");
        bool hudUp = gm != null && gm.playerData != null && hero != null && (gm.gameState == GameState.PLAYING || gm.gameState == GameState.PAUSED);
        if (hudUp)
        {
            PlayerData pd = gm.playerData;
            sb.Append("\"health\":{\"value\":").Append(Mathf.Max(0, pd.health)).Append(",\"evidence\":\"hud\"},");
            sb.Append("\"health_max\":{\"value\":").Append(pd.maxHealth).Append(",\"evidence\":\"hud\"},");
            sb.Append("\"soul\":{\"fraction\":").Append(F(Math.Round(pd.maxMP > 0 ? pd.MPCharge / (double)pd.maxMP : 0.0, 2))).Append(",\"evidence\":\"hud gauge\"},");
            sb.Append("\"geo\":{\"value\":").Append(pd.geo).Append(",\"evidence\":\"hud\"},");
        }
        sb.Append("\"text\":[");
        if (dialogue)
        {
            sb.Append("{\"kind\":\"dialogue\",\"blocking\":true,\"content\":");
            Str(sb, dialogueText ?? "");
            sb.Append('}');
        }
        sb.Append("]},");
        sb.Append("\"rules\":[\"Walk left/right, jump (the longer the button is held the higher, up to a full jump), strike with the nail left/right/up and, in the air, down.\",\"Enemies and spikes hurt on touch; with no health left the knight dies and wakes at the last resting place.\",\"A room is left by walking, falling or climbing out through an opening at its border.\",\"Screen x increases right and screen y increases down. Each action holds for the requested frames; the game is stopped while the model decides.\"],");
        sb.Append("\"capabilities\":{\"global_debug\":true,\"pixels_to_model\":false,\"hidden_own_state\":false},");
        sb.Append("\"disclosure\":{\"visibility\":\"geometry mode: the game's collision shapes inside the rectangle of the game plane that the camera shows; nothing outside that rectangle. Occlusion by scenery is NOT applied.\",");
        sb.Append("\"geometry\":\"terrain = collision outlines rasterised at ").Append(F(step)).Append(" units and cut into rectangles (top = a surface, bottom = a ceiling); doors, hazards, enemies and the knight = bounds of their colliders; all clipped to the camera rectangle\",");
        sb.Append("\"can_exceed_human_view\":[\"collision without art (invisible walls) and openings hidden behind foreground art (secret rooms)\",\"dark rooms are reported like lit rooms\",\"things covered by foreground scenery, fog or particles are reported (no occlusion)\",\"a collider that is active while its sprite is faded out\",\"decoration that looks dangerous is not reported as a hazard\",\"exact collision geometry and hit boxes\"],\"accepted\":\"no occlusion, decoration that looks dangerous not reported as a hazard, and exact geometry: accepted by the user on 2026-11-19 (what a human player can see and understand is fair to give)\",\"names\":\"no internal object or room names: look is a kind word (wall, terrain, enemy, hazard, knight), spikes or acid for the game's own hazard types, door only for objects that look like doors; ids are <room code>/<letter><number> with an opaque room code\",\"screen_space_ui\":\"cells touched by the bounding rectangles of HUD-camera renderers and canvas graphics with visible alpha carry the unknown bit; actors, hazards and doors wholly under one are not reported (view.window.ui_rects lists the rectangles)\",");
        sb.Append("\"unsupported\":\"scenery, background and particles are not reported; menus, map and inventory screens are not reported\",");
        sb.Append("\"resolution\":\"").Append(cell).Append(" px cells on a ").Append(viewW).Append(" x ").Append(viewH).Append(" screen; the game window is taken as ").Append(NominalW).Append(" x ").Append(NominalH).Append("\",");
        sb.Append("\"debug\":\"debug_observation.json / observations.jsonl hold global state; plumbing and evaluation only\"}}");
        return sb.ToString();
    }

    public int ObjectCount
    {
        get { return objects.Count; }
    }

    public int RectCount
    {
        get { return rects.Count; }
    }
}

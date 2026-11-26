// Added by the Honing devmode study (2026-11-25). NOT part of the author's project. Used only by HoningBridge.
//
// Read-only instrumentation, all of it off unless its knob is set:
//   HONING_TIMING=1|2     PlayerLoop phase timing per action window -> <session>/timing.jsonl (2: every subsystem)
//   HONING_FRAME_LOG=1|2  exact state per played frame -> <session>/frames.jsonl (2: with the full dumps, not only
//                         their hashes)
//   HONING_INTROSPECT=1   what in the loaded scenes could depend on the wall clock (state machines' real-time waits,
//                         audio waits, animators, particle systems) -> <session>/introspection_*.json
// Nothing here changes game state: it reads fields and writes files.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using HutongGames.PlayMaker;
using UnityEngine;
using UnityEngine.LowLevel;

// Timestamps inserted into Unity's PlayerLoop. Each segment is the time from one marker to the next marker that
// runs, attributed to the subsystem that follows the first marker. Marker 0 starts a frame.
public static class HoningLoop
{
    public static string[] Names = new string[0];
    public static Action<long[], long> FrameDone;   // (segment ticks of the finished frame, its total ticks)
    public static Action EndOfLoop;                  // runs at the very end of each PlayerLoop iteration
    public static Action AfterPreloading;            // runs right after EarlyUpdate.UpdatePreloading
    public static Action BeforeRender, AfterRender;  // around PostLateUpdate.FinishFrameRendering (the cameras draw there)
    public static Action BeforeInvisible, AfterInvisible;   // around EarlyUpdate.RendererNotifyInvisible
    private static long[] seg = new long[0];
    private static long frameStart = -1, lastTime;
    private static int last = -1;
    private static bool timingOn;

    // Rebuilds the current player loop: an end-of-loop hook, and (timing) markers before the subsystems.
    public static void Install(bool timing, bool detailed)
    {
        timingOn = timing;
        PlayerLoopSystem root = PlayerLoop.GetCurrentPlayerLoop();
        var names = new List<string>();
        var top = new PlayerLoopSystem[root.subSystemList.Length];
        for (int i = 0; i < top.Length; i++)
        {
            PlayerLoopSystem group = root.subSystemList[i];
            PlayerLoopSystem[] original = group.subSystemList ?? new PlayerLoopSystem[0];
            var subs = new List<PlayerLoopSystem>();
            for (int j = 0; j < original.Length; j++)
            {
                string name = group.type.Name + "." + (original[j].type != null ? original[j].type.Name : "?");
                if (timing && (j == 0 || detailed || Key(name)))
                {
                    subs.Add(Marker(names.Count));
                    names.Add(name);
                }
                if (name == "PostLateUpdate.FinishFrameRendering")
                {
                    subs.Add(new PlayerLoopSystem { type = typeof(HoningLoop), updateDelegate = () => { if (BeforeRender != null) BeforeRender(); } });
                }
                if (name == "EarlyUpdate.RendererNotifyInvisible")
                {
                    subs.Add(new PlayerLoopSystem { type = typeof(HoningLoop), updateDelegate = () => { if (BeforeInvisible != null) BeforeInvisible(); } });
                }
                subs.Add(original[j]);
                if (name == "EarlyUpdate.UpdatePreloading")
                {
                    subs.Add(new PlayerLoopSystem { type = typeof(HoningLoop), updateDelegate = () => { if (AfterPreloading != null) AfterPreloading(); } });
                }
                if (name == "PostLateUpdate.FinishFrameRendering")
                {
                    subs.Add(new PlayerLoopSystem { type = typeof(HoningLoop), updateDelegate = () => { if (AfterRender != null) AfterRender(); } });
                }
                if (name == "EarlyUpdate.RendererNotifyInvisible")
                {
                    subs.Add(new PlayerLoopSystem { type = typeof(HoningLoop), updateDelegate = () => { if (AfterInvisible != null) AfterInvisible(); } });
                }
            }
            if (timing && original.Length == 0)
            {
                subs.Add(Marker(names.Count));
                names.Add(group.type.Name);
            }
            if (i == top.Length - 1)
            {
                subs.Add(new PlayerLoopSystem { type = typeof(HoningLoop), updateDelegate = () => { if (EndOfLoop != null) EndOfLoop(); } });
                if (timing)
                {
                    subs.Add(Marker(names.Count));
                    names.Add("AfterPlayerLoop(gap)");
                }
            }
            group.subSystemList = subs.ToArray();
            top[i] = group;
        }
        root.subSystemList = top;
        Names = names.ToArray();
        seg = new long[Names.Length];
        PlayerLoop.SetPlayerLoop(root);
    }

    private static bool Key(string n)
    {
        return n.Contains("Render") || n.Contains("Present") || n.Contains("Physics") || n.Contains("Animation") || n.Contains("ScriptRun")
               || n.Contains("Canvas") || n.Contains("Director") || n.Contains("FrameComplete") || n.Contains("GraphicsFence") || n.Contains("GPU")
               || n.Contains("HoningBridge") || n.Contains("Preloading") || n.Contains("Audio") || n.Contains("Particle") || n.Contains("Video")
               || n.Contains("Input") || n.Contains("WaitForTargetFPS") || n.Contains("UIElements") || n.Contains("Tilemap") || n.Contains("Sprite");
    }

    private static PlayerLoopSystem Marker(int index)
    {
        return new PlayerLoopSystem { type = typeof(HoningLoop), updateDelegate = () => Mark(index) };
    }

    private static void Mark(int index)
    {
        long now = Stopwatch.GetTimestamp();
        if (last >= 0)
        {
            seg[last] += now - lastTime;
        }
        if (index == 0)
        {
            if (frameStart >= 0 && FrameDone != null)
            {
                FrameDone(seg, now - frameStart);
            }
            seg = new long[Names.Length];
            frameStart = now;
        }
        last = index;
        lastTime = now;
    }

    public static double Ms(long ticks)
    {
        return ticks * 1000.0 / Stopwatch.Frequency;
    }
}

public static class HoningProbe
{
    public static string R(double v)
    {
        return Math.Round(v, 4).ToString("R", CultureInfo.InvariantCulture);
    }

    public static string F(float v)
    {
        return v.ToString("R", CultureInfo.InvariantCulture);
    }

    public static string D(double v)
    {
        return v.ToString("R", CultureInfo.InvariantCulture);
    }

    // FNV-1a over the UTF-16 code units of a text
    public static ulong Hash(StringBuilder text, int from)
    {
        ulong h = 14695981039346656037UL;
        for (int i = from; i < text.Length; i++)
        {
            h = (h ^ text[i]) * 1099511628211UL;
        }
        return h;
    }

    // ------------------------------------------------------------------------------------------------ state dump
    private static readonly Dictionary<Type, FieldInfo[]> fieldCache = new Dictionary<Type, FieldInfo[]>();
    private static Dictionary<int, string> lastObjects;

    private static FieldInfo[] Fields(Type t)
    {
        FieldInfo[] result;
        if (fieldCache.TryGetValue(t, out result))
        {
            return result;
        }
        var list = new List<FieldInfo>();
        for (Type c = t; c != null && c != typeof(MonoBehaviour) && c != typeof(object); c = c.BaseType)
        {
            foreach (FieldInfo f in c.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                Type ft = f.FieldType;
                if (ft == typeof(bool) || ft == typeof(int) || ft == typeof(float) || ft == typeof(Vector2) || ft == typeof(Vector3) || ft.IsEnum
                    || ft == typeof(string) || ft == typeof(double) || ft == typeof(long))
                {
                    list.Add(f);
                }
            }
        }
        list.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
        result = list.ToArray();
        fieldCache[t] = result;
        return result;
    }

    // every plain field (bool, int, float, vector, enum, string) of an object, exact
    public static void DumpFields(StringBuilder b, object o)
    {
        b.Append('{');
        bool first = true;
        foreach (FieldInfo f in Fields(o.GetType()))
        {
            object v = f.GetValue(o);
            Type t = f.FieldType;
            b.Append(first ? "\"" : ",\"").Append(f.Name).Append("\":");
            first = false;
            if (t == typeof(bool)) b.Append((bool)v ? '1' : '0');
            else if (t == typeof(int)) b.Append((int)v);
            else if (t == typeof(long)) b.Append((long)v);
            else if (t == typeof(float)) b.Append('"').Append(F((float)v)).Append('"');
            else if (t == typeof(double)) b.Append('"').Append(D((double)v)).Append('"');
            else if (t == typeof(Vector2)) { var w = (Vector2)v; b.Append("\"").Append(F(w.x)).Append(' ').Append(F(w.y)).Append('"'); }
            else if (t == typeof(Vector3)) { var w = (Vector3)v; b.Append("\"").Append(F(w.x)).Append(' ').Append(F(w.y)).Append(' ').Append(F(w.z)).Append('"'); }
            else if (t.IsEnum) b.Append('"').Append(v).Append('"');
            else HoningVisible.Str(b, (string)v);
        }
        b.Append('}');
    }

    private static string PathOf(Transform t)
    {
        string p = t.name;
        int depth = 0;
        while (t.parent != null && depth < 3)
        {
            t = t.parent;
            p = t.name + "/" + p;
            depth++;
        }
        return p;
    }

    private static void Vec(StringBuilder b, Vector3 p)
    {
        b.Append(F(p.x)).Append(' ').Append(F(p.y));
    }

    // One line per frame. Hashes of: every rigidbody (position, velocity, state), every state machine (active state),
    // every enemy (health), the knight's controller fields, the sprite animators, the player data; the knight and
    // the clocks in clear. level 2 adds the dumps themselves.
    public static void FrameLine(StringBuilder b, int level, string tag, int frame, int window, int fixedStepsInFrame, int fixedStepsAll)
    {
        b.Append("{\"tag\":\"").Append(tag).Append("\",\"f\":").Append(frame).Append(",\"w\":").Append(window)
         .Append(",\"tc\":").Append(UnityEngine.Time.frameCount)
         .Append(",\"t\":\"").Append(D(HoningClock.EngineTimeAsDouble)).Append("\",\"ft\":\"").Append(D(HoningClock.EngineFixedTimeAsDouble))
         .Append("\",\"dt\":\"").Append(F(HoningClock.EngineDeltaTime)).Append("\",\"ts\":\"").Append(F(HoningClock.EngineTimeScale))
         .Append("\",\"udt\":\"").Append(F(UnityEngine.Time.unscaledDeltaTime)).Append("\",\"ut\":\"").Append(F(UnityEngine.Time.unscaledTime))
         .Append("\",\"rt\":\"").Append(F(UnityEngine.Time.realtimeSinceStartup))
         .Append("\",\"wall\":\"").Append(F(HoningClock.EngineUnscaledDeltaTime))
         .Append("\",\"fs\":").Append(fixedStepsInFrame).Append(",\"fsAll\":").Append(fixedStepsAll)
         .Append(",\"scene\":\"").Append(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name)
         .Append("\",\"scenes\":").Append(UnityEngine.SceneManagement.SceneManager.sceneCount)
         .Append(",\"rng\":").Append(JsonUtility.ToJson(UnityEngine.Random.state));
        GameManager gm = GameManager.UnsafeInstance;
        HeroController hero = HeroController.SilentInstance;
        Camera cam = HoningVisible.GameCamera();
        if (cam != null)
        {
            b.Append(",\"cam\":\"");
            Vec(b, cam.transform.position);
            b.Append('"');
        }
        if (gm != null)
        {
            b.Append(",\"gs\":\"").Append(gm.gameState).Append('"');
            PlayerData pd = gm.playerData;
            if (pd != null)
            {
                b.Append(",\"hp\":").Append(pd.health).Append(",\"mp\":").Append(pd.MPCharge).Append(",\"geo\":").Append(pd.geo);
            }
        }
        var scratch = new StringBuilder(4096);
        if (hero != null)
        {
            Rigidbody2D rb = hero.GetComponent<Rigidbody2D>();
            b.Append(",\"p\":\"");
            Vec(b, hero.transform.position);
            if (rb != null)
            {
                b.Append(' ');
                Vec(b, rb.position);
                b.Append(' ');
                Vec(b, rb.velocity);
            }
            b.Append('"');
            scratch.Length = 0;
            DumpFields(scratch, hero);
            if (hero.cState != null)
            {
                DumpFields(scratch, hero.cState);
            }
            tk2dSpriteAnimator anim = hero.GetComponent<tk2dSpriteAnimator>();
            if (anim != null && anim.CurrentClip != null)
            {
                scratch.Append(anim.CurrentClip.name).Append(':').Append(anim.CurrentFrame).Append(':').Append(F(anim.ClipTimeSeconds));
            }
            b.Append(",\"hH\":\"").Append(Hash(scratch, 0).ToString("x16")).Append('"');
            if (level >= 2)
            {
                b.Append(",\"hero\":");
                HoningVisible.Str(b, scratch.ToString());
            }
        }
        // rigidbodies
        scratch.Length = 0;
        Rigidbody2D[] bodies = UnityEngine.Object.FindObjectsOfType<Rigidbody2D>();
        Array.Sort(bodies, (x, y) => x.GetInstanceID().CompareTo(y.GetInstanceID()));
        foreach (Rigidbody2D r in bodies)
        {
            scratch.Append(r.name).Append('|');
            Vec(scratch, r.position);
            scratch.Append('|');
            Vec(scratch, r.velocity);
            scratch.Append('|').Append(F(r.rotation)).Append('|').Append((int)r.bodyType).Append(r.simulated ? 's' : '-').Append(r.IsSleeping() ? 'z' : '-');
            Vec(scratch, r.transform.position);
            scratch.Append(';');
        }
        b.Append(",\"nRb\":").Append(bodies.Length).Append(",\"rbH\":\"").Append(Hash(scratch, 0).ToString("x16")).Append('"');
        if (level >= 2)
        {
            b.Append(",\"rb\":");
            HoningVisible.Str(b, scratch.ToString());
        }
        // state machines
        scratch.Length = 0;
        PlayMakerFSM[] fsms = UnityEngine.Object.FindObjectsOfType<PlayMakerFSM>();
        Array.Sort(fsms, (x, y) => x.GetInstanceID().CompareTo(y.GetInstanceID()));
        foreach (PlayMakerFSM fsm in fsms)
        {
            scratch.Append(fsm.gameObject.name).Append('.').Append(fsm.FsmName).Append('=').Append(fsm.ActiveStateName).Append(';');
        }
        b.Append(",\"nFsm\":").Append(fsms.Length).Append(",\"fsmH\":\"").Append(Hash(scratch, 0).ToString("x16")).Append('"');
        if (level >= 2)
        {
            b.Append(",\"fsm\":");
            HoningVisible.Str(b, scratch.ToString());
        }
        // enemies
        scratch.Length = 0;
        HealthManager[] hms = UnityEngine.Object.FindObjectsOfType<HealthManager>();
        Array.Sort(hms, (x, y) => x.GetInstanceID().CompareTo(y.GetInstanceID()));
        foreach (HealthManager hm in hms)
        {
            scratch.Append(hm.name).Append('|').Append(hm.hp).Append(hm.isDead ? 'd' : '-').Append('|');
            Vec(scratch, hm.transform.position);
            tk2dSpriteAnimator a = hm.GetComponent<tk2dSpriteAnimator>();
            if (a != null && a.CurrentClip != null)
            {
                scratch.Append('|').Append(a.CurrentClip.name).Append(':').Append(a.CurrentFrame);
            }
            scratch.Append(';');
        }
        b.Append(",\"nEn\":").Append(hms.Length).Append(",\"enH\":\"").Append(Hash(scratch, 0).ToString("x16")).Append('"');
        if (level >= 2)
        {
            b.Append(",\"en\":");
            HoningVisible.Str(b, scratch.ToString());
        }
        // every sprite animator (clip and frame) and the number of objects
        scratch.Length = 0;
        tk2dSpriteAnimator[] anims = UnityEngine.Object.FindObjectsOfType<tk2dSpriteAnimator>();
        Array.Sort(anims, (x, y) => x.GetInstanceID().CompareTo(y.GetInstanceID()));
        foreach (tk2dSpriteAnimator a in anims)
        {
            scratch.Append(a.name).Append('=').Append(a.CurrentClip != null ? a.CurrentClip.name : "").Append(':').Append(a.CurrentClip != null ? a.CurrentFrame : -1).Append(';');
        }
        b.Append(",\"nAn\":").Append(anims.Length).Append(",\"anH\":\"").Append(Hash(scratch, 0).ToString("x16")).Append('"');
        if (level >= 2)
        {
            b.Append(",\"an\":");
            HoningVisible.Str(b, scratch.ToString());
        }
        Transform[] gos = UnityEngine.Object.FindObjectsOfType<Transform>();
        b.Append(",\"nGo\":").Append(gos.Length);
        if (level >= 2)
        {
            // which objects appeared and disappeared since the previous logged frame (names; at most 12 each)
            var now = new Dictionary<int, string>(gos.Length);
            foreach (Transform t in gos)
            {
                now[t.GetInstanceID()] = t.name;
            }
            if (lastObjects != null)
            {
                int shown = 0;
                b.Append(",\"appeared\":[");
                foreach (KeyValuePair<int, string> kv in now)
                {
                    if (!lastObjects.ContainsKey(kv.Key) && shown < 12)
                    {
                        b.Append(shown++ > 0 ? "," : "");
                        HoningVisible.Str(b, kv.Value);
                    }
                }
                shown = 0;
                b.Append("],\"gone\":[");
                foreach (KeyValuePair<int, string> kv in lastObjects)
                {
                    if (!now.ContainsKey(kv.Key) && shown < 12)
                    {
                        b.Append(shown++ > 0 ? "," : "");
                        HoningVisible.Str(b, kv.Value);
                    }
                }
                b.Append("]");
            }
            lastObjects = now;
        }
        b.Append("}\n");
    }

    // ------------------------------------------------------------------------------------------------ introspection
    // What in the loaded scenes could depend on the wall clock.
    public static string Introspect()
    {
        var b = new StringBuilder(65536);
        b.Append("{\"scene\":\"").Append(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name).Append("\",\"frameCount\":").Append(UnityEngine.Time.frameCount);
        b.Append(",\"time\":{\"fixedDeltaTime\":\"").Append(F(UnityEngine.Time.fixedDeltaTime)).Append("\",\"maximumDeltaTime\":\"").Append(F(UnityEngine.Time.maximumDeltaTime))
         .Append("\",\"captureFramerate\":").Append(UnityEngine.Time.captureFramerate).Append(",\"timeScale\":\"").Append(F(UnityEngine.Time.timeScale))
         .Append("\",\"targetFrameRate\":").Append(Application.targetFrameRate).Append(",\"vSyncCount\":").Append(QualitySettings.vSyncCount)
         .Append(",\"renderFrameInterval\":").Append(UnityEngine.Rendering.OnDemandRendering.renderFrameInterval).Append(",\"batchmode\":").Append(Application.isBatchMode ? 1 : 0)
         .Append(",\"backgroundLoadingPriority\":\"").Append(Application.backgroundLoadingPriority).Append("\"}");
        b.Append(",\"physics2d\":{\"simulationMode\":\"").Append(Physics2D.simulationMode).Append("\",\"autoSyncTransforms\":").Append(Physics2D.autoSyncTransforms ? 1 : 0)
         .Append(",\"velocityIterations\":").Append(Physics2D.velocityIterations).Append(",\"positionIterations\":").Append(Physics2D.positionIterations)
         .Append(",\"useMultithreading\":").Append(Physics2D.jobOptions.useMultithreading ? 1 : 0).Append('}');
        b.Append(",\"graphics\":{\"device\":\"").Append(SystemInfo.graphicsDeviceType).Append("\",\"name\":");
        HoningVisible.Str(b, SystemInfo.graphicsDeviceName);
        b.Append(",\"screen\":[").Append(Screen.width).Append(',').Append(Screen.height).Append("]}");
        b.Append(",\"audio\":{\"outputSampleRate\":").Append(AudioSettings.outputSampleRate).Append(",\"speakerMode\":\"").Append(AudioSettings.speakerMode).Append("\"}");
        // state machines: actions with a real-time flag that is set, and actions that wait for a sound
        var counts = new SortedDictionary<string, int>();
        var realTime = new List<string>();
        var audioWait = new List<string>();
        var visibilityEvents = new List<string>();
        var actionTypes = new SortedDictionary<string, int>();
        int fsmCount = 0, stateCount = 0, actionCount = 0, failed = 0;
        foreach (PlayMakerFSM fsm in Resources.FindObjectsOfTypeAll<PlayMakerFSM>())
        {
            if (!fsm.gameObject.scene.IsValid())
            {
                continue;
            }
            fsmCount++;
            FsmState[] states;
            try
            {
                states = fsm.FsmStates;
            }
            catch (Exception)
            {
                failed++;
                continue;
            }
            if (states == null)
            {
                continue;
            }
            try
            {
                foreach (FsmTransition tr in fsm.FsmGlobalTransitions)
                {
                    if (tr.EventName != null && tr.EventName.Contains("VISIBLE") && visibilityEvents.Count < 300)
                    {
                        visibilityEvents.Add(PathOf(fsm.transform) + " [" + fsm.FsmName + "] global: " + tr.EventName + " -> " + tr.ToState);
                    }
                }
            }
            catch (Exception)
            {
                failed++;
            }
            foreach (FsmState state in states)
            {
                stateCount++;
                if (state.Transitions != null)
                {
                    foreach (FsmTransition tr in state.Transitions)
                    {
                        if (tr.EventName != null && tr.EventName.Contains("VISIBLE") && visibilityEvents.Count < 300)
                        {
                            visibilityEvents.Add(PathOf(fsm.transform) + " [" + fsm.FsmName + "] " + state.Name + ": " + tr.EventName + " -> " + tr.ToState);
                        }
                    }
                }
                FsmStateAction[] actions;
                try
                {
                    actions = state.Actions;
                }
                catch (Exception)
                {
                    failed++;
                    continue;
                }
                if (actions == null)
                {
                    continue;
                }
                foreach (FsmStateAction action in actions)
                {
                    if (action == null)
                    {
                        continue;
                    }
                    actionCount++;
                    Type t = action.GetType();
                    {
                        int n;
                        actionTypes.TryGetValue(t.Name, out n);
                        actionTypes[t.Name] = n + 1;
                    }
                    string where = PathOf(fsm.transform) + " [" + fsm.FsmName + "] " + state.Name + ": " + t.Name;
                    foreach (FieldInfo f in t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
                    {
                        string n = f.Name.ToLowerInvariant();
                        if (n == "realtime" || n == "ignoretimescale" || n == "ignorecurveoffset" && false)
                        {
                            object v = f.GetValue(action);
                            bool on = v is bool ? (bool)v : v is FsmBool ? ((FsmBool)v).Value : false;
                            string key = t.Name + "." + f.Name + (on ? "=true" : "=false");
                            int c;
                            counts.TryGetValue(key, out c);
                            counts[key] = c + 1;
                            if (on && realTime.Count < 400)
                            {
                                realTime.Add(where);
                            }
                        }
                    }
                    if (t.Name.StartsWith("AudioPlay") || t.Name == "WaitForAudioFinished" || t.Name.Contains("VideoPlayer"))
                    {
                        string detail = "";
                        foreach (FieldInfo f in t.GetFields(BindingFlags.Instance | BindingFlags.Public))
                        {
                            object v = f.GetValue(action);
                            if (v is FsmEvent)
                            {
                                detail += " " + f.Name + "=" + ((FsmEvent)v).Name;
                            }
                            else if (v is FsmBool && f.Name.ToLowerInvariant().Contains("wait"))
                            {
                                detail += " " + f.Name + "=" + ((FsmBool)v).Value;
                            }
                        }
                        int c;
                        counts.TryGetValue(t.Name, out c);
                        counts[t.Name] = c + 1;
                        if (detail != "" && audioWait.Count < 400)
                        {
                            audioWait.Add(where + detail);
                        }
                    }
                }
            }
        }
        b.Append(",\"fsm\":{\"machines\":").Append(fsmCount).Append(",\"states\":").Append(stateCount).Append(",\"actions\":").Append(actionCount).Append(",\"failed\":").Append(failed).Append(",\"flagCounts\":{");
        bool first = true;
        foreach (KeyValuePair<string, int> kv in counts)
        {
            b.Append(first ? "" : ",");
            first = false;
            HoningVisible.Str(b, kv.Key);
            b.Append(':').Append(kv.Value);
        }
        b.Append("},\"realTimeOn\":[");
        for (int i = 0; i < realTime.Count; i++)
        {
            b.Append(i > 0 ? "," : "");
            HoningVisible.Str(b, realTime[i]);
        }
        b.Append("],\"audioWithEvent\":[");
        for (int i = 0; i < audioWait.Count; i++)
        {
            b.Append(i > 0 ? "," : "");
            HoningVisible.Str(b, audioWait[i]);
        }
        b.Append("],\"visibilityEvents\":[");
        for (int i = 0; i < visibilityEvents.Count; i++)
        {
            b.Append(i > 0 ? "," : "");
            HoningVisible.Str(b, visibilityEvents[i]);
        }
        b.Append("],\"actionTypes\":{");
        first = true;
        foreach (KeyValuePair<string, int> kv in actionTypes)
        {
            b.Append(first ? "" : ",");
            first = false;
            HoningVisible.Str(b, kv.Key);
            b.Append(':').Append(kv.Value);
        }
        b.Append("}}");
        // every object, in the order of its instance id
        Transform[] all = UnityEngine.Object.FindObjectsOfType<Transform>();
        Array.Sort(all, (x, y) => x.GetInstanceID().CompareTo(y.GetInstanceID()));
        b.Append(",\"objects\":[");
        for (int i = 0; i < all.Length; i++)
        {
            b.Append(i > 0 ? "," : "");
            HoningVisible.Str(b, all[i].GetInstanceID() + ":" + all[i].gameObject.scene.name + ":" + PathOf(all[i]));
        }
        b.Append("]");
        // animators, particle systems, rigidbodies, video players
        var modes = new SortedDictionary<string, int>();
        foreach (Animator a in Resources.FindObjectsOfTypeAll<Animator>())
        {
            if (!a.gameObject.scene.IsValid())
            {
                continue;
            }
            string key = "Animator " + a.updateMode + " " + a.cullingMode;
            int c;
            modes.TryGetValue(key, out c);
            modes[key] = c + 1;
        }
        foreach (ParticleSystem p in Resources.FindObjectsOfTypeAll<ParticleSystem>())
        {
            if (!p.gameObject.scene.IsValid())
            {
                continue;
            }
            string key = "ParticleSystem unscaled=" + (p.main.useUnscaledTime ? 1 : 0) + " collision=" + (p.collision.enabled ? 1 : 0) + " trigger=" + (p.trigger.enabled ? 1 : 0)
                         + " autoSeed=" + (p.useAutoRandomSeed ? 1 : 0);
            int c;
            modes.TryGetValue(key, out c);
            modes[key] = c + 1;
        }
        foreach (Rigidbody2D r in Resources.FindObjectsOfTypeAll<Rigidbody2D>())
        {
            if (!r.gameObject.scene.IsValid())
            {
                continue;
            }
            string key = "Rigidbody2D " + r.bodyType + " " + r.interpolation + " " + r.collisionDetectionMode + " " + r.sleepMode;
            int c;
            modes.TryGetValue(key, out c);
            modes[key] = c + 1;
        }
        foreach (UnityEngine.Video.VideoPlayer v in Resources.FindObjectsOfTypeAll<UnityEngine.Video.VideoPlayer>())
        {
            string key = "VideoPlayer timeReference=" + v.timeReference + " timeSource=" + v.timeSource;
            int c;
            modes.TryGetValue(key, out c);
            modes[key] = c + 1;
        }
        foreach (MonoBehaviour m in Resources.FindObjectsOfTypeAll<MonoBehaviour>())
        {
            if (m == null || !m.gameObject.scene.IsValid())
            {
                continue;
            }
            string key = "Behaviour " + m.GetType().FullName;
            int c;
            modes.TryGetValue(key, out c);
            modes[key] = c + 1;
        }
        b.Append(",\"components\":{");
        first = true;
        foreach (KeyValuePair<string, int> kv in modes)
        {
            b.Append(first ? "" : ",");
            first = false;
            HoningVisible.Str(b, kv.Key);
            b.Append(':').Append(kv.Value);
        }
        b.Append("}}");
        return b.ToString();
    }
}

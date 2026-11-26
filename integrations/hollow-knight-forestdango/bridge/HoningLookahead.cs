// Added by the Honing devmode study (2026-11-25). NOT part of the author's project. Used only by HoningBridge / HoningDev.
//
// Lookahead by process copy: "if the knight plays these windows from this moment, what happens?", and then the game
// is back at the present, exactly.
//
// The developer command {"action":"dev","op":"lookahead","name":"<dir name>","plan":[<windows>], ...} makes the game
// process copy itself (the operating system's fork: the copy shares all memory with the original until one of them
// writes to it, so nothing has to be saved or restored and nothing can be forgotten: physics, state machines,
// coroutines, the random generator, everything is in the copy). The COPY plays the planned windows at full speed,
// writes what it saw into <session>/lookahead/<name>/ (the same files a session has: observations.jsonl,
// visible_observations.jsonl, frames.jsonl with HONING_FRAME_LOG) and ends. The ORIGINAL waits for the copy to end
// and answers the command; its own state has not changed.
//   plan      windows as in action.json: {"action":"keys","frames":8,"hold":["RIGHT"],"press":["JUMP"]} ...
//   seconds   how long the original waits for the copy before it stops it (default 30 s of wall time)
//   value     if not 0, the copy re-seeds the game's random generator with it first: "a future" instead of "the
//             future" (with 0 the copy has the original's generator, so it sees exactly what will happen)
//   name      directory name under <session>/lookahead/ (default la<n>)
// Only without a window (-batchmode -nographics): a copy of a process that owns a window and a graphics context would
// share them with the original. Only the thread that runs the game exists in the copy (the engine's worker threads do
// not), so the copy turns the garbage collector off for its short life and must not wait for a worker.
//
// Whether a harness may hand simulated futures to the decision model is the user's decision; this file only provides
// the capability.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using UnityEngine;

public static class HoningLookahead
{
    [DllImport("libc", SetLastError = true)]
    private static extern int fork();

    [DllImport("libc", SetLastError = true)]
    private static extern int waitpid(int pid, out int status, int options);

    [DllImport("libc")]
    private static extern void _exit(int status);

    [DllImport("libc", SetLastError = true)]
    private static extern int kill(int pid, int sig);

    [Serializable]
    private class Request
    {
        public string op, name;
        public float seconds;
        public int value;
        public bool flag;
        public HoningBridge.Command[] plan;
    }

    public static bool InChild;
    public static string Dir;
    public static int Count;
    private static HoningBridge.Command[] plan;
    private static int next;
    private static long childStart;
    private static int plannedFrames;

    // In the original: forks, waits, returns the answer. In the copy: returns null (the bridge then plays the plan).
    public static string Run(string json, HoningBridge bridge, string session)
    {
        Request r;
        try
        {
            r = JsonUtility.FromJson<Request>(json);
        }
        catch (Exception e)
        {
            return "{\"ok\":false,\"error\":\"unreadable lookahead request: " + e.Message.Replace("\"", "'") + "\"}";
        }
        if (r.plan == null || r.plan.Length == 0)
        {
            return "{\"ok\":false,\"error\":\"empty plan\"}";
        }
        if (!Application.isBatchMode && !r.flag)
        {
            return "{\"ok\":false,\"error\":\"lookahead copies the game process; run without a window (-batchmode -nographics)\"}";
        }
        Count++;
        string name = string.IsNullOrEmpty(r.name) ? "la" + Count : r.name;
        string dir = Path.Combine(session, "lookahead", name);
        Directory.CreateDirectory(dir);
        int frames = 0;
        foreach (HoningBridge.Command c in r.plan)
        {
            frames += c.frames;
        }
        bridge.FlushAll();
        // The copy has only this thread. The garbage collector stops the world by signalling every thread it knows,
        // and aborts when one of them is not there ("pthread_kill failed at suspend", measured): it is switched off
        // before the copy is made, so that the copy never collects, and switched on again in the original.
        var gcBefore = UnityEngine.Scripting.GarbageCollector.GCMode;
        UnityEngine.Scripting.GarbageCollector.GCMode = UnityEngine.Scripting.GarbageCollector.Mode.Disabled;
        double limit = r.seconds > 0f ? r.seconds : 5.0 + frames * 0.05;
        string ready = Path.Combine(dir, "ready"), progress = Path.Combine(dir, "progress");
        long t0 = Stopwatch.GetTimestamp();
        int attempts = 0, stuckAtStart = 0, stuckLater = 0, pid = -1, status = 0;
        bool ended = false, killed = false;
        double forkMs = 0.0;
        string lastStuck = "";
        while (attempts < 8)
        {
            attempts++;
            foreach (string f in Directory.GetFiles(dir))
            {
                File.Delete(f);
            }
            long tf = Stopwatch.GetTimestamp();
            pid = fork();
            if (pid < 0)
            {
                UnityEngine.Scripting.GarbageCollector.GCMode = gcBefore;
                return "{\"ok\":false,\"error\":\"fork failed, errno " + Marshal.GetLastWin32Error() + "\"}";
            }
            if (pid == 0)
            {
                // ---- the copy
                InChild = true;
                Dir = dir;
                plan = r.plan;
                next = 0;
                plannedFrames = frames;
                childStart = Stopwatch.GetTimestamp();
                UnityEngine.Debug.unityLogger.logEnabled = false;   // the engine log file belongs to the original
                // The sound system's mixing thread does not exist in the copy. If it held one of the mixer's locks at
                // the instant of the copy, the copy would wait for that lock for ever at its first sound, or the first
                // time a sound source is removed (measured: about one copy in eight, all of them in FMOD's critical
                // sections). So the copy goes through both paths once, here: it plays and stops a silent sound, and
                // makes and removes a sound source. If that comes back, the locks are free for good (nothing else
                // can take them); if not, the original sees no "ready" file, ends this copy and makes another.
                if (probe != null)
                {
                    probe.Play();
                    probe.Stop();
                    var go = new GameObject("Honing lookahead probe (copy)");
                    AudioSource second = go.AddComponent<AudioSource>();
                    second.playOnAwake = false;
                    second.volume = 0f;
                    second.clip = probe.clip;
                    second.Play();
                    go.SetActive(false);
                    UnityEngine.Object.DestroyImmediate(go);
                }
                File.WriteAllText(ready, "1");
                if (r.value != 0)
                {
                    UnityEngine.Random.InitState(r.value);
                }
                return null;
            }
            // ---- the original
            forkMs = HoningLoop.Ms(Stopwatch.GetTimestamp() - tf);
            long t1 = Stopwatch.GetTimestamp();
            bool isReady = false;
            while (HoningLoop.Ms(Stopwatch.GetTimestamp() - t1) < 120.0)
            {
                if (File.Exists(ready))
                {
                    isReady = true;
                    break;
                }
                if (waitpid(pid, out status, 1) == pid)
                {
                    break;      // the copy ended before it was ready
                }
                Thread.SpinWait(50);
            }
            if (!isReady)
            {
                kill(pid, 9);
                waitpid(pid, out status, 0);
                stuckAtStart++;
                continue;
            }
            // wait for the copy to end; it writes its frame number into "progress" after every window
            long lastChange = Stopwatch.GetTimestamp();
            string lastProgress = "";
            bool stuck = false;
            ended = false;
            while (true)
            {
                int got = waitpid(pid, out status, 1);      // WNOHANG
                if (got == pid)
                {
                    ended = true;
                    break;
                }
                if (got < 0)
                {
                    break;
                }
                string now = "";
                try
                {
                    if (File.Exists(progress))
                    {
                        now = File.ReadAllText(progress);
                    }
                }
                catch (Exception)
                {
                }
                if (now != lastProgress)
                {
                    lastProgress = now;
                    lastChange = Stopwatch.GetTimestamp();
                }
                else if (HoningLoop.Ms(Stopwatch.GetTimestamp() - lastChange) > 1500.0)
                {
                    stuck = true;
                }
                if (stuck || HoningLoop.Ms(Stopwatch.GetTimestamp() - t1) > limit * 1000.0)
                {
                    // SIGABRT first: the engine's own handler then writes where the copy was stuck into the log
                    kill(pid, 6);
                    long t3 = Stopwatch.GetTimestamp();
                    while (waitpid(pid, out status, 1) != pid)
                    {
                        if (HoningLoop.Ms(Stopwatch.GetTimestamp() - t3) > 3000.0)
                        {
                            kill(pid, 9);
                            waitpid(pid, out status, 0);
                            break;
                        }
                        Thread.Sleep(1);
                    }
                    killed = true;
                    lastStuck = "after " + lastProgress.Trim();
                    break;
                }
                Thread.Sleep(0);
            }
            if (killed && stuck && attempts < 8)
            {
                stuckLater++;
                killed = false;
                continue;       // the original has not moved: simply look again
            }
            break;
        }
        UnityEngine.Scripting.GarbageCollector.GCMode = gcBefore;
        long t2 = Stopwatch.GetTimestamp();
        bool exited = (status & 0x7f) == 0;
        int code = (status >> 8) & 0xff;
        string done = "null";
        try
        {
            string path = Path.Combine(dir, "done.json");
            if (File.Exists(path))
            {
                done = File.ReadAllText(path);
            }
        }
        catch (Exception)
        {
        }
        var sb = new StringBuilder();
        sb.Append("{\"ok\":").Append(ended && exited && (code == 0 || code == 4) && !killed ? "true" : "false")
          .Append(",\"complete\":").Append(ended && exited && code == 0 && !killed ? "true" : "false").Append(",\"name\":\"").Append(name).Append("\",\"pid\":").Append(pid)
          .Append(",\"planned_windows\":").Append(r.plan.Length).Append(",\"planned_frames\":").Append(frames)
          .Append(",\"copies_made\":").Append(attempts).Append(",\"stuck_at_start\":").Append(stuckAtStart).Append(",\"stuck_later\":").Append(stuckLater)
          .Append(",\"fork_ms\":").Append(HoningProbe.R(forkMs)).Append(",\"total_ms\":").Append(HoningProbe.R(HoningLoop.Ms(t2 - t0)))
          .Append(",\"exit_code\":").Append(exited ? code : -1).Append(",\"signal\":").Append(exited ? 0 : status & 0x7f).Append(",\"stopped_by_timeout\":").Append(killed ? "true" : "false")
          .Append(",\"reseeded\":").Append(r.value).Append(",\"dir\":");
        HoningVisible.Str(sb, dir);
        sb.Append(",\"copy\":").Append(done).Append('}');
        return sb.ToString();
    }

    // A silent sound source of the bridge's own, made once at the start (only without a window): what a copy plays
    // to find out whether the sound mixer's lock is free.
    private static AudioSource probe;

    public static void Prepare()
    {
        if (probe != null || !Application.isBatchMode)
        {
            return;
        }
        var go = new GameObject("Honing lookahead probe");
        UnityEngine.Object.DontDestroyOnLoad(go);
        probe = go.AddComponent<AudioSource>();
        probe.playOnAwake = false;
        probe.volume = 0f;
        probe.clip = AudioClip.Create("honing-probe", 64, 1, 44100, false);
    }

    // In the copy, after every window: a sign of life for the original
    public static void Progress(int framesPlayed)
    {
        try
        {
            File.WriteAllText(Path.Combine(Dir, "progress"), framesPlayed.ToString());
        }
        catch (Exception)
        {
        }
    }

    // In the copy, called by the game's scene-load code just before it would load a scene (overlay, SceneLoad.cs and
    // the other load sites): a copy has no loading thread, so its future ends here. The original is told how far the
    // copy got.
    public static void BeforeSceneLoad(string scene)
    {
        if (InChild && endCopy != null)
        {
            endCopy("room change: the copy stops before loading " + scene);
        }
    }

    public static Action<string> endCopy;       // set by the bridge: flush its logs and call Finish

    // In the copy: the next planned window (its id is set to the observation it answers), or null when the plan is done.
    public static HoningBridge.Command NextWindow(int sequence)
    {
        if (plan == null || next >= plan.Length)
        {
            return null;
        }
        HoningBridge.Command c = plan[next++];
        c.id = sequence;
        if (string.IsNullOrEmpty(c.action))
        {
            c.action = "keys";
        }
        return c;
    }

    // In the copy: write the summary and end the process at once (no engine shutdown: everything else is the original's).
    public static void Finish(string note, int framesPlayed)
    {
        try
        {
            double ms = HoningLoop.Ms(Stopwatch.GetTimestamp() - childStart);
            File.WriteAllText(Path.Combine(Dir, "done.json"),
                "{\"windows_played\":" + next + ",\"planned_frames\":" + plannedFrames + ",\"frame_at_end\":" + framesPlayed + ",\"wall_ms\":" + HoningProbe.R(ms)
                + ",\"note\":\"" + (note ?? "").Replace("\"", "'") + "\"}");
        }
        catch (Exception)
        {
        }
        _exit(note == null ? 0 : note.StartsWith("room change") ? 4 : 3);
    }
}

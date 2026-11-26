// Added by the Honing devmode study (2026-11-25). NOT part of the author's project.
//
// A virtual wall clock for the game's own scripts (knob HONING_CLOCK=virtual; default "real" = the engine's clock,
// exactly as before).
//
// Why. The bridge fixes the GAME time step (Time.captureFramerate: every frame is 1/HONING_FRAME_HZ s of game time),
// but several things in the game are timed with the WALL clock on purpose (Time.unscaledDeltaTime,
// Time.realtimeSinceStartup, WaitForSecondsRealtime, PlayMaker's FsmTime.RealtimeSinceStartup): the freeze after the
// knight is hit (GameManager.FreezeMoment), the black-screen wait of a room change, menu fades, real-time waits of
// state machines. Their length in FRAMES therefore depends on how fast the machine renders: at 8 frames per second a
// 0.25 s freeze is 2 frames, at 2000 frames per second it is 500 frames. With the virtual clock every frame counts as
// exactly 1/HONING_FRAME_HZ s of wall time as well, i.e. the game behaves as on a machine that renders exactly
// HONING_FRAME_HZ frames per second, whatever the real speed.
//
// How. C# gives a type declared in an assembly's own source precedence over an imported type of the same full name
// (compiler warning CS0436). The three types below are compiled into Assembly-CSharp, so every script of the game
// and every PlayMaker action compiled with it binds to them; nothing in the author's files is edited. With
// HONING_CLOCK=real each member forwards to the engine's own (same values, same calls). Code in other assemblies
// (PlayMaker.dll, InControl.dll, Assembly-CSharp-firstpass.dll, the engine itself) keeps the real clock.
extern alias UECore;
extern alias PMDll;
using System;
using System.Globalization;
using RealTime = UECore::UnityEngine.Time;

public static class HoningClock
{
    public static readonly bool Virtual;
    public static readonly double Step;        // seconds of wall time one frame stands for
    private static int lastFrame = -1;
    private static double now;                 // virtual wall time at the start of the current frame
    private static double offset;              // subtracted from what scripts see (not used yet)
    // A developer control may let one frame count as more wall time (HONING_DEV_SKIP_FADE); 0 = the normal step.
    public static double StepOverride;

    static HoningClock()
    {
        string session = Environment.GetEnvironmentVariable("HONING_SESSION_DIR");
        string clock = Environment.GetEnvironmentVariable("HONING_CLOCK");
        string hz = Environment.GetEnvironmentVariable("HONING_FRAME_HZ");
        int n;
        if (string.IsNullOrEmpty(hz) || !int.TryParse(hz, NumberStyles.Integer, CultureInfo.InvariantCulture, out n) || n <= 0)
        {
            n = 30;
        }
        Step = 1.0 / n;
        Virtual = !string.IsNullOrEmpty(session) && (clock == "virtual" || Environment.GetEnvironmentVariable("HONING_FAST") == "1");
    }

    private static void Sync()
    {
        int f = RealTime.frameCount;
        if (f != lastFrame)
        {
            if (lastFrame >= 0)
            {
                now += (f - lastFrame) * (StepOverride > 0.0 ? StepOverride : Step);
            }
            lastFrame = f;
        }
    }

    public static double Now
    {
        get
        {
            Sync();
            return now - offset;
        }
    }

    // the engine's own values, whatever the knob (for logs and for the bridge's own timeouts)
    public static double EngineTimeAsDouble { get { return RealTime.timeAsDouble; } }
    public static double EngineFixedTimeAsDouble { get { return RealTime.fixedTimeAsDouble; } }
    public static float EngineDeltaTime { get { return RealTime.deltaTime; } }
    public static float EngineTimeScale { get { return RealTime.timeScale; } }
    public static float EngineUnscaledDeltaTime { get { return RealTime.unscaledDeltaTime; } }
    public static float EngineRealtimeSinceStartup { get { return RealTime.realtimeSinceStartup; } }
    public static int EngineFrameCount { get { return RealTime.frameCount; } }
}

namespace UnityEngine
{
    // Shadows UnityEngine.Time for the code compiled into Assembly-CSharp. See the text at the top.
    public static class Time
    {
        public static float time { get { return RealTime.time; } }
        public static double timeAsDouble { get { return RealTime.timeAsDouble; } }
        public static float timeSinceLevelLoad { get { return RealTime.timeSinceLevelLoad; } }
        public static double timeSinceLevelLoadAsDouble { get { return RealTime.timeSinceLevelLoadAsDouble; } }
        public static float deltaTime { get { return RealTime.deltaTime; } }
        public static float fixedTime { get { return RealTime.fixedTime; } }
        public static double fixedTimeAsDouble { get { return RealTime.fixedTimeAsDouble; } }
        public static float fixedDeltaTime { get { return RealTime.fixedDeltaTime; } set { RealTime.fixedDeltaTime = value; } }
        public static float maximumDeltaTime { get { return RealTime.maximumDeltaTime; } set { RealTime.maximumDeltaTime = value; } }
        public static float smoothDeltaTime { get { return RealTime.smoothDeltaTime; } }
        public static float maximumParticleDeltaTime { get { return RealTime.maximumParticleDeltaTime; } set { RealTime.maximumParticleDeltaTime = value; } }
        public static float timeScale { get { return RealTime.timeScale; } set { RealTime.timeScale = value; } }
        public static int frameCount { get { return RealTime.frameCount; } }
        public static int renderedFrameCount { get { return RealTime.renderedFrameCount; } }
        public static float captureDeltaTime { get { return RealTime.captureDeltaTime; } set { RealTime.captureDeltaTime = value; } }
        public static int captureFramerate { get { return RealTime.captureFramerate; } set { RealTime.captureFramerate = value; } }
        public static bool inFixedTimeStep { get { return RealTime.inFixedTimeStep; } }

        // ---- the wall clock: virtual when HONING_CLOCK=virtual
        public static float unscaledDeltaTime
        {
            get { return HoningClock.Virtual ? (RealTime.inFixedTimeStep ? RealTime.fixedDeltaTime : (float)(HoningClock.StepOverride > 0.0 ? HoningClock.StepOverride : HoningClock.Step)) : RealTime.unscaledDeltaTime; }
        }

        public static float fixedUnscaledDeltaTime
        {
            get { return HoningClock.Virtual ? RealTime.fixedDeltaTime : RealTime.fixedUnscaledDeltaTime; }
        }

        public static float unscaledTime
        {
            get { return HoningClock.Virtual ? (float)HoningClock.Now : RealTime.unscaledTime; }
        }

        public static double unscaledTimeAsDouble
        {
            get { return HoningClock.Virtual ? HoningClock.Now : RealTime.unscaledTimeAsDouble; }
        }

        public static float fixedUnscaledTime
        {
            get { return HoningClock.Virtual ? (float)HoningClock.Now : RealTime.fixedUnscaledTime; }
        }

        public static double fixedUnscaledTimeAsDouble
        {
            get { return HoningClock.Virtual ? HoningClock.Now : RealTime.fixedUnscaledTimeAsDouble; }
        }

        public static float realtimeSinceStartup
        {
            get { return HoningClock.Virtual ? (float)HoningClock.Now : RealTime.realtimeSinceStartup; }
        }

        public static double realtimeSinceStartupAsDouble
        {
            get { return HoningClock.Virtual ? HoningClock.Now : RealTime.realtimeSinceStartupAsDouble; }
        }
    }

    // The engine's own WaitForSecondsRealtime (UnityEngine.CoreModule, 2020.3) is this managed class; the copy reads
    // the clock through the Time above, so it follows the knob.
    public class WaitForSecondsRealtime : CustomYieldInstruction
    {
        public float waitTime { get; set; }
        private float waitUntilTime = -1f;

        public override bool keepWaiting
        {
            get
            {
                if (waitUntilTime < 0f)
                {
                    waitUntilTime = Time.realtimeSinceStartup + waitTime;
                }
                bool wait = Time.realtimeSinceStartup < waitUntilTime;
                if (!wait)
                {
                    Reset();
                }
                return wait;
            }
        }

        public WaitForSecondsRealtime(float time)
        {
            waitTime = time;
        }

        public override void Reset()
        {
            waitUntilTime = -1f;
        }
    }
}

namespace HutongGames.PlayMaker
{
    // Shadows PlayMaker's FsmTime for the actions compiled into Assembly-CSharp (the only member they use).
    public static class FsmTime
    {
        public static float RealtimeSinceStartup
        {
            get { return HoningClock.Virtual ? (float)HoningClock.Now : PMDll::HutongGames.PlayMaker.FsmTime.RealtimeSinceStartup; }
        }
    }
}

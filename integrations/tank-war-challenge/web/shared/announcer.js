// A fighting-game announcer for the human challenge: a big line over the arena ("ROUND 1", "FIGHT!",
// "K.O.!") and the same line spoken with the browser's speech synthesis, low and slow. In-play callouts
// ("CLOSE ONE!") use a small line in the arena's top-right corner instead, so they never cover the play.
const VOICES = ["Daniel", "Alex", "Fred", "Google UK English Male", "Microsoft David", "Microsoft Guy"];
export function createAnnouncer(banner, corner = banner) {
    const settings = { enabled: true, volume: 0.6 };
    const speech = globalThis.speechSynthesis;
    let voice = null;
    const hide = new Map();
    const last = new Map(),
        timers = new Set();
    const pick = () => {
        const voices = speech?.getVoices() ?? [];
        voice =
            VOICES.map((name) => voices.find((v) => v.name.startsWith(name))).find(Boolean) ??
            voices.find((v) => v.lang?.startsWith("en")) ??
            null;
    };
    pick();
    speech?.addEventListener?.("voiceschanged", pick);
    function show(element, text, tone) {
        element.textContent = text;
        element.dataset.tone = tone;
        element.classList.remove("show");
        void element.offsetWidth; // restart the animation
        element.classList.add("show");
        clearTimeout(hide.get(element));
        hide.set(element, setTimeout(() => element.classList.remove("show"), 1500));
    }
    function speak(text, interrupt) {
        if (!speech) return;
        if (interrupt) speech.cancel();
        const line = new SpeechSynthesisUtterance(text);
        if (voice) line.voice = voice;
        line.pitch = 0.55;
        line.rate = 0.9;
        line.volume = settings.volume;
        speech.speak(line);
    }
    return {
        set(next) {
            Object.assign(settings, next);
            if (!settings.enabled) this.clear();
        },
        // tone: "hot" (red), "cool" (blue) or "gold"; cooldown_ms keeps a line from repeating too often.
        say(text, { spoken = text, tone = "gold", interrupt = false, cooldownMs = 0, delayMs = 0, inCorner = false } = {}) {
            if (!settings.enabled) return;
            const run = () => {
                const now = performance.now();
                if (cooldownMs && now - (last.get(text) ?? -Infinity) < cooldownMs) return;
                last.set(text, now);
                show(inCorner ? corner : banner, text, tone);
                speak(spoken, interrupt);
            };
            if (!delayMs) return run();
            const timer = setTimeout(() => (timers.delete(timer), run()), delayMs);
            timers.add(timer);
        },
        clear() {
            for (const timer of timers) clearTimeout(timer);
            timers.clear();
            speech?.cancel();
            banner.classList.remove("show");
            corner.classList.remove("show");
        },
    };
}

// Sound effects for the human challenge, synthesized with the Web Audio API (no audio files): shells,
// bounces, shield and armor hits, explosions, round starts and results. Browsers only start audio after a
// click or key press, so the context is created by unlock(), which the page calls on the first one.
export function createSounds({ volume = 0.6, muted = false } = {}) {
    const settings = { volume, muted };
    let ctx = null,
        master = null,
        noiseBuffer = null,
        lastBounce = 0,
        lastDryFire = 0;
    const live = () => (ctx && !settings.muted && ctx.state === "running" ? ctx : null);
    function envelope(node, start, peak, duration) {
        const gain = ctx.createGain();
        gain.gain.setValueAtTime(0.0001, start);
        gain.gain.exponentialRampToValueAtTime(peak, start + 0.005);
        gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
        node.connect(gain).connect(master);
    }
    function tone(type, from, to, duration, peak, delay = 0) {
        if (!live()) return;
        const start = ctx.currentTime + delay,
            osc = ctx.createOscillator();
        osc.type = type;
        osc.frequency.setValueAtTime(from, start);
        osc.frequency.exponentialRampToValueAtTime(to, start + duration);
        envelope(osc, start, peak, duration);
        osc.start(start);
        osc.stop(start + duration + 0.02);
    }
    function noise(filterType, from, to, duration, peak, delay = 0) {
        if (!live()) return;
        const start = ctx.currentTime + delay,
            source = ctx.createBufferSource(),
            filter = ctx.createBiquadFilter();
        source.buffer = noiseBuffer;
        filter.type = filterType;
        filter.frequency.setValueAtTime(from, start);
        filter.frequency.exponentialRampToValueAtTime(to, start + duration);
        source.connect(filter);
        envelope(filter, start, peak, duration);
        source.start(start);
        source.stop(start + duration + 0.02);
    }
    return {
        unlock() {
            if (!ctx) {
                const AudioContext = globalThis.AudioContext || globalThis.webkitAudioContext;
                if (!AudioContext) return;
                ctx = new AudioContext();
                master = ctx.createGain();
                master.gain.value = settings.volume;
                master.connect(ctx.destination);
                noiseBuffer = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
                const data = noiseBuffer.getChannelData(0);
                for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
            }
            if (ctx.state === "suspended") void ctx.resume();
        },
        set(next) {
            Object.assign(settings, next);
            if (master) master.gain.value = settings.volume;
        },
        shot(player) {
            const pitch = player === 0 ? 1 : 0.82;
            tone("square", 340 * pitch, 110 * pitch, 0.09, 0.16);
            noise("bandpass", 2400, 700, 0.08, 0.28);
        },
        // Trying to fire with every shell of the magazine in flight: a quiet click, at most every 0.15 s.
        dryFire() {
            if (!live() || ctx.currentTime - lastDryFire < 0.15) return;
            lastDryFire = ctx.currentTime;
            noise("highpass", 6000, 3000, 0.025, 0.07);
            tone("square", 2400, 1800, 0.02, 0.03);
        },
        bounce() {
            if (!live() || ctx.currentTime - lastBounce < 0.04) return;
            lastBounce = ctx.currentTime;
            tone("triangle", 1500, 1100, 0.035, 0.05);
        },
        shield() {
            tone("sine", 1760, 1700, 0.28, 0.15);
            tone("sine", 2640, 2500, 0.2, 0.07);
        },
        armor() {
            tone("sawtooth", 200, 70, 0.2, 0.18);
            noise("lowpass", 900, 200, 0.15, 0.22);
        },
        crumble() {
            noise("lowpass", 700, 90, 0.45, 0.5);
            tone("sine", 70, 40, 0.35, 0.3);
            noise("bandpass", 3000, 900, 0.12, 0.12, 0.05);
        },
        explosion() {
            noise("lowpass", 1800, 60, 0.8, 0.65);
            tone("sine", 110, 35, 0.7, 0.4);
        },
        roundStart() {
            tone("triangle", 660, 660, 0.08, 0.1);
            tone("triangle", 990, 990, 0.12, 0.1, 0.12);
        },
        win() {
            [523, 659, 784, 1047].forEach((f, i) => tone("triangle", f, f, 0.16, 0.12, i * 0.11));
        },
        lose() {
            [392, 330, 262].forEach((f, i) => tone("triangle", f, f * 0.98, 0.22, 0.12, i * 0.16));
        },
        draw() {
            tone("triangle", 440, 440, 0.2, 0.1);
            tone("triangle", 440, 440, 0.2, 0.1, 0.25);
        },
    };
}

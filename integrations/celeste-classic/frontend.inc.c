/* Included before main() in the pinned SDL2 frontend. No controller code runs while paused. */
#include <ctype.h>
#include <limits.h>
extern void Honing_update(void);
extern void Honing_draw(void);
extern void Honing_begin(unsigned seed);
extern void Honing_state(FILE* out);
static const char* honing_dir = NULL;
static unsigned long honing_id = 1, honing_ticks = 0, honing_episode_ticks = 0;
static unsigned long honing_rejected = 0, honing_resets = 0;
static unsigned honing_mask = 0, honing_frames = 0;
static Uint64 honing_last_valid = 0, honing_timeout_ms = 240000;
static void* honing_initial = NULL;
static int honing_capture = 0;

static void honing_path(char* out, const char* name) {
    if (snprintf(out, 4096, "%s/%s", honing_dir, name) >= 4096) exit(12);
}
static void honing_publish(const char* status) {
    char path[4096], temporary[4096];
    honing_path(path, "observation.json"); honing_path(temporary, "observation.tmp");
    FILE* out = fopen(temporary, "w");
    if (!out) exit(12);
    fprintf(out, "{\"id\":%lu,\"status\":\"%s\",\"ticks\":%lu,\"episodeTicks\":%lu,"
        "\"setupTicks\":60,\"lastMask\":%u,\"lastFrames\":%u,\"resets\":%lu,\"rejected\":%lu,",
        honing_id, status, honing_ticks, honing_episode_ticks, honing_mask,
        honing_frames, honing_resets, honing_rejected);
    Honing_state(out);
    fprintf(out, "}\n");
    if (ferror(out) || fclose(out) || rename(temporary, path)) exit(12);
}
static int honing_number(const char* text, unsigned long* value) {
    if (!*text) return 0;
    for (const char* p = text; *p; p++) if (!isdigit((unsigned char)*p)) return 0;
    errno = 0; *value = strtoul(text, NULL, 10);
    return errno == 0 && *value <= 1000000000UL;
}
static void honing_start(void) {
    honing_dir = getenv("HONING_SESSION_DIR");
    if (!honing_dir || !*honing_dir) { honing_dir = NULL; return; }
    const char* timeout = getenv("HONING_WATCHDOG_SECONDS");
    unsigned long seconds = 240;
    if (timeout && (!honing_number(timeout, &seconds) || seconds < 1 || seconds > 3600)) exit(12);
    honing_timeout_ms = seconds * 1000;
    honing_capture = getenv("HONING_CAPTURE") && !strcmp(getenv("HONING_CAPTURE"), "1");
    unsigned long seed = 7;
    const char* seed_text = getenv("HONING_SEED");
    if (seed_text && !honing_number(seed_text, &seed)) exit(12);
    Celeste_P8_load_state(initial_game_state);
    Honing_begin((unsigned)seed);
    buttons_state = 0;
    for (int i = 0; i < 60; i++) { Honing_update(); Honing_draw(); }
    honing_initial = malloc(Celeste_P8_get_state_size());
    if (!honing_initial) exit(12);
    Celeste_P8_save_state(honing_initial);
    SDL_Flip(screen);
    honing_last_valid = SDL_GetTicks64();
    honing_publish("paused");
}
static void honing_loop(void) {
    /* Deadline is checked on every poll, including a flood of invalid/stale commands. */
    if (SDL_GetTicks64() - honing_last_valid >= honing_timeout_ms) {
        honing_publish("watchdog"); running = 0; return;
    }
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) if (ev.type == SDL_QUIT) {
        honing_publish("window_closed"); running = 0; return;
    }
    char path[4096]; honing_path(path, "command.txt");
    FILE* in = fopen(path, "rb");
    if (!in) { SDL_Flip(screen); SDL_Delay(10); return; }
    char command[256];
    size_t size = fread(command, 1, sizeof(command) - 1, in);
    int too_long = fgetc(in) != EOF;
    fclose(in); remove(path); command[size] = 0;
    char op[16], seq[32], mask[32], count[32], extra;
    unsigned long id = 0, bits = 0, ticks = 0;
    const char* error = NULL;
    if (too_long || memchr(command, 0, size) ||
        sscanf(command, "%15s %31s %31s %31s %c", op, seq, mask, count, &extra) != 4 ||
        !honing_number(seq, &id) || !honing_number(mask, &bits) || !honing_number(count, &ticks))
        error = "syntax";
    else if (id != honing_id) error = "stale_or_future_id";
    else if (!strcmp(op, "step")) {
        if (bits > 63 || (bits & 3) == 3 || (bits & 12) == 12 || ticks < 1 || ticks > 30)
            error = "input_range";
    } else if ((strcmp(op, "reset") && strcmp(op, "stop")) || bits || ticks) error = "operation";
    if (error) {
        honing_rejected++;
        char name[80]; snprintf(name, sizeof name, "rejected-%06lu.raw", honing_rejected);
        honing_path(path, name); FILE* raw = fopen(path, "wb");
        if (!raw) exit(12);
        fwrite(command, 1, size, raw); fclose(raw);
        honing_path(path, "native_rejections.jsonl"); FILE* log = fopen(path, "a");
        if (!log) exit(12);
        fprintf(log, "{\"reason\":\"%s\",\"id\":%lu,\"ticks\":%lu}\n", error, honing_id, honing_ticks);
        fclose(log); honing_publish("paused"); SDL_Delay(10); return;
    }
    honing_last_valid = SDL_GetTicks64();
    honing_mask = (unsigned)bits; honing_frames = (unsigned)ticks;
    if (!strcmp(op, "stop")) { honing_id++; honing_publish("stopped"); running = 0; return; }
    if (!strcmp(op, "reset")) {
        Celeste_P8_load_state(honing_initial); honing_episode_ticks = 0; honing_resets++;
        buttons_state = 0; /* Reset performs no draw: drawing mutates the restored game state. */
    } else {
        buttons_state = (Uint16)bits;
        for (unsigned long i = 0; i < ticks; i++) {
            Honing_update(); Honing_draw(); honing_ticks++; honing_episode_ticks++;
            if (honing_capture) {
                char name[80]; snprintf(name, sizeof name, "frames/%06lu.bmp", honing_ticks);
                honing_path(path, name);
                if (SDL_SaveBMP(screen, path)) exit(12);
            }
            SDL_Flip(screen);
        }
    }
    honing_id++; honing_publish("paused");
}

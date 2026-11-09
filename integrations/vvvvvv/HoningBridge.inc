// Included by main.cpp to reuse its original fixed/render loop without replacing physics.
// Honing experimental modification; original VVVVVV is copyright Terry Cavanagh.
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>

static std::string honing_dir;
static std::string honing_action = "wait";
static int honing_id = 1;
static int honing_tick = 0;
static int honing_milliseconds = 0;
static int honing_epoch = 0;
static int honing_capture_index = 0;
static int honing_action_tick = 0;
static std::string honing_last_action = "reset";
static int honing_last_ticks = 0;

bool honing_active(void) { return getenv("HONING_SESSION_DIR") != NULL; }

void honing_input(void)
{
    if (!honing_active()) return;
    game.press_left = honing_action == "left" || honing_action == "flip_left";
    game.press_right = honing_action == "right" || honing_action == "flip_right";
    // A flip is a one-tick edge. Consecutive flip commands have a release tick.
    game.press_action = honing_action_tick == 1 && honing_action.find("flip") == 0;
    game.press_interact = false;
}

static void honing_one_tick(void)
{
    key.isActive = true;
    accumulator = 0;
    timePrev = 0;
    time_ = game.get_timestep();
    deltaloop();
}

static void honing_reset(void)
{
    honing_action = "wait";
    honing_action_tick = 0;
    script.startgamemode(Start_TIMETRIAL_SPACESTATION1);
    gamestate_funcs = get_gamestate_funcs(game.gamestate, &num_gamestate_funcs);
    gamestate_func_index = -1;
    meta_func_index = 0;
    loop_assign_active_funcs();
    // Run the original trial countdown/fade, not a fabricated game state.
    for (int i = 0; i < 165; ++i) honing_one_tick();
    honing_tick = 0;
    honing_milliseconds = 0;
}

static void honing_capture(void)
{
    if (getenv("HONING_CAPTURE") == NULL) return;
    SDL_Texture* previous = SDL_GetRenderTarget(gameScreen.m_renderer);
    SDL_SetRenderTarget(gameScreen.m_renderer, graphics.gameTexture);
    SDL_Surface* surface = SDL_RenderReadPixels(gameScreen.m_renderer, NULL);
    if (surface == NULL) VVV_exit(2);
    char name[80];
    SDL_snprintf(name, sizeof(name), "/frames/%06d.bmp", honing_capture_index++);
    if (!SDL_SaveBMP(surface, (honing_dir + name).c_str())) VVV_exit(2);
    SDL_DestroySurface(surface);
    SDL_SetRenderTarget(gameScreen.m_renderer, previous);
}

static void honing_observe(void)
{
    const int p = obj.getplayer();
    if (p < 0 || p >= (int)obj.entities.size()) VVV_exit(2);
    const entclass& e = obj.entities[p];
    std::ofstream out((honing_dir + "/observation.tmp").c_str());
    out << std::setprecision(8) << "{\"id\":" << honing_id << ",\"tick\":" << honing_tick
        << ",\"epoch\":" << honing_epoch << ",\"engineFrame\":" << game.framecounter
        << ",\"simulationMilliseconds\":" << honing_milliseconds
        << ",\"tickMilliseconds\":" << game.get_timestep()
        << ",\"lastAction\":\"" << honing_last_action << "\",\"lastActionTicks\":" << honing_last_ticks
        << ",\"room\":[" << game.roomx << ',' << game.roomy << "]"
        << ",\"player\":{\"x\":" << e.xp << ",\"y\":" << e.yp
        << ",\"vx\":" << e.vx << ",\"vy\":" << e.vy
        << ",\"width\":" << e.w << ",\"height\":" << e.h
        << ",\"onGround\":" << e.onground << ",\"onRoof\":" << e.onroof << "}"
        << ",\"gravity\":" << game.gravitycontrol << ",\"deathSequence\":" << game.deathseq
        << ",\"deaths\":" << game.deathcounts << ",\"flips\":" << game.totalflips
        << ",\"hasControl\":" << (game.hascontrol ? "true" : "false")
        << ",\"checkpoint\":[" << game.saverx << ',' << game.savery << ',' << game.savex << ',' << game.savey << ',' << game.savegc << "]"
        << ",\"localOriginTiles\":[" << e.xp / 8 - 4 << ',' << e.yp / 8 - 4 << "]"
        << ",\"localSolidRows\":[";
    for (int y = -4; y <= 4; ++y) {
        if (y != -4) out << ',';
        out << '"';
        for (int x = -4; x <= 4; ++x) {
            int tx = e.xp / 8 + x, ty = e.yp / 8 + y;
            out << (tx < 0 || tx >= 40 || ty < 0 || ty >= 29 ? '?' : (map.collide(tx, ty, false) ? '#' : (map.collide(tx, ty, true) ? '!' : '.')));
        }
        out << '"';
    }
    out << "],\"entities\":[";
    int count = 0;
    for (size_t i = 0; i < obj.entities.size(); ++i) {
        const entclass& other = obj.entities[i];
        if ((int)i == p || other.invis || other.type == EntityType_INVALID || other.xp < 0 || other.xp >= 320 || other.yp < 0 || other.yp >= 232) continue;
        if (count++ != 0) out << ',';
        out << "[" << other.type << ',' << other.xp << ',' << other.yp << ',' << other.w << ',' << other.h << ',' << (other.harmful ? 1 : 0) << ']';
    }
    out << "]}";
    out.close();
    if (!out || rename((honing_dir + "/observation.tmp").c_str(), (honing_dir + "/observation.json").c_str())) VVV_exit(2);
}

void honing_run(void)
{
    honing_dir = getenv("HONING_SESSION_DIR");
    SDL_SetWindowTitle(gameScreen.m_window, "VVVVVV - experimental Honing bridge");
    SDL_SetRenderVSync(gameScreen.m_renderer, 0);
    game.over30mode = false;
    honing_reset();
    honing_observe();
    Uint64 last_command = SDL_GetTicks();
    Uint64 idle_timeout = 600000;
    const char* timeout_text = getenv("HONING_IDLE_TIMEOUT_MS");
    if (timeout_text != NULL) {
        char* end = NULL;
        long value = strtol(timeout_text, &end, 10);
        if (*end != '\0' || value < 100 || value > 600000) VVV_exit(2);
        idle_timeout = (Uint64)value;
    }
    for (;;) {
        // Check on every pass: a flood of rejected commands must not keep an orphan alive.
        if (SDL_GetTicks() - last_command > idle_timeout) VVV_exit(3);
        std::ifstream input((honing_dir + "/command.txt").c_str());
        if (!input) {
            SDL_Delay(10);
            continue;
        }
        std::string line, extra, action;
        std::ostringstream contents;
        contents << input.rdbuf();
        line = contents.str();
        input.close();
        remove((honing_dir + "/command.txt").c_str());
        std::istringstream command(line);
        int id = 0, ticks = -1;
        bool valid = line.size() <= 128 && bool(command >> id >> action >> ticks) && !(command >> extra);
        bool control = action == "stop" || action == "reset";
        bool offered = action == "wait" || action == "left" || action == "right" || action == "flip" || action == "flip_left" || action == "flip_right";
        valid = valid && id == honing_id && ((control && ticks == 0) || (offered && ticks >= 2 && ticks <= 120));
        if (!valid) {
            std::ofstream error((honing_dir + "/native_error.txt").c_str(), std::ios::app);
            error << "Rejected command; paused observation " << honing_id << ": " << line << '\n';
            continue;
        }
        last_command = SDL_GetTicks();
        honing_last_action = action;
        honing_last_ticks = ticks;
        if (action == "stop") VVV_exit(0);
        if (action == "reset") { honing_epoch++; honing_reset(); }
        else {
            honing_action = action;
            for (int i = 0; i < ticks; ++i) {
                honing_action_tick = i;
                honing_milliseconds += game.get_timestep();
                honing_one_tick();
                honing_tick++;
                honing_capture();
            }
            honing_action = "wait";
        }
        honing_id++;
        honing_observe();
    }
}

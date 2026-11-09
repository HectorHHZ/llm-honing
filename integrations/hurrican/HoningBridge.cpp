// Hurrican remains the simulation and renderer; this module only schedules input.
#include "HoningBridge.hpp"
#include "Gameplay.hpp"
#include "Main.hpp"
#include "Timer.hpp"
#include "Tileengine.hpp"
#include "GegnerClass.hpp"
#include "Gegner_Helper.hpp"
#include "Projectiles.hpp"
#include <chrono>
#include "opengl.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
int sequence = 0;
int tick = 0;
int remaining = 0;
int action = 0;
bool initialized = false;
bool advancing = false;
int idleSeconds = 240;
std::string inputBuffer;
std::string directory;

void observation(const char *error = nullptr) {
    const auto &p = Player[0];
    int primaryProjectiles = 0;
    for (const auto *shot : Projectiles.projectiles)
        if (shot->pParent == &Player[0] && shot->ShotArt >= SPREADSHOT && shot->ShotArt <= LASERSHOTBIG2)
            ++primaryProjectiles;
    std::ostringstream out;
    out << "HONING {\"id\":" << sequence << ",\"tick\":" << tick
        << ",\"stage\":" << Stage << ",\"mode\":" << static_cast<int>(SpielZustand)
        << ",\"primaryProjectiles\":" << primaryProjectiles
        << ",\"timeLimit\":" << TileEngine.Timelimit
        << ",\"camera\":[" << TileEngine.XOffset << ',' << TileEngine.YOffset << ']'
        << ",\"player\":{\"x\":" << p.xpos << ",\"y\":" << p.ypos
        << ",\"vx\":" << p.xspeed << ",\"vy\":" << p.yspeed
        << ",\"energy\":" << p.Energy << ",\"lives\":" << p.Lives
        << ",\"shotDelay\":" << p.ShotDelay << ",\"autoFireCount\":" << p.AutoFireCount
        << ",\"score\":" << p.Score << ",\"canJump\":" << (p.JumpPossible ? "true" : "false")
        << ",\"action\":" << static_cast<int>(p.Handlung)
        << ",\"facing\":" << Direction::asInt(p.Blickrichtung) << "},\"enemies\":[";
    std::vector<const GegnerClass *> nearby;
    for (const auto &enemy : Gegner.enemies)
        if (enemy->Active && enemy->Destroyable && enemy->IsOnScreen()) nearby.push_back(enemy.get());
    std::sort(nearby.begin(), nearby.end(), [&p](auto a, auto b) {
        auto distance = [&p](auto e) { return (e->xPos-p.xpos)*(e->xPos-p.xpos)+(e->yPos-p.ypos)*(e->yPos-p.ypos); };
        return distance(a) < distance(b);
    });
    for (size_t i = 0; i < std::min<size_t>(6, nearby.size()); ++i) {
        const auto *e = nearby[i];
        if (i) out << ',';
        out << "{\"type\":" << e->GegnerArt << ",\"x\":" << e->xPos << ",\"y\":" << e->yPos
            << ",\"energy\":" << e->Energy << '}';
    }
    out << ']';
    if (error) out << ",\"error\":\"" << error << '"';
    out << '}';
    std::cout << out.str() << std::endl;
}
}

bool HoningEnabled() { return std::getenv("HONING_HURRICAN_SESSION") != nullptr; }

bool HoningBeforeTick() {
    if (!HoningEnabled()) return true;
    if (!initialized) {
        directory = std::getenv("HONING_HURRICAN_SESSION");
        if (const char *setting = std::getenv("HONING_HURRICAN_IDLE_SECONDS")) {
            std::istringstream value(setting);
            std::string extra;
            if (!(value >> idleSeconds) || (value >> extra) || idleSeconds < 1 || idleSeconds > 3600) return false;
        }
        if (!GameInit2()) return false;
        NUMPLAYERS = 1;
        RunningTutorial = false;
        Skill = SKILL_MEDIUM;
        Stage = 1;
        NewStage = 1;
        InitNewGame();
        InitNewGameLevel();
        // The original weapon arms on shoot release. Present one neutral startup
        // heartbeat before the first observation, as a normal unheld keyboard does.
        Timer.update();
        Timer.setSpeedFactor(Timer.GetMoveSpeed()/60.0f);
        Heartbeat();
        initialized = true;
        observation();
    }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(idleSeconds);
    while (remaining == 0) {
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (milliseconds <= 0) return false;
        // No simulation update while waiting; invalid traffic never renews the absolute deadline.
        struct pollfd input {STDIN_FILENO, POLLIN, 0};
        int ready = poll(&input, 1, static_cast<int>(milliseconds));
        if (ready <= 0) return false;
        char character;
        if (read(STDIN_FILENO, &character, 1) != 1) return false;
        if (character != '\n') {
            if (inputBuffer.size() >= 256) return false;
            inputBuffer += character;
            continue;
        }
        std::string line = inputBuffer;
        inputBuffer.clear();
        std::istringstream command(line);
        std::string verb, extra;
        int id = -1, frames = 0, choice = -1;
        if (!(command >> verb >> id) || id != sequence) { observation("stale or malformed command"); continue; }
        if (verb == "stop" && !(command >> extra)) return false;
        if (verb == "observe" && !(command >> extra)) { observation(); continue; }
        if (verb != "step" || !(command >> frames >> choice) || (command >> extra)
            || frames < 1 || frames > 120 || choice < 0 || choice > 9) {
            observation("invalid step"); continue;
        }
        ++sequence;
        remaining = frames;
        action = choice;
    }
    Timer.update();
    Timer.setSpeedFactor(Timer.GetMoveSpeed()/60.0f);
    advancing = true;
    return true;
}

void HoningAfterTick() {
    if (!HoningEnabled() || !advancing) return;
    advancing = false;
    ++tick;
    if (--remaining == 0) observation();
}

std::bitset<MAX_AKTIONEN> HoningActions() {
    std::bitset<MAX_AKTIONEN> keys;
    // wait, left, right, shoot, left+shoot, right+shoot, jump,
    // left+jump+shoot, right+jump+shoot, crouch+shoot.
    if (action == 1 || action == 4 || action == 7) keys[AKTION_LINKS] = true;
    if (action == 2 || action == 5 || action == 8) keys[AKTION_RECHTS] = true;
    if (action == 6 || action == 7 || action == 8) keys[AKTION_JUMP] = true;
    if (action == 3 || action == 4 || action == 5 || action == 7 || action == 8 || action == 9) keys[AKTION_SHOOT] = true;
    if (action == 9) keys[AKTION_DUCKEN] = true;
    return keys;
}

void HoningCapture(SDL_Window *window) {
    if (!HoningEnabled() || !advancing || tick % 2 != 0 || !std::getenv("HONING_HURRICAN_CAPTURE")) return;
    int width, height;
    SDL_GL_GetDrawableSize(window, &width, &height);
    std::vector<unsigned char> pixels(width * height * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    char name[64];
    snprintf(name, sizeof(name), "/frame%06d.ppm", tick/2);
    std::ofstream file(directory + name, std::ios::binary);
    file << "P6\n" << width << ' ' << height << "\n255\n";
    for (int row = height-1; row >= 0; --row)
        file.write(reinterpret_cast<char *>(pixels.data()+row*width*3), width*3);
}

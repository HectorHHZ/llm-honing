/* SPDX-License-Identifier: MIT
 * Local, opt-in Taisei v1.4.6 stage bridge. Included from stage.c.
 */
#include "honing_command.h"

static unsigned honing_id, honing_remaining, honing_flags;

static const char *honing_dir(void) {
    return getenv("HONING_SESSION_DIR");
}

static void honing_path(char *buf, size_t size, const char *name) {
    if(snprintf(buf, size, "%s/%s", honing_dir(), name) >= (int)size) exit(70);
}

static void honing_observe(void) {
    char tmp[4096], path[4096];
    honing_path(tmp, sizeof(tmp), "observation.tmp");
    honing_path(path, sizeof(path), "observation.json");
    FILE *f = fopen(tmp, "w");
    if(!f) exit(71);
    fprintf(f, "{\"id\":%u,\"frame\":%d,\"fps\":60,\"stage\":%u,\"gameover\":%d,"
        "\"viewport\":[%d,%d],\"player\":{\"x\":%.3f,\"y\":%.3f,\"lives\":%d,\"bombs\":%d,\"score\":%llu},",
        honing_id, global.frames, global.stage->id, global.gameover, VIEWPORT_W, VIEWPORT_H,
        creal(global.plr.pos), cimag(global.plr.pos), global.plr.lives, global.plr.bombs,
        (unsigned long long)global.plr.points);
    /* Bound context by reporting the nearest eight currently on-screen enemy bullets.
     * This is a partial current-state sensor; it exposes neither future scripts nor RNG.
     */
    Projectile *near[8] = {0};
    unsigned total = 0;
    for(Projectile *p = global.projs.first; p; p = p->next) {
        if(p->type != PROJ_ENEMY || creal(p->pos) < 0 || creal(p->pos) > VIEWPORT_W ||
            cimag(p->pos) < 0 || cimag(p->pos) > VIEWPORT_H) continue;
        ++total;
        for(unsigned i = 0; i < 8; ++i) {
            if(!near[i] || cabs(p->pos - global.plr.pos) < cabs(near[i]->pos - global.plr.pos)) {
                for(unsigned j = 7; j > i; --j) near[j] = near[j - 1];
                near[i] = p;
                break;
            }
        }
    }
    fprintf(f, "\"visibleEnemyBullets\":%u,\"nearestBullets\":[", total);
    for(unsigned i = 0; i < 8 && near[i]; ++i) {
        Projectile *p = near[i];
        cmplx velocity = p->move.velocity;
        fprintf(f, "%s[%.2f,%.2f,%.2f,%.2f]", i ? "," : "",
            creal(p->pos), cimag(p->pos), creal(velocity), cimag(velocity));
    }
    fprintf(f, "]}\n");
    if(fclose(f) || rename(tmp, path)) exit(72);
}

static int honing_gate(void) {
    if(!honing_dir()) return 1;
    if(!honing_remaining) {
        ++honing_id;
        honing_observe();
        Uint64 start = SDL_GetTicks();
        for(;;) {
            char path[4096], buf[256];
            honing_path(path, sizeof(path), "action.json");
            FILE *f = fopen(path, "r");
            if(f) {
                size_t count = fread(buf, 1, sizeof(buf) - 1, f);
                int extra = fgetc(f);
                fclose(f);
                remove(path);
                buf[count] = 0;
                HoningCommand command;
                if(extra == EOF && !memchr(buf, 0, count) && honing_parse_command(buf, honing_id, &command)) {
                    if(command.stop) { taisei_quit(); return 0; }
                    honing_remaining = command.frames;
                    honing_flags = command.flags;
                    break;
                }
                honing_path(path, sizeof(path), "rejected-command.raw");
                f = fopen(path, "wb");
                if(!f) exit(73);
                fwrite(buf, 1, count, f);
                fclose(f);
                honing_path(path, sizeof(path), "rejected-command.json");
                f = fopen(path, "w");
                if(!f) exit(73);
                fprintf(f, "{\"id\":%u,\"frame\":%d,\"reason\":\"invalid or stale canonical command\"}\n", honing_id, global.frames);
                fclose(f);
            }
            /* Wall-clock timeout cannot advance the paused simulation. */
            if(SDL_GetTicks() - start > 240000) exit(74);
            SDL_Delay(5);
        }
    }
    --honing_remaining;
    return 1;
}

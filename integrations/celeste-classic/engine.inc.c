/* Included in the caller's pinned celeste.c to access its existing state. */
static unsigned long honing_updates = 0, honing_draws = 0;

void Honing_update(void) { Celeste_P8_update(); honing_updates++; }
void Honing_draw(void) { Celeste_P8_draw(); honing_draws++; }
void Honing_begin(unsigned seed) {
    Celeste_P8_set_rndseed(seed);
    Celeste_P8_init();
    begin_game(); /* Fixed first-room setup, no gameplay inputs or room skipping. */
}
void Honing_state(FILE* out) {
    fprintf(out, "\"nativeUpdates\":%lu,\"nativeDraws\":%lu,\"clockFrame\":%d,"
        "\"room\":%d,\"deaths\":%d,\"freezeTicks\":%d,\"restartPending\":%s,\"player\":",
        honing_updates, honing_draws, frames, level_index(), deaths, freeze,
        will_restart ? "true" : "false");
    OBJ* player = NULL;
    bool completed = false;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (objects[i].active && objects[i].type == OBJ_PLAYER) player = &objects[i];
        if (objects[i].active && objects[i].type == OBJ_FLAG && objects[i].show) completed = true;
    }
    if (player) fprintf(out, "{\"x\":%.6g,\"y\":%.6g,\"vx\":%.6g,\"vy\":%.6g,"
        "\"dashes\":%d,\"grounded\":%s}", (double)player->x, (double)player->y,
        (double)player->spd.x, (double)player->spd.y, player->djump,
        OBJ_is_solid(player, 0, 1) ? "true" : "false");
    else fprintf(out, "null");
    fprintf(out, ",\"summitFlagTouched\":%s,\"tiles\":[", completed ? "true" : "false");
    for (int y = 0; y < 16; y++) {
        if (y) fputc(',', out);
        fputc('"', out);
        for (int x = 0; x < 16; x++) {
            int t = tile_at(x, y);
            fputc(t == 17 ? '^' : t == 27 ? 'v' : t == 43 ? '<' : t == 59 ? '>' :
                  P8fget(t, 0) ? '#' : '.', out);
        }
        fputc('"', out);
    }
    fprintf(out, "],\"entities\":[");
    bool first = true;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        OBJ* obj = &objects[i];
        if (!obj->active || obj->type == OBJ_PLAYER || obj->type == OBJ_SMOKE ||
            obj->type == OBJ_ROOM_TITLE || obj->type == OBJ_LIFEUP) continue;
        fprintf(out, "%s{\"type\":\"%s\",\"x\":%.6g,\"y\":%.6g}", first ? "" : ",",
            OBJ_PROP(obj).nam, (double)obj->x, (double)obj->y);
        first = false;
    }
    fprintf(out, "]");
}

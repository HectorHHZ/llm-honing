/* SPDX-License-Identifier: MIT */
#pragma once
#include <ctype.h>
#include <stdio.h>
#include <string.h>

typedef struct HoningCommand {
    unsigned id;
    unsigned frames;
    unsigned flags;
    int stop;
} HoningCommand;

/* This private transport accepts the controller's canonical JSON only. */
static int honing_parse_command(const char *text, unsigned expected, HoningCommand *out) {
    unsigned id, frames;
    char action[24];
    int n = 0;
    if(sscanf(text, "{\"id\":%u,\"action\":\"%23[a-z_]\",\"frames\":%u}%n",
        &id, action, &frames, &n) != 3 || !n) return 0;
    char canonical[128];
    int length = snprintf(canonical, sizeof(canonical), "{\"id\":%u,\"action\":\"%s\",\"frames\":%u}", id, action, frames);
    if(length != n || strncmp(canonical, text, n)) return 0;
    while(isspace((unsigned char)text[n])) ++n;
    if(text[n] || id != expected || frames < 1 || frames > 120) return 0;
    const char *names[] = {"stay", "left", "right", "up", "down", "up_left", "up_right", "down_left", "down_right", "stop"};
    const unsigned directions[] = {0, 4, 8, 1, 2, 5, 9, 6, 10, 0};
    for(unsigned i = 0; i < sizeof(names) / sizeof(*names); ++i) {
        if(!strcmp(action, names[i])) {
            *out = (HoningCommand){id, frames, directions[i] | 16 | 32, i == 9};
            return 1;
        }
    }
    return 0;
}

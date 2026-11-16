"""Execute the production command loop; optional cases use the pinned engine source.

The portable harness substitutes SDL and game callbacks, not command parsing or
input scheduling. HONING_CELESTE_SOURCE enables original-engine physics checks.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).parents[1] / "integrations/celeste-classic"
pytestmark = pytest.mark.unit


def compile_harness(tmp_path: Path, engine: bool = False) -> Path:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required")
    upstream = os.environ.get("HONING_CELESTE_SOURCE")
    if engine and not upstream:
        pytest.skip("set HONING_CELESTE_SOURCE to the pinned external ccleste checkout")
    if engine:
        revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        assert revision == "261d96f15af430b8111abc7a5250229246654f52"
        expected = subprocess.check_output(["git", "-C", str(upstream), "show", "HEAD:celeste.c"])
        assert (Path(upstream) / "celeste.c").read_bytes() == expected
    source = tmp_path / "controls.c"
    prelude = r"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <stdarg.h>
typedef unsigned short Uint16;
typedef unsigned long long Uint64;
typedef struct { int type; } SDL_Event;
enum { SDL_QUIT = 256 };
static Uint16 buttons_state;
static int running = 1;
static void *screen, *initial_game_state;
static Uint64 SDL_GetTicks64(void) { return 1000; }
static int SDL_PollEvent(SDL_Event *e) { (void)e; return 0; }
static void SDL_Flip(void *s) { (void)s; }
static void SDL_Delay(unsigned ms) { (void)ms; }
static int SDL_SaveBMP(void *s, const char *p) { (void)s; (void)p; return 0; }
"""
    if engine:
        prelude += '#include "celeste.c"\n#include "engine.inc.c"\n'
        prelude += r"""static int test_tiles[16][16];
static int callback(CELESTE_P8_CALLBACK_TYPE type, ...) {
    va_list args; va_start(args, type);
    int result = 0;
    if (type == CELESTE_P8_BTN) result = !!(buttons_state & (1u << va_arg(args, int)));
    else if (type == CELESTE_P8_MGET) {
        int x = va_arg(args, int), y = va_arg(args, int);
        result = y == 15 ? 1 : (x>=0 && x<16 && y>=0 && y<16 ? test_tiles[y][x] : 0);
    } else if (type == CELESTE_P8_FGET) {
        int tile = va_arg(args, int), flag = va_arg(args, int);
        result = tile == 1 && flag == 0;
    }
    va_end(args); return result;
}
static OBJ *player(void) {
    for (int i=0; i<MAX_OBJECTS; i++)
        if (objects[i].active && objects[i].type == OBJ_PLAYER) return &objects[i];
    assert(0); return NULL;
}
static void setup(void) {
    Celeste_P8_set_call_func(callback); Honing_begin(7);
    memset(objects, 0, sizeof objects);
    init_object(OBJ_PLAYER, 64, 112);
    Honing_update(); Honing_draw();
    assert(OBJ_is_solid(player(), 0, 1));
}
"""
    else:
        prelude += r"""static unsigned latched, frozen, jump_edges, dash_edges, updates, draws;
static unsigned masks[128];
static void Honing_update(void) {
    masks[updates++] = buttons_state;
    if (frozen) { frozen--; return; }
    unsigned edges = buttons_state & 48u;
    jump_edges += !!(edges & ~latched & 16u);
    dash_edges += !!(edges & ~latched & 32u);
    latched = edges;
}
static void Honing_draw(void) { draws++; }
static void Honing_begin(unsigned seed) { (void)seed; }
static void Honing_state(FILE *out) { fprintf(out, "\"fixture\":true"); }
static unsigned Honing_held_edges(void) { return latched; }
static void Celeste_P8_load_state(void *s) { (void)s; latched=0; }
static size_t Celeste_P8_get_state_size(void) { return 1; }
static void Celeste_P8_save_state(void *s) { (void)s; }
static void setup(void) {}
"""
    # Production includes expect external linkage for these entry points.
    if not engine:
        prelude = prelude.replace("static void Honing_", "void Honing_").replace(
            "static unsigned Honing_", "unsigned Honing_"
        )
    prelude += '#include "frontend.inc.c"\n'
    common = r"""static void command(const char *op, unsigned mask, unsigned ticks) {
    char path[4096]; honing_path(path, "command.txt");
    FILE *f=fopen(path, "w"); assert(f);
    fprintf(f, "%s %lu %u %u\n", op, honing_id, mask, ticks); fclose(f);
    honing_loop();
}
int main(int argc, char **argv) {
    assert(argc == 2); honing_dir=argv[1]; honing_last_valid=1000; setup();
"""
    if engine:  # noqa: SIM108 - Keep the two multiline C programs readable.
        checks = r"""    /* A real dash creates two hit-stop ticks and consumes the native dash. */
    command("step", 36, 1);
    assert(player()->p_dash && player()->djump == 0 && freeze > 0);
    unsigned long before = honing_ticks;
    command("step", 36, 1); /* Entire budget lies in hit-stop: release still pending. */
    assert(player()->p_dash && buttons_state == 4 && honing_ticks == before+1);
    command("step", 36, 1); /* Last freeze tick must not swallow the pending release. */
    assert(player()->p_dash && buttons_state == 4 && freeze == 0);
    command("step", 36, 1); /* Original PLAYER_update finally samples release. */
    assert(!player()->p_dash && buttons_state == 4);
    command("step", 36, 1); /* Native edge accepted; no dash while availability is zero. */
    assert(player()->p_dash && player()->djump == 0 && freeze == 0);
    /* Wait for an actual native landing/refill; do not edit the player's fields. */
    for (int i=0; i<30 && (!OBJ_is_solid(player(),0,1) || !player()->djump); i++)
        command("step", 0, 6);
    assert(OBJ_is_solid(player(),0,1) && player()->djump == 1);
    command("step", 36, 1);
    assert(player()->djump == 0 && freeze > 0);
    /* A separate original-engine fixture checks repeated jumps after landing. */
    setup(); buttons_state=0;
    command("step", 16, 30);
    for (int i=0; i<10 && !OBJ_is_solid(player(),0,1); i++) command("step", 16, 6);
    assert(OBJ_is_solid(player(),0,1) && player()->p_jump);
    command("step", 18, 2);
    assert(player()->p_jump && player()->spd.y < 0 && buttons_state == 18);
    assert(honing_updates == honing_draws);
    test_tiles[0][0]=43; test_tiles[0][1]=59; test_tiles[0][2]=17; test_tiles[0][3]=27;
    assert(spikes_at(0,0,1,1,-1,0) && !spikes_at(0,0,1,1,1,0));
    assert(spikes_at(15,0,1,1,1,0) && !spikes_at(15,0,1,1,-1,0));
    FILE *state=tmpfile(); assert(state); Honing_state(state); rewind(state);
    char json[32768]; size_t bytes=fread(json,1,sizeof(json)-1,state); json[bytes]=0; fclose(state);
    assert(strstr(json,"\"tiles\":[\"><^v"));
    puts("original engine: hit-stop release, zero-dash rejection, landing/refill, repeated jump passed");
"""
    else:
        checks = r"""    command("step", 17, 3); command("step", 18, 3);
    assert(jump_edges == 2 && updates == 6 && draws == 6);
    assert(masks[3] == 2 && masks[4] == 18 && masks[5] == 18);
    command("step", 18, 1); /* release-only, exact requested duration */
    assert(masks[6] == 2 && jump_edges == 2);
    command("step", 18, 1);
    assert(masks[7] == 18 && jump_edges == 3);
    command("step", 36, 1); assert(dash_edges == 1);
    frozen=3;
    command("step", 36, 2); command("step", 36, 3);
    assert(dash_edges == 2 && updates == 14 && draws == 14);
    for (int i=9; i<13; i++) assert(masks[i] == 4);
    assert(masks[13] == 36);
    unsigned long id=honing_id, ticks=honing_ticks;
    command("step", 3, 1); /* opposing directions rejected by actual parser */
    assert(honing_id == id && honing_ticks == ticks && honing_rejected == 1);
    command("step", 36, 0);
    assert(honing_id == id && honing_ticks == ticks && honing_rejected == 2);
    /* Every offered edge option, plus a raw combined jump/dash command. */
    unsigned choices[]={16,17,18,33,34,36,40,37,38,41,42,49};
    for (unsigned i=0; i<sizeof choices/sizeof choices[0]; i++) {
        unsigned mask=choices[i], before=updates;
        latched=mask&48u; buttons_state=mask;
        command("step",mask,2);
        assert(updates==before+2 && masks[before]==(mask&~48u) && masks[before+1]==mask);
    }
    puts("production parser/scheduler: repeat, compound, short-budget, freeze and rejection passed");
"""
    source.write_text(prelude + common + checks + "return 0;\n}\n")
    binary = tmp_path / "controls"
    command = [compiler, "-std=c99", "-Wall", "-Wextra", str(source), "-I", str(HERE)]
    if engine:
        command += ["-I", str(upstream)]
    command += ["-lm", "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("engine", [False, True], ids=["portable-production-loop", "original-engine"])
def test_compiled_control_loop(tmp_path: Path, engine: bool) -> None:
    binary = compile_harness(tmp_path, engine)
    session = tmp_path / "session"
    session.mkdir()
    result = subprocess.run([str(binary), str(session)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_request_describes_release_budget_and_spike_directions() -> None:
    spec = importlib.util.spec_from_file_location("classic_controls_run", HERE / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = module.request_for({"id": 1}, 1)
    text = request["parts"]["essential"][0]
    assert "release" in text and "including any release tick" in text
    assert "repeating a held button does not retrigger" not in text


def load_run():
    spec = importlib.util.spec_from_file_location("classic_timeout_run", HERE / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("timeout", ["4000", "nan", "inf", "0", "-1"])
def test_cli_invalid_timeout_rejects_before_side_effects(tmp_path: Path, timeout: str) -> None:
    run = load_run()
    output = tmp_path / "uncreated"
    argv = ["run.py", "--game", "unused-game", "--output", str(output), "--timeout", timeout]
    with (
        patch.object(run.sys, "argv", argv),
        patch.object(run, "DecisionClient") as client,
        patch.object(run, "Native") as native,
        pytest.raises(SystemExit) as error,
    ):
        run.main()
    assert error.value.code == 2
    assert not client.called and not native.called and not output.exists()


def test_cli_timeout_boundary_propagates_full_native_margin(tmp_path: Path) -> None:
    run = load_run()
    argv = ["run.py", "--game", "unused-game", "--output", str(tmp_path / "run"), "--timeout", "3570"]
    with (
        patch.object(run.sys, "argv", argv),
        patch.object(run, "DecisionClient"),
        patch.object(run, "Native") as native,
    ):
        native.return_value.start.side_effect = RuntimeError("injected bounded exit")
        with pytest.raises(RuntimeError, match="injected bounded exit"):
            run.main()
    assert native.call_args.kwargs["watchdog"] == 3600
    native.return_value.close.assert_called_once()

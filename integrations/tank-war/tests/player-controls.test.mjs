import test from "node:test";
import assert from "node:assert/strict";
import {
    directionControls,
    mountPointerControls,
    freeMoveControls,
} from "../web/shared/pointer-controls.js";
import { Game, TURN_SPEED } from "../web/game/engine.js";
test("screen direction uses responsive bounded steering and shortest-angle wrapping", () => {
    assert.deepEqual(directionControls(0, 1, 0), { drive: 1, turn: 0 });
    assert.deepEqual(directionControls(0, 0, 1), { drive: 0, turn: 1 });
    assert.deepEqual(directionControls(0, 0, -1), { drive: 0, turn: -1 });
    assert.equal(directionControls(Math.PI - 0.005, -1, -0.005).turn, 0);
    assert.equal(directionControls(0, 1, 0, 0.4).drive, 0.4);
    assert.ok(directionControls(0, 1, 0.1).turn > 0.8);
    assert.deepEqual(directionControls(0, 0, 0), { drive: 0, turn: 0 });
    for (let a = -Math.PI; a < Math.PI; a += 0.1) {
        const c = directionControls(a, 1, 2);
        assert.ok(c.drive >= 0 && c.drive <= 1);
        assert.ok(Math.abs(c.turn) <= 1);
    }
});
test("hover steers without pressing; release preserves direction; leave stops; click fires independently", () => {
    class Element extends EventTarget {
        style = {};
        classList = { add() {}, remove() {} };
        setPointerCapture() {}
        focus() {}
        getBoundingClientRect() {
            return { left: 0, top: 0, width: 112, height: 112 };
        }
    }
    const bus = new EventTarget(),
        doc = new EventTarget();
    const oldAdd = globalThis.addEventListener,
        oldDoc = globalThis.document;
    globalThis.addEventListener = bus.addEventListener.bind(bus);
    globalThis.document = doc;
    try {
        const canvas = new Element(),
            stick = new Element(),
            knob = new Element(),
            fire = new Element();
        let enabled = true;
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            width: 900,
            height: 630,
        });
        const tank = { angle: 0, x: 100, y: 100 };
        const p = mountPointerControls({
            canvas,
            stick,
            knob,
            fire,
            enabled: () => enabled,
            tank: () => tank,
        });
        const event = (el, type, opts = {}) => {
            const e = new Event(type, { cancelable: true });
            Object.assign(e, {
                pointerId: 1,
                pointerType: "mouse",
                button: 0,
                buttons: 0,
                clientX: 200,
                clientY: 100,
                ...opts,
            });
            el.dispatchEvent(e);
        };
        event(canvas, "pointermove");
        let c = p.sample();
        assert.equal(c.drive, 1);
        assert.equal(c.active, true);
        assert.equal(c.fire, false);
        event(canvas, "pointerdown");
        assert.equal(p.sample().fire, true);
        event(canvas, "pointerup");
        c = p.sample();
        assert.equal(c.fire, false);
        assert.equal(c.drive, 1);
        tank.x = 195;
        assert.equal(p.sample().drive, 0);
        tank.x = 100;
        assert.equal(p.sample().drive, 1);
        event(canvas, "pointerleave");
        assert.equal(p.sample().active, false);
        canvas.getBoundingClientRect = () => ({
            left: 10,
            top: 20,
            width: 450,
            height: 315,
        });
        event(canvas, "pointermove", { clientX: 110, clientY: 70 });
        assert.equal(p.sample().drive, 1);
        p.releaseMouse();
        assert.equal(p.sample().active, false);
        event(stick, "pointerdown", {
            pointerId: 2,
            pointerType: "touch",
            clientX: 56,
            clientY: 56,
        });
        assert.equal(p.sample().drive, 0); // Touching the fixed center must not move the tank.
        event(stick, "pointermove", {
            pointerId: 2,
            pointerType: "touch",
            clientX: 150,
            clientY: 56,
        });
        event(fire, "pointerdown", { pointerId: 3, pointerType: "touch" });
        assert.equal(p.sample().drive, 1);
        assert.equal(p.sample().fire, true);
        event(fire, "pointerup", { pointerId: 3 });
        assert.equal(p.sample().fire, false);
        assert.equal(p.sample().active, true);
        event(stick, "pointercancel", { pointerId: 2 });
        assert.equal(p.sample().active, false);
        event(stick, "pointerdown", {
            pointerId: 4,
            pointerType: "touch",
            clientX: 56,
            clientY: 56,
        });
        assert.equal(p.sample().drive, 0);
        event(stick, "pointermove", {
            pointerId: 2,
            pointerType: "touch",
            clientX: 150,
            clientY: 56,
        });
        assert.equal(p.sample().drive, 0); // Old fingers cannot steer the new gesture.
        event(stick, "pointermove", {
            pointerId: 4,
            pointerType: "touch",
            clientX: 100,
            clientY: 56,
        });
        assert.equal(p.sample().drive, 1);
        event(fire, "pointerdown", { pointerId: 5, pointerType: "touch" });
        event(stick, "pointerup", { pointerId: 4 });
        assert.equal(p.sample().drive, 0);
        assert.equal(p.sample().fire, true);
        event(fire, "pointerup", { pointerId: 5 });
        event(stick, "pointerdown", {
            pointerId: 6,
            pointerType: "touch",
            clientX: 100,
            clientY: 56,
        });
        assert.equal(p.sample().drive, 1); // Edge presses use the same fixed center.
        bus.dispatchEvent(new Event("resize"));
        assert.equal(p.sample().active, false);
        event(fire, "pointerdown", { pointerId: 3 });
        bus.dispatchEvent(new Event("blur"));
        assert.equal(p.sample().fire, false);
        enabled = false;
        event(canvas, "pointermove");
        event(canvas, "pointerdown");
        assert.equal(p.sample().active, false);
        assert.equal(p.sample().fire, false);
    } finally {
        globalThis.addEventListener = oldAdd;
        globalThis.document = oldDoc;
    }
});
test("only responsive human controls receive 68 percent faster than keyboard rotation", () => {
    const game = new Game(3);
    game.start();
    const a = game.tanks[0].angle,
        b = game.tanks[1].angle;
    game.step(1 / 60, {
        0: { drive: 0, turn: 1, fire: false, responsiveTurn: true },
        1: { drive: 0, turn: 1, fire: false },
    });
    const delta = (x, y) => Math.atan2(Math.sin(x - y), Math.cos(x - y));
    assert.ok(
        Math.abs(delta(game.tanks[0].angle, a) - (TURN_SPEED * 1.68) / 60) <
            1e-8,
    );
    assert.ok(Math.abs(delta(game.tanks[1].angle, b) - TURN_SPEED / 60) < 1e-8);
    game.step(1 / 60, { 0: { drive: 0, turn: 0, fire: false } });
    assert.equal(game.tanks[0].turnSpeed, TURN_SPEED);
});

test("public human turn boost doubles both pointer and keyboard speed without changing the bot", () => {
    for (const responsiveTurn of [false, true]) {
        const game = new Game(3);
        game.start();
        const a = game.tanks[0].angle,
            b = game.tanks[1].angle;
        game.step(1 / 60, {
            0: {
                drive: 0,
                turn: 1,
                fire: false,
                responsiveTurn,
                fastHumanTurn: true,
            },
            1: { drive: 0, turn: 1, fire: false },
        });
        const delta = (x, y) => Math.atan2(Math.sin(x - y), Math.cos(x - y));
        const expected = (TURN_SPEED * (responsiveTurn ? 1.68 : 1) * 2) / 60;
        assert.ok(Math.abs(delta(game.tanks[0].angle, a) - expected) < 1e-8);
        assert.ok(
            Math.abs(delta(game.tanks[1].angle, b) - TURN_SPEED / 60) < 1e-8,
        );
        game.step(1 / 60, { 0: { drive: 0, turn: 0, fire: false } });
        assert.equal(game.tanks[0].turnSpeed, TURN_SPEED);
    }
});

test("hybrid keys move in screen directions and opposing keys cancel", () => {
    const held = new Set(["KeyW", "KeyD"]);
    assert.deepEqual(
        freeMoveControls((k) => held.has(k)),
        { moveX: 1, moveY: -1 },
    );
    held.add("KeyS");
    held.add("KeyA");
    assert.deepEqual(
        freeMoveControls((k) => held.has(k)),
        { moveX: 0, moveY: 0 },
    );
    assert.deepEqual(
        freeMoveControls((k) => k === "ArrowLeft"),
        { moveX: -1, moveY: 0 },
    );
});
function hybridGame() {
    const game = new Game(3);
    game.start();
    game.maze = { ...game.maze, walls: [] };
    Object.assign(game.tanks[0], { x: 200, y: 200, angle: 0 });
    Object.assign(game.tanks[1], { x: 700, y: 500 });
    return game;
}
test("hybrid movement is independent of aim; shots follow the cursor after movement", () => {
    const game = hybridGame();
    game.step(1 / 60, {
        0: {
            freeMove: true,
            moveX: 0,
            moveY: -1,
            aimPoint: { x: 500, y: 200 },
            fire: true,
        },
        1: {},
    });
    const t = game.tanks[0],
        b = game.bullets[0];
    assert.equal(t.x, 200);
    assert.ok(t.y < 200);
    assert.equal(t.hullAngle, -Math.PI / 2);
    const aim = Math.atan2(200 - t.y, 500 - t.x);
    assert.equal(t.angle, aim);
    assert.ok(b);
    assert.ok(Math.abs(Math.atan2(b.vy, b.vx) - aim) < 1e-10);
    game.step(1 / 60, { 0: { drive: 0, turn: 0, fire: false }, 1: {} });
    assert.equal(t.hullAngle, undefined); // Switching back restores the fixed barrel.
});
test("hybrid diagonals have normal speed and cannot cross walls or another tank", () => {
    const straight = hybridGame(),
        diagonal = hybridGame();
    straight.step(1 / 60, { 0: { freeMove: true, moveX: 1, moveY: 0 }, 1: {} });
    diagonal.step(1 / 60, { 0: { freeMove: true, moveX: 1, moveY: 1 }, 1: {} });
    const travelled = (g) => Math.hypot(g.tanks[0].x - 200, g.tanks[0].y - 200);
    assert.ok(Math.abs(travelled(straight) - travelled(diagonal)) < 1e-8);
    const wall = hybridGame();
    wall.maze.walls = [{ x: 225, y: 100, w: 6, h: 200 }];
    for (let i = 0; i < 60; i++)
        wall.step(1 / 60, { 0: { freeMove: true, moveX: 1, moveY: 0 }, 1: {} });
    assert.ok(wall.tanks[0].x <= 203 + 1e-8);
    const pair = hybridGame();
    pair.tanks[1].x = 250;
    pair.tanks[1].y = 200;
    for (let i = 0; i < 60; i++)
        pair.step(1 / 60, { 0: { freeMove: true, moveX: 1, moveY: 0 }, 1: {} });
    assert.ok(pair.tanks[1].x - pair.tanks[0].x >= 44 - 1e-8);
});

import { manualPlayers } from "../web/shared/player-controls.js";
import { adapter } from "../web/game/adapter.js";
test("local arrows stay with red in all three blue control schemes", () => {
    for (const scheme of ["mouse", "keyboard", "hybrid"]) {
        const inputs = manualPlayers({
            adapter,
            mode: "local",
            scheme,
            touch: false,
            held: (k) => ["ArrowUp", "ArrowRight", "Enter"].includes(k),
            pointer: { active: false, fire: false },
        });
        assert.equal(inputs[0].drive, 0);
        assert.equal(inputs[0].turn, 0);
        assert.equal(inputs[0].fire, false);
        assert.equal(inputs[0].moveX ?? 0, 0);
        assert.equal(inputs[0].moveY ?? 0, 0);
        assert.equal(inputs[1].drive, 1);
        assert.equal(inputs[1].turn, 1);
        assert.equal(inputs[1].fire, true);
    }
});
test("mode routing isolates keyboard and mouse, combines hybrid and preserves touch and AI", () => {
    const setup = {
        adapter,
        mode: "human",
        touch: false,
        held: (k) => ["KeyW", "KeyD", "Space"].includes(k),
        pointer: {
            active: true,
            drive: 0.5,
            turn: -1,
            fire: true,
            aimPoint: { x: 400, y: 400 },
        },
    };
    const mouse = manualPlayers({ ...setup, scheme: "mouse" })[0];
    assert.equal(mouse.drive, 0.5);
    assert.equal(mouse.turn, -1);
    const keyboard = manualPlayers({
        ...setup,
        scheme: "keyboard",
        held: () => false,
    })[0];
    assert.equal(keyboard.drive, 0);
    assert.equal(keyboard.fire, false);
    const hybrid = manualPlayers({ ...setup, scheme: "hybrid" })[0];
    assert.equal(hybrid.moveX, 1);
    assert.equal(hybrid.moveY, -1);
    assert.deepEqual(hybrid.aimPoint, { x: 400, y: 400 });
    assert.equal(hybrid.fire, true);
    const touch = manualPlayers({
        ...setup,
        scheme: "keyboard",
        touch: true,
    })[0];
    assert.equal(touch.drive, 0.5);
    assert.equal(touch.fire, true);
    assert.deepEqual(manualPlayers({ ...setup, mode: "ai" }), {});
});

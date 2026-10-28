import { freeMoveControls } from "./pointer-controls.js";

// Local red keeps arrows/Enter even when blue uses a mouse or touch control.
export function manualPlayers({ adapter, mode, scheme, touch, held, pointer }) {
    if (mode === "ai") return {};
    const blueHeld = (key) =>
        (mode !== "local" || !key.startsWith("Arrow")) && held(key);
    const keyboard = touch || scheme === "keyboard";
    const blue = {
        ...adapter.manual(0, mode, (key) => keyboard && blueHeld(key)),
        fastHumanTurn: true,
    };
    if (!touch && scheme === "hybrid") {
        Object.assign(blue, freeMoveControls(blueHeld), {
            freeMove: true,
            drive: 0,
            turn: 0,
            aimPoint: pointer?.aimPoint,
            fire: !!pointer?.fire,
        });
    } else if (pointer?.active && (touch || scheme === "mouse")) {
        Object.assign(blue, {
            drive: pointer.drive,
            turn: pointer.turn,
            responsiveTurn: true,
        });
    }
    if (pointer?.fire && (touch || scheme !== "keyboard")) blue.fire = true;
    const manual = { 0: blue };
    if (mode === "local")
        manual[1] = { ...adapter.manual(1, mode, held), fastHumanTurn: true };
    return manual;
}

export function playerControlHelp(scheme, mode, touch) {
    const blue = touch
        ? "Drag joystick to steer · Fire to shoot"
        : scheme === "hybrid"
          ? "WASD to move · Mouse to aim · Left click to fire"
          : scheme === "keyboard"
            ? "W/S drive · A/D turn · Space fire"
            : "Move mouse to steer · Left click to fire";
    return `${blue}${mode === "local" ? " · Red: arrows to drive/turn, Enter to fire" : scheme !== "mouse" && !touch ? " · Arrow keys also supported" : ""}`;
}

// Convert a screen-direction vector into the original hull/drive controls.
// The engine still applies bounded rotation, collision and fire-rate limits.
export function directionControls(angle, x, y, strength = 1) {
    if (
        !Number.isFinite(angle) ||
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        Math.hypot(x, y) < 0.001
    )
        return { drive: 0, turn: 0 };
    const error = Math.atan2(
        Math.sin(Math.atan2(y, x) - angle),
        Math.cos(Math.atan2(y, x) - angle),
    );
    return {
        drive:
            Math.abs(error) > Math.PI / 3
                ? 0
                : Math.max(0, Math.cos(error)) *
                  Math.max(0, Math.min(1, strength)),
        turn:
            Math.abs(error) < 0.02
                ? 0
                : Math.max(-1, Math.min(1, error / 0.12)),
    };
}
export function freeMoveControls(held) {
    return {
        moveX:
            Number(held("KeyD") || held("ArrowRight")) -
            Number(held("KeyA") || held("ArrowLeft")),
        moveY:
            Number(held("KeyS") || held("ArrowDown")) -
            Number(held("KeyW") || held("ArrowUp")),
    };
}
export function mountPointerControls({
    canvas,
    stick,
    zone = stick,
    knob,
    fire,
    enabled,
    tank,
}) {
    let steer = null,
        mouse = null,
        stickPointer = null,
        origin = null,
        tap = false;
    const firing = new Set();
    const paint = (x = 0, y = 0) => {
        if (knob) knob.style.transform = `translate(${x}px, ${y}px)`;
    };
    const hideStick = () => {
        origin = null;
        zone?.classList.remove("active");
        paint();
    };
    const reset = () => {
        steer = null;
        mouse = null;
        stickPointer = null;
        hideStick();
        tap = false;
        firing.clear();
        paint();
        fire?.classList.remove("pressed");
    };
    const readMouse = (e) => {
        if (e.pointerType !== "mouse" || !enabled()) return;
        mouse = { x: e.clientX, y: e.clientY };
    };
    canvas.addEventListener("pointermove", readMouse);
    canvas.addEventListener("pointerdown", (e) => {
        if (e.pointerType !== "mouse" || e.button !== 0 || !enabled()) return;
        e.preventDefault();
        canvas.focus({ preventScroll: true });
        readMouse(e);
        tap = true;
    });
    const cancelMouse = (e) => {
        if (e.pointerType === "mouse") mouse = null;
    };
    canvas.addEventListener("pointerleave", cancelMouse);
    canvas.addEventListener("pointercancel", cancelMouse);
    const moveStick = (e) => {
        const r = stick.getBoundingClientRect(),
            radius = r.width * 0.34;
        let x = e.clientX - origin.x,
            y = e.clientY - origin.y;
        const length = Math.hypot(x, y),
            scale = length > radius ? radius / length : 1;
        x *= scale;
        y *= scale;
        paint(x, y);
        steer =
            length < 8
                ? null
                : { x, y, strength: Math.min(1, (length - 8) / (radius - 8)) };
    };
    zone?.addEventListener("pointerdown", (e) => {
        if (!enabled() || stickPointer !== null || e.button > 0) return;
        e.preventDefault();
        mouse = null;
        stickPointer = e.pointerId;
        const r = stick.getBoundingClientRect();
        origin = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
        zone.classList.add("active");
        zone.setPointerCapture(e.pointerId);
        moveStick(e);
    });
    zone?.addEventListener("pointermove", (e) => {
        if (e.pointerId === stickPointer) {
            e.preventDefault();
            moveStick(e);
        }
    });
    const releaseStick = (e) => {
        if (e.pointerId === stickPointer) {
            stickPointer = null;
            steer = null;
            hideStick();
        }
    };
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
        zone?.addEventListener(name, releaseStick);
    fire?.addEventListener("pointerdown", (e) => {
        if (!enabled() || e.button > 0) return;
        e.preventDefault();
        firing.add(e.pointerId);
        tap = true;
        fire.classList.add("pressed");
        fire.setPointerCapture(e.pointerId);
    });
    const releaseFire = (e) => {
        firing.delete(e.pointerId);
        if (!firing.size) fire.classList.remove("pressed");
    };
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
        fire?.addEventListener(name, releaseFire);
    addEventListener("blur", reset);
    addEventListener("resize", reset);
    addEventListener("pagehide", reset);
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) reset();
    });
    return {
        reset,
        releaseMouse() {
            mouse = null;
        },
        sample() {
            if (!enabled()) {
                reset();
                return { active: false, drive: 0, turn: 0, fire: false };
            }
            let direction = steer,
                aimPoint = null;
            if (mouse && stickPointer === null) {
                const r = canvas.getBoundingClientRect(),
                    t = tank();
                aimPoint = {
                    x: ((mouse.x - r.left) * 900) / r.width,
                    y: ((mouse.y - r.top) * 630) / r.height,
                };
                const x = aimPoint.x - t.x,
                    y = aimPoint.y - t.y;
                // Stop at the cursor rather than orbiting or overshooting it.
                direction =
                    Math.hypot(x, y) > 14
                        ? {
                              x,
                              y,
                              strength: Math.min(
                                  1,
                                  (Math.hypot(x, y) - 14) / 45,
                              ),
                          }
                        : null;
            }
            const move = direction
                ? directionControls(
                      tank().angle,
                      direction.x,
                      direction.y,
                      direction.strength,
                  )
                : { drive: 0, turn: 0 };
            const shoot = tap || firing.size > 0;
            tap = false;
            return {
                ...move,
                active: !!mouse || stickPointer !== null,
                fire: shoot,
                aimPoint,
            };
        },
    };
}

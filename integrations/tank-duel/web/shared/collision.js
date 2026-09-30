// The complete ink silhouette, including the barrel, fits inside this disk.
export const TANK_RADIUS = 22;
export const SHELL_RADIUS = 3;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export function circleTouchesWall(x, y, radius, wall) {
    const dx = x - clamp(x, wall.x, wall.x + wall.w);
    const dy = y - clamp(y, wall.y, wall.y + wall.h);
    return dx * dx + dy * dy < radius * radius - 1e-8;
}
export function moveTankPosition(maze, tank, dx, dy, dt, other, speed) {
    if (!Number.isFinite(dt) || dt <= 0) return;
    const length = Math.max(1, Math.hypot(dx, dy));
    const mx = (dx / length) * speed * dt,
        my = (dy / length) * speed * dt;
    // Bound spatial travel, not frame duration: a delayed frame cannot skip a wall.
    const steps = Math.max(1, Math.ceil(Math.hypot(mx, my) / 2));
    const free = (x, y) =>
        !maze.walls.some((w) => circleTouchesWall(x, y, TANK_RADIUS, w)) &&
        (!other?.alive ||
            Math.hypot(x - other.x, y - other.y) >= TANK_RADIUS * 2 - 1e-8);
    for (let i = 0; i < steps; i++) {
        const x = tank.x + mx / steps;
        if (free(x, tank.y)) tank.x = x;
        const y = tank.y + my / steps;
        if (free(tank.x, y)) tank.y = y;
    }
}
export function advanceShell(shell, dt, walls, onBounce = () => {}) {
    if (!Number.isFinite(dt) || dt <= 0) return shell;
    const steps = Math.max(
        1,
        Math.ceil((Math.hypot(shell.vx, shell.vy) * dt) / 2),
    );
    const step = dt / steps;
    for (let i = 0; i < steps; i++) {
        let x = shell.x + shell.vx * step;
        if (walls.some((w) => circleTouchesWall(x, shell.y, SHELL_RADIUS, w))) {
            shell.vx *= -1;
            x = shell.x;
            shell.bounces++;
            onBounce(shell);
        }
        let y = shell.y + shell.vy * step;
        if (walls.some((w) => circleTouchesWall(x, y, SHELL_RADIUS, w))) {
            shell.vy *= -1;
            y = shell.y;
            shell.bounces++;
            onBounce(shell);
        }
        shell.x = x;
        shell.y = y;
        shell.age += step;
    }
    return shell;
}

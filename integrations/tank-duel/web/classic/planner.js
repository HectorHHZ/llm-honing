import { STRATEGIES, rangeCost } from "./strategies.js";
import {
    SHELL_LIFETIME,
    DRIVE,
    TURNS,
    SPEED,
    TURN_SPEED,
    RADIUS,
    clamp,
    angleDelta,
    distance,
    route,
    cellOf,
    center,
    blocked,
    tankMotion,
    DRIVE_INPUT,
    TURN_INPUT,
    forecastMovement,
    forecastShot,
    findAimDirections,
    aimContext,
    avoidanceContext,
    forecastCurrentControls,
} from "./engine.js";
export const PLAN_PROTOCOL = "plans-v1";
export const PLAN_DURATION = 0.65;
const round = (n) => Math.round(n * 10) / 10;
export const motionId = (drive, turn) => drive + "_" + turn;
export function visible(maze, a, b, radius = 3) {
    const d = distance(a, b);
    for (let s = 0; s < d; s += 3)
        if (
            blocked(
                maze,
                a.x + ((b.x - a.x) * s) / d,
                a.y + ((b.y - a.y) * s) / d,
                radius,
            )
        )
            return false;
    return true;
}
function arrival(game, id, delay) {
    const t = game.tanks[id],
        p = { ...t },
        dt = 1 / 60;
    for (let k = 0; k < Math.ceil(delay / dt); k++) {
        const c = forecastCurrentControls(game, t, (k + 1) * dt);
        tankMotion(
            game.maze,
            p,
            DRIVE_INPUT[c.drive],
            TURN_INPUT[c.turn],
            dt,
            game.tanks[1 - id],
        );
    }
    return p;
}
export function buildPlans(
    game,
    id,
    { reactionDelay = 0.3, strategy = STRATEGIES.baseline } = {},
) {
    const t = game.tanks[id],
        enemy = game.tanks[1 - id],
        p = arrival(game, id, reactionDelay),
        los = visible(game.maze, p, enemy);
    const aim = findAimDirections(game.maze, t, enemy, {
            limit: strategy.multiAim ? 6 : 3,
        }),
        arrivalAim = reactionDelay
            ? findAimDirections(game.maze, p, enemy, {
                  limit: strategy.multiAim ? 6 : 3,
              })
            : aim;
    const path = route(game.maze, cellOf(p), cellOf(enemy));
    let target = enemy;
    if (!los) {
        target = center(cellOf(p));
        for (const cell of path) {
            const next = center(cell);
            if (!visible(game.maze, p, next, RADIUS + 2)) break;
            target = next;
        }
    }
    const firingLane = arrivalAim.best !== null;
    const targetHeading =
            arrivalAim.best?.angle ??
            Math.atan2(target.y - p.y, target.x - p.x),
        error = angleDelta(targetHeading, p.angle);
    const duration = strategy.adaptiveHold
            ? clamp(reactionDelay + 0.15, PLAN_DURATION, 1.2)
            : PLAN_DURATION,
        actions = [];
    for (const drive of Object.keys(DRIVE))
        for (const turn of Object.keys(TURNS)) {
            const variants = [];
            for (const heading of strategy.multiAim && arrivalAim.options.length
                ? arrivalAim.options.map((a) => a.angle)
                : [targetHeading]) {
                const error = angleDelta(heading, p.angle);
                // A bounded key press at physical angular speed, not a snapped/independent turret.
                const direction = TURN_INPUT[turn],
                    toward = direction && Math.sign(error) === direction;
                const turnDuration = direction
                    ? Math.round(
                          Math.min(
                              duration,
                              toward ? Math.abs(error) / TURN_SPEED : 0.18,
                          ) * 60,
                      ) / 60
                    : 0;
                const commandDuration = Math.max(
                    turnDuration + 1 / 60,
                    drive === "stop"
                        ? duration
                        : clamp(distance(p, target) / SPEED, 0.1, duration),
                );
                const controls = { drive, turn },
                    movement = forecastMovement(game, id, controls, {
                        reactionDelay,
                        controlDuration: commandDuration,
                        turnDuration,
                        horizon:
                            reactionDelay +
                            Math.max(
                                commandDuration + 0.2,
                                strategy.safetyHorizon || 0,
                            ),
                    });
                const fireDelay = turnDuration;
                const shot = forecastShot(game, id, controls, {
                    reactionDelay,
                    controlDuration: commandDuration,
                    turnDuration,
                    fireDelay,
                    horizon: 3,
                });
                const errorAfter = Math.abs(
                    angleDelta(heading, movement.angle),
                );
                const progress =
                    distance(p, target) -
                    Math.hypot(movement.x - target.x, movement.y - target.y);
                const hitPenalty =
                    movement.hitTime === null
                        ? 0
                        : 1500 - 100 * movement.hitTime;
                let score = progress - errorAfter * 50 - hitPenalty;
                if (firingLane) {
                    score =
                        (shot.enemyHitTime !== null && shot.selfHitTime === null
                            ? 180
                            : 0) -
                        errorAfter * 85 +
                        (distance(p, enemy) > 180 ? progress * 0.2 : 0) -
                        hitPenalty;
                }
                if (
                    strategy.shotSpeedWeight &&
                    shot.enemyHitTime !== null &&
                    shot.selfHitTime === null
                )
                    score +=
                        Math.max(0, 3 - shot.enemyHitTime - fireDelay) *
                        strategy.shotSpeedWeight;
                const rangeAfter = Math.hypot(
                    movement.x - enemy.x,
                    movement.y - enemy.y,
                );
                const spacingProgress =
                    rangeCost(distance(p, enemy), strategy) -
                    rangeCost(rangeAfter, strategy);
                if (los || firingLane)
                    score += spacingProgress * strategy.spacingWeight;
                if (strategy.clearanceWeight)
                    score -=
                        Math.max(
                            0,
                            strategy.clearanceMargin - movement.clearance,
                        ) * strategy.clearanceWeight;
                // Prefer useful fire without driving into point-blank range.
                if (strategy.spacingWeight && rangeAfter < 100)
                    score -= (100 - rangeAfter) * 2;
                // Do not repeatedly drive into a wall or turn away from the chosen corridor.
                if (drive !== "stop" && movement.travel < 5) score -= 45;
                const allowFire =
                    shot.canFire &&
                    shot.selfHitTime === null &&
                    shot.enemyHitTime !== null;
                const plan = {
                    id: motionId(drive, turn),
                    controls,
                    duration: commandDuration,
                    turnDuration,
                    fireDelay,
                    movement,
                    shot,
                    score,
                    allowFire,
                    target: { x: target.x, y: target.y },
                    aimHeading: heading,
                    rangeAfter,
                    spacingProgress,
                };
                plan.description = `Drive ${drive} for ${Math.round(commandDuration * 1000)}ms; turn ${turn} for ${Math.round(turnDuration * 1000)}ms; ${firingLane ? "verified hit direction" : "waypoint"} error after ${Math.round((errorAfter * 180) / Math.PI)}deg; target progress ${Math.round(progress)}px; existing shell collision ${movement.hitTime === null ? "none" : Math.round(movement.hitTime * 1000) + "ms"}; shot after turn: enemy hit ${shot.enemyHitTime === null ? "none" : Math.round(shot.enemyHitTime * 1000) + "ms"}, self hit ${shot.selfHitTime === null ? "none" : Math.round(shot.selfHitTime * 1000) + "ms"}.`;
                if (strategy.multiAim)
                    plan.description += ` Chosen barrel heading ${round((heading * 180) / Math.PI)}deg; enemy range after ${round(rangeAfter)}px; preferred range ${strategy.spacingWeight ? `${strategy.minRange}-${strategy.maxRange}px` : "unrestricted"}; spacing improvement ${round(spacingProgress)}px.`;
                if (strategy.clearanceWeight)
                    plan.description += ` Minimum shell clearance ${round(movement.clearance)}px; prefer at least ${strategy.clearanceMargin}px. Enemy range ${round(rangeAfter)}px; comfortable range ${strategy.minRange}-${strategy.maxRange}px.`;
                variants.push(plan);
            }
            actions.push(
                variants.reduce((best, p) => (p.score > best.score ? p : best)),
            );
        }
    const best = Math.max(...actions.map((a) => a.score));
    const admissible = actions
        .filter((a) => a.score >= best - 12)
        .map((a) => a.id);
    const bullets = game.bullets.map((b) => [
        b.id,
        b.owner === id ? "self" : "enemy",
        round(b.x),
        round(b.y),
        round(b.vx),
        round(b.vy),
        b.bounces,
        round(SHELL_LIFETIME - b.age),
    ]);
    let aimingText =
        aimContext(aim) +
        aimContext(
            arrivalAim,
            `AIM AT PREDICTED RESPONSE POSITION x=${round(p.x)},y=${round(p.y)},heading=${round((p.angle * 180) / Math.PI)}deg AFTER ${Math.round(reactionDelay * 1000)}ms`,
        );
    if (strategy.allAnglePrompt)
        aimingText = aimingText.replaceAll(
            "Rotate the hull toward the first verified direction",
            "Compare the verified directions; each motion specifies its selected barrel heading",
        );
    const bulletText = `All ${bullets.length} bullets [id,owner,x,y,vx,vy,bounces,remaining_s]: ${JSON.stringify(bullets)}. `;
    const weaponState =
        avoidanceContext(game, id) +
        `One hit kills; own shells are lethal. Weapon mechanically available ammo=${5 - game.bullets.filter((b) => b.owner === id).length}/5, cooldown=${Math.round(t.cooldown * 1000)}ms. ` +
        `Me (${round(t.x)},${round(t.y)}), heading=${round((t.angle * 180) / Math.PI)}deg; enemy (${round(enemy.x)},${round(enemy.y)}). ` +
        `Verified aim at predicted response position: ${JSON.stringify(strategy.allAnglePrompt ? arrivalAim.options : arrivalAim.best)}. ` +
        bulletText +
        `Use the SELECTED motion's new-shot forecast, including its turn and fire delay, NOT another candidate or current heading. All walls were included. Enemy held stationary; horizon 3s; movement can invalidate an aim hint. Latest state is checked again before firing.`;
    const state =
        `STRATEGY ${strategy.id}: ${strategy.description} ` +
        aimingText +
        avoidanceContext(game, id) +
        `Tank duel. One hit kills. ALL shells including own are lethal. Fixed barrel rotates with hull at ${round((TURN_SPEED * 180) / Math.PI)}deg/s; drive ${SPEED}px/s. No instantaneous aiming. Full tank radius ${RADIUS}px, shell radius 3px.\n` +
        `Me: x=${round(t.x)}, y=${round(t.y)}, heading=${round((t.angle * 180) / Math.PI)}deg. Enemy: x=${round(enemy.x)}, y=${round(enemy.y)}. Enemy visible=${los}. ` +
        `Target waypoint: x=${round(target.x)}, y=${round(target.y)}. Turn goal=${firingLane ? "verified firing direction" : "navigation waypoint"}; target hull heading=${round((targetHeading * 180) / Math.PI)}deg, relative error=${round((error * 180) / Math.PI)}deg after estimated ${Math.round(reactionDelay * 1000)}ms response delay (positive right). ` +
        `Round ${game.round}, time ${round(game.elapsed)}s, remaining ${round(game.timeLeft)}s. Ammo ${5 - game.bullets.filter((b) => b.owner === id).length}/5, cooldown ${Math.round(t.cooldown * 1000)}ms. Current executed controls ${t.executedDrive}/${t.executedTurn}, expire in ${Math.round(Math.max(0, t.commandUntil - game.elapsed) * 1000)}ms. Enemy heading ${round((enemy.angle * 180) / Math.PI)}deg. ` +
        `Walls [x,y,w,h]: ${JSON.stringify(game.maze.walls.map((w) => [w.x, w.y, w.w, w.h]))}. Forecasts hold the enemy stationary and apply this plan until expiry, then stop; shot horizon 3 seconds after firing. Future enemy movement and later decisions are unknown. ` +
        `Choose a JOINT drive/turn plan. Smaller heading error and positive target progress are better. Existing-shell safety first. Rotate, then assess firing for that same plan. ` +
        bulletText +
        `Execution constraints admit plans ${admissible.join(",")}; raw model probabilities for all nine plans remain recorded.`;
    return {
        state,
        actions,
        admissible,
        bullets,
        aim,
        arrivalAim,
        reactionDelay,
        enemyVisible: los,
        target,
        request: {
            protocol: PLAN_PROTOCOL,
            state,
            weapon_state: weaponState,
            aim: { current: aim, predicted: arrivalAim },
            motions: actions.map((a) => ({
                id: a.id,
                description: a.description,
                drive: a.controls.drive,
                turn: a.controls.turn,
                shot: {
                    can_fire: a.shot.canFire,
                    self_hit_ms:
                        a.shot.selfHitTime === null
                            ? null
                            : Math.round(a.shot.selfHitTime * 1000),
                    enemy_hit_ms:
                        a.shot.enemyHitTime === null
                            ? null
                            : Math.round(a.shot.enemyHitTime * 1000),
                    fire_after_ms: Math.round(a.fireDelay * 1000),
                },
            })),
            admissible_motion_ids: admissible,
        },
    };
}
export function applyPlanDecision(
    game,
    id,
    response,
    strategy = STRATEGIES.baseline,
) {
    const current = buildPlans(game, id, { reactionDelay: 0, strategy }),
        requested = response.motion_id;
    let plan = current.actions.find((a) => a.id === requested);
    if (!plan) throw Error("Unknown motion plan");
    // Revalidate stale geometry. Never silently use a weapon result conditioned on a different motion.
    const retainIntent =
        strategy.revalidateSlack &&
        plan.movement.hitTime === null &&
        plan.movement.clearance >= 0 &&
        plan.score >=
            Math.max(...current.actions.map((p) => p.score)) -
                strategy.revalidateSlack &&
        (plan.controls.drive === "stop" || plan.movement.travel >= 5);
    if (!current.admissible.includes(requested) && retainIntent) {
        game.emit("plan_intent_retained", {
            player: id,
            motion_id: requested,
            score: plan.score,
            safety_clearance: plan.movement.clearance,
        });
    }
    if (!current.admissible.includes(requested) && !retainIntent) {
        const chosen = current.admissible.reduce(
            (best, k) =>
                response.probabilities.motion[k] >
                response.probabilities.motion[best]
                    ? k
                    : best,
            current.admissible[0],
        );
        plan = current.actions.find((a) => a.id === chosen);
        game.emit("plan_revalidate", {
            player: id,
            requested,
            executed: chosen,
            fire_cancelled: response.controls.fire,
        });
    }
    const fire =
        response.controls.fire && plan.id === requested && plan.allowFire;
    if (response.controls.fire && !fire)
        game.emit("plan_veto", {
            player: id,
            reason:
                plan.id !== requested
                    ? "motion_changed"
                    : "fresh_shot_unavailable",
        });
    game.setPlan(id, {
        ...plan,
        motionId: plan.id,
        controls: { ...plan.controls, fire },
    });
    return {
        motion_id: plan.id,
        controls: { ...plan.controls, fire },
        turn_ms: Math.round(plan.turnDuration * 1000),
        fire_after_ms: Math.round(plan.fireDelay * 1000),
    };
}

// The two leaderboards and player names on the server side.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { Leaderboard, cleanName } from "../leaderboard.mjs";
import { MatchScheduler } from "../match-scheduler.mjs";

const file = () => path.join(mkdtempSync(path.join(tmpdir(), "board-")), "leaderboard.json");

test("fair and modified rounds go to separate boards, ranked by wins", () => {
    const board = new Leaderboard(file());
    board.record("a", "Ace", { winner: 0, modified: false });
    board.record("a", "Ace", { winner: 0, modified: false });
    board.record("b", "Bo", { winner: 0, modified: false });
    board.record("b", "Bo", { winner: 1, modified: false });
    board.record("a", "Ace", { winner: 0, modified: true });
    const top = board.top();
    assert.deepEqual(top.fair.map((r) => [r.name, r.wins, r.losses]), [["Ace", 2, 0], ["Bo", 1, 1]]);
    assert.deepEqual(top.modified.map((r) => [r.name, r.wins]), [["Ace", 1]]);
});

test("a rename follows the player onto both boards and survives a restart", async () => {
    const f = file(), board = new Leaderboard(f);
    board.record("a", "Ace", { winner: 0, modified: false });
    board.record("a", "Ace", { winner: null, modified: true });
    board.rename("a", "Captain");
    assert.equal(board.top().fair[0].name, "Captain");
    assert.equal(board.top().modified[0].draws, 1);
    await new Promise((r) => setTimeout(r, 400));
    const again = new Leaderboard(f);
    assert.equal(again.nameFor("a"), "Captain");
    assert.equal(again.top().fair[0].wins, 1);
    assert.ok(JSON.parse(readFileSync(f, "utf8")).boards);
});

test("names are 1 to 20 letters, digits, spaces and a little punctuation", () => {
    assert.equal(cleanName("  Calm   Owl 7 "), "Calm Owl 7");
    assert.equal(cleanName("坦克大王"), "坦克大王");
    assert.equal(cleanName(""), null);
    assert.equal(cleanName("<script>"), null);
    assert.equal(cleanName("x".repeat(21)), null);
});

test("the scheduler renames, refuses a taken name, reuses a saved name and reports each round once", () => {
    const scheduler = new MatchScheduler({ capacity: 2 });
    const results = [];
    scheduler.savedName = (owner) => (owner === "o1" ? "Saved Name" : null);
    scheduler.onResult = (s, r) => results.push([s.nickname, r]);
    const t1 = randomUUID(), t2 = randomUUID();
    assert.equal(scheduler.join(t1, "o1").nickname, "Saved Name");
    scheduler.join(t2, "o2");
    assert.throws(() => scheduler.rename(t2, "Saved Name"), /already has that name/);
    assert.equal(scheduler.rename(t2, "Other").nickname, "Other");
    assert.throws(() => scheduler.rename(t2, "<bad>"), /1 to 20/);
    const ticket = scheduler.beginRound(t1, 1).round_ticket;
    scheduler.finishRound(t1, ticket, { winner: 0, modified: true, extra: "ignored" });
    scheduler.finishRound(t1, ticket, { winner: 1 });
    assert.deepEqual(results, [["Saved Name", { winner: 0, modified: true }]]);
    clearInterval(scheduler.sweeper);
});

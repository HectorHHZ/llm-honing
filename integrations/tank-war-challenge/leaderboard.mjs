// Two leaderboards for the human challenge, kept in data/leaderboard.json beside this file: rounds played
// fair, and rounds played with modifiers. A player is their browser's owner cookie; the board shows the name
// they chose last. Results come from the player's browser, which runs the game, so the board is for fun.
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";

export const BOARDS = ["fair", "modified"];
const NAME = /^[\p{L}\p{N} _.'!-]{1,20}$/u;

export function cleanName(name) {
    if (typeof name !== "string") return null;
    const trimmed = name.normalize("NFKC").replace(/\s+/g, " ").trim();
    return NAME.test(trimmed) ? trimmed : null;
}

export class Leaderboard {
    constructor(file) {
        this.file = file;
        this.timer = null;
        this.data = { names: {}, boards: { fair: {}, modified: {} } };
        try {
            const saved = JSON.parse(readFileSync(file, "utf8"));
            this.data = {
                names: saved.names ?? {},
                boards: { fair: saved.boards?.fair ?? {}, modified: saved.boards?.modified ?? {} },
            };
        } catch {}
    }
    nameFor(owner) {
        return this.data.names[owner] ?? null;
    }
    rename(owner, name) {
        this.data.names[owner] = name;
        for (const board of BOARDS) if (this.data.boards[board][owner]) this.data.boards[board][owner].name = name;
        this.save();
    }
    record(owner, name, { winner, modified }) {
        const rows = this.data.boards[modified ? "modified" : "fair"];
        const row = (rows[owner] ??= { name, wins: 0, losses: 0, draws: 0, rounds: 0, updated: 0 });
        row.name = name;
        row.rounds++;
        if (winner === 0) row.wins++;
        else if (winner === 1) row.losses++;
        else row.draws++;
        row.updated = Date.now();
        this.save();
    }
    top(limit = 10) {
        const rank = (rows) =>
            Object.values(rows)
                .sort((a, b) => b.wins - a.wins || b.wins / b.rounds - a.wins / a.rounds || a.updated - b.updated)
                .slice(0, limit)
                .map(({ name, wins, losses, draws, rounds }) => ({ name, wins, losses, draws, rounds }));
        return { fair: rank(this.data.boards.fair), modified: rank(this.data.boards.modified) };
    }
    save() {
        clearTimeout(this.timer);
        this.timer = setTimeout(() => {
            mkdirSync(path.dirname(this.file), { recursive: true });
            writeFileSync(`${this.file}.tmp`, JSON.stringify(this.data));
            renameSync(`${this.file}.tmp`, this.file);
        }, 300);
    }
}

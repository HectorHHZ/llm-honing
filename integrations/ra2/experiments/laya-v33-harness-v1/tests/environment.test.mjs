import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
    chmodSync,
    copyFileSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { assertSupportedPlatform, validateReleaseIdentity } from "../check.mjs";
import { HF_MODEL, HF_REVISION } from "../runtime.mjs";
import { SOURCE_REVISION, WEIGHTS_SHA256 } from "../cpu-client.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ENVIRONMENT_KEYS = [
    "RA2_LAB_DIR",
    "RA2_NODE_BIN",
    "RA2_SDK_DIR",
    "RA2_GAME_DIR",
    "RA2_ENV_FILE",
    "LAYA_RELEASE_DIR",
    "LAYA_MODEL_PATH",
    "LAYA_SOURCE_DIR",
    "LAYA_VENV_DIR",
    "LAYA_PYTHON",
    "LAYA_DEVICE",
    "SUPALOSA_DIR",
    "USE_TF",
    "TOKENIZERS_PARALLELISM",
    "HF_HUB_OFFLINE",
];
const probe = `console.log(JSON.stringify({ entry: process.argv[1].split("/").at(-1),
    argv: process.argv.slice(2), environment: Object.fromEntries(${JSON.stringify(
        ENVIRONMENT_KEYS,
    )}.map(k => [k, process.env[k]])) }));`;

function fixture(t, system = "Darwin") {
    const dir = mkdtempSync(join(tmpdir(), "ra2 harness environment "));
    t.after(() => rmSync(dir, { recursive: true, force: true }));
    const experiment = join(
        dir,
        "checkout",
        "integrations",
        "ra2",
        "experiments",
        "laya-v33-harness-v1",
    );
    const home = join(dir, "another user"),
        bin = join(dir, "tools");
    mkdirSync(experiment, { recursive: true });
    mkdirSync(home);
    mkdirSync(bin);
    for (const name of ["run.sh", "env.sh", "setup-linux.sh"])
        copyFileSync(join(ROOT, name), join(experiment, name));
    for (const name of ["run.mjs", "check.mjs", "report.mjs"])
        writeFileSync(join(experiment, name), probe);
    writeFileSync(
        join(bin, "uname"),
        `#!/bin/bash\nif [[ "$1" == -s ]]; then echo '${system}'; else echo x86_64; fi\n`,
    );
    chmodSync(join(bin, "uname"), 0o755);
    const environment = {
        ...process.env,
        HOME: home,
        PATH: `${bin}:/usr/bin:/bin`,
        RA2_NODE_BIN: process.execPath,
    };
    for (const key of ENVIRONMENT_KEYS)
        if (key !== "RA2_NODE_BIN") delete environment[key];
    const run = (args = [], overrides = {}, script = "run.sh") =>
        spawnSync("/bin/bash", [join(experiment, script), ...args], {
            cwd: dir,
            env: { ...environment, ...overrides },
            encoding: "utf8",
            timeout: 10000,
        });
    return { dir, experiment, home, bin, run };
}

function result(output) {
    assert.equal(output.status, 0, output.stderr || output.error?.message);
    return JSON.parse(output.stdout);
}

test("launcher relocates SDK independently of cwd and derives lab paths from the user's home", (t) => {
    const f = fixture(t);
    const output = result(f.run(["--help"]));
    const env = output.environment;
    assert.equal(env.RA2_LAB_DIR, join(f.home, "ra2-lab"));
    assert.equal(
        env.LAYA_PYTHON,
        join(f.home, "ra2-lab/laya-v33-cpu-venv/bin/python"),
    );
    assert.equal(
        resolve(env.RA2_SDK_DIR),
        resolve(f.experiment, "../../engine/ra2-game-env-0.1.1/js/package"),
    );
    assert.equal(env.LAYA_MODEL_PATH, env.LAYA_RELEASE_DIR);
    assert.deepEqual(output.argv, ["--help"]);
    assert.equal(existsSync(env.RA2_LAB_DIR), false);
});

test("launcher preserves explicit paths with spaces and passes CLI arguments without rewriting", (t) => {
    const f = fixture(t, "Linux");
    const overrides = {
        RA2_LAB_DIR: join(f.dir, "custom lab"),
        RA2_SDK_DIR: join(f.dir, "custom sdk"),
        LAYA_VENV_DIR: join(f.dir, "custom venv"),
        RA2_GAME_DIR: join(f.dir, "custom assets"),
        LAYA_DEVICE: "cuda",
    };
    const args = [
        "--run",
        "--seat",
        "blue",
        "--ticks=30",
        "--out",
        "test-output",
    ];
    const output = result(f.run(args, overrides));
    assert.deepEqual(output.argv, args);
    for (const key of [
        "RA2_LAB_DIR",
        "RA2_SDK_DIR",
        "LAYA_VENV_DIR",
        "RA2_GAME_DIR",
    ])
        assert.equal(output.environment[key], overrides[key]);
    assert.equal(
        output.environment.LAYA_PYTHON,
        join(overrides.LAYA_VENV_DIR, "bin/python"),
    );
    assert.equal(output.environment.LAYA_DEVICE, "cpu");
    assert.equal(output.environment.HF_HUB_OFFLINE, "1");
});

test("configuration is explicit, exported and never read from an incidental .env", (t) => {
    const f = fixture(t);
    writeFileSync(
        join(f.experiment, ".env"),
        "echo 'INCIDENTAL FILE EXECUTED'; exit 77\n",
    );
    assert.equal(f.run(["--help"]).status, 0);
    const config = join(f.dir, "chosen environment.sh");
    writeFileSync(
        config,
        'RA2_LAB_DIR="$HOME/custom lab"\nLAYA_VENV_DIR="$RA2_LAB_DIR/custom venv"\n',
    );
    const output = result(f.run(["--help"], { RA2_ENV_FILE: config }));
    assert.equal(output.environment.RA2_LAB_DIR, join(f.home, "custom lab"));
    assert.equal(
        output.environment.LAYA_PYTHON,
        join(f.home, "custom lab/custom venv/bin/python"),
    );
    const missing = f.run(["--help"], {
        RA2_ENV_FILE: join(f.dir, "missing.env"),
    });
    assert.notEqual(missing.status, 0);
    assert.match(missing.stderr, /Cannot read RA2_ENV_FILE/);
});

test("host help/default/report do not require Linux, Python, assets or model directories", (t) => {
    const f = fixture(t);
    assert.equal(result(f.run()).entry, "run.mjs");
    assert.equal(result(f.run(["--help"])).entry, "run.mjs");
    const output = result(f.run(["--report", "runs/example"]));
    assert.equal(output.entry, "report.mjs");
    assert.deepEqual(output.argv, ["runs/example"]);
    assert.equal(existsSync(join(f.home, "ra2-lab")), false);
});

test("host run/check are rejected before their JavaScript entry points execute", (t) => {
    const f = fixture(t);
    for (const args of [["--run"], ["--check"]]) {
        const output = f.run(args);
        assert.notEqual(output.status, 0);
        assert.equal(output.stdout, "");
        assert.match(output.stderr, /require x86_64 Linux/);
    }
});

test("Linux --check forwards only check arguments to the read-only entry point", (t) => {
    const f = fixture(t, "Linux");
    const output = result(f.run(["--check"]));
    assert.equal(output.entry, "check.mjs");
    assert.deepEqual(output.argv, []);
});

test("Node selection respects an explicit executable and otherwise falls back to PATH", (t) => {
    const f = fixture(t);
    symlinkSync(process.execPath, join(f.bin, "node"));
    assert.equal(
        result(f.run(["--help"], { RA2_NODE_BIN: "" })).environment
            .RA2_NODE_BIN,
        join(f.bin, "node"),
    );
    const missing = f.run(["--help"], {
        RA2_NODE_BIN: join(f.dir, "missing-node"),
    });
    assert.notEqual(missing.status, 0);
    assert.match(missing.stderr, /Node executable was not found/);
});

test("setup is opt-in and help needs no environment, network, model or writable lab directory", (t) => {
    const f = fixture(t);
    for (const args of [[], ["--help"], ["-h"]]) {
        const output = f.run(
            args,
            { RA2_ENV_FILE: join(f.dir, "missing.env") },
            "setup-linux.sh",
        );
        assert.equal(output.status, 0, output.stderr);
        assert.match(output.stdout, /Usage: bash setup-linux.sh --install/);
    }
    assert.equal(f.run(["--unknown"], {}, "setup-linux.sh").status, 2);
    const install = f.run(["--install"], {}, "setup-linux.sh");
    assert.notEqual(install.status, 0);
    assert.match(install.stderr, /require x86_64 Linux/);
    assert.equal(existsSync(join(f.home, "ra2-lab")), false);
});

test("preflight supports only the intended engine platform", () => {
    assert.doesNotThrow(() => assertSupportedPlatform("linux", "x64"));
    assert.throws(
        () => assertSupportedPlatform("darwin", "x64"),
        /x86_64 Linux/,
    );
    assert.throws(
        () => assertSupportedPlatform("linux", "arm64"),
        /x86_64 Linux/,
    );
});

test("preflight checks source, weights and release revision without loading the model", () => {
    const release = {
        repo: HF_MODEL,
        weightsSha256: WEIGHTS_SHA256,
        sourceRevision: SOURCE_REVISION,
    };
    const snapshot = { ...release, revision: HF_REVISION };
    assert.doesNotThrow(() => validateReleaseIdentity(release, snapshot));
    assert.throws(
        () =>
            validateReleaseIdentity(release, { ...snapshot, revision: "main" }),
        /revision/,
    );
    assert.throws(
        () =>
            validateReleaseIdentity(
                { ...release, weightsSha256: "other" },
                snapshot,
            ),
        /identity/,
    );
    assert.throws(
        () =>
            validateReleaseIdentity(release, {
                ...snapshot,
                sourceRevision: "other",
            }),
        /identity/,
    );
    assert.throws(() => validateReleaseIdentity(release, null), /identity/);
});

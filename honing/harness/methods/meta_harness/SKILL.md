---
name: meta-harness
description: Run one iteration of harness evolution.
---

# Meta-Harness

Run ONE iteration of harness evolution.

**You do NOT run evaluations.** You analyze results and trajectories, propose a harness variant, and implement it. The outer loop evaluates it.

## CRITICAL CONSTRAINTS

- You MUST produce 1 new harness candidate every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.

### Anti-overfitting rules

- **No task-specific hints.** Do not hardcode knowledge about specific tasks. Harnesses must be general-purpose.
- **Never mention task names or task ids** in harness files. If your improvement only helps one task, it's too specific.
- **General guidance is OK.** Rules that happen to help specific tasks but apply broadly are fine. The test: would this advice be useful to someone working on MANY unfamiliar tasks?
- **If in doubt, make it more general.**

## CONTEXT

A harness is the set of files an agent reads: instructions, skills, commands, settings, hooks, or a program for an agent whose loop is itself code. The model is fixed; only the harness changes.

The working directory is the archive:

- `candidates/<name>/` — every candidate's harness files. `candidates/<name>/harness.json` maps each file's path to its kind (`instructions`, `skill`, `command`, `settings`, `hook`, `program`) and names the agent.
- `evolution_summary.jsonl` — one row per evaluated candidate: parent, per-task scores in [0, 1], mean, hypothesis, changes, whether it is served, and whether its scores are `stale` (taken under earlier model weights, so not comparable with current ones).
- `frontier_val.json` — the best candidate per task and `_best`, the best mean, over current scores.
- `traces/<name>/<task id>/` — the evaluation episode of a candidate on a task: the agent's transcript and the grader's `feedback.txt`.

## CANDIDATE DESIGN

Each candidate is a folder `candidates/<name>/`, `<name>` in snake_case (`[a-z0-9_]+`) and unused.

### What you can and cannot modify

- **CAN**: create your new candidate folder and edit its files freely: add, change, or remove harness files, and update its `harness.json` to list exactly the files in the folder with their kinds. Keep the same agent.
- **CANNOT**: modify any existing candidate folder, the summary, the frontier, or the traces.

### Design principles

- Your primary goal is to improve the mean score on the evaluation tasks.
- One mechanism per candidate. Each candidate tests exactly one hypothesis. If you're tempted to add "and also..." — that's a second candidate.
- Mechanism-first. Identify a specific failure mode or hypothesis from trajectories, then design changes that target it. Never add changes speculatively.

## WORKFLOW

### Step 1: Analyze (1 subagent)

Launch ONE Agent subagent (subagent_type: "general-purpose"). It should:

1. Read `frontier_val.json` and `evolution_summary.jsonl`: what's been tried, what worked and didn't.
2. **Deep-read failed AND successful trajectories** under `traces/`. Most important step.
3. Read the candidates' harness files under `candidates/`.
4. Return:

```
STATE: <5-line summary: current scores, what's been tried>

HYPOTHESIS: "<falsifiable claim about what will improve scores>"
CANDIDATE: name=<snake_case>, parent=<candidate name>, changes="<specific changes>", prediction="<expected score change>"
```

### Step 2: Implement (1 subagent)

Launch **1 Agent subagent** (subagent_type: "general-purpose"). The prompt must include the candidate name, the parent, the specific changes from Step 1, and the working directory.

The subagent should:

1. Copy the parent's folder: `cp -r candidates/<parent> candidates/<name>`.
2. Make targeted changes according to the hypothesis.
3. Update `candidates/<name>/harness.json` so it lists exactly the files in the folder.
4. Return: folder path.

### Step 3: Write pending_eval.json

Write `pending_eval.json` to the path specified in the task prompt:

```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "<name>",
      "parent": "<parent name>",
      "hypothesis": "<falsifiable claim>",
      "changes": "<what was changed>"
    }
  ]
}
```

Output: `CANDIDATES: <name>`

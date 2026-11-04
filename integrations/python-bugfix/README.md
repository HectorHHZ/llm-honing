# python-bugfix

A small coding task suite for harness evolution. Each task is a Python function with one
bug. An agent edits the task's workspace, and `CommandGrader` scores the fraction of hidden
cases that pass, with feedback a reflective method such as GEPA can read.

Load it with `honing.harness.tasks.load_task_suite("integrations/python-bugfix/tasks/suite.jsonl")`.

## Layout

```text
tasks/
  suite.jsonl            one line per task: prompt, workspace, grader command, split
  gepa-live.jsonl        the 11 tasks the live GEPA run uses (scripts/e2e/gepa_live.py)
  <task_id>/workspace/   what the agent starts from: the function with its bug
  <task_id>/reference/   a correct fix, used only by tests/test_example_tasks.py
  grader/grade.py        runs each hidden case in a child process inside the workspace
  grader/cases/          the hidden cases, one file per task
config/
  gepa-live-seed/        the seed harness for the live GEPA run
```

Paths in `suite.jsonl` and `gepa-live.jsonl` are relative to `tasks/`, and the manifest
loader refuses a path that leaves that directory, so the grader sits next to the tasks.

| Task | Manifest | Split |
| --- | --- | --- |
| `clamp` | `suite.jsonl` | train |
| `mean` | `suite.jsonl` | val |
| `reverse_words` | `suite.jsonl` | heldout |
| `word_count` | `gepa-live.jsonl` | train |
| `max_index` | `gepa-live.jsonl` | train |
| `in_range` | `gepa-live.jsonl` | train |
| `average` | `gepa-live.jsonl` | train |
| `find_user` | `gepa-live.jsonl` | train |
| `count_between` | `gepa-live.jsonl` | train |
| `last_word` | `gepa-live.jsonl` | val |
| `clamp_index` | `gepa-live.jsonl` | val |
| `median` | `gepa-live.jsonl` | val |
| `count_tag` | `gepa-live.jsonl` | val |
| `overlap` | `gepa-live.jsonl` | val |

`gepa-live.jsonl` is a separate manifest so the live run keeps its own 6 train / 5 val split.

## How grading works

`CommandGrader` copies `grader/` to a directory outside the agent's workspace and runs
`python3 -I grade.py cases/<task_id>.json` there, with the workspace in `HONING_WORKSPACE`.
`grade.py` reads its cases, deletes its copy of them, and then runs each case in a child
process started in the workspace; it never imports the agent's code itself. Its last line
of stdout is `{"score": ..., "feedback": ...}`.

## Not a security boundary

The agent's code runs as the same user as the grader, so it can still reach the hidden
cases, for example through a parent process's `/proc/<pid>/cwd`.
`tests/test_example_tasks.py` keeps that read as a strict xfail. Do not optimize against
this suite with GEPA or RL until the sandboxes in #16 land.

## Adding a task

Add `tasks/<task_id>/workspace/`, `tasks/<task_id>/reference/` and
`tasks/grader/cases/<task_id>.json`, then one line in `tasks/suite.jsonl`. Every task shares
`grader/`, and a task's content hash covers its whole `grader_files` directory, so changing
any task's cases changes every task's hash. Giving each task its own hidden cases needs a
change to `CommandGrader` and is left to a follow-up.

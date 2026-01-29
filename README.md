Below is a **complete, ready-to-drop-in `README.md`** for the app in its current, tested-ready state. It is written to be accurate to the authoritative files you shared and reflects the actual behavior (DAG execution, unified diffs, validators, context packs, planner mode, dry run, artifacts).

---

# AI Orchestrator for Automated Code Refactoring

This project is an **AI-driven software development orchestrator**.
It automates large-scale code changes (refactors, restructures, boilerplate generation) while enforcing **deterministic validation, policy checks, and version control discipline**.

The system is designed to replace linear prompting with a **reproducible, auditable SDLC pipeline** driven by LLMs, validators, and DAG execution.

---

## Key Capabilities

* **Automated code changes** using LLMs (preferred format: unified diffs)
* **DAG-based execution** (single-node by default, multi-node via planner)
* **Bounded context at scale** via context packs and cached module summaries
* **Deterministic validation gates** (tests, lint, typecheck, build, policies)
* **Policy-as-code enforcement** (architecture boundaries, forbidden imports, secrets scanning)
* **Automatic patch repair** when diffs fail to apply
* **Full artifact logging** for every run and node
* **Git-native workflow** with controlled commits

---

## Requirements

* Python 3.10+
* Git (must be run inside a git repository)
* An OpenAI-compatible API key available in your environment
* A valid `.ai-orchestrator.json` configuration file in the repo root

---

## Installation

Clone or install the orchestrator package into your environment (editable install recommended):

```bash
pip install -e .
```

Ensure your repository contains a config file:

```bash
.ai-orchestrator.json
```

---

## Configuration (`.ai-orchestrator.json`)

Minimal example:

```json
{
  "model": "gpt-5.1-codex-mini",
  "test_command": "pytest",
  "max_files_per_run": 5,
  "include_globs": ["**/*.py"],
  "exclude_globs": ["**/__pycache__/**"],
  "validators": [
    { "name": "tests", "command": "pytest" }
  ]
}
```

### Advanced configuration (validators + architecture policies)

```json
{
  "test_command": "pytest",
  "validators": [
    { "name": "lint", "command": "ruff ." },
    { "name": "tests", "command": "pytest" },
    { "name": "layering", "kind": "policy" },
    { "name": "secrets", "kind": "policy" }
  ],
  "architecture": {
    "layering": {
      "forbidden_imports": {
        "ui": ["backend", "db"],
        "cli": ["db"]
      }
    },
    "secrets": {
      "deny_patterns": ["AKIA[0-9A-Z]{16}"]
    }
  }
}
```

---

## Basic Usage

### Run a phase (default: single-node DAG)

```bash
ai-dev run oop_refactor
```

What happens:

1. A new branch is created (e.g. `ai/oop_refactor`)
2. Files are selected deterministically
3. A bounded **context pack** is built
4. The LLM generates unified diffs
5. Patches are applied (with repair if needed)
6. Validators run (tests, policies, etc.)
7. Changes are committed if validation passes

---

### Dry run (no changes applied)

```bash
ai-dev run oop_refactor --dry-run
```

* Generates context, prompts, and patches
* **Does not** apply patches, run validators, or commit
* Useful for inspection and debugging

---

## Planner Mode (Multi-node DAG)

Planner mode lets the model propose a **small execution plan** (DAG) instead of a single linear step.

```bash
ai-dev run oop_refactor --use-plan
```

Typical planner output:

```
scaffold → implement → test → fix
```

Each node:

* Runs with its own context pack
* Can have its own validators
* Commits independently (default policy)

Planner output is **strict-schema validated** before execution.

---

## Context Packs (Context at Scale)

For each node, the orchestrator builds a **bounded context pack** containing:

* Architecture / constraints excerpts
* Cached module summaries (hash-based, reusable)
* Targeted file snippets
* Failure hints from previous nodes (if any)

Artifacts written:

```
.ai-orchestrator/
├── context_pack.json                 # latest pointer
├── context_pack_index.json           # run index
└── runs/
    └── <timestamp>-<run>/
        └── nodes/
            └── <node_id>/
                └── context_pack.json
```

Phases consume this via:

```python
ctx.state["context_pack"]
```

---

## Validators and Policy Enforcement

### Supported validator types

1. **Shell validators**

   * lint, typecheck, tests, build
2. **Policy validators**

   * forbidden imports
   * dependency boundaries
   * secrets scanning (git diff–based)

Validators:

* Run in order
* Stop on first failure by default
* Can be overridden per DAG node

Results are written to:

```
.ai-orchestrator/validator_results.json   # latest pointer
.ai-orchestrator/runs/.../nodes/.../validator_results.json
```

---

## Patch Application and Repair

* Preferred format: **unified diffs**
* Applied using `git apply --whitespace=nowarn --recount`
* If a patch fails:

  * Diagnostics are captured
  * Relevant file snippets are extracted
  * The LLM attempts up to **two repair passes**
* All attempts are logged as artifacts

---

## Artifacts and Auditing

Every run is fully auditable:

```
.ai-orchestrator/runs/<timestamp>-<run>/
├── run.json
├── summary.json
└── nodes/
    └── <node_id>/
        ├── selected_files.json
        ├── context_pack.json
        ├── patches.json
        ├── validator_results.json
        ├── patch_apply_error.txt
        └── patch_repair_attempt_*.json
```

Nothing is hidden or implicit.

---

## Git Workflow

* Runs on a dedicated branch (`ai/<run_name>`)
* Commits are created:

  * **per node** (default, recommended for refactors), or
  * **once at the end** (configurable)
* Dry runs do not touch git

---

## What This Tool Is For

* Large refactors across many files
* Boilerplate generation with strong guarantees
* Enforcing architectural rules automatically
* Replacing manual “review-driven” iteration with deterministic gates

## What It Is Not (Yet)

* A runtime agent inside your production app
* A substitute for human architectural decisions
* A fully autonomous system without validation

---

## Next Steps (Recommended)

* Add more phases (API scaffold, migrations, CI, containers)
* Add stronger policy validators (license checks, perf budgets)
* Integrate with CI to run orchestrator in “proposal mode”

---


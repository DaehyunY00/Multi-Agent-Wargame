# AGENTS.md

## Project
This repository implements a tactical turn-based wargame simulation framework
with LLM-driven multi-agent decision making.

## Primary architecture
- `src/wargame/core`: hex grid, terrain, enums, shared models
- `src/wargame/combat`: Lanchester-based combat resolution
- `src/wargame/engine`: state manager, fog of war, turn advancement
- `src/wargame/agents`: LLM wrappers, prompts, action parser, baseline agents
- `src/wargame/orchestrator`: blue/red/white turn loop
- `src/wargame/scenarios`: scenario schema, loader, preset scenarios
- `src/wargame/analysis`: metrics, statistical summaries, plots
- `tests`: pytest unit/integration tests

## Technical constraints
- Python 3.11+
- Keep the engine independent from LLM backends
- LLM must only decide actions; combat outcome must be resolved by engine logic
- Use dataclasses and type hints
- Prefer pure Python + NumPy + SciPy + pandas
- Do not introduce LangChain or large orchestration frameworks
- Keep modules small and testable

## Domain constraints
- Fog of war is faction-specific
- Action outputs must be validated against a strict schema
- Invalid LLM outputs must fall back safely
- Logs must be saved as JSONL
- Scenario definitions should be loadable from YAML

## Testing
Before considering a task done:
1. add or update pytest tests
2. run unit tests relevant to touched files
3. run formatting/lint/type checks if configured
4. explain any residual risks or TODOs

## Done means
A task is complete only if:
- code is implemented in the correct module
- tests pass for the changed behavior
- public interfaces are documented
- no placeholder stubs remain unless explicitly requested

## Working style
- Make the smallest change that fully solves the task
- Do not refactor unrelated files
- Preserve deterministic behavior where seeds are provided
- Prefer explicit interfaces over hidden coupling
# Repository Conventions

- Treat `Research_plan.md` as the source of truth for module boundaries.
- Keep the simulation engine independent from any LLM backend implementation.
- Do not introduce LangChain or similar orchestration frameworks.
- Prefer `dataclasses`, `Enum`, and explicit type hints across modules.
- Keep this stage limited to placeholders, interfaces, and typed scaffolding.

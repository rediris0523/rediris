# RedIris Task Center Module

This directory preserves the inherited Moirai-era task-center foundation for the RedIris candidate repository.

The old source README contained stale prior-project and netuid-era wording. In this candidate, the module should be read as the RedIris task coordination foundation, not as proof of a live deployed task center.

The internal package path remains `rediris.task_center` to avoid breaking imports.

## Candidate Role

In RedIris, a task center would coordinate canon/performance/state-transition challenges.

Possible responsibilities:

- publish public character/world challenges;
- define allowed source ranges and output schemas;
- track miner submissions;
- route submissions to validators;
- support score archive and Lifecycle Ledger concepts;
- expose non-sensitive challenge, score, miner, and validator APIs;
- help separate local concept simulation from future Bittensor deployment.

## Important Boundary

Do not read this module as live deployment evidence.

This candidate does not claim:

- live RedIris task-center operation;
- wallet use;
- subtensor calls;
- registered netuid;
- submitted weights;
- live emissions;
- production readiness;
- commercial readiness;
- rights clearance;
- active public endpoint availability.

## Files To Review

- `task_center_main.py`: FastAPI task-center entrypoint.
- `api/`: audit, miners, scores, tasks, and validators routes.
- `services/`: lifecycle, miner health, miner selection, score archive, dispatcher, validator, idle reward paths.
- `schemas/`: inherited task/audit/dataset/miner/score schemas.
- `scripts/`: inherited simulation and helper scripts. Do not run scripts that require wallets, subtensor, chain access, external endpoints, or private local keys.
- `config.example.yml`: placeholder config example for future controlled development.

## Safe Local Review

For U4 / MC05 review, read files only.

Before any future execution:

- prepare an isolated local config;
- keep wallet, hotkey, coldkey, API tokens, and provider credentials out of the repo;
- do not call subtensor, public endpoints, or `set_weights` in documentation review;
- start with local/no-chain examples;
- document any external call in a non-sensitive ledger.

## Related Docs

- `../../README.md`
- `../../docs/PUBLIC_CLAIMS_BOUNDARY.md`
- `../../docs/RUNTIME_HOLD_AND_NO_CHAIN_TEST_PLAN.md`
- `../../docs/KNOWN_LIMITATIONS.md`
- `../../docs/CANON_DELTA_PACK_SCHEMA.md`
- `../../docs/CANON_INTEGRITY_SIGNAL_SCHEMA.md`

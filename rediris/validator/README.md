# RedIris Validator Module

This directory preserves the inherited Moirai-era validator service foundation for the RedIris candidate repository.

The internal package path remains `rediris.validator` to avoid breaking imports. Public-facing documentation should refer to this as the RedIris validator foundation.

## Candidate Role

In RedIris, validators score miner-submitted canon/performance/state-transition packages and output `Canon Integrity Signal`.

Validator scoring should focus on:

- canon compatibility;
- character identity consistency;
- creative contribution;
- source / memory recall;
- performance continuity;
- future evolvability;
- safety / rights / overcopy risk.

The inherited service currently contains FastAPI routes, audit/score APIs, Bittensor sync hooks, task processing, score cache/calculation, quality evaluation, and weight sync service paths. These are preserved as developer handoff foundation, not as proof of current RedIris validator operation.

## Important Boundary

Do not read this module as live validator or emissions evidence.

This candidate does not claim:

- live RedIris validator operation;
- wallet use;
- subtensor calls;
- `set_weights` execution;
- registered netuid;
- submitted weights;
- live emissions;
- production readiness;
- commercial readiness;
- rights clearance.

## Files To Review

- `validator_main.py`: FastAPI validator entrypoint.
- `api/`: audit and score routes.
- `services/`: Bittensor sync, task processor, weight sync, score cache/calculator, quality/dataset/audit validators.
- `schemas/`: inherited audit and score schemas.
- `config.example.yml`: placeholder config example for future controlled development.

## Safe Local Review

For U4 / MC05 review, read files only. Do not run validator services or weight sync flows unless a future development task explicitly authorizes an isolated local runtime.

Before any future execution:

- prepare an isolated local config;
- keep wallet, hotkey, coldkey, API tokens, and provider credentials out of the repo;
- do not call subtensor or `set_weights` in documentation review;
- start with no-chain, no-wallet local tests;
- document any external call in a non-sensitive ledger.

## Related Docs

- `../../README.md`
- `../../docs/PUBLIC_CLAIMS_BOUNDARY.md`
- `../../docs/RUNTIME_HOLD_AND_NO_CHAIN_TEST_PLAN.md`
- `../../docs/KNOWN_LIMITATIONS.md`
- `../../docs/CANON_INTEGRITY_SIGNAL_SCHEMA.md`

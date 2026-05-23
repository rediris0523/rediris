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

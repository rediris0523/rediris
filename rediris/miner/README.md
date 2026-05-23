# RedIris Miner Module

This directory preserves the inherited Moirai-era miner service foundation for the RedIris candidate repository.

The internal package path remains `rediris.miner` to avoid breaking imports. Public-facing documentation should refer to this as the RedIris miner foundation.

## Candidate Role

In RedIris, miners are expected to propose validated AI-native IP state transitions, not just generate more content.

Possible miner artifacts include:

- `Canon Delta Pack`
- `Canon Repair Pack`
- `Character Performance Pack`
- `New Character Candidate Pack`

The inherited service currently contains FastAPI routes, queue management, GPU/training support, dataset handling, Bittensor sync hooks, task monitoring, and HuggingFace upload paths. These are preserved as developer handoff foundation, not as proof of current RedIris deployment.

## Important Boundary

Do not read this module as live deployment evidence.

This candidate does not claim:

- live RedIris miner operation;
- wallet use;
- subtensor calls;
- registered netuid;
- submitted weights;
- live emissions;
- HuggingFace upload success;
- production readiness;
- commercial readiness;
- rights clearance.

## Files To Review

- `miner_main.py`: FastAPI miner entrypoint.
- `api/`: miner API routes.
- `services/`: queue, GPU, task monitor, dataset, training, inference, and Bittensor sync services.
- `schemas/`: inherited request/response schemas.
- `config.example.yml`: placeholder config example for future controlled development.

## Safe Local Review

For U4 / MC05 review, read files only. Do not run miner services unless a future development task explicitly authorizes an isolated local runtime.

Before any future execution:

- prepare a local config outside secrets-bearing paths;
- keep wallet, hotkey, coldkey, API tokens, and provider credentials out of the repo;
- avoid external endpoints unless explicitly authorized;
- start with no-chain, no-wallet local tests;
- document any external call in a non-sensitive ledger.

## Related Docs

- `../../README.md`
- `../../docs/PUBLIC_CLAIMS_BOUNDARY.md`
- `../../docs/RUNTIME_HOLD_AND_NO_CHAIN_TEST_PLAN.md`
- `../../docs/KNOWN_LIMITATIONS.md`
- `../../docs/CANON_DELTA_PACK_SCHEMA.md`

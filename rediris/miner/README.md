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

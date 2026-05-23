# RedIris

RedIris is a proposed creative-consensus subnet for AI-native IP.

It does not try to reward generic content volume. It focuses on a harder question: when AI characters, worlds, and stories keep changing, which changes deserve to become part of the accepted Canon State?

The proposed Bittensor intelligence commodity for RedIris is:

> validated AI-native IP state transitions

Miners propose state-transition packages for a living character/world system. Validators score whether those transitions preserve identity, memory, style, performance continuity, safety boundaries, and future evolvability.

## Repository Status

This repository is an adaptation candidate / concept implementation base.

RedIris builds on prior Moirai-era public repository and hackathon/subnet experiment evidence. The current codebase preserves the original Python/FastAPI/Bittensor-oriented miner, validator, task-center, and common module structure for future development.

Start here before public review:

- [Public Claims Boundary](docs/PUBLIC_CLAIMS_BOUNDARY.md)
- [Runtime HOLD And No-Chain Test Plan](docs/RUNTIME_HOLD_AND_NO_CHAIN_TEST_PLAN.md)
- [Known Limitations](docs/KNOWN_LIMITATIONS.md)
- [Legacy Materials Excluded](docs/LEGACY_MATERIALS_EXCLUDED.md)
- [Chinese README](docs/README_CN.md)

Important boundary:

- This repository does not prove that RedIris is live.
- This repository does not prove submitted Bittensor weights.
- This repository does not prove live emissions.
- This repository does not prove production readiness.
- This repository does not prove commercial readiness.
- This repository does not prove rights clearance.
- This repository does not prove mainnet readiness or testnet readiness.
- This repository does not prove official partner endorsement.
- This repository does not prove final visual canon for any character.

The internal Python package name is now `rediris` in this candidate. The inherited Moirai-era module structure is preserved, but imports and package paths have been renamed to match the RedIris project name.

## What RedIris Is

RedIris is a mechanism for maintaining and evolving AI-native character/world systems.

It is designed for:

- AI characters;
- virtual humans;
- game NPCs;
- interactive narrative agents;
- long-horizon role identity;
- world-state continuity;
- character performance consistency;
- source and style fidelity;
- creative regression testing;
- lifecycle governance for living IP.

## What RedIris Is Not

RedIris is not:

- a normal content-generation network;
- a simple prompt chain;
- a pure model-training leaderboard;
- a generic agent-tool marketplace;
- a legal rights-clearance system;
- a claim of current deployment;
- a claim that the 13 seed characters have final public canon;
- a claim that inherited Moirai-era PDFs or images are RedIris public proof.

## Core Objects

### Canon State

`Canon State` is the current accepted state of a living AI-native IP system. It can include character identity, memory, world rules, accepted scenes, relationship state, style and performance contracts, safety boundaries, rights/overcopy boundaries, and lifecycle state.

### Canon Delta Pack

A `Canon Delta Pack` is a miner-submitted candidate change to the Canon State.

It may include a scene continuation, dialogue, world event, relationship update, timeline state change, character motivation update, emotional arc, narrative branch, source-grounded rationale, and declared dependencies on prior canon.

### Canon Repair Pack

A `Canon Repair Pack` is a miner-submitted patch for contradiction, drift, weak performance, or faulty memory.

### Character Performance Pack

A `Character Performance Pack` captures performance continuity direction. It can describe gesture, expression, rhythm, speech pattern, listening behavior, visual continuity, voice direction, interactive persona rules, and storyboard/prompt constraints without generating final media.

### New Character Candidate Pack

A `New Character Candidate Pack` proposes a new character entering or challenging the world. It should describe identity premise, world role, relationship hooks, interaction grammar, conflict/value contribution, non-overlap with existing characters, consistency tests, and safety/rights risk.

### Canon Integrity Signal

A `Canon Integrity Signal` is validator scoring output for a submitted pack. It is not final truth, legal certification, rights clearance, or final canon by itself.

It should measure:

- canon compatibility;
- character identity consistency;
- creative contribution;
- source / memory recall;
- performance continuity;
- future evolvability;
- safety / rights / overcopy risk.

### Lifecycle Ledger

A `Lifecycle Ledger` is a non-sensitive record of accepted, repaired, rejected, archived, revived, merged, or challenged character/world state transitions.

## Miner Role

Miners propose candidate state transitions for AI-native IP.

A miner does not merely submit "more content." A miner submits a structured package that can be tested against public rules and hidden continuity checks. The useful output is not just a scene, image prompt, or dialogue fragment; it is an evidence-bearing proposal for how the living IP should evolve.

Example miner work:

- submit a Canon Delta Pack for a character's next decision;
- submit a Canon Repair Pack that resolves a contradiction;
- submit a Character Performance Pack that preserves voice, gesture, and persona;
- submit a New Character Candidate Pack that can challenge the current cast.

## Validator Role

Validators score candidate transitions and produce Canon Integrity Signals.

Validators should not be framed as generic taste judges. They test whether a proposed state transition preserves the same character, the same world, and the same creative identity while still adding meaningful future value.

A validator can use:

- public challenge rules;
- hidden canon facts;
- contradiction traps;
- style drift tests;
- source/memory recall checks;
- duplicate and overcopy detection;
- safety and rights-risk flags;
- commit-reveal or equivalent anti-cheat flow;
- score traces suitable for later audit.

## Bittensor Mechanism Alignment

RedIris has a clear proposed subnet work unit:

1. A task center publishes a character/world challenge.
2. Miners commit to and reveal structured canon/performance/state-transition packages.
3. Validators apply public scoring rules and hidden continuity tests.
4. Validators output Canon Integrity Signals.
5. Validator signals can become weight candidates / validator weights in a future subnet design.
6. Yuma Consensus or Yuma-style consensus can aggregate validator judgments.
7. In the proposed mechanism, emissions can reward miners whose state transitions improve canon continuity.

This is proposed mechanism wording. This candidate repository does not claim that RedIris has submitted weights, called `set_weights`, used a wallet, touched subtensor, or received live emissions.

## 13 Seed Characters As Genesis State

RedIris can start with 13 seed characters as genesis state.

They are not permanent protagonists and not final public canon. They are starting state for lifecycle testing. Over time, characters may become:

- `seed`
- `active`
- `ascendant`
- `branching`
- `dormant`
- `marginal`
- `retired`
- `revived`
- `challenged`
- `replaced_or_merged`

This makes the creative system evolvable: characters can grow, sleep, be repaired, return, split into branches, or be challenged by new candidates.

## Relationship To Moirai-Era Work

RedIris builds on prior Moirai-era public repository and hackathon/subnet experiment evidence.

The safe lineage statement is:

> RedIris builds from prior Moirai-era public repository and hackathon/subnet experiment evidence, and shifts the focus from distributed content generation to validated AI-native IP state transitions.

This repository preserves the Moirai-era code structure because it already separates miner, validator, task center, and common Bittensor-facing utilities. RedIris adapts that foundation toward canon continuity, character performance, creative regression, and lifecycle governance.

Inherited Moirai-era PDFs, images, and log screenshots are excluded from this public-clean package. See `docs/LEGACY_MATERIALS_EXCLUDED.md`.

## Repository Structure

```text
.
├── README.md
├── docs/
├── examples/
├── rediris/
│   ├── common/
│   ├── miner/
│   ├── task_center/
│   └── validator/
└── LICENSE
```

Key inherited modules:

- `rediris/miner/`: FastAPI miner service foundation.
- `rediris/validator/`: FastAPI validator service foundation.
- `rediris/task_center/`: task lifecycle, miner selection, scoring, and coordination foundation.
- `rediris/common/`: shared config, Bittensor, database, auth, crypto, middleware, logging, and utility paths.

## Reading The Repo Locally

For a safe local read:

1. Start with this `README.md`.
2. Read `docs/PUBLIC_CLAIMS_BOUNDARY.md`.
3. Read `docs/RUNTIME_HOLD_AND_NO_CHAIN_TEST_PLAN.md`.
4. Read `docs/KNOWN_LIMITATIONS.md`.
5. Read `docs/README_CN.md` if Chinese explanation is needed.
6. Read schema docs under `docs/`.
7. Read sample JSON under `examples/`.
8. Treat `rediris/*/README.md` as inherited module notes with RedIris caveats.
9. Read `docs/LEGACY_MATERIALS_EXCLUDED.md` for legacy material exclusion context.

Do not treat config examples as live deployment instructions. Do not run wallet, subtensor, external endpoint, HuggingFace upload, provider/model calls, or `set_weights` flows unless a future development task explicitly authorizes and isolates that runtime.

## Local Development Boundary

This candidate keeps inherited runtime dependencies in place for developer handoff. U4/U4.1 did not add new dependencies.

Before any real execution, a future developer should:

- read `docs/RUNTIME_HOLD_AND_NO_CHAIN_TEST_PLAN.md`;
- read `docs/KNOWN_LIMITATIONS.md`;
- resolve missing/incomplete imports, including absent `rediris/common/models/`;
- replace placeholder configs with a controlled local config;
- keep secrets out of the repository;
- keep `rediris` as the package name unless a future migration has a specific compatibility reason to change it;
- run only no-chain, no-wallet local tests first;
- document any external endpoint, wallet, provider, model, or chain use in a non-sensitive ledger.

## Documentation And Examples

- `docs/README_CN.md`: Chinese public/team explanation.
- `docs/PUBLIC_CLAIMS_BOUNDARY.md`: claims boundary for judges and public readers.
- `docs/RUNTIME_HOLD_AND_NO_CHAIN_TEST_PLAN.md`: runtime HOLD and no-chain plan.
- `docs/KNOWN_LIMITATIONS.md`: current known limitations.
- `docs/CANON_DELTA_PACK_SCHEMA.md`: schema notes for miner-side pack types.
- `docs/CANON_INTEGRITY_SIGNAL_SCHEMA.md`: schema notes for validator outputs.
- `examples/canon_delta_pack.example.json`: illustrative miner package.
- `examples/canon_integrity_signal.example.json`: illustrative validator signal.
- `docs/LEGACY_MATERIALS_EXCLUDED.md`: boundary for excluded Moirai-era PDFs/assets.

## License

This candidate preserves the original repository license file. Public claims and rights around generated or future creative content remain separate from code licensing and must be validated before any commercial or public release claim.

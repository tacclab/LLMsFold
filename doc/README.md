# Documentation Index

This folder contains architecture and component-level technical docs for LLMsFold.

## System
- `doc/architecture.md`: End-to-end runtime flow and service interactions.

## Components
- `doc/components/main-cli.md`: CLI entrypoint and orchestration lifecycle.
- `doc/components/generator.md`: Core generation loop and scoring flow.
- `doc/components/nvidia-client.md`: NVIDIA Boltz request/poll/parse logic.
- `doc/components/pocket.md`: Pocket discovery and residue extraction.
- `doc/components/chemistry.md`: Chemistry utilities and SMILES parsing.
- `doc/components/pubchem-service.md`: PubChem novelty and patent signals.
- `doc/components/config-and-schemas.md`: Settings and validation models.
- `doc/components/clients-factory.md`: Shared client caching and shutdown.

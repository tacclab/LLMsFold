# System Architecture

## Runtime Flow
```mermaid
flowchart LR
    CLI[main.py] --> CFG[src.core.config]
    CLI --> CHEM[src.chemistry]
    CLI --> GEN[src.generator]

    GEN --> PROMPT[src.prompt]
    GEN --> POCKET[src.pocket]
    GEN --> BOLTZ[src.nvidia_client]
    GEN --> PUBCHEM[src.services.pubchem]
    GEN --> SCHEMAS[src.schemas]

    BOLTZ --> NVAPI[(NVIDIA Boltz API)]
    PUBCHEM --> PCAPI[(PubChem PUG REST)]
    GEN --> CSV[(unified_report.csv)]
```

## Request-Level Sequence
```mermaid
sequenceDiagram
    participant User
    participant CLI as main.py
    participant Gen as MoleculeGenerator
    participant LLM as Groq Chat API
    participant Boltz as NVIDIA Boltz API
    participant PubChem as PubChem API

    User->>CLI: run pipeline
    CLI->>CLI: load env + parse args
    CLI->>Gen: run(options)
    loop iteration 1..N
        Gen->>LLM: generate SMILES list
        LLM-->>Gen: text payload
        Gen->>Gen: parse/filter/validate
        Gen->>Boltz: score valid molecules
        Boltz-->>Gen: affinity + confidence metrics
        Gen->>Gen: compute similarity and reward
    end
    Gen->>PubChem: novelty checks per SMILES
    PubChem-->>Gen: CID + patent counts
    Gen-->>CLI: report path
```

## Data Contracts
- Runtime options are validated by `PipelineOptions`.
- Boltz responses are normalized by `BoltzPrediction` and `BoltzLigandAffinity`.
- Final per-molecule rows are represented by `MoleculeRecord` plus patent fields from `PatentCheckResult`.

## Failure Boundaries
- Missing env keys: startup validation error in `main.py`.
- External API errors: handled in service/client layers and downgraded to empty/default results where possible.
- No surviving molecules: generator returns `None` and logs warning instead of writing a report.

# Component: PubChem Service (`src/services/pubchem.py`)

## Purpose
Adds novelty and patent-related indicators to final candidate molecules.

## Public API
- `PubChemService.check_patents(smiles)`
- `check_pubchem_patents(smiles)` (legacy tuple wrapper)

## Request Flow
```mermaid
flowchart TD
    A[SMILES] --> B[Resolve CID from /compound/smiles/.../cids]
    B --> C{CID found?}
    C -->|Yes| D[Fetch identity PatentID xrefs]
    C -->|No| E[identity_patents = 0]
    D --> F[Run substructure search]
    E --> F
    F --> G{202 waiting list key?}
    G -->|Yes| H[Poll listkey endpoint up to 3 times]
    G -->|No| I[substructure_patents = 0]
    H --> J[Count top 10 matching CIDs]
    I --> K[Build PatentCheckResult]
    J --> K
```

## Interpretation
- `pubchem_cid` present implies known compound identity.
- `identity_patents` approximates direct identity patent signal.
- `substructure_patents` approximates broad prior-art density.

## Error Handling
- HTTP and parsing failures are caught and mapped to empty defaults (`PatentCheckResult()`).

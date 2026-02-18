# Component: NVIDIA Client (`src/nvidia_client.py`)

## Purpose
Wraps NVIDIA Boltz API calls and computes molecule property rows for downstream ranking.

## Public API
- `BoltzClient.make_nvcf_call(smiles, sequence, pocket_residues=None)`
- `BoltzClient.compute_properties(smiles_list, sequence, pocket_residues=None)`

## Request/Response Flow
```mermaid
flowchart TD
    A[SMILES + protein sequence] --> B[Build Boltz payload]
    B --> C[POST /boltz2/predict]
    C --> D{HTTP 200 or 202?}
    D -->|200| E[Parse response JSON]
    D -->|202| F[Poll status endpoint]
    F --> E
    D -->|Other| G[Return None]
    E --> H[Validate with BoltzPrediction schema]
    H --> I[Extract affinity/confidence values]
    I --> J[Compute RDKit descriptors + QED + SAS]
    J --> K[Build MoleculeRecord rows]
```

## External Endpoints
- Predict: `https://health.api.nvidia.com/v1/biology/mit/boltz2/predict`
- Polling: `https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{task_id}`

## Behavior Notes
- Polling checks every 5 seconds until completion/error.
- Missing `nvcf-reqid` on `202` is treated as failure (`None`).
- Schema validation failures are safely downgraded to `None`.
- `compute_properties` skips invalid SMILES and failed predictions.

## Output Columns Produced
- Boltz metrics: `pTM`, `ipTM`, `Confidence`, `pLDDT`, `Affinity_Prob`, `pIC50`, `IC50_uM`
- Chemistry descriptors: `MolWt`, `LogP`, `QED`, `SAS`, `TPSA`, `H_Acceptors`, `H_Donors`, `Rotatable_Bonds`

# Component: Pocket Utilities (`src/pocket.py`)

## Purpose
Detects target binding pockets and maps nearby residues used as optional generation constraints.

## Main Functions
- `get_binding_pockets_and_residues(pdb_path, output_dir="results")`
- `setup_p2rank()` and `get_p2rank_pocket(...)` (legacy P2Rank helpers)

## Pocket Discovery Flow
```mermaid
flowchart TD
    A[Input PDB] --> B[DeepChem ConvexHullPocketFinder]
    B --> C{Pockets found?}
    C -->|No| D[Return No pockets found, Unknown]
    C -->|Yes| E[Compute centers and approximate volumes]
    E --> F[Write all_pockets.csv]
    F --> G[Interactive pocket selection]
    G --> H[Load PDB with RDKit]
    H --> I[Collect residues within 8A]
    I --> J[Return center string + residue list]
```

## Selection Model
- User can select specific pocket ID.
- `0` selects largest approximate volume automatically.
- Selection is interactive via `input()`, so automation should use `--no-pocket` in CLI.

## Output Shape
- Pocket descriptor: `Center: x, y, z`
- Residue list: `ALA10_A, GLY27_B, ...`

## Legacy P2Rank Path
- `setup_p2rank` searches for `prank` executable under project directories and makes it executable.
- `get_p2rank_pocket` executes `prank predict` and parses top `residue_ids` from generated CSV.

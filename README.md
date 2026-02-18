# AI Molecular Generation Framework (LLMsFold)
<p align="center">
  <img src="images/LLMsFold.png" alt="LLMsFold Logo" width="300">
 
</p>

<p align="center">
  <strong>AI Molecular Generation Framework (LLMsFold)</strong>
</p>

An automated drug discovery pipeline that uses Large Language Models (LLMs) and the *NVIDIA Boltz-2* biological prediction engine to iteratively design and evaluate novel molecules.

## Key Features
* **Dynamic PDB Parsing**: Automatically extracts protein sequences from standard `.pdb` files for docking and affinity prediction.
* **LLM-Driven Generation**: Leverages `Llama-3.3-70B` (via Groq) to propose novel SMILES strings based on existing molecular leads.
* **Boltz-2 Integration**: Predicts binding probability (`Affinity_Prob`), `pIC50`, and structural confidence metrics (`ipTM`, `pLDDT`) via NVIDIA Health APIs.
* **RL Feedback Loop**: Implements a Reinforcement Learning-style loop where the best-performing molecules are used to refine the next generation of candidates.

--- 

<p align="center">
  
  
  <a href="#"><img src="https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?logo=python&logoColor=white" alt="Python Versions"></a>
  
  <a href="https://groq.com/"><img src="https://img.shields.io/badge/LLM-Llama--3.3--70B-orange.svg" alt="LLM: Llama-3.3"></a>
  
  <a href="https://health.api.nvidia.com/"><img src="https://img.shields.io/badge/Predictor-NVIDIA%20Boltz--2-76b900.svg?logo=nvidia&logoColor=white" alt="NVIDIA Boltz-2"></a>
</p>

## Repository Structure

```text
molgen-framework/
├── .env                  # Private: API keys and default file paths
├── pyproject.toml        # Modern Project Metadata (managed by uv)
├── uv.lock               # Deterministic dependency lockfile
├── main.py               # Main CLI entry point (Orchestration)
├── run.sh                # Automation wrapper (loads .env and executes)
├── sascorer.py           # RDKit Contrib script for synthetic accessibility
├── fpscores.pkl.gz       # Fragment scores for SAS calculation
├── src/                  # All source code lives here
│   ├── __init__.py
│   ├── utils.py          # PDB parsing, SMILES validation, Similarity metrics
│   ├── nvidia_client.py  # Async Boltz-2 API wrapper & polling logic
│   └── generator.py      # Core RL loop and LLM prompt logic
├── data/                 # Input data folder
│   ├── acvr1_R206H_clean.pdb  # Your specific target protein
│   └── few_shot_smiles1.csv   # Initial reference molecules
└── results/              # Auto-generated CSV reports 
``` 

## Installation
* Using uv (Recommended)

* This project is managed by uv for high-performance dependency management.


## Clone the repository

```bash
git clone https://github.com/your-username/molgen-framework.git
cd molgen-framework
```
## Install dependencies and create environment

* If you prefer not to use uv, you can install via pip:

```bash
pip install -r requirements.txt
Required Scoring Assets
```
* The framework requires external RDKit scoring scripts for Synthetic Accessibility (SAS):

```bash
wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py
wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz
```
## Versioning
* LLMsFold follows Semantic Versioning. We use pyproject.toml to manage project metadata. To check your current environment and versioning info:

```bash
uv pip list | grep molgen
```
## Quickstart
* MolGen is designed to be an automated pipeline. You can run it via the CLI or use the core components in your own scripts.

1. Configure your environment

* Create a .env file in the root directory:

```bash
GROQ_API_KEY="your_groq_key"
NVIDIA_API_KEY="your_nvidia_key"
PDB_FILE="data/acvr1_R206H_clean.pdb"
FEW_SHOT_CSV="data/few_shot_smiles.csv"
OUTPUT_DIR="results"
LLM_MODEL="llama-3.3-70b-versatile"
LLM_TEMPERATURE=0.8
```
2. Run the automated pipeline

```bash
# Using the shell wrapper
chmod +x run.sh
./run.sh

# Or using uv directly
uv run --env-file .env main.py --iters 3 --samples 5
```

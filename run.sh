#!/bin/sh

print_banner() {
cat <<'EOF'
+------------------------------------------------------------------------------+
|  _      _     __  __      ______    _     _                                 |
| | |    | |   |  \/  |    |  ____|  | |   | |                                |
| | |    | |   | \  / |___ | |__ ___ | | __| |                                |
| | |    | |   | |\/| / __||  __/ _ \| |/ _` |                                |
| | |____| |___| |  | \__ \| | | (_) | | (_| |                                |
| |______|_____|_|  |_|___/|_|  \___/|_|\__,_|                                |
|                                                                              |
|                                  LLMsFold                                    |
+------------------------------------------------------------------------------+
Authors: W. W. Waththe Liyanage, Fabio Bove, Dario Righelli, Salvatore Romano
         Rosa Visone, Marilena V. Iorio, Pietro Lio, Cristian Taccioli
EOF
}

print_banner
export LLMSFOLD_BANNER_SHOWN=1

# Load environment variables from .env.
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
    echo "Successfully loaded configuration from .env"
else
    echo "Warning: .env file not found. Ensure keys are exported manually."
fi

# Validation for required API keys
if [ -z "$GROQ_API_KEY" ] || [ -z "$NVIDIA_API_KEY" ]; then
    echo "Error: Missing API keys. Please check your .env file."
    exit 1
fi

# Set defaults if not provided in .env
PDB_PATH=${PDB_FILE:-"data/acvr1_R206H_clean.pdb"}
CSV_PATH=${FEW_SHOT_CSV:-"data/few_shot_smiles_patent.csv"}
OUT_DIR=${OUTPUT_DIR:-"results"}

echo "------------------------------------------------"
echo "Target PDB: $PDB_PATH"
echo "Reference CSV: $CSV_PATH"
echo "Output: $OUT_DIR"
echo "------------------------------------------------"


python3 main.py \
    --pdb "$PDB_PATH" \
    --csv "$CSV_PATH" \
    --out "$OUT_DIR"

#!/bin/bash

# Load environment variables from .env 
if [ -f .env ]; then
    export $(cat .env | xargs)
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
PDB_PATH=${PDB_FILE:-"data/target.pdb"}
CSV_PATH=${FEW_SHOT_CSV:-"data/few_shot_smiles1.csv"}
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
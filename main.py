import asyncio
import os
import argparse
from src.utils import extract_sequence_from_pdb
from src.nvidia_client import BoltzClient
from src.generator import generate_molecules_unified

async def main():
    # Setup CLI Arguments
    parser = argparse.ArgumentParser(description="AI Molecular Generation Framework")
    parser.add_argument("--pdb", type=str, default=os.getenv("PDB_FILE", "data/target.pdb"), help="Path to PDB file")
    parser.add_argument("--csv", type=str, default=os.getenv("FEW_SHOT_CSV", "data/few_shot_smiles1.csv"), help="Path to few-shot CSV")
    parser.add_argument("--out", type=str, default=os.getenv("OUTPUT_DIR", "results"), help="Output directory")
    parser.add_argument("--iters", type=int, default=3, help="Number of RL iterations")
    parser.add_argument("--samples", type=int, default=5, help="Samples per iteration")
    parser.add_argument("--no-pocket", action="store_false", dest="use_pocket", help="Disable pocket-aware generation")
    parser.set_defaults(use_pocket=True)
    args = parser.parse_args()

    # API Keys from Environment
    groq_key = os.getenv("GROQ_API_KEY")
    nv_key = os.getenv("NVIDIA_API_KEY")

    if not groq_key or not nv_key:
        print("Error: GROQ_API_KEY and NVIDIA_API_KEY must be set in environment variables.")
        return

    if not os.path.exists(args.pdb):
        print(f"Error: PDB file not found at {args.pdb}")
        return

    # 1. Extract sequence from PDB
    print(f"Extracting sequence from {args.pdb}...")
    protein_seq = extract_sequence_from_pdb(args.pdb)
    
    # 2. Initialize Boltz Client
    boltz = BoltzClient(api_key=nv_key)
    
    # 3. Run Pipeline
    await generate_molecules_unified( pdb_path=args.pdb,
        boltz_client=boltz,
        output_dir=args.out,
        protein_sequence=protein_seq,
        few_shot_csv=args.csv,
        max_iterations=args.iters,
        max_samples=args.samples,
        use_pocket_data=args.use_pocket
    )

if __name__ == "__main__":
    asyncio.run(main())
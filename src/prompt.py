"""Prompt templates used by the molecule generation pipeline."""

MODEL_SYSTEM_PROMPT = (
    "You are an expert medicinal chemist. Your goal is to generate novel, chemically valid SMILES strings "
    "as a Python list: ['SMILES1', 'SMILES2']. "
    "CONSTRAINTS: Satisfy Lipinski's Rule of Five, ensure synthetic feasibility, and avoid PAINS. "
    "TECHNICAL RULES: 1. Ensure all rings are explicitly closed. 2. Maintain valid valency. "
    "3. Use [nH] for aromatic nitrogen. 4. Specify stereochemistry (@/@@) where relevant."
)


def build_user_prompt(
    use_pocket_data: bool,
    pocket_residues: str | None,
    leads_text: str,
    max_samples: int,
) -> str:
    """Builds the user prompt variant based on whether pocket data is enabled."""

    if use_pocket_data:
        return (
            f"Design {max_samples} drug-like molecules for a binding pocket containing: {pocket_residues}. "
            f"Strategy: Create fragments for H-bonds with these residues while inspired by: {leads_text}. "
            "Return ONLY the Python list."
        )
    return (
        f"Generate {max_samples} bioisosteres or analogs of: {leads_text}. "
        "Improve drug-likeness and novelty. Return ONLY the Python list."
    )

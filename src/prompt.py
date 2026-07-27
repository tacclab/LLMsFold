"""Prompt templates used by the molecule generation pipeline."""

MODEL_SYSTEM_PROMPT = (
    "You are an expert medicinal chemist. Your goal is to generate novel, chemically valid SMILES strings "
    "as a Python list: ['SMILES1', 'SMILES2']. "
    "CONSTRAINTS: Satisfy Lipinski's Rule of Five, ensure synthetic feasibility, and avoid PAINS. "
    "TECHNICAL RULES: 1. Ensure all rings are explicitly closed. 2. Maintain valid valency. "
    "3. Use [nH] for aromatic nitrogen. 4. Specify stereochemistry (@/@@) where relevant. "
    "You may also be shown molecules to AVOID because they failed on one axis of the design goal "
    "(binding affinity vs. synthesizability) even though they succeeded on the other — do not repeat "
    "their scaffolds or substructures."
)


def _build_avoid_clause(avoid_hard_to_synthesize: str, avoid_weak_binders: str) -> str:
    """Builds the contrastive AVOID clause from negative-example pools."""

    avoid_parts = []
    if avoid_hard_to_synthesize:
        avoid_parts.append(
            f"strong binders that were too hard to synthesize, like: {avoid_hard_to_synthesize}"
        )
    if avoid_weak_binders:
        avoid_parts.append(
            f"easily synthesizable molecules that bound too weakly, like: {avoid_weak_binders}"
        )
    if not avoid_parts:
        return ""
    return f"AVOID repeating the failure patterns of {'; and '.join(avoid_parts)}. "


def _build_already_proposed_clause(already_proposed: str) -> str:
    """Builds the exact-repeat clause from the run's proposal history."""

    if not already_proposed:
        return ""
    return f"ALREADY PROPOSED, do not repeat these exact structures verbatim: {already_proposed}. "


def build_user_prompt(
    use_pocket_data: bool,
    pocket_residues: str | None,
    leads_text: str,
    max_samples: int,
    avoid_hard_to_synthesize: str = "",
    avoid_weak_binders: str = "",
    already_proposed: str = "",
) -> str:
    """Builds the user prompt variant based on whether pocket data is enabled."""

    avoid_clause = _build_avoid_clause(avoid_hard_to_synthesize, avoid_weak_binders)
    already_proposed_clause = _build_already_proposed_clause(already_proposed)

    if use_pocket_data:
        return (
            f"Design {max_samples} drug-like molecules for a binding pocket containing: {pocket_residues}. "
            f"Strategy: Create fragments for H-bonds with these residues while inspired by: {leads_text}. "
            f"{avoid_clause}{already_proposed_clause}"
            "Return ONLY the Python list."
        )
    return (
        f"Generate {max_samples} bioisosteres or analogs of: {leads_text}. "
        f"Improve drug-likeness and novelty. {avoid_clause}{already_proposed_clause}"
        "Return ONLY the Python list."
    )

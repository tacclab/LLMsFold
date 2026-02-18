"""Core molecule generation loop and orchestration."""

import asyncio
import os
import re
from functools import lru_cache
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import FilterCatalog, rdFingerprintGenerator

from src.chemistry import (
    calculate_reward,
    get_max_similarity,
    parse_smiles_from_text,
    passes_lipinski,
)
from src.clients import get_cached_groq_client
from src.core.config import get_generator_settings
from src.core.constants import (
    ADJ_AFFINITY_THRESHOLD,
    CONTEXT_LEADS_WINDOW,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SAMPLES,
    MORGAN_FINGERPRINT_RADIUS,
    SAS_SCORE_MAX,
    SEED_SMILES_LIMIT,
    UNIFIED_REPORT_FILENAME,
)
from src.core.logging import get_logger
from src.nvidia_client import BoltzClient
from src.pocket import get_binding_pockets_and_residues
from src.prompt import MODEL_SYSTEM_PROMPT, build_user_prompt
from src.schemas import PipelineOptions
from src.services import PubChemService

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_filter_catalog() -> FilterCatalog.FilterCatalog:
    """Creates and caches PAINS/BRENK filter catalogs."""

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


class MoleculeGenerator:
    """Coordinates LLM generation, Boltz scoring, and IP checks."""

    def __init__(
        self,
        groq_api_key: str,
        boltz_client: BoltzClient,
        pubchem_service: PubChemService | None = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        llm_temperature: float = DEFAULT_LLM_TEMPERATURE,
    ) -> None:
        """Initializes a generator with cached service clients."""

        self._groq_client = get_cached_groq_client(groq_api_key)
        self._boltz_client = boltz_client
        self._pubchem_service = pubchem_service or PubChemService()
        self._filter_catalog = _get_filter_catalog()
        self._llm_model = llm_model
        self._llm_temperature = llm_temperature

    @staticmethod
    def _extract_residue_indices(pocket_residues: str | list[str] | None) -> list[int]:
        """Converts residue descriptors into integer residue indices."""

        residue_list = pocket_residues.split(",") if isinstance(pocket_residues, str) else (pocket_residues or [])
        clean_indices: list[int] = []
        for residue in residue_list:
            match = re.search(r"\d+", str(residue))
            if match:
                clean_indices.append(int(match.group()))
        return clean_indices

    @staticmethod
    def _build_user_prompt(
        use_pocket_data: bool,
        pocket_residues: str | None,
        leads_text: str,
        max_samples: int,
    ) -> str:
        """Compatibility wrapper around prompt templates module."""

        return build_user_prompt(use_pocket_data, pocket_residues, leads_text, max_samples)

    @staticmethod
    def _post_process_scores(results_df: pd.DataFrame, target_fps: list[Any]) -> pd.DataFrame:
        """Adds similarity and reward columns used for ranking."""

        if results_df.empty:
            return results_df

        metrics = results_df["SMILES"].apply(
            lambda smiles: {"MaxSim": get_max_similarity(smiles, target_fps)}
        ).apply(pd.Series)
        enriched = pd.concat([results_df, metrics], axis=1)
        enriched["adj_affinity"] = enriched["Affinity_Prob"].apply(
            lambda value: value if value > ADJ_AFFINITY_THRESHOLD else 0
        )
        enriched["score"] = enriched.apply(calculate_reward, axis=1)
        return enriched[enriched["SAS"] <= SAS_SCORE_MAX]

    async def run(self, options: PipelineOptions) -> str | None:
        """Executes iterative candidate generation and writes the final CSV.

        Args:
            options: Validated runtime parameters for one run.

        Returns:
            Path to generated report when successful, otherwise `None`.
        """

        few_shot_data = pd.read_csv(options.few_shot_csv, sep=";")
        positives = few_shot_data["Smiles"].dropna().tolist()[:SEED_SMILES_LIMIT]

        fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_FINGERPRINT_RADIUS
        )
        target_fps = [
            fingerprint_generator.GetFingerprint(Chem.MolFromSmiles(smiles))
            for smiles in positives
            if Chem.MolFromSmiles(smiles)
        ]

        global_registry: list[dict[str, Any]] = []
        context_leads = positives.copy()

        pocket_residues: str | None = None
        clean_indices: list[int] = []
        if options.use_pocket_data:
            pocket_coords, pocket_residues = get_binding_pockets_and_residues(
                options.pdb_path,
                options.output_dir,
            )
            clean_indices = self._extract_residue_indices(pocket_residues)
            logger.info("Targeting Pocket at %s", pocket_coords)
            logger.info("Cleaned Residue Indices: %s", clean_indices)
        else:
            logger.info("Running in Few-Shot mode (Ignoring pocket constraints).")

        for iteration in range(1, options.max_iterations + 1):
            logger.info("--- ITERATION %s ---", iteration)
            leads_text = ", ".join(context_leads[-CONTEXT_LEADS_WINDOW:])
            user_content = self._build_user_prompt(
                options.use_pocket_data,
                pocket_residues,
                leads_text,
                options.max_samples,
            )

            try:
                completion = self._groq_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self._llm_temperature,
                )

                raw_content = completion.choices[0].message.content or ""
                new_smiles = parse_smiles_from_text(raw_content)

                valid_smiles: list[str] = []
                for smiles in new_smiles:
                    mol = Chem.MolFromSmiles(smiles, sanitize=True)
                    if not mol:
                        logger.warning("Skipping invalid/unkekulizable SMILES: %s", smiles)
                        continue
                    if self._filter_catalog.HasMatch(mol):
                        continue
                    if passes_lipinski(mol):
                        valid_smiles.append(smiles)

                results_df = await self._boltz_client.compute_properties(
                    valid_smiles,
                    options.protein_sequence,
                    pocket_residues=clean_indices if options.use_pocket_data else None,
                )

                scored = self._post_process_scores(results_df, target_fps)
                if not scored.empty:
                    global_registry.extend(scored.to_dict("records"))
            except Exception as exc:  # noqa: BLE001
                logger.error("Error in iteration %s: %s", iteration, exc, exc_info=True)
                continue

        if not global_registry:
            logger.warning(
                "No molecules were successfully generated. "
                "Check LLM connectivity or SMILES validation."
            )
            return None

        final_hits = pd.DataFrame(global_registry).drop_duplicates(subset="SMILES")
        logger.info("Verifying IP status for %s unique candidates...", len(final_hits))

        patent_checks = await asyncio.gather(
            *(self._pubchem_service.check_patents(smiles) for smiles in final_hits["SMILES"]),
        )
        ip_df = pd.DataFrame([item.to_report_row() for item in patent_checks])
        final_hits = pd.concat([final_hits.reset_index(drop=True), ip_df], axis=1)
        final_hits = final_hits.sort_values(by=["Is_Novel", "score"], ascending=[False, False])

        os.makedirs(options.output_dir, exist_ok=True)
        report_path = os.path.join(options.output_dir, UNIFIED_REPORT_FILENAME)
        final_hits.to_csv(report_path, index=False)
        logger.info("Workflow Complete. Results in %s", report_path)
        return report_path


async def generate_molecules_unified(
    pdb_path: str,
    boltz_client: BoltzClient,
    output_dir: str,
    protein_sequence: str,
    few_shot_csv: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    use_pocket_data: bool = True,
) -> str | None:
    """Backward-compatible function wrapper for the original API."""

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY must be set in environment variables.")

    options = PipelineOptions(
        pdb_path=pdb_path,
        few_shot_csv=few_shot_csv,
        output_dir=output_dir,
        protein_sequence=protein_sequence,
        max_iterations=max_iterations,
        max_samples=max_samples,
        use_pocket_data=use_pocket_data,
    )

    runtime_settings = get_generator_settings()
    generator = MoleculeGenerator(
        groq_api_key=groq_api_key,
        boltz_client=boltz_client,
        llm_model=runtime_settings.llm_model,
        llm_temperature=runtime_settings.llm_temperature,
    )
    return await generator.run(options)

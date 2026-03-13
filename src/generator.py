"""Core molecule generation loop and orchestration."""

import json
import re
import subprocess
from pathlib import Path
from functools import lru_cache
from time import perf_counter
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import FilterCatalog, rdFingerprintGenerator

from src.chemistry import (
    calculate_heuristic_score,
    get_max_similarity,
    parse_smiles_from_text,
    passes_lipinski,
)
from src.clients import get_cached_groq_client
from src.core.config import GeneratorSettings, get_generator_settings
from src.core.constants import (
    ADJ_AFFINITY_THRESHOLD,
    CONTEXT_LEADS_WINDOW,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    MORGAN_FINGERPRINT_RADIUS,
    SAS_SCORE_MAX,
    SEED_SMILES_LIMIT,
    UNIFIED_REPORT_FILENAME,
)
from src.core.exceptions import PocketDetectionError
from src.core.logging import get_logger
from src.core.messages import (
    invalid_smiles,
    no_molecules_generated,
    persist_structure_unavailable,
    pocket_detection_unavailable,
)
from src.core.progress import gather_with_progress, make_progress_bar
from src.nvidia_client import BoltzClient
from src.pocket import get_binding_pockets_and_residues
from src.prompt import MODEL_SYSTEM_PROMPT, build_user_prompt
from src.schemas import (
    BestStructureRecord,
    PipelineOptions,
    PocketContact,
    ScoredMoleculeRecord,
    UnifiedReportRow,
)
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
        settings: GeneratorSettings | None = None,
    ) -> None:
        """Initializes a generator with cached service clients."""

        self._settings = settings or get_generator_settings()
        self._groq_client = get_cached_groq_client(groq_api_key)
        self._boltz_client = boltz_client
        self._pubchem_service = pubchem_service or PubChemService(settings=self._settings)
        self._filter_catalog = _get_filter_catalog()
        self._llm_model = llm_model
        self._llm_temperature = llm_temperature
        self._adj_affinity_threshold = self._settings.adj_affinity_threshold
        self._sas_score_max = self._settings.sas_score_max
        self._best_structure_affinity_threshold = self._settings.best_structure_affinity_threshold

    @staticmethod
    def _extract_residue_contacts(pocket_residues: str | list[str] | None) -> list[PocketContact]:
        """Converts residue descriptors into chain-aware residue contacts."""

        residue_list = (
            pocket_residues.split(",")
            if isinstance(pocket_residues, str)
            else (pocket_residues or [])
        )
        contacts: list[PocketContact] = []
        for residue in residue_list:
            match = re.search(r"([A-Z]{3})(\d+)_([A-Za-z0-9])", str(residue).strip())
            if match:
                contacts.append(
                    PocketContact(chain_id=match.group(3), residue_index=int(match.group(2)))
                )
        return contacts

    @staticmethod
    def _enrich_scores(
        results_df: pd.DataFrame,
        target_fps: list[Any],
        *,
        adj_affinity_threshold: float = ADJ_AFFINITY_THRESHOLD,
    ) -> pd.DataFrame:
        """Adds similarity and reward columns used for ranking."""

        if results_df.empty:
            return results_df

        metrics = (
            results_df["SMILES"]
            .apply(lambda smiles: {"MaxSim": get_max_similarity(smiles, target_fps)})
            .apply(pd.Series)
        )
        enriched = pd.concat([results_df, metrics], axis=1)
        enriched["adj_affinity"] = enriched["Affinity_Prob"].apply(
            lambda value: value if value > adj_affinity_threshold else 0
        )
        enriched["score"] = enriched.apply(calculate_heuristic_score, axis=1)
        return enriched

    @staticmethod
    def _candidate_id(index: int) -> str:
        """Builds a human-friendly candidate id from 1-based rank."""

        return f"candidate-{index:04d}"

    @staticmethod
    def _persist_best_structures(
        best_structures_by_smiles: dict[str, BestStructureRecord],
        final_hits: pd.DataFrame,
        output_dir: Path,
        threshold: float | None,
        pocket_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persists docked structures for high-affinity molecules when enabled."""

        if threshold is None:
            return 0

        base_dir = output_dir / "best"
        saved_count = 0
        skipped_low_affinity = 0
        for row in final_hits.to_dict("records"):
            affinity = float(row.get("Affinity_Prob", 0.0))
            smiles = str(row.get("SMILES", ""))
            candidate_id = str(row.get("Candidate_ID", ""))
            if not smiles or not candidate_id:
                continue
            if affinity <= threshold:
                logger.info(
                    "Skipping structure save for %s: Affinity_Prob=%.3f <= BEST_STRUCTURE_AFFINITY_THRESHOLD=%.3f",
                    candidate_id,
                    affinity,
                    threshold,
                )
                skipped_low_affinity += 1
                continue

            structure = best_structures_by_smiles.get(smiles)
            if structure is None:
                logger.warning(persist_structure_unavailable(smiles, candidate_id))
                continue

            data_dir = base_dir / candidate_id / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "candidate_id": candidate_id,
                "smiles": smiles,
                "affinity_prob": affinity,
            }
            if pocket_metadata is not None:
                metadata["selected_pocket"] = pocket_metadata
            (data_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (data_dir / "structure.cif").write_text(structure.structure, encoding="utf-8")
            saved_count += 1

        if skipped_low_affinity:
            logger.info(
                "%s structure(s) skipped: Affinity_Prob below BEST_STRUCTURE_AFFINITY_THRESHOLD=%.3f. "
                "Lower BEST_STRUCTURE_AFFINITY_THRESHOLD in .env to save more structures.",
                skipped_low_affinity,
                threshold,
            )
        return saved_count

    @staticmethod
    def _post_process_scores(
        results_df: pd.DataFrame,
        target_fps: list[Any],
        *,
        adj_affinity_threshold: float = ADJ_AFFINITY_THRESHOLD,
        sas_score_max: float = SAS_SCORE_MAX,
    ) -> pd.DataFrame:
        """Applies the configured score thresholds to enriched candidate rows."""

        enriched = MoleculeGenerator._enrich_scores(
            results_df,
            target_fps,
            adj_affinity_threshold=adj_affinity_threshold,
        )
        if enriched.empty:
            return enriched
        return enriched[enriched["SAS"] <= sas_score_max]

    async def run(self, options: PipelineOptions) -> str | None:
        """Executes iterative candidate generation and writes the final CSV.

        Args:
            options: Validated runtime parameters for one run.

        Returns:
            Path to generated report when successful, otherwise `None`.
        """

        total_steps = 5
        with make_progress_bar(
            total=total_steps,
            desc="Pipeline steps",
            unit="step",
            leave=True,
        ) as step_progress:
            step_started_at = perf_counter()
            logger.info(
                "Step 1/%s: loading few-shot examples from %s", total_steps, options.few_shot_csv
            )
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
            step_progress.update(1)
            step_progress.set_postfix_str("few-shot ready")
            logger.info(
                "Step 1/%s complete in %.2fs: loaded %s seed molecules and %s fingerprints",
                total_steps,
                perf_counter() - step_started_at,
                len(positives),
                len(target_fps),
            )

            context_leads = positives.copy()

            step_started_at = perf_counter()
            pocket_residues: str | None = None
            residue_contacts: list[PocketContact] = []
            selected_pocket_metadata: dict[str, Any] | None = None
            use_pocket_data = options.use_pocket_data
            if use_pocket_data:
                logger.info(
                    "Step 2/%s: detecting pocket constraints from %s",
                    total_steps,
                    options.pdb_path,
                )
            else:
                logger.info(
                    "Step 2/%s: pocket detection disabled; using few-shot mode", total_steps
                )

            if use_pocket_data:
                try:
                    pocket_coords, pocket_residues = get_binding_pockets_and_residues(
                        options.pdb_path,
                        options.output_dir,
                    )
                except (PocketDetectionError, subprocess.SubprocessError) as exc:
                    logger.warning(pocket_detection_unavailable(exc))
                    use_pocket_data = False
                else:
                    residue_contacts = self._extract_residue_contacts(pocket_residues)
                    selected_pocket_metadata = {
                        "coordinates": pocket_coords,
                        "residues": pocket_residues,
                        "residue_contacts": [
                            contact.model_dump() for contact in residue_contacts
                        ],
                    }
                    logger.info(
                        "Using pocket constraints from %s with %s residue contacts",
                        pocket_coords,
                        len(residue_contacts),
                    )

            if not use_pocket_data:
                logger.info("Running few-shot mode without pocket constraints")

            step_progress.update(1)
            step_progress.set_postfix_str("constraints ready")
            logger.info(
                "Step 2/%s complete in %.2fs: mode=%s residue_contacts=%s",
                total_steps,
                perf_counter() - step_started_at,
                "pocket-aware" if use_pocket_data else "few-shot",
                len(residue_contacts),
            )

            logger.info(
                "Scoring thresholds: affinity>%.3f SAS<=%.3f",
                self._adj_affinity_threshold,
                self._sas_score_max,
            )

            step_started_at = perf_counter()
            logger.info(
                "Step 3/%s: running %s generation iterations with %s samples each",
                total_steps,
                options.max_iterations,
                options.max_samples,
            )
            global_registry: list[ScoredMoleculeRecord] = []
            best_structures_by_smiles: dict[str, BestStructureRecord] = {}
            with make_progress_bar(
                total=options.max_iterations,
                desc="Iterations",
                unit="iter",
            ) as iteration_progress:
                for iteration in range(1, options.max_iterations + 1):
                    iteration_started_at = perf_counter()
                    logger.info("Iteration %s/%s started", iteration, options.max_iterations)
                    leads_text = ", ".join(context_leads[-CONTEXT_LEADS_WINDOW:])
                    user_content = build_user_prompt(
                        use_pocket_data,
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
                        logger.info(
                            "Iteration %s/%s produced %s candidate SMILES",
                            iteration,
                            options.max_iterations,
                            len(new_smiles),
                        )

                        valid_smiles: list[str] = []
                        invalid_smiles_count = 0
                        catalog_filtered_count = 0
                        lipinski_filtered_count = 0
                        for smiles in make_progress_bar(
                            new_smiles,
                            desc=f"Iteration {iteration} filtering",
                            unit="mol",
                        ):
                            mol = Chem.MolFromSmiles(smiles, sanitize=True)
                            if not mol:
                                logger.warning(invalid_smiles(smiles))
                                invalid_smiles_count += 1
                                continue
                            if self._filter_catalog.HasMatch(mol):
                                catalog_filtered_count += 1
                                continue
                            if not passes_lipinski(mol):
                                lipinski_filtered_count += 1
                                continue
                            valid_smiles.append(smiles)

                        logger.info(
                            "Iteration %s/%s retained %s/%s candidates after filtering",
                            iteration,
                            options.max_iterations,
                            len(valid_smiles),
                            len(new_smiles),
                        )

                        results_df, best_structures = await self._boltz_client.compute_properties(
                            valid_smiles,
                            options.protein_sequence,
                            pocket_residues=residue_contacts if use_pocket_data else None,
                        )

                        for item in best_structures:
                            existing = best_structures_by_smiles.get(item.smiles)
                            if existing is None or item.affinity_prob > existing.affinity_prob:
                                best_structures_by_smiles[item.smiles] = item

                        enriched = self._enrich_scores(
                            results_df,
                            target_fps,
                            adj_affinity_threshold=self._adj_affinity_threshold,
                        )
                        rejected_sas_count = (
                            int((enriched["SAS"] > self._sas_score_max).sum())
                            if not enriched.empty
                            else 0
                        )
                        low_affinity_count = (
                            int((enriched["Affinity_Prob"] <= self._adj_affinity_threshold).sum())
                            if not enriched.empty
                            else 0
                        )
                        scored = (
                            enriched[enriched["SAS"] <= self._sas_score_max]
                            if not enriched.empty
                            else enriched
                        )

                        logger.info(
                            "Iteration %s/%s complete in %.2fs: parsed=%s valid=%s "
                            "boltz_scored=%s kept=%s invalid=%s catalog_filtered=%s "
                            "lipinski_filtered=%s boltz_failed=%s rejected_sas=%s "
                            "low_affinity=%s",
                            iteration,
                            options.max_iterations,
                            perf_counter() - iteration_started_at,
                            len(new_smiles),
                            len(valid_smiles),
                            len(results_df),
                            len(scored),
                            invalid_smiles_count,
                            catalog_filtered_count,
                            lipinski_filtered_count,
                            max(len(valid_smiles) - len(results_df), 0),
                            rejected_sas_count,
                            low_affinity_count,
                        )

                        if not scored.empty:
                            scored_records = [
                                ScoredMoleculeRecord.model_validate(row)
                                for row in scored.to_dict("records")
                            ]
                            global_registry.extend(scored_records)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Error in iteration %s: %s", iteration, exc, exc_info=True)
                    finally:
                        iteration_progress.update(1)
                        iteration_progress.set_postfix_str(f"iter={iteration}")

            step_progress.update(1)
            step_progress.set_postfix_str("generation complete")
            logger.info(
                "Step 3/%s complete in %.2fs: collected %s scored candidates",
                total_steps,
                perf_counter() - step_started_at,
                len(global_registry),
            )

            if not global_registry:
                logger.warning(no_molecules_generated())
                return None

            final_hits = pd.DataFrame.from_records(
                [record.model_dump() for record in global_registry]
            ).drop_duplicates(subset="SMILES")
            logger.info(
                "Deduplicated scored candidates down to %s unique molecules",
                len(final_hits),
            )

            step_started_at = perf_counter()
            logger.info(
                "Step 4/%s: verifying IP status for %s unique candidates",
                total_steps,
                len(final_hits),
            )
            patent_checks = await gather_with_progress(
                [self._pubchem_service.check_patents(smiles) for smiles in final_hits["SMILES"]],
                desc="Patent checks",
                unit="mol",
            )
            ip_df = pd.DataFrame([item.to_report_row() for item in patent_checks])
            final_hits = pd.concat([final_hits.reset_index(drop=True), ip_df], axis=1)
            final_hits = (
                final_hits.assign(
                    _pubchem_known_rank=final_hits["PubChem_Known"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"no": 0, "yes": 1})
                    .fillna(1)
                )
                .sort_values(by=["_pubchem_known_rank", "score"], ascending=[True, False])
                .drop(columns=["_pubchem_known_rank"])
            )
            final_hits["Candidate_ID"] = [
                self._candidate_id(index) for index in range(1, len(final_hits) + 1)
            ]
            final_hits = pd.DataFrame.from_records(
                [
                    UnifiedReportRow.model_validate(row).model_dump()
                    for row in final_hits.to_dict("records")
                ]
            )
            step_progress.update(1)
            step_progress.set_postfix_str("ip verified")
            logger.info(
                "Step 4/%s complete in %.2fs: ranked %s final candidates",
                total_steps,
                perf_counter() - step_started_at,
                len(final_hits),
            )

            step_started_at = perf_counter()
            logger.info("Step 5/%s: writing unified report to %s", total_steps, options.output_dir)
            options.output_dir.mkdir(parents=True, exist_ok=True)
            report_path = options.output_dir / UNIFIED_REPORT_FILENAME
            final_hits.to_csv(report_path, index=False)
            saved_structures = self._persist_best_structures(
                best_structures_by_smiles,
                final_hits,
                options.output_dir,
                self._best_structure_affinity_threshold,
                selected_pocket_metadata,
            )
            step_progress.update(1)
            step_progress.set_postfix_str("report written")
            logger.info(
                "Step 5/%s complete in %.2fs: wrote %s rows to %s (best_structures_saved=%s)",
                total_steps,
                perf_counter() - step_started_at,
                len(final_hits),
                report_path,
                saved_structures,
            )
            logger.info("Workflow Complete. Results in %s", report_path)
            return str(report_path)

"""Core molecule generation loop and orchestration."""

import json
import random
import re
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
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
    NEGATIVE_LEADS_WINDOW,
    REFERENCE_BASELINE_FILENAME,
    RUN_MANIFEST_FILENAME,
    SAS_SCORE_MAX,
    SAS_SCORE_MIN,
    SIMILARITY_PENALTY_FRACTION,
    SIMILARITY_PENALTY_THRESHOLD,
    UNIFIED_REPORT_FILENAME,
)
from src.core.exceptions import PocketDetectionError
from src.core.logging import get_logger
from src.core.messages import (
    holdout_evaluator_failed,
    holdout_evaluator_unavailable,
    invalid_smiles,
    no_molecules_generated,
    no_reference_scores_returned,
    persist_structure_unavailable,
    pocket_detection_unavailable,
    reference_scoring_failed,
    scaffold_analysis_failed,
    user_pocket_unavailable,
)
from src.core.progress import gather_with_progress, make_progress_bar
from src.diversity import ScaffoldAnalyzer
from src.evaluators import (
    HoldoutCandidate,
    HoldoutContext,
    HoldoutEvaluator,
    PoseBustersEvaluator,
    VinaEvaluator,
)
from src.manifest import IterationRecord, RunManifest, build_run_manifest, file_digest
from src.nvidia_client import BoltzClient
from src.pocket import (
    get_binding_pockets_and_residues,
    get_pocket_from_center,
    get_pocket_from_ligand,
    get_pocket_from_residues,
)
from src.prompt import MODEL_SYSTEM_PROMPT, build_user_prompt
from src.schemas import (
    BestStructureRecord,
    PipelineOptions,
    PocketContact,
    ReferenceBaselineRow,
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
        holdout_evaluators: Sequence[HoldoutEvaluator] | None = None,
    ) -> None:
        """Initializes a generator with cached service clients.

        `boltz_client` is the in-loop optimizer: its scores steer generation.
        `holdout_evaluators` are kept strictly apart from it and run only
        once generation has finished, so nothing they report can have
        influenced what was generated.
        """

        self._settings = settings or get_generator_settings()
        self._groq_client = get_cached_groq_client(
            groq_api_key, max_retries=self._settings.llm_max_retries
        )
        self._boltz_client = boltz_client
        self._pubchem_service = pubchem_service or PubChemService(settings=self._settings)
        self._filter_catalog = _get_filter_catalog()
        self._llm_model = llm_model
        self._llm_temperature = llm_temperature
        self._adj_affinity_threshold = self._settings.adj_affinity_threshold
        self._sas_score_max = self._settings.sas_score_max
        self._iptm_threshold = self._settings.iptm_threshold
        self._plddt_threshold = self._settings.plddt_threshold
        self._random_seed = self._settings.random_seed
        self._best_structure_affinity_threshold = self._settings.best_structure_affinity_threshold
        self._lead_diversity_max_sim = self._settings.lead_diversity_max_sim
        self._similarity_penalty_threshold = self._settings.similarity_penalty_threshold
        self._similarity_penalty_fraction = self._settings.similarity_penalty_fraction
        self._already_proposed_window = self._settings.already_proposed_window
        self._holdout_evaluators: list[HoldoutEvaluator] = list(
            holdout_evaluators
            if holdout_evaluators is not None
            else (
                PoseBustersEvaluator(),
                VinaEvaluator(receptor_pdbqt=self._settings.vina_receptor_pdbqt),
            )
        )

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
        sas_score_max: float = SAS_SCORE_MAX,
        similarity_penalty_threshold: float = SIMILARITY_PENALTY_THRESHOLD,
        similarity_penalty_fraction: float = SIMILARITY_PENALTY_FRACTION,
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
        enriched["Adj_Binding_Probability"] = enriched["Boltz2_Binding_Probability"].apply(
            lambda value: value if value > adj_affinity_threshold else 0
        )
        enriched["synth_factor"] = enriched["SAS"].apply(
            lambda sas: max(
                0.0,
                min(1.0, (sas_score_max - sas) / (sas_score_max - SAS_SCORE_MIN)),
            )
        )
        enriched["score"] = enriched.apply(
            lambda row: calculate_heuristic_score(
                row,
                similarity_penalty_threshold=similarity_penalty_threshold,
                similarity_penalty_fraction=similarity_penalty_fraction,
            ),
            axis=1,
        )
        return enriched

    @staticmethod
    def _candidate_id(index: int) -> str:
        """Builds a human-friendly candidate id from 1-based rank."""

        return f"candidate-{index:04d}"

    @staticmethod
    def _select_negative_leads(
        enriched: pd.DataFrame,
        *,
        adj_affinity_threshold: float,
        sas_score_max: float,
        known_hard_to_synthesize: Sequence[str],
        known_weak_binders: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        """Finds new contrastive negative examples for feedback into the prompt.

        Identifies two failure modes that a single-axis score would hide:
        strong binders that are too hard to synthesize, and easily
        synthesizable molecules that bind too weakly.

        Returns:
            Tuple of (new hard-to-synthesize SMILES, new weak-binder SMILES),
            excluding any SMILES already present in the known pools.
        """

        if enriched.empty:
            return [], []

        hard_to_synthesize_mask = (
            enriched["Boltz2_Binding_Probability"] > adj_affinity_threshold
        ) & (enriched["SAS"] > sas_score_max)
        weak_binder_mask = (enriched["SAS"] <= sas_score_max) & (
            enriched["Boltz2_Binding_Probability"] <= adj_affinity_threshold
        )

        candidate_hard_to_synthesize = (
            enriched.loc[hard_to_synthesize_mask]
            .sort_values("Boltz2_Binding_Probability", ascending=False)["SMILES"]
            .tolist()
        )
        candidate_weak_binders = (
            enriched.loc[weak_binder_mask].sort_values("SAS", ascending=True)["SMILES"].tolist()
        )

        new_hard_to_synthesize = [
            smiles
            for smiles in candidate_hard_to_synthesize
            if smiles not in known_hard_to_synthesize
        ]
        new_weak_binders = [
            smiles for smiles in candidate_weak_binders if smiles not in known_weak_binders
        ]
        return new_hard_to_synthesize, new_weak_binders

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
            affinity = float(row.get("Boltz2_Binding_Probability", 0.0))
            smiles = str(row.get("SMILES", ""))
            candidate_id = str(row.get("Candidate_ID", ""))
            if not smiles or not candidate_id:
                continue
            if affinity <= threshold:
                logger.info(
                    "Skipping structure save for %s: Boltz2_Binding_Probability=%.3f <= "
                    "BEST_STRUCTURE_AFFINITY_THRESHOLD=%.3f",
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
                "%s structure(s) skipped: Boltz2_Binding_Probability below "
                "BEST_STRUCTURE_AFFINITY_THRESHOLD=%.3f. "
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
            sas_score_max=sas_score_max,
        )
        if enriched.empty:
            return enriched
        return enriched[enriched["SAS"] <= sas_score_max]

    def _build_manifest(self, options: PipelineOptions) -> tuple[RunManifest, Path]:
        """Creates the provenance manifest for a run and its output path.

        The scoring gates are surfaced as their own `thresholds` block rather
        than being left inside the settings dump, because they are the values
        a reader needs in order to check that the numbers in a published
        table were produced under the thresholds the text claims.
        """

        return (
            build_run_manifest(
                run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                started_at=datetime.now(timezone.utc),
                settings=self._settings,
                thresholds={
                    "adj_affinity_threshold": self._adj_affinity_threshold,
                    "sas_score_max": self._sas_score_max,
                    "sas_score_min": SAS_SCORE_MIN,
                    "iptm_threshold": self._iptm_threshold,
                    "plddt_threshold": self._plddt_threshold,
                    "best_structure_affinity_threshold": self._best_structure_affinity_threshold,
                    "lead_diversity_max_sim": self._lead_diversity_max_sim,
                    "already_proposed_window": self._already_proposed_window,
                    "seed_smiles_limit": self._settings.seed_smiles_limit,
                    "extra_prompt_constraints": self._settings.extra_prompt_constraints,
                    "similarity_penalty_threshold": self._similarity_penalty_threshold,
                    "similarity_penalty_fraction": self._similarity_penalty_fraction,
                },
                inputs={
                    "pdb_path": str(options.pdb_path),
                    "pdb_sha256": file_digest(options.pdb_path),
                    "few_shot_csv": str(options.few_shot_csv),
                    "few_shot_csv_sha256": file_digest(options.few_shot_csv),
                    "output_dir": str(options.output_dir),
                    "target_chain_id": options.target_chain_id,
                    "protein_sequence_length": len(options.protein_sequence),
                    "mapped_residue_count": len(options.residue_position_map),
                    "max_iterations": options.max_iterations,
                    "max_samples": options.max_samples,
                    "use_pocket_data": options.use_pocket_data,
                    "score_references": options.score_references,
                    "llm_model": self._llm_model,
                    "llm_temperature": self._llm_temperature,
                    "random_seed": self._random_seed,
                },
            ),
            options.output_dir / RUN_MANIFEST_FILENAME,
        )

    @staticmethod
    def _resolve_pocket(options: PipelineOptions) -> tuple[str, str, dict[str, float] | None]:
        """Resolves the binding pocket from whichever source the user chose.

        Every mode returns the same triple, so the chain filtering and
        sequence-position translation that follow are identical whether the
        site was detected from geometry or supplied by the user.
        """

        mode = options.pocket_mode
        if mode == "residues" and options.pocket_residues:
            return get_pocket_from_residues(
                options.pdb_path,
                options.pocket_residues,
                default_chain=options.target_chain_id,
            )
        if mode == "ligand" and options.pocket_ligand:
            return get_pocket_from_ligand(
                options.pdb_path,
                options.pocket_ligand,
                contact_distance=options.pocket_contact_distance,
            )
        if mode == "center" and options.pocket_center is not None:
            return get_pocket_from_center(
                options.pdb_path,
                options.pocket_center,
                contact_distance=options.pocket_contact_distance,
            )
        return get_binding_pockets_and_residues(
            options.pdb_path,
            options.output_dir,
            backend="deepchem",
            pocket_index=-1 if options.pocket_index is None else options.pocket_index,
        )

    async def _score_reference_molecules(
        self,
        reference_smiles: Sequence[str],
        reference_names: Sequence[str],
        reference_fps: Sequence[Any],
        options: PipelineOptions,
        residue_contacts: Sequence[PocketContact] | None,
    ) -> pd.DataFrame:
        """Scores the few-shot reference inhibitors on the candidate scoring path.

        Published candidate tables are only interpretable against a baseline
        computed the same way, so the references go through the same Boltz
        call, the same pocket constraints, and the same composite score as
        generated molecules.

        Reference rows are returned to the caller for their own report and
        are never merged into the candidate registry, the similarity-penalty
        pool, or the prompt feedback pools -- a reference leaking into those
        would both bias generation toward the seeds and make a known
        inhibitor indistinguishable from a generated candidate in the final
        ranking.

        Returns:
            One validated `ReferenceBaselineRow` per reference that Boltz
            scored successfully, or an empty frame when none could be scored.
        """

        results_df, _ = await self._boltz_client.compute_properties(
            list(reference_smiles),
            options.protein_sequence,
            pocket_residues=residue_contacts,
        )
        if results_df.empty:
            logger.warning(no_reference_scores_returned())
            return pd.DataFrame()

        # `strict=True`: the three sequences are appended in lockstep when the
        # seed file is read, so a length mismatch means names or fingerprints
        # have silently desynchronized from their molecules.
        fp_by_smiles = dict(zip(reference_smiles, reference_fps, strict=True))
        name_by_smiles = dict(zip(reference_smiles, reference_names, strict=True))

        rows: list[dict[str, Any]] = []
        for row in results_df.to_dict("records"):
            smiles = str(row["SMILES"])
            # Leave-one-out: a reference compared against the full seed set
            # would always find itself at similarity 1.0, which is not the
            # quantity generated candidates are measured on.
            own_fp = fp_by_smiles.get(smiles)
            other_fps = [fp for other, fp in fp_by_smiles.items() if other != smiles]
            if own_fp is not None and other_fps:
                max_sim = max(DataStructs.BulkTanimotoSimilarity(own_fp, other_fps))
            else:
                max_sim = 0.0

            affinity = float(row["Boltz2_Binding_Probability"])
            sas = float(row["SAS"])
            adj_binding_probability = affinity if affinity > self._adj_affinity_threshold else 0.0
            synth_factor = max(
                0.0,
                min(1.0, (self._sas_score_max - sas) / (self._sas_score_max - SAS_SCORE_MIN)),
            )
            score = calculate_heuristic_score(
                pd.Series(
                    {
                        "Adj_Binding_Probability": adj_binding_probability,
                        "MaxSim": max_sim,
                        "synth_factor": synth_factor,
                    }
                ),
                similarity_penalty_threshold=self._similarity_penalty_threshold,
                similarity_penalty_fraction=self._similarity_penalty_fraction,
            )

            passes_affinity = affinity > self._adj_affinity_threshold
            passes_sas = sas <= self._sas_score_max
            passes_iptm = float(row["Boltz2_ipTM"]) >= self._iptm_threshold
            passes_plddt = float(row["Boltz2_Complex_pLDDT"]) >= self._plddt_threshold

            rows.append(
                ReferenceBaselineRow.model_validate(
                    {
                        **row,
                        "Reference_Name": name_by_smiles.get(smiles, smiles),
                        "MaxSim_To_Other_References": max_sim,
                        "Adj_Binding_Probability": adj_binding_probability,
                        "synth_factor": synth_factor,
                        "score": score,
                        "Passes_Affinity_Gate": passes_affinity,
                        "Passes_SAS_Gate": passes_sas,
                        "Passes_ipTM_Gate": passes_iptm,
                        "Passes_pLDDT_Gate": passes_plddt,
                        "Passes_All_Gates": (
                            passes_affinity and passes_sas and passes_iptm and passes_plddt
                        ),
                    }
                ).model_dump()
            )

        return pd.DataFrame.from_records(rows)

    def _run_holdout_evaluation(
        self,
        final_hits: pd.DataFrame,
        best_structures_by_smiles: dict[str, BestStructureRecord],
        options: PipelineOptions,
        pocket_metadata: dict[str, Any] | None,
        manifest: RunManifest,
    ) -> pd.DataFrame:
        """Evaluates finished candidates with predictors that did not select them.

        This runs only after the generation loop has exited and the candidate
        set is final. That ordering is the independence guarantee: by the time
        any holdout number exists, every prompt has been sent, every lead pool
        is closed, and the similarity registry is frozen, so no holdout score
        can have shaped the molecules it is judging.

        Evaluators that are unavailable are recorded in the manifest with the
        reason, rather than being silently skipped -- a reader needs to know
        which independent checks actually ran before reading the columns.

        Returns:
            Holdout columns aligned to `final_hits`, or an empty frame when no
            evaluator produced anything.
        """

        if not self._settings.enable_holdout_evaluation:
            manifest.holdout = {"enabled": False}
            logger.info("Holdout evaluation disabled via ENABLE_HOLDOUT_EVALUATION")
            return pd.DataFrame()

        candidates = [
            HoldoutCandidate(
                smiles=smiles,
                candidate_id=f"candidate-{position:04d}",
                structure=(
                    structure.structure
                    if (structure := best_structures_by_smiles.get(smiles)) is not None
                    else None
                ),
            )
            for position, smiles in enumerate(final_hits["SMILES"], start=1)
        ]
        with_structures = sum(1 for candidate in candidates if candidate.structure)
        context = HoldoutContext(
            protein_pdb_path=Path(options.pdb_path),
            docking_box=(pocket_metadata or {}).get("docking_box"),
        )

        summary: dict[str, Any] = {
            "enabled": True,
            "candidates": len(candidates),
            "candidates_with_structures": with_structures,
            "evaluators": {},
        }
        merged: dict[str, dict[str, Any]] = {}

        for evaluator in self._holdout_evaluators:
            availability = evaluator.availability()
            record: dict[str, Any] = {
                "available": availability.available,
                "reason": availability.reason,
                "evaluated": 0,
            }
            if availability.available:
                started_at = perf_counter()
                try:
                    produced = evaluator.evaluate(candidates, context)
                except Exception as exc:  # noqa: BLE001 - third-party failure modes vary
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    logger.warning(holdout_evaluator_failed(evaluator.name, exc), exc_info=True)
                    produced = {}
                record["evaluated"] = len(produced)
                record["seconds"] = round(perf_counter() - started_at, 3)
                for smiles, columns in produced.items():
                    merged.setdefault(smiles, {}).update(columns)
                logger.info(
                    "Holdout evaluator '%s' scored %s/%s candidate(s) in %.2fs",
                    evaluator.name,
                    len(produced),
                    len(candidates),
                    record["seconds"],
                )
            else:
                logger.warning(holdout_evaluator_unavailable(evaluator.name, availability.reason))
            summary["evaluators"][evaluator.name] = record

        if not any(entry["evaluated"] for entry in summary["evaluators"].values()):
            logger.warning(
                "No independent evaluator scored any candidate. Every number in this "
                "report comes from the same predictor that selected the molecules."
            )

        manifest.holdout = summary
        if not merged:
            return pd.DataFrame()

        return pd.DataFrame(
            [merged.get(smiles, {}) for smiles in final_hits["SMILES"]],
            index=final_hits.index,
        )

    async def run(self, options: PipelineOptions) -> str | None:
        """Executes one pipeline run and persists its provenance manifest.

        The manifest is written before any scoring starts and again when the
        run ends, so an interrupted or failed run still leaves behind the
        configuration it was using.

        Args:
            options: Validated runtime parameters for one run.

        Returns:
            Path to generated report when successful, otherwise `None`.
        """

        run_started_at = perf_counter()
        manifest, manifest_path = self._build_manifest(options)
        manifest.write(manifest_path)
        logger.info("Run manifest initialized at %s", manifest_path)

        try:
            return await self._run_pipeline(options, manifest)
        except Exception:
            manifest.status = "failed"
            raise
        finally:
            # Recorded here rather than at build time because the endpoint
            # only discloses anything once it has answered a request. Guarded
            # on type so a stand-in client cannot put an unserializable object
            # into the manifest.
            provenance = getattr(self._boltz_client, "provenance", None)
            if isinstance(provenance, dict):
                manifest.code["boltz_endpoint"] = provenance
            manifest.finished_at = datetime.now(timezone.utc).isoformat()
            manifest.elapsed_seconds = perf_counter() - run_started_at
            manifest.write(manifest_path)
            logger.info("Run manifest finalized at %s (status=%s)", manifest_path, manifest.status)

    async def _run_pipeline(self, options: PipelineOptions, manifest: RunManifest) -> str | None:
        """Executes iterative candidate generation and writes the final CSV.

        Args:
            options: Validated runtime parameters for one run.
            manifest: Provenance record updated in place as the run proceeds.

        Returns:
            Path to generated report when successful, otherwise `None`.
        """

        total_steps = 5
        if self._random_seed is not None:
            random.seed(self._random_seed)
            np.random.seed(self._random_seed)
            logger.info("Run seeded with RANDOM_SEED=%s", self._random_seed)
        else:
            logger.info(
                "No RANDOM_SEED configured; run is not seeded and may not be exactly reproducible"
            )
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
            seed_frame = few_shot_data.dropna(subset=["Smiles"]).head(
                self._settings.seed_smiles_limit
            )
            # Seed files carry a human-readable name only when they were built
            # from known inhibitors; generated-seed files have SMILES alone.
            name_column = next(
                (column for column in ("Common_Name", "Name") if column in seed_frame.columns),
                None,
            )
            # Canonicalize seeds the same way every LLM-generated SMILES is
            # canonicalized (via SmilesString validation), so a seed and a
            # differently-written duplicate proposed later are recognized as
            # the same molecule for similarity/dedup purposes.
            positives: list[str] = []
            positive_mols: list[Any] = []
            positive_names: list[str] = []
            for position, (_, seed_row) in enumerate(seed_frame.iterrows(), start=1):
                mol = Chem.MolFromSmiles(str(seed_row["Smiles"]).strip())
                if mol is None:
                    continue
                positives.append(Chem.MolToSmiles(mol))
                positive_mols.append(mol)
                raw_name = str(seed_row[name_column]).strip() if name_column else ""
                positive_names.append(
                    raw_name
                    if raw_name and raw_name.lower() != "nan"
                    else f"reference-{position:02d}"
                )

            fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=MORGAN_FINGERPRINT_RADIUS
            )
            target_fps = [fingerprint_generator.GetFingerprint(mol) for mol in positive_mols]
            step_progress.update(1)
            step_progress.set_postfix_str("few-shot ready")
            manifest.inputs["seed_molecule_count"] = len(positives)
            manifest.inputs["seed_names"] = list(positive_names)
            logger.info(
                "Step 1/%s complete in %.2fs: loaded %s seed molecules and %s fingerprints",
                total_steps,
                perf_counter() - step_started_at,
                len(positives),
                len(target_fps),
            )

            context_leads = positives.copy()
            lead_fingerprints = list(target_fps)
            # Similarity-penalty pool: starts as the seed molecules and grows with
            # every molecule that clears the scoring gates in any iteration, so
            # later candidates are penalized for resembling anything registered so
            # far -- not just the fixed seed set.
            registry_fps: list[Any] = list(target_fps)
            registry_smiles_seen: set[str] = set(positives)
            anchor_leads = positives[:2]
            negative_leads_hard_to_synthesize: list[str] = []
            negative_leads_weak_binders: list[str] = []

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
                    pocket_coords, pocket_residues, pocket_box_dims = self._resolve_pocket(options)
                except (PocketDetectionError, subprocess.SubprocessError, TypeError) as exc:
                    # Automatic detection may fail for reasons the user cannot
                    # act on, so a run falls back to few-shot mode. A pocket
                    # the user named is different: continuing without it would
                    # spend the whole API budget on a site they did not ask
                    # for, while the report still looked normal.
                    if options.pocket_mode not in {"auto", "index"}:
                        raise PocketDetectionError(
                            user_pocket_unavailable(options.pocket_mode, exc)
                        ) from exc
                    logger.warning(pocket_detection_unavailable(exc))
                    use_pocket_data = False
                else:
                    detected_contacts = self._extract_residue_contacts(pocket_residues)
                    on_target_chain = [
                        contact
                        for contact in detected_contacts
                        if contact.chain_id == options.target_chain_id
                    ]
                    dropped_off_target = len(detected_contacts) - len(on_target_chain)
                    if dropped_off_target:
                        logger.warning(
                            "Dropped %s pocket residue contact(s) outside target chain "
                            "'%s' (only that chain's numbering matches the sequence "
                            "submitted to Boltz)",
                            dropped_off_target,
                            options.target_chain_id,
                        )

                    # Detected contacts carry PDB residue numbers (e.g. 203-498);
                    # Boltz expects 1-based positions within the submitted
                    # sequence (e.g. 1-294). Translate through the position map
                    # built from the same residues used to construct that
                    # sequence, and drop anything with no mapped position
                    # (missing/disordered residues) rather than ever sending a
                    # raw PDB number as if it were a sequence position.
                    residue_contacts = []
                    dropped_unmapped = 0
                    for contact in on_target_chain:
                        sequence_index = options.residue_position_map.get(contact.residue_index)
                        if sequence_index is None:
                            dropped_unmapped += 1
                            continue
                        residue_contacts.append(
                            PocketContact(chain_id=contact.chain_id, residue_index=sequence_index)
                        )
                    if dropped_unmapped:
                        logger.warning(
                            "Dropped %s pocket residue contact(s) with no position in the "
                            "submitted sequence (PDB residue number falls in a gap/"
                            "disordered region, or no residue_position_map was provided)",
                            dropped_unmapped,
                        )

                    selected_pocket_metadata = {
                        "coordinates": pocket_coords,
                        "residues": pocket_residues,
                        "target_chain_id": options.target_chain_id,
                        "residue_contacts": [contact.model_dump() for contact in residue_contacts],
                    }
                    if pocket_box_dims is not None:
                        selected_pocket_metadata["docking_box"] = pocket_box_dims
                    logger.info(
                        "Using pocket constraints from %s with %s residue contacts",
                        pocket_coords,
                        len(residue_contacts),
                    )

            if not use_pocket_data:
                logger.info("Running few-shot mode without pocket constraints")

            manifest.pocket = {
                # Distinguishes a pocket found by geometry from one the user
                # supplied, and both from a run that fell back to no pocket at
                # all. These support very different claims about how the
                # binding site was identified.
                "mode": options.pocket_mode
                if use_pocket_data
                else ("detection_failed" if options.use_pocket_data else "disabled"),
                "user_specified": use_pocket_data and options.pocket_mode != "auto",
                "selection": {
                    "residues": options.pocket_residues,
                    "ligand": options.pocket_ligand,
                    "center": list(options.pocket_center) if options.pocket_center else None,
                    "index": options.pocket_index,
                    "contact_distance": options.pocket_contact_distance,
                },
                "backend": "deepchem"
                if use_pocket_data and options.pocket_mode in {"auto", "index"}
                else None,
                "residue_contact_count": len(residue_contacts),
                "selected_pocket": selected_pocket_metadata,
            }

            # Optional baseline stage. Run here, before generation, so the
            # references are scored against exactly the pocket constraints the
            # candidates will face, and so the baseline survives even if the
            # generation loop later fails.
            reference_baseline = pd.DataFrame()
            reference_seconds = 0.0
            if options.score_references and positives:
                reference_started_at = perf_counter()
                logger.info(
                    "Scoring %s reference inhibitor(s) as a baseline: %s",
                    len(positives),
                    ", ".join(positive_names),
                )
                try:
                    reference_baseline = await self._score_reference_molecules(
                        positives,
                        positive_names,
                        target_fps,
                        options,
                        residue_contacts if use_pocket_data else None,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(reference_scoring_failed(exc), exc_info=True)
                reference_seconds = perf_counter() - reference_started_at
                if not reference_baseline.empty:
                    passing = int(reference_baseline["Passes_All_Gates"].sum())
                    # Written as soon as it exists rather than alongside the
                    # candidate report, so the baseline survives a failure in
                    # the generation loop that follows.
                    options.output_dir.mkdir(parents=True, exist_ok=True)
                    reference_path = options.output_dir / REFERENCE_BASELINE_FILENAME
                    reference_baseline.to_csv(reference_path, index=False)
                    logger.info(
                        "Reference baseline scored in %.2fs: %s/%s reference(s) returned, "
                        "%s would pass every candidate gate; written to %s",
                        reference_seconds,
                        len(reference_baseline),
                        len(positives),
                        passing,
                        reference_path,
                    )
                    if passing < len(reference_baseline):
                        logger.warning(
                            "%s/%s reference inhibitor(s) would be rejected by this run's "
                            "own thresholds -- review whether the gates are set too "
                            "aggressively before interpreting candidate rankings",
                            len(reference_baseline) - passing,
                            len(reference_baseline),
                        )
                    manifest.reference_baseline = {
                        "requested": len(positives),
                        "scored": len(reference_baseline),
                        "passing_all_gates": passing,
                        "report": REFERENCE_BASELINE_FILENAME,
                        "seconds": reference_seconds,
                    }
                else:
                    manifest.reference_baseline = {
                        "requested": len(positives),
                        "scored": 0,
                        "passing_all_gates": 0,
                        "report": None,
                        "seconds": reference_seconds,
                    }
            elif options.score_references:
                logger.warning(
                    "Reference scoring requested but no valid seed molecules were loaded"
                )

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
                "Scoring thresholds: affinity>%.3f SAS<=%.3f ipTM>=%.3f pLDDT>=%.3f",
                self._adj_affinity_threshold,
                self._sas_score_max,
                self._iptm_threshold,
                self._plddt_threshold,
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
            scored_rows_by_smiles: dict[str, dict[str, Any]] = {}
            llm_time_total = 0.0
            boltz_time_total = 0.0
            with make_progress_bar(
                total=options.max_iterations,
                desc="Iterations",
                unit="iter",
            ) as iteration_progress:
                for iteration in range(1, options.max_iterations + 1):
                    iteration_started_at = perf_counter()
                    iteration_record = IterationRecord(iteration=iteration, status="succeeded")
                    logger.info("Iteration %s/%s started", iteration, options.max_iterations)
                    recent_leads = context_leads[-CONTEXT_LEADS_WINDOW:]
                    combined_leads = list(dict.fromkeys(anchor_leads + recent_leads))
                    prompt_leads = combined_leads[:CONTEXT_LEADS_WINDOW]
                    leads_text = ", ".join(prompt_leads)
                    prompt_hard_to_synthesize = negative_leads_hard_to_synthesize[
                        -NEGATIVE_LEADS_WINDOW:
                    ]
                    prompt_weak_binders = negative_leads_weak_binders[-NEGATIVE_LEADS_WINDOW:]
                    avoid_hard_to_synthesize = ", ".join(prompt_hard_to_synthesize)
                    avoid_weak_binders = ", ".join(prompt_weak_binders)
                    prompt_already_proposed = list(scored_rows_by_smiles.keys())[
                        -self._already_proposed_window :
                    ]
                    already_proposed_text = ", ".join(prompt_already_proposed)
                    logger.info(
                        "Iteration %s/%s prompt context: %s lead(s), %s "
                        "hard-to-synthesize, %s weak-binder negative example(s), "
                        "%s already-proposed molecule(s)",
                        iteration,
                        options.max_iterations,
                        len(prompt_leads),
                        len(prompt_hard_to_synthesize),
                        len(prompt_weak_binders),
                        len(prompt_already_proposed),
                    )
                    user_content = build_user_prompt(
                        use_pocket_data,
                        pocket_residues,
                        leads_text,
                        options.max_samples,
                        avoid_hard_to_synthesize,
                        avoid_weak_binders,
                        already_proposed_text,
                        extra_constraints=self._settings.extra_prompt_constraints,
                    )

                    try:
                        llm_started_at = perf_counter()
                        completion_kwargs: dict[str, Any] = {
                            "model": self._llm_model,
                            "messages": [
                                {"role": "system", "content": MODEL_SYSTEM_PROMPT},
                                {"role": "user", "content": user_content},
                            ],
                            "temperature": self._llm_temperature,
                            # Without this the provider assumes the model's
                            # full context length as the expected output and
                            # rejects the request against its per-minute
                            # output-token budget before running anything.
                            "max_tokens": self._settings.llm_max_tokens,
                        }
                        if self._random_seed is not None:
                            completion_kwargs["seed"] = self._random_seed
                        completion = self._groq_client.chat.completions.create(**completion_kwargs)
                        llm_elapsed = perf_counter() - llm_started_at
                        llm_time_total += llm_elapsed

                        iteration_record.llm_seconds = llm_elapsed

                        raw_content = completion.choices[0].message.content or ""
                        new_smiles, invalid_at_parse_count = parse_smiles_from_text(raw_content)
                        iteration_record.proposed = len(new_smiles)
                        logger.info(
                            "Iteration %s/%s: LLM generation took %.2fs, produced %s "
                            "candidate SMILES (%s unparseable proposal(s) discarded before "
                            "filtering)",
                            iteration,
                            options.max_iterations,
                            llm_elapsed,
                            len(new_smiles),
                            invalid_at_parse_count,
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

                        iteration_record.valid_after_filters = len(valid_smiles)
                        logger.info(
                            "Iteration %s/%s retained %s/%s candidates after filtering",
                            iteration,
                            options.max_iterations,
                            len(valid_smiles),
                            len(new_smiles),
                        )

                        already_scored_smiles = [
                            smiles for smiles in valid_smiles if smiles in scored_rows_by_smiles
                        ]
                        new_to_score = [
                            smiles for smiles in valid_smiles if smiles not in scored_rows_by_smiles
                        ]
                        if already_scored_smiles:
                            logger.info(
                                "Iteration %s/%s reusing %s cached Boltz score(s), "
                                "submitting %s new candidate(s)",
                                iteration,
                                options.max_iterations,
                                len(already_scored_smiles),
                                len(new_to_score),
                            )

                        boltz_started_at = perf_counter()
                        results_df, best_structures = await self._boltz_client.compute_properties(
                            new_to_score,
                            options.protein_sequence,
                            pocket_residues=residue_contacts if use_pocket_data else None,
                        )
                        boltz_elapsed = perf_counter() - boltz_started_at
                        boltz_time_total += boltz_elapsed
                        logger.info(
                            "Iteration %s/%s: Boltz scoring took %.2fs for %s new "
                            "candidate(s) (%s reused from cache)",
                            iteration,
                            options.max_iterations,
                            boltz_elapsed,
                            len(new_to_score),
                            len(already_scored_smiles),
                        )

                        for row in results_df.to_dict("records"):
                            scored_rows_by_smiles[row["SMILES"]] = row

                        if already_scored_smiles:
                            cached_rows = [
                                scored_rows_by_smiles[smiles] for smiles in already_scored_smiles
                            ]
                            results_df = pd.concat(
                                [results_df, pd.DataFrame(cached_rows)], ignore_index=True
                            )

                        for item in best_structures:
                            existing = best_structures_by_smiles.get(item.smiles)
                            if existing is None or item.affinity_prob > existing.affinity_prob:
                                best_structures_by_smiles[item.smiles] = item

                        iteration_record.boltz_seconds = boltz_elapsed
                        iteration_record.boltz_scored = len(results_df)

                        enriched = self._enrich_scores(
                            results_df,
                            registry_fps,
                            adj_affinity_threshold=self._adj_affinity_threshold,
                            sas_score_max=self._sas_score_max,
                            similarity_penalty_threshold=self._similarity_penalty_threshold,
                            similarity_penalty_fraction=self._similarity_penalty_fraction,
                        )
                        rejected_sas_count = (
                            int((enriched["SAS"] > self._sas_score_max).sum())
                            if not enriched.empty
                            else 0
                        )
                        low_affinity_count = (
                            int(
                                (
                                    enriched["Boltz2_Binding_Probability"]
                                    <= self._adj_affinity_threshold
                                ).sum()
                            )
                            if not enriched.empty
                            else 0
                        )
                        low_iptm_count = (
                            int((enriched["Boltz2_ipTM"] < self._iptm_threshold).sum())
                            if not enriched.empty
                            else 0
                        )
                        low_plddt_count = (
                            int((enriched["Boltz2_Complex_pLDDT"] < self._plddt_threshold).sum())
                            if not enriched.empty
                            else 0
                        )
                        scored = (
                            enriched[
                                (enriched["SAS"] <= self._sas_score_max)
                                & (enriched["Boltz2_ipTM"] >= self._iptm_threshold)
                                & (enriched["Boltz2_Complex_pLDDT"] >= self._plddt_threshold)
                            ]
                            if not enriched.empty
                            else enriched
                        )

                        iteration_record.kept = len(scored)

                        new_hard_to_synthesize, new_weak_binders = self._select_negative_leads(
                            enriched,
                            adj_affinity_threshold=self._adj_affinity_threshold,
                            sas_score_max=self._sas_score_max,
                            known_hard_to_synthesize=negative_leads_hard_to_synthesize,
                            known_weak_binders=negative_leads_weak_binders,
                        )
                        negative_leads_hard_to_synthesize.extend(new_hard_to_synthesize)
                        negative_leads_weak_binders.extend(new_weak_binders)
                        if new_hard_to_synthesize or new_weak_binders:
                            logger.info(
                                "Iteration %s/%s flagged %s hard-to-synthesize and %s "
                                "weak-binder negative example(s) for feedback",
                                iteration,
                                options.max_iterations,
                                len(new_hard_to_synthesize),
                                len(new_weak_binders),
                            )

                        logger.info(
                            "Iteration %s/%s complete in %.2fs (llm=%.2fs, boltz=%.2fs): "
                            "parsed=%s valid=%s boltz_scored=%s kept=%s invalid=%s "
                            "catalog_filtered=%s lipinski_filtered=%s boltz_failed=%s "
                            "rejected_sas=%s low_affinity=%s low_iptm=%s low_plddt=%s",
                            iteration,
                            options.max_iterations,
                            perf_counter() - iteration_started_at,
                            llm_elapsed,
                            boltz_elapsed,
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
                            low_iptm_count,
                            low_plddt_count,
                        )

                        if not scored.empty:
                            scored_records = [
                                ScoredMoleculeRecord.model_validate(row)
                                for row in scored.to_dict("records")
                            ]
                            global_registry.extend(scored_records)

                            # Grow the similarity-penalty pool with this iteration's newly
                            # registered molecules so the *next* iteration's penalty check
                            # sees everything scored so far, not just the original seeds.
                            new_registry_entries = 0
                            for record in scored_records:
                                if record.SMILES in registry_smiles_seen:
                                    continue
                                registry_smiles_seen.add(record.SMILES)
                                registry_mol = Chem.MolFromSmiles(record.SMILES)
                                if registry_mol is None:
                                    continue
                                registry_fps.append(
                                    fingerprint_generator.GetFingerprint(registry_mol)
                                )
                                new_registry_entries += 1
                            if new_registry_entries:
                                logger.info(
                                    "Iteration %s/%s added %s new molecule(s) to the "
                                    "similarity-penalty registry (%s total)",
                                    iteration,
                                    options.max_iterations,
                                    new_registry_entries,
                                    len(registry_fps),
                                )

                            # Rank leads from the cumulative registry (all iterations so
                            # far), not just this iteration's batch, so a strong candidate
                            # found early keeps steering later prompts even after weaker
                            # molecules are scored in between.
                            best_score_by_smiles: dict[str, float] = {}
                            for record in global_registry:
                                existing_score = best_score_by_smiles.get(record.SMILES)
                                if existing_score is None or record.score > existing_score:
                                    best_score_by_smiles[record.SMILES] = record.score
                            top_leads = [
                                smiles
                                for smiles, _ in sorted(
                                    best_score_by_smiles.items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            ][: CONTEXT_LEADS_WINDOW * 3]
                            new_leads: list[str] = []
                            for candidate_lead in top_leads:
                                if len(new_leads) >= CONTEXT_LEADS_WINDOW:
                                    break
                                if candidate_lead in context_leads:
                                    continue
                                candidate_mol = Chem.MolFromSmiles(candidate_lead)
                                if candidate_mol is None:
                                    continue
                                candidate_fp = fingerprint_generator.GetFingerprint(candidate_mol)
                                if lead_fingerprints:
                                    max_sim_to_pool = max(
                                        DataStructs.TanimotoSimilarity(candidate_fp, fp)
                                        for fp in lead_fingerprints
                                    )
                                    if max_sim_to_pool > self._lead_diversity_max_sim:
                                        continue
                                new_leads.append(candidate_lead)
                                lead_fingerprints.append(candidate_fp)
                            if new_leads:
                                context_leads.extend(new_leads)
                                logger.info(
                                    "Iteration %s/%s fed %s new, structurally-distinct lead(s) "
                                    "back into context",
                                    iteration,
                                    options.max_iterations,
                                    len(new_leads),
                                )
                            else:
                                logger.info(
                                    "Iteration %s/%s: no sufficiently novel leads to feed back "
                                    "(all candidates too similar to existing pool)",
                                    iteration,
                                    options.max_iterations,
                                )
                    except Exception as exc:  # noqa: BLE001
                        # A failed iteration is recorded rather than only
                        # logged: a run that reports N iterations but silently
                        # completed fewer would overstate both its cost and
                        # the optimization its final candidates received.
                        iteration_record.status = "failed"
                        iteration_record.error = f"{type(exc).__name__}: {exc}"
                        logger.error("Error in iteration %s: %s", iteration, exc, exc_info=True)
                    finally:
                        manifest.iterations.append(iteration_record)
                        iteration_progress.update(1)
                        iteration_progress.set_postfix_str(f"iter={iteration}")

            step_progress.update(1)
            step_progress.set_postfix_str("generation complete")
            logger.info(
                "Step 3/%s complete in %.2fs (llm_total=%.2fs, boltz_total=%.2fs): "
                "collected %s scored candidates",
                total_steps,
                perf_counter() - step_started_at,
                llm_time_total,
                boltz_time_total,
                len(global_registry),
            )

            if not global_registry:
                manifest.status = "no_candidates"
                manifest.timings = {
                    "llm_generation_seconds": llm_time_total,
                    "boltz_scoring_seconds": boltz_time_total,
                    "reference_scoring_seconds": reference_seconds,
                }
                logger.warning(no_molecules_generated())
                return None

            final_hits = pd.DataFrame.from_records(
                [record.model_dump() for record in global_registry]
            ).drop_duplicates(subset="SMILES")
            logger.info(
                "Deduplicated scored candidates down to %s unique molecules",
                len(final_hits),
            )

            # Scaffold analysis is best-effort: it runs after every API call
            # has been paid for, so a parsing failure here must leave the
            # report intact rather than discard a completed run.
            try:
                scaffold_analyzer = ScaffoldAnalyzer(positives)
                scaffold_columns = pd.DataFrame(
                    [scaffold_analyzer.annotate(smiles) for smiles in final_hits["SMILES"]],
                    index=final_hits.index,
                )
                final_hits = pd.concat([final_hits, scaffold_columns], axis=1)
                manifest.diversity = scaffold_analyzer.summarize(list(final_hits["SMILES"]))
                logger.info(
                    "Scaffold analysis: %s unique Murcko scaffold(s) across %s molecule(s) "
                    "(diversity=%s, novel_scaffold_fraction=%s, internal_diversity=%s)",
                    manifest.diversity.get("unique_murcko_scaffolds"),
                    manifest.diversity.get("molecules"),
                    manifest.diversity.get("scaffold_diversity"),
                    manifest.diversity.get("novel_scaffold_fraction"),
                    manifest.diversity.get("internal_diversity"),
                )
                scaffold_diversity = manifest.diversity.get("scaffold_diversity")
                if scaffold_diversity is not None and scaffold_diversity < 0.5:
                    logger.warning(
                        "Scaffold diversity is %.2f: over half the candidates share a core "
                        "with another candidate. Consider lowering "
                        "SIMILARITY_PENALTY_THRESHOLD or LEAD_DIVERSITY_MAX_SIM.",
                        scaffold_diversity,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(scaffold_analysis_failed(exc), exc_info=True)

            holdout_columns = self._run_holdout_evaluation(
                final_hits,
                best_structures_by_smiles,
                options,
                selected_pocket_metadata,
                manifest,
            )
            if not holdout_columns.empty:
                final_hits = pd.concat([final_hits, holdout_columns], axis=1)

            step_started_at = perf_counter()
            logger.info(
                "Step 4/%s: verifying IP status via PubChem for %s unique candidates",
                total_steps,
                len(final_hits),
            )
            patent_checks = await gather_with_progress(
                [self._pubchem_service.check_patents(smiles) for smiles in final_hits["SMILES"]],
                desc="Patent checks",
                unit="mol",
            )
            pubchem_time_total = perf_counter() - step_started_at
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
                "Step 4/%s complete in %.2fs (pubchem=%.2fs): ranked %s final candidates",
                total_steps,
                perf_counter() - step_started_at,
                pubchem_time_total,
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
            manifest.status = "completed"
            manifest.results = {
                "report": UNIFIED_REPORT_FILENAME,
                "scored_records": len(global_registry),
                "unique_candidates": len(final_hits),
                "structures_saved": saved_structures,
                "iterations_succeeded": sum(
                    1 for record in manifest.iterations if record.status == "succeeded"
                ),
                "iterations_failed": sum(
                    1 for record in manifest.iterations if record.status == "failed"
                ),
            }
            manifest.timings = {
                "llm_generation_seconds": llm_time_total,
                "boltz_scoring_seconds": boltz_time_total,
                "reference_scoring_seconds": reference_seconds,
                "pubchem_seconds": pubchem_time_total,
            }
            logger.info(
                "Timing breakdown: llm_generation=%.2fs boltz_scoring=%.2fs "
                "reference_scoring=%.2fs pubchem=%.2fs",
                llm_time_total,
                boltz_time_total,
                reference_seconds,
                pubchem_time_total,
            )
            logger.info("Workflow Complete. Results in %s", report_path)
            return str(report_path)

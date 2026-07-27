"""NVIDIA Boltz client for async molecular property prediction."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import perf_counter

import httpx
import pandas as pd
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt

from src.clients import get_cached_http_client
from src.core.config import GeneratorSettings, get_generator_settings
from src.core.logging import get_logger
from src.core.messages import (
    boltz_structure_missing,
    boltz_task_timeout,
    invalid_boltz_response,
)
from src.core.progress import make_progress_bar
from src.sa_score import sa_score_mol
from src.schemas import (
    BestStructureRecord,
    BoltzA3M,
    BoltzContact,
    BoltzLigandAffinity,
    BoltzLigandRequest,
    BoltzMsa,
    BoltzMsaUniRef90,
    BoltzPocketConstraint,
    BoltzPolymer,
    BoltzPrediction,
    BoltzRequest,
    MoleculeRecord,
    PocketContact,
)

logger = get_logger(__name__)


class _BoltzRateLimitError(Exception):
    """Raised when Boltz responds with HTTP 429."""

    def __init__(self, response: httpx.Response, operation: str) -> None:
        self.response = response
        self.operation = operation
        self.retry_after_seconds = _parse_retry_after(response.headers.get("Retry-After"))
        super().__init__(f"Boltz rate limited during {operation}")


def _parse_retry_after(value: str | None) -> float | None:
    """Parses `Retry-After` seconds or HTTP-date header values."""

    if not value:
        return None

    try:
        return max(float(value), 0.0)
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)




class BoltzResult:
    """Validated Boltz prediction plus optional docked structure payload."""

    def __init__(self, prediction: BoltzPrediction, best_structure: BestStructureRecord | None = None) -> None:
        self.prediction = prediction
        self.best_structure = best_structure

class BoltzClient:
    """Async wrapper around NVIDIA Boltz prediction API."""

    def __init__(
        self,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        settings: GeneratorSettings | None = None,
    ) -> None:
        """Initializes the client.

        Args:
            api_key: NVIDIA API key.
            http_client: Optional preconfigured async client.
        """

        self.api_key = api_key
        self.public_url = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
        self.status_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{task_id}"
        self._settings = settings or get_generator_settings()
        self._client = http_client or get_cached_http_client()

    def _get_retry_delay(self, retry_state: RetryCallState) -> float:
        """Computes retry delay for Boltz 429 responses."""

        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        min_wait = self._settings.boltz_retry_min_wait_seconds
        max_wait = self._settings.boltz_retry_max_wait_seconds
        if max_wait < min_wait:
            max_wait = min_wait

        if isinstance(exc, _BoltzRateLimitError) and exc.retry_after_seconds is not None:
            return min(max(exc.retry_after_seconds, min_wait), max_wait)

        delay = min_wait * (2 ** (retry_state.attempt_number - 1))
        return min(delay, max_wait)

    def _log_rate_limit_retry(self, retry_state: RetryCallState) -> None:
        """Logs upcoming retry attempts after Boltz 429 responses."""

        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        if not isinstance(exc, _BoltzRateLimitError):
            return

        sleep_for = retry_state.next_action.sleep if retry_state.next_action else 0.0
        logger.warning(
            "Boltz returned HTTP 429 during %s; retrying in %.1fs (attempt %s/%s)",
            exc.operation,
            sleep_for,
            retry_state.attempt_number + 1,
            self._settings.boltz_retry_attempts,
        )

    async def _request_with_retry(
        self,
        request: Callable[[], Awaitable[httpx.Response]],
        operation: str,
    ) -> httpx.Response | None:
        """Retries Boltz requests on HTTP 429 using the configured backoff policy."""

        retryer = AsyncRetrying(
            retry=retry_if_exception_type(_BoltzRateLimitError),
            wait=self._get_retry_delay,
            stop=stop_after_attempt(self._settings.boltz_retry_attempts),
            before_sleep=self._log_rate_limit_retry,
            reraise=True,
        )

        try:
            async for attempt in retryer:
                with attempt:
                    response = await request()
                    if response.status_code == 429:
                        raise _BoltzRateLimitError(response, operation)
                    return response
        except _BoltzRateLimitError:
            logger.warning(
                "Boltz kept returning HTTP 429 during %s after %s attempts",
                operation,
                self._settings.boltz_retry_attempts,
            )
            return None

        return None

    def _build_contacts(
        self, pocket_residues: Sequence[PocketContact] | None
    ) -> list[BoltzContact]:
        """Converts validated residue contacts into Boltz payload contacts.

        Boltz always assigns id "A" to the single polymer submitted in the
        request regardless of that chain's original PDB label, so every
        contact maps to "A" here. Callers are responsible for filtering
        `pocket_residues` down to the chain that `sequence` was extracted
        from before reaching this point (see `MoleculeGenerator.run`).
        """

        return [
            BoltzContact(id="A", residue_index=normalized.residue_index)
            for normalized in pocket_residues or []
        ]

    async def _poll_task(self, task_id: str, headers: dict[str, str]) -> dict[str, object] | None:
        """Polls long-running NVCF task until completion, failure, or timeout."""

        elapsed = 0.0
        while elapsed < self._settings.boltz_max_wait_seconds:
            await asyncio.sleep(self._settings.boltz_poll_interval_seconds)
            elapsed += self._settings.boltz_poll_interval_seconds
            status_response = await self._request_with_retry(
                lambda: self._client.get(
                    self.status_url.format(task_id=task_id),
                    headers=headers,
                    timeout=self._settings.boltz_request_timeout_seconds,
                ),
                operation=f"status poll for task {task_id}",
            )
            if status_response is None:
                return None
            if status_response.status_code == 200:
                return status_response.json()
            if status_response.status_code >= 400:
                return None

        logger.error(boltz_task_timeout(task_id, self._settings.boltz_max_wait_seconds))
        return None

    async def make_nvcf_call(
        self,
        smiles: str,
        sequence: str,
        pocket_residues: Sequence[PocketContact] | None = None,
    ) -> BoltzResult | None:
        """Submits a Boltz prediction and validates response structure.

        Args:
            smiles: Ligand SMILES to evaluate.
            sequence: Target protein sequence.
            pocket_residues: Optional validated residue contacts for pocket constraints.

        Returns:
            Parsed Boltz prediction and optional docked structure payload when successful.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "NVCF-POLL-SECONDS": str(self._settings.boltz_poll_header_seconds),
            "Content-Type": "application/json",
        }

        contacts = self._build_contacts(pocket_residues)
        payload = BoltzRequest(
            polymers=[
                BoltzPolymer(
                    id="A",
                    sequence=sequence,
                    msa=BoltzMsa(
                        uniref90=BoltzMsaUniRef90(a3m=BoltzA3M(alignment=f">seq1\n{sequence}"))
                    ),
                )
            ],
            ligands=[BoltzLigandRequest(smiles=smiles, id="L1")],
            constraints=[BoltzPocketConstraint(binder="L1", contacts=contacts)] if contacts else [],
            recycling_steps=self._settings.boltz_recycling_steps,
            sampling_steps=self._settings.boltz_sampling_steps,
            diffusion_samples=self._settings.boltz_diffusion_samples,
            step_scale=self._settings.boltz_step_scale,
            without_potentials=self._settings.boltz_without_potentials,
        )

        response = await self._request_with_retry(
            lambda: self._client.post(
                self.public_url,
                json=payload.model_dump(mode="json"),
                headers=headers,
                timeout=self._settings.boltz_request_timeout_seconds,
            ),
            operation=f"predict request for {smiles}",
        )
        if response is None:
            return None
        raw_data: dict[str, object] | None
        if response.status_code == 202:
            task_id = response.headers.get("nvcf-reqid")
            if not task_id:
                return None
            raw_data = await self._poll_task(task_id, headers)
        elif response.status_code == 200:
            raw_data = response.json()
        else:
            return None

        if raw_data is None:
            return None

        try:
            prediction = BoltzPrediction.model_validate(raw_data)
        except ValidationError:
            logger.warning(invalid_boltz_response(smiles))
            return None

        best_structure = self._extract_best_structure(smiles, prediction)
        return BoltzResult(prediction=prediction, best_structure=best_structure)

    @staticmethod
    def _extract_best_structure(
        smiles: str,
        prediction: BoltzPrediction,
    ) -> BestStructureRecord | None:
        """Extracts the mmCIF structure from the first Boltz structure entry."""

        if not prediction.structures:
            logger.warning(boltz_structure_missing(smiles))
            return None

        affinity = 0.0
        if prediction.affinities:
            ligand_affinity = next(iter(prediction.affinities.values()))
            if ligand_affinity.affinity_probability_binary:
                affinity = ligand_affinity.affinity_probability_binary[0]

        return BestStructureRecord(
            candidate_id="pending",
            smiles=smiles,
            affinity_prob=affinity,
            structure=prediction.structures[0].structure,
        )

    async def compute_properties(
        self,
        smiles_list: list[str],
        sequence: str,
        pocket_residues: Sequence[PocketContact] | None = None,
    ) -> tuple[pd.DataFrame, list[BestStructureRecord]]:
        """Evaluates candidate molecules and returns computed properties.

        Args:
            smiles_list: Candidate SMILES strings.
            sequence: Target protein sequence.
            pocket_residues: Optional validated residue contacts.

        Returns:
            Tuple of dataframe rows and optional docked structures from Boltz responses.
        """

        if not smiles_list:
            logger.info("No candidates available for Boltz scoring")
            return pd.DataFrame(), []

        logger.info("Submitting %s candidates to Boltz", len(smiles_list))
        records: list[MoleculeRecord] = []
        best_structures: list[BestStructureRecord] = []
        with make_progress_bar(
            total=len(smiles_list), desc="Boltz scoring", unit="mol"
        ) as progress:
            for index, smiles in enumerate(smiles_list, start=1):
                progress.set_postfix_str(smiles)
                scoring_started_at = perf_counter()
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    logger.warning("Skipping invalid molecule before Boltz scoring: %s", smiles)
                    progress.update(1)
                    continue

                logger.debug("Submitting Boltz evaluation for %s", smiles)
                result = await self.make_nvcf_call(
                    smiles=smiles,
                    sequence=sequence,
                    pocket_residues=pocket_residues,
                )
                if result is None:
                    logger.warning(
                        "Boltz returned no prediction for %s after %.2fs",
                        smiles,
                        perf_counter() - scoring_started_at,
                    )
                    progress.update(1)
                    continue

                prediction = result.prediction

                ptm = prediction.ptm_scores[0] if prediction.ptm_scores else 0.0
                iptm = prediction.iptm_scores[0] if prediction.iptm_scores else 0.0
                confidence = (
                    prediction.confidence_scores[0] if prediction.confidence_scores else 0.0
                )
                plddt = (
                    prediction.complex_plddt_scores[0] if prediction.complex_plddt_scores else 0.0
                )

                # `affinity_probability_binary` (binder/non-binder classifier output) and
                # `affinity_pic50` (regression output) are two independent Boltz-2 model
                # heads, not one derived from the other. `IC50_uM` below *is* derived,
                # but only from pIC50: IC50[M] = 10^-pIC50, converted here to micromolar.
                ligand_affinity = (
                    next(iter(prediction.affinities.values()))
                    if prediction.affinities
                    else BoltzLigandAffinity(affinity_probability_binary=[], affinity_pic50=[])
                )
                affinity_probability = (
                    ligand_affinity.affinity_probability_binary[0]
                    if ligand_affinity.affinity_probability_binary
                    else 0.0
                )
                pic50 = ligand_affinity.affinity_pic50[0] if ligand_affinity.affinity_pic50 else 0.0
                ic50_um = pow(10, -pic50) * 1e6

                logger.info(
                    "Boltz result %s/%s for %s in %.2fs: affinity=%.3f pIC50=%.3f "
                    "pTM=%.3f ipTM=%.3f pLDDT=%.3f",
                    index,
                    len(smiles_list),
                    smiles,
                    perf_counter() - scoring_started_at,
                    affinity_probability,
                    pic50,
                    ptm,
                    iptm,
                    plddt,
                )

                record = MoleculeRecord(
                    SMILES=smiles,
                    pTM=ptm,
                    ipTM=iptm,
                    Confidence=confidence,
                    pLDDT=plddt,
                    Affinity_Prob=affinity_probability,
                    pIC50=pic50,
                    IC50_uM=ic50_um,
                    MolWt=round(Descriptors.MolWt(mol), 2),
                    LogP=round(Descriptors.MolLogP(mol), 2),
                    QED=QED.qed(mol),
                    SAS=round(sa_score_mol(mol), 3),
                    TPSA=round(Descriptors.TPSA(mol), 2),
                    H_Acceptors=int(Descriptors.NumHAcceptors(mol)),
                    H_Donors=int(Descriptors.NumHDonors(mol)),
                    Rotatable_Bonds=int(Descriptors.NumRotatableBonds(mol)),
                )
                records.append(record)
                if result.best_structure is not None:
                    best_structures.append(result.best_structure)
                progress.update(1)

        logger.info(
            "Boltz scoring complete: %s/%s candidates succeeded",
            len(records),
            len(smiles_list),
        )
        return pd.DataFrame.from_records([record.model_dump() for record in records]), best_structures

"""NVIDIA Boltz client for async molecular property prediction."""

import asyncio
from typing import Any

import httpx
import pandas as pd
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

import sascorer
from src.clients import get_cached_http_client
from src.core.logging import get_logger
from src.schemas import BoltzLigandAffinity, BoltzPrediction, MoleculeRecord

logger = get_logger(__name__)


class BoltzClient:
    """Async wrapper around NVIDIA Boltz prediction API."""

    def __init__(self, api_key: str, http_client: httpx.AsyncClient | None = None) -> None:
        """Initializes the client.

        Args:
            api_key: NVIDIA API key.
            http_client: Optional preconfigured async client.
        """

        self.api_key = api_key
        self.public_url = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
        self.status_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{task_id}"
        self._client = http_client or get_cached_http_client()

    async def _poll_task(self, task_id: str, headers: dict[str, str]) -> dict[str, Any] | None:
        """Polls long-running NVCF task until completion or failure."""

        while True:
            await asyncio.sleep(5)
            status_response = await self._client.get(
                self.status_url.format(task_id=task_id),
                headers=headers,
                timeout=400,
            )
            if status_response.status_code == 200:
                return status_response.json()
            if status_response.status_code >= 400:
                return None

    async def make_nvcf_call(
        self,
        smiles: str,
        sequence: str,
        pocket_residues: list[int] | None = None,
    ) -> BoltzPrediction | None:
        """Submits a Boltz prediction and validates response structure.

        Args:
            smiles: Ligand SMILES to evaluate.
            sequence: Target protein sequence.
            pocket_residues: Optional residue indices for pocket constraints.

        Returns:
            Parsed `BoltzPrediction` payload when successful.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "NVCF-POLL-SECONDS": "300",
            "Content-Type": "application/json",
        }

        polymers = [
            {
                "id": "A",
                "molecule_type": "protein",
                "sequence": sequence,
                "msa": {"uniref90": {"a3m": {"alignment": f">seq1\n{sequence}", "format": "a3m"}}},
            }
        ]
        ligands = [{"smiles": smiles, "id": "L1", "predict_affinity": True}]

        constraints: list[dict[str, Any]] = []
        if pocket_residues:
            constraints.append(
                {
                    "constraint_type": "pocket",
                    "binder": "L1",
                    "contacts": [{"id": "A", "residue_index": residue} for residue in pocket_residues],
                }
            )

        payload = {
            "polymers": polymers,
            "ligands": ligands,
            "constraints": constraints,
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "step_scale": 1.2,
            "without_potentials": True,
        }

        response = await self._client.post(self.public_url, json=payload, headers=headers, timeout=400)
        raw_data: dict[str, Any] | None
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
            return BoltzPrediction.model_validate(raw_data)
        except ValidationError:
            return None

    async def compute_properties(
        self,
        smiles_list: list[str],
        sequence: str,
        pocket_residues: list[int] | None = None,
    ) -> pd.DataFrame:
        """Evaluates candidate molecules and returns computed properties.

        Args:
            smiles_list: Candidate SMILES strings.
            sequence: Target protein sequence.
            pocket_residues: Optional residue constraints.

        Returns:
            Dataframe with one row per successfully evaluated molecule.
        """

        records: list[dict[str, Any]] = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            logger.info("Evaluating SMILES: %s", smiles)
            prediction = await self.make_nvcf_call(
                smiles=smiles,
                sequence=sequence,
                pocket_residues=pocket_residues,
            )
            if prediction is None:
                continue

            ptm = prediction.ptm_scores[0] if prediction.ptm_scores else 0.0
            iptm = prediction.iptm_scores[0] if prediction.iptm_scores else 0.0
            confidence = prediction.confidence_scores[0] if prediction.confidence_scores else 0.0
            plddt = prediction.complex_plddt_scores[0] if prediction.complex_plddt_scores else 0.0

            ligand_affinity = (
                next(iter(prediction.affinities.values()))
                if prediction.affinities
                else BoltzLigandAffinity()
            )
            affinity_probability = (
                ligand_affinity.affinity_probability_binary[0]
                if ligand_affinity.affinity_probability_binary
                else 0.0
            )
            pic50 = ligand_affinity.affinity_pic50[0] if ligand_affinity.affinity_pic50 else 0.0
            ic50_um = pow(10, -pic50) * 1e6

            logger.info("  > Predicted pTM:           %.3f", ptm)
            logger.info("  > Predicted ipTM:          %.3f", iptm)
            logger.info("  > Confidence Score:        %.3f", confidence)
            logger.info("  > Average pLDDT:           %.3f", plddt)
            logger.info("  > Affinity Probability:    %.3f", affinity_probability)
            logger.info("  > Predicted pIC50:         %.3f", pic50)
            logger.info("  > Predicted IC50 (µM):     %.3f", ic50_um)

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
                QED= QED.qed(mol),
                SAS=round(sascorer.calculateScore(mol), 3),
                TPSA=round(Descriptors.TPSA(mol), 2),
                H_Acceptors=int(Descriptors.NumHAcceptors(mol)),
                H_Donors=int(Descriptors.NumHDonors(mol)),
                Rotatable_Bonds=int(Descriptors.NumRotatableBonds(mol)),
            )
            records.append(record.model_dump())

        return pd.DataFrame.from_records(records)

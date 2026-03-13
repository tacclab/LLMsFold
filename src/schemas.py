"""Pydantic schemas used by the generation pipeline."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from rdkit import Chem


def _validate_smiles(value: str) -> str:
    """Normalizes and validates a SMILES string with RDKit."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("SMILES must be a non-empty string")
    if Chem.MolFromSmiles(normalized) is None:
        raise ValueError(f"Invalid SMILES: {normalized}")
    return normalized


SmilesString = Annotated[str, AfterValidator(_validate_smiles)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class GeneratedMolecule(BaseModel):
    """Single molecule proposal returned by the language model."""

    model_config = ConfigDict(extra="ignore")

    smiles: SmilesString

    @classmethod
    def from_raw(cls, payload: Any) -> "GeneratedMolecule":
        """Builds a validated molecule proposal from raw model payloads."""

        if isinstance(payload, cls):
            return payload
        if isinstance(payload, str):
            return cls(smiles=payload)
        if isinstance(payload, dict):
            if "smiles" in payload:
                return cls(smiles=payload["smiles"])
            if "SMILES" in payload:
                return cls(smiles=payload["SMILES"])
        return cls.model_validate(payload)


class ModelOutput(BaseModel):
    """Normalized LLM output containing validated molecule proposals."""

    model_config = ConfigDict(extra="ignore")

    molecules: list[GeneratedMolecule] = Field(min_length=1)

    @field_validator("molecules")
    @classmethod
    def _deduplicate_molecules(cls, molecules: list[GeneratedMolecule]) -> list[GeneratedMolecule]:
        """Preserves order while removing duplicate SMILES proposals."""

        unique_molecules: list[GeneratedMolecule] = []
        seen_smiles: set[str] = set()
        for molecule in molecules:
            if molecule.smiles in seen_smiles:
                continue
            seen_smiles.add(molecule.smiles)
            unique_molecules.append(molecule)
        return unique_molecules

    @classmethod
    def from_raw_payload(cls, payload: Any) -> "ModelOutput":
        """Coerces common LLM payload shapes into validated molecule proposals."""

        raw_items = cls._extract_raw_items(payload)
        molecules: list[GeneratedMolecule] = []
        for item in raw_items:
            try:
                molecules.append(GeneratedMolecule.from_raw(item))
            except ValidationError:
                continue
        return cls(molecules=molecules)

    @staticmethod
    def _extract_raw_items(payload: Any) -> list[Any]:
        """Returns raw molecule items from list/dict/string payloads."""

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("molecules", "smiles", "SMILES"):
                if key not in payload:
                    continue
                raw_value = payload[key]
                return raw_value if isinstance(raw_value, list) else [raw_value]
            if "smiles" in payload or "SMILES" in payload:
                return [payload]
        if isinstance(payload, str):
            return [payload]
        return []

    def smiles_list(self) -> list[str]:
        """Returns the validated SMILES strings in output order."""

        return [molecule.smiles for molecule in self.molecules]


class PipelineOptions(BaseModel):
    """Validated runtime options for the molecule generation pipeline.

    Attributes:
        pdb_path: Path to input PDB structure file.
        few_shot_csv: Path to few-shot seed molecules CSV.
        output_dir: Folder where reports are written.
        protein_sequence: Protein sequence extracted from the target PDB.
        max_iterations: Number of iterative generation rounds.
        max_samples: Number of LLM proposals per round.
        use_pocket_data: If `True`, includes pocket residue constraints.
    """

    pdb_path: Path
    few_shot_csv: Path
    output_dir: Path
    protein_sequence: str = Field(min_length=1)
    max_iterations: int = Field(default=3, ge=1)
    max_samples: int = Field(default=5, ge=1)
    use_pocket_data: bool = True

    @field_validator("protein_sequence")
    @classmethod
    def _normalize_protein_sequence(cls, sequence: str) -> str:
        """Removes whitespace and uppercases the protein sequence."""

        normalized = "".join(sequence.split()).upper()
        if not normalized:
            raise ValueError("protein_sequence must contain at least one residue")
        return normalized


class PocketContact(BaseModel):
    """Validated pocket residue contact used across generator/client layers."""

    chain_id: str = Field(min_length=1, max_length=1)
    residue_index: int = Field(ge=1)

    @field_validator("chain_id")
    @classmethod
    def _normalize_chain_id(cls, chain_id: str) -> str:
        """Trims and uppercases a single chain identifier."""

        normalized = chain_id.strip().upper()
        if len(normalized) != 1 or not normalized.isalnum():
            raise ValueError("chain_id must be a single alphanumeric character")
        return normalized


class BoltzLigandAffinity(BaseModel):
    """Subset of Boltz affinity values used for ranking."""

    model_config = ConfigDict(extra="ignore")

    affinity_probability_binary: list[Probability]
    affinity_pic50: list[float]


class BoltzContact(BaseModel):
    """Pocket contact entry expected by the NVIDIA Boltz payload."""

    id: str = Field(min_length=1)
    residue_index: int = Field(ge=1)


class BoltzPocketConstraint(BaseModel):
    """Pocket constraint portion of a Boltz request payload."""

    constraint_type: Literal["pocket"] = "pocket"
    binder: str = Field(min_length=1)
    contacts: list[BoltzContact]


class BoltzA3M(BaseModel):
    """Minimal A3M alignment block used in Boltz requests."""

    alignment: str = Field(min_length=1)
    format: Literal["a3m"] = "a3m"


class BoltzMsaUniRef90(BaseModel):
    """UniRef90 alignment container used by NVIDIA Boltz."""

    a3m: BoltzA3M


class BoltzMsa(BaseModel):
    """MSA wrapper used by the Boltz polymer definition."""

    uniref90: BoltzMsaUniRef90


class BoltzPolymer(BaseModel):
    """Protein chain request entry for NVIDIA Boltz."""

    id: str = Field(min_length=1)
    molecule_type: Literal["protein"] = "protein"
    sequence: str = Field(min_length=1)
    msa: BoltzMsa


class BoltzLigandRequest(BaseModel):
    """Ligand request entry for NVIDIA Boltz."""

    smiles: SmilesString
    id: str = Field(min_length=1)
    predict_affinity: bool = True


class BoltzRequest(BaseModel):
    """Validated request payload sent to NVIDIA Boltz."""

    polymers: list[BoltzPolymer]
    ligands: list[BoltzLigandRequest]
    constraints: list[BoltzPocketConstraint] = Field(default_factory=list)
    recycling_steps: int = Field(ge=1)
    sampling_steps: int = Field(ge=1)
    diffusion_samples: int = Field(ge=1)
    step_scale: float = Field(gt=0.0)
    without_potentials: bool = True


class BoltzPrediction(BaseModel):
    """Normalized NVIDIA Boltz response payload."""

    model_config = ConfigDict(extra="ignore")

    ptm_scores: list[Probability]
    iptm_scores: list[Probability]
    confidence_scores: list[Probability]
    complex_plddt_scores: list[Probability]
    affinities: dict[str, BoltzLigandAffinity] = Field(default_factory=dict)
    structure: str | None = None


class BestStructureRecord(BaseModel):
    """Docked structure payload persisted for high-affinity candidates."""

    candidate_id: str = Field(min_length=1)
    smiles: SmilesString
    affinity_prob: Probability
    evaluation: dict[str, object] = Field(default_factory=dict)
    pdb: str | None = None
    structure: str | None = None


class MoleculeRecord(BaseModel):
    """Computed features and predicted activity for a candidate molecule."""

    model_config = ConfigDict(extra="forbid")

    SMILES: SmilesString
    pTM: Probability
    ipTM: Probability
    Confidence: Probability
    pLDDT: Probability
    Affinity_Prob: Probability
    pIC50: float
    IC50_uM: NonNegativeFloat
    MolWt: Annotated[float, Field(gt=0.0)]
    LogP: float
    QED: Probability
    SAS: Annotated[float, Field(ge=0.0, le=10.0)]
    TPSA: NonNegativeFloat
    H_Acceptors: NonNegativeInt
    H_Donors: NonNegativeInt
    Rotatable_Bonds: NonNegativeInt


class ScoredMoleculeRecord(MoleculeRecord):
    """Final scored row produced before novelty enrichment."""

    MaxSim: Probability
    adj_affinity: Probability
    score: NonNegativeFloat
    Candidate_ID: str | None = None


class PatentReportRow(BaseModel):
    """Validated novelty-related columns appended to the final report."""

    PubChem_CID: int | Literal["N/A"]
    Identity_Patents: NonNegativeInt
    Substructure_Patents: NonNegativeInt
    PubChem_Known: Literal["Yes", "No"]
    PubChem_Novelty_Note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_pubchem_consistency(self) -> "PatentReportRow":
        """Ensures PubChem flags remain aligned with the CID field."""

        if self.PubChem_Known == "No" and self.PubChem_CID != "N/A":
            raise ValueError("PubChem_CID must be 'N/A' when PubChem_Known is 'No'")
        if self.PubChem_Known == "Yes" and self.PubChem_CID == "N/A":
            raise ValueError("PubChem_CID must be present when PubChem_Known is 'Yes'")
        return self


class UnifiedReportRow(ScoredMoleculeRecord):
    """Complete validated row written to the final unified CSV report."""

    PubChem_CID: int | Literal["N/A"]
    Identity_Patents: NonNegativeInt
    Substructure_Patents: NonNegativeInt
    PubChem_Known: Literal["Yes", "No"]
    PubChem_Novelty_Note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_pubchem_consistency(self) -> "UnifiedReportRow":
        """Ensures report rows keep CID and novelty flags consistent."""

        if self.PubChem_Known == "No" and self.PubChem_CID != "N/A":
            raise ValueError("PubChem_CID must be 'N/A' when PubChem_Known is 'No'")
        if self.PubChem_Known == "Yes" and self.PubChem_CID == "N/A":
            raise ValueError("PubChem_CID must be present when PubChem_Known is 'Yes'")
        return self


class PatentCheckResult(BaseModel):
    """PubChem presence and patent-like identity/substructure signals."""

    pubchem_cid: int | None = None
    identity_patents: NonNegativeInt = 0
    substructure_patents: NonNegativeInt = 0

    @field_validator("pubchem_cid")
    @classmethod
    def _normalize_pubchem_cid(cls, cid: int | None) -> int | None:
        """Treats non-positive CIDs as missing values."""

        if cid is None or cid <= 0:
            return None
        return cid

    def to_report_row(self) -> dict[str, int | str]:
        """Formats values as final CSV columns.

        Returns:
            A dictionary with report-ready keys.
        """

        pubchem_known: Literal["Yes", "No"] = (
            "No" if (self.pubchem_cid is None or self.pubchem_cid == 0) else "Yes"
        )
        row = PatentReportRow(
            PubChem_CID=self.pubchem_cid if self.pubchem_cid else "N/A",
            Identity_Patents=self.identity_patents,
            Substructure_Patents=self.substructure_patents,
            PubChem_Known=pubchem_known,
            PubChem_Novelty_Note="Absence from PubChem does not establish legal novelty.",
        )
        return row.model_dump()

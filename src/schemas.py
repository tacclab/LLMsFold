"""Pydantic schemas used by the generation pipeline."""

from pydantic import BaseModel, ConfigDict, Field


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

    pdb_path: str
    few_shot_csv: str
    output_dir: str
    protein_sequence: str
    max_iterations: int = Field(default=3, ge=1)
    max_samples: int = Field(default=5, ge=1)
    use_pocket_data: bool = True


class BoltzLigandAffinity(BaseModel):
    """Subset of Boltz affinity values used for ranking."""

    model_config = ConfigDict(extra="ignore")

    affinity_probability_binary: list[float] = Field(default_factory=lambda: [0.0])
    affinity_pic50: list[float] = Field(default_factory=lambda: [0.0])


class BoltzPrediction(BaseModel):
    """Normalized NVIDIA Boltz response payload."""

    model_config = ConfigDict(extra="ignore")

    ptm_scores: list[float] = Field(default_factory=lambda: [0.0])
    iptm_scores: list[float] = Field(default_factory=lambda: [0.0])
    confidence_scores: list[float] = Field(default_factory=lambda: [0.0])
    complex_plddt_scores: list[float] = Field(default_factory=lambda: [0.0])
    affinities: dict[str, BoltzLigandAffinity] = Field(default_factory=dict)


class MoleculeRecord(BaseModel):
    """Computed features and predicted activity for a candidate molecule."""

    SMILES: str
    pTM: float
    ipTM: float
    Confidence: float
    pLDDT: float
    Affinity_Prob: float
    pIC50: float
    IC50_uM: float
    MolWt: float
    LogP: float
    QED: float
    SAS: float
    TPSA: float
    H_Acceptors: int
    H_Donors: int
    Rotatable_Bonds: int


class PatentCheckResult(BaseModel):
    """PubChem presence and patent-like identity/substructure signals."""

    pubchem_cid: int | None = None
    identity_patents: int = 0
    substructure_patents: int = 0

    def to_report_row(self) -> dict[str, int | str]:
        """Formats values as final CSV columns.

        Returns:
            A dictionary with report-ready keys.
        """

        pubchem_known = "No" if (self.pubchem_cid is None or self.pubchem_cid == 0) else "Yes"
        return {
            "PubChem_CID": self.pubchem_cid if self.pubchem_cid else "N/A",
            "Identity_Patents": self.identity_patents,
            "Substructure_Patents": self.substructure_patents,
            "PubChem_Known": pubchem_known,
            "PubChem_Novelty_Note": "Absence from PubChem does not establish legal novelty.",
        }

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

from src.core.constants import DEFAULT_POCKET_CONTACT_DISTANCE


def _validate_smiles(value: str) -> str:
    """Validates a SMILES string with RDKit and returns its canonical form.

    Canonicalizing here -- the single point every SMILES in the pipeline
    passes through via `SmilesString` -- ensures the same molecule written
    two different ways is cached, deduplicated, and counted as one entry
    everywhere downstream (Boltz score cache, global registry, lead pools,
    final report), instead of being treated as two distinct molecules.
    """

    normalized = value.strip()
    if not normalized:
        raise ValueError("SMILES must be a non-empty string")
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {normalized}")
    return Chem.MolToSmiles(mol)


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
    invalid_count: int = 0

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
        invalid_count = 0
        for item in raw_items:
            try:
                molecules.append(GeneratedMolecule.from_raw(item))
            except ValidationError:
                invalid_count += 1
                continue
        return cls(molecules=molecules, invalid_count=invalid_count)

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
        target_chain_id: PDB chain id that `protein_sequence` was extracted
            from. Detected pocket residues are filtered to this chain before
            being remapped onto Boltz's single-polymer id ("A").
        residue_position_map: Maps PDB residue numbers (e.g. 203-498) onto
            1-based positions within `protein_sequence` (e.g. 1-294). Pocket
            residues without an entry here fall in a gap/disordered region
            and must not be sent to Boltz.
        score_references: If `True`, the few-shot reference inhibitors are
            scored through the same pipeline as generated candidates and
            written to their own baseline report. Reference molecules never
            enter the candidate ranking, the similarity registry, or the
            prompt feedback pools.
        pocket_residues: Explicit residue selection, e.g. ``"A:203, A:207"``.
        pocket_ligand: Residue name of a bound ligand whose surroundings
            define the pocket, e.g. ``"ATP"`` or ``"ATP:A:401"``.
        pocket_center: Explicit pocket center in the structure's own frame.
        pocket_index: 0-based index into the detected pocket list, to pin a
            specific pocket instead of accepting automatic selection.
        pocket_contact_distance: Radius used by the ligand and center modes
            when collecting surrounding residues.
    """

    pdb_path: Path
    few_shot_csv: Path
    output_dir: Path
    protein_sequence: str = Field(min_length=1)
    max_iterations: int = Field(default=3, ge=1)
    max_samples: int = Field(default=5, ge=1)
    use_pocket_data: bool = True
    target_chain_id: str = Field(default="A", min_length=1, max_length=1)
    residue_position_map: dict[int, int] = Field(default_factory=dict)
    score_references: bool = False
    pocket_residues: str | None = None
    pocket_ligand: str | None = None
    pocket_center: tuple[float, float, float] | None = None
    pocket_index: int | None = Field(default=None, ge=0)
    pocket_contact_distance: float = Field(default=DEFAULT_POCKET_CONTACT_DISTANCE, gt=0.0)

    @property
    def pocket_mode(self) -> Literal["disabled", "residues", "ligand", "center", "index", "auto"]:
        """Reports how this run's binding pocket is chosen.

        Recorded in the manifest so a reader can tell a geometry-driven
        pocket from one the user supplied -- the two support very different
        claims about how the site was found.
        """

        if not self.use_pocket_data:
            return "disabled"
        if self.pocket_residues:
            return "residues"
        if self.pocket_ligand:
            return "ligand"
        if self.pocket_center is not None:
            return "center"
        if self.pocket_index is not None:
            return "index"
        return "auto"

    @model_validator(mode="after")
    def _validate_single_pocket_source(self) -> "PipelineOptions":
        """Rejects more than one pocket source at a time.

        Silently preferring one source over another would let a run report a
        pocket the user did not ask for, so an ambiguous request fails here.
        """

        chosen = [
            name
            for name, value in (
                ("pocket_residues", self.pocket_residues),
                ("pocket_ligand", self.pocket_ligand),
                ("pocket_center", self.pocket_center),
                ("pocket_index", self.pocket_index),
            )
            if value is not None and value != ""
        ]
        if len(chosen) > 1:
            raise ValueError(
                "Give only one pocket source; received " + " and ".join(sorted(chosen))
            )
        if chosen and not self.use_pocket_data:
            raise ValueError(
                f"{chosen[0]} was given but pocket use is disabled; drop one of the two"
            )
        return self

    @field_validator("protein_sequence")
    @classmethod
    def _normalize_protein_sequence(cls, sequence: str) -> str:
        """Removes whitespace and uppercases the protein sequence."""

        normalized = "".join(sequence.split()).upper()
        if not normalized:
            raise ValueError("protein_sequence must contain at least one residue")
        return normalized

    @field_validator("target_chain_id")
    @classmethod
    def _normalize_target_chain_id(cls, chain_id: str) -> str:
        """Trims and uppercases the target chain identifier."""

        normalized = chain_id.strip().upper()
        if len(normalized) != 1 or not normalized.isalnum():
            raise ValueError("target_chain_id must be a single alphanumeric character")
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

    affinity_pred_value: list[float] = Field(default_factory=list)
    affinity_probability_binary: list[Probability] = Field(default_factory=list)
    model_1_affinity_pred_value: list[float] = Field(default_factory=list)
    model_1_affinity_probability_binary: list[Probability] = Field(default_factory=list)
    model_2_affinity_pred_value: list[float] = Field(default_factory=list)
    model_2_affinity_probability_binary: list[Probability] = Field(default_factory=list)
    affinity_pic50: list[float] = Field(default_factory=list)


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


class BoltzStructure(BaseModel):
    """A single structure prediction in mmCIF format returned by Boltz."""

    model_config = ConfigDict(extra="ignore")

    structure: str
    format: str = "mmcif"
    name: str | None = None
    source: str | None = None


class BoltzPrediction(BaseModel):
    """Normalized NVIDIA Boltz response payload."""

    model_config = ConfigDict(extra="ignore")

    structures: list[BoltzStructure] = Field(default_factory=list)
    ptm_scores: list[Probability] = Field(default_factory=list)
    iptm_scores: list[Probability] = Field(default_factory=list)
    confidence_scores: list[Probability] = Field(default_factory=list)
    complex_plddt_scores: list[Probability] = Field(default_factory=list)
    affinities: dict[str, BoltzLigandAffinity] = Field(default_factory=dict)


class BestStructureRecord(BaseModel):
    """Docked structure payload persisted for high-affinity candidates."""

    candidate_id: str = Field(min_length=1)
    smiles: SmilesString
    affinity_prob: Probability
    structure: str  # mmCIF content


class MoleculeRecord(BaseModel):
    """Computed features and predicted activity for a candidate molecule.

    Columns carrying a `Boltz2_` prefix are outputs of the Boltz-2 predictor,
    not measurements: naming the model in the column keeps a predicted value
    from being read as an experimental one, and leaves room for a second
    evaluator's columns to sit beside them without ambiguity. The unprefixed
    columns are deterministic RDKit descriptors of the molecule itself, which
    depend on no model.

    `Boltz2_Binding_Probability` (a binder/non-binder classifier head) and
    `Boltz2_Predicted_Affinity_pIC50` (a regression head) are independent
    model outputs. `Boltz2_Predicted_Affinity_uM` is not a third output: it
    is the algebraic transform 10^-pIC50 of the regression head, expressed in
    micromolar, and is uncalibrated against experimental data.
    """

    model_config = ConfigDict(extra="forbid")

    SMILES: SmilesString
    Boltz2_pTM: Probability
    Boltz2_ipTM: Probability
    Boltz2_Confidence: Probability
    Boltz2_Complex_pLDDT: Probability
    Boltz2_Binding_Probability: Probability
    Boltz2_Predicted_Affinity_pIC50: float
    Boltz2_Predicted_Affinity_uM: NonNegativeFloat
    MolWt: Annotated[float, Field(gt=0.0)]
    LogP: float
    QED: Probability
    SAS: Annotated[float, Field(ge=0.0, le=10.0)]
    TPSA: NonNegativeFloat
    H_Acceptors: NonNegativeInt
    H_Donors: NonNegativeInt
    Rotatable_Bonds: NonNegativeInt


class ScoredMoleculeRecord(MoleculeRecord):
    """Final scored row produced before novelty enrichment.

    `Adj_Binding_Probability` is `Boltz2_Binding_Probability` after the
    configured cutoff is applied; it is a reward term, not a prediction of
    its own.
    """

    MaxSim: Probability
    Adj_Binding_Probability: Probability
    synth_factor: Probability
    score: NonNegativeFloat
    Candidate_ID: str | None = None


class ReferenceBaselineRow(MoleculeRecord):
    """A reference inhibitor scored through the candidate scoring path.

    Emitted to its own report so published candidate tables can be read
    against a baseline computed with the identical predictor, pocket
    constraints, and composite score, rather than against values quoted
    from the literature under different assumptions.

    `MaxSim_To_Other_References` is deliberately leave-one-out: a reference
    compared against the full seed set would always score 1.0 against
    itself, which is not the quantity a generated molecule is measured on.

    The gate columns record whether this run's thresholds would have
    admitted an established inhibitor, which is the direct check on whether
    those thresholds are set at a defensible level.
    """

    Reference_Name: str = Field(min_length=1)
    Role: Literal["reference"] = "reference"
    MaxSim_To_Other_References: Probability
    Adj_Binding_Probability: Probability
    synth_factor: Probability
    score: NonNegativeFloat
    Passes_Affinity_Gate: bool
    Passes_SAS_Gate: bool
    Passes_ipTM_Gate: bool
    Passes_pLDDT_Gate: bool
    Passes_All_Gates: bool


PubChemClassification = Literal[
    "Novel/Not in PubChem",
    "Known Drug/Medication",
    "Patent-Referenced",
    "Known Compound (Unclassified)",
]


class PatentReportRow(BaseModel):
    """Validated novelty-related columns appended to the final report."""

    PubChem_CID: int | Literal["N/A"]
    Identity_Patents: NonNegativeInt
    Substructure_Patents: NonNegativeInt
    PubChem_Known: Literal["Yes", "No"]
    PubChem_Classification: PubChemClassification
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
    """Complete validated row written to the final unified CSV report.

    The scaffold columns answer a different question from the PubChem ones:
    PubChem says whether this exact molecule is already known, while
    `Scaffold_Is_Novel` says whether its core differs from every core the
    model was shown. A candidate can be absent from PubChem while reusing a
    seed scaffold outright. They default to `None` because scaffold analysis
    is best-effort and must not fail a completed run.
    """

    Murcko_Scaffold: str | None = None
    Generic_Scaffold: str | None = None
    Scaffold_Is_Novel: bool | None = None
    Generic_Scaffold_Is_Novel: bool | None = None
    Max_Scaffold_Sim_To_Seeds: Probability | None = None

    # Holdout columns. These come from evaluators that had no influence on
    # generation, so unlike every other score here they are not something the
    # pipeline optimized toward. They are `None` when the corresponding
    # evaluator was unavailable or could not handle a given pose, which is
    # reported rather than filled in.
    PoseBusters_Pass: bool | None = None
    PoseBusters_Checks_Passed: NonNegativeInt | None = None
    PoseBusters_Checks_Total: NonNegativeInt | None = None
    PoseBusters_Failed_Checks: str | None = None
    PoseBusters_Bond_Orders_Assigned: bool | None = None
    Vina_Score_kcal_per_mol: float | None = None
    Vina_Minimized_kcal_per_mol: float | None = None
    Vina_Minimization_Gain: float | None = None

    PubChem_CID: int | Literal["N/A"]
    Identity_Patents: NonNegativeInt
    Substructure_Patents: NonNegativeInt
    PubChem_Known: Literal["Yes", "No"]
    PubChem_Classification: PubChemClassification
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
    has_drug_info: bool = False

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

        classification: PubChemClassification
        if pubchem_known == "No":
            classification = "Novel/Not in PubChem"
        elif self.has_drug_info:
            classification = "Known Drug/Medication"
        elif self.identity_patents > 0 or self.substructure_patents > 0:
            classification = "Patent-Referenced"
        else:
            classification = "Known Compound (Unclassified)"

        row = PatentReportRow(
            PubChem_CID=self.pubchem_cid if self.pubchem_cid else "N/A",
            Identity_Patents=self.identity_patents,
            Substructure_Patents=self.substructure_patents,
            PubChem_Known=pubchem_known,
            PubChem_Classification=classification,
            PubChem_Novelty_Note=(
                "Classification reflects PubChem's own annotation coverage "
                "(Drug and Medication Information section presence, plus identity/"
                "substructure patent cross-references); it is not a formal drug "
                "registry lookup or a legal freedom-to-operate determination."
            ),
        )
        return row.model_dump()

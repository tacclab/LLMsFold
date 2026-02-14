"""Shared pytest fixtures for the project test suite."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure local `src/` imports resolve when tests run from project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schemas import PipelineOptions


@pytest.fixture
def few_shot_csv(tmp_path: Path) -> Path:
    """Creates a semicolon-delimited few-shot file used by generator tests."""

    csv_path = tmp_path / "few_shot.csv"
    pd.DataFrame({"Smiles": ["CCO", "CCN", "CCC"]}).to_csv(csv_path, sep=";", index=False)
    return csv_path


@pytest.fixture
def pipeline_options(tmp_path: Path, few_shot_csv: Path) -> PipelineOptions:
    """Builds default validated options for workflow-oriented tests."""

    return PipelineOptions(
        pdb_path=str(tmp_path / "protein.pdb"),
        few_shot_csv=str(few_shot_csv),
        output_dir=str(tmp_path / "results"),
        protein_sequence="MKT",
        max_iterations=1,
        max_samples=2,
        use_pocket_data=True,
    )


@pytest.fixture
def scored_dataframe() -> pd.DataFrame:
    """Provides a minimal scored dataframe compatible with generator output."""

    return pd.DataFrame(
        [
            {
                "SMILES": "CCO",
                "pTM": 0.9,
                "ipTM": 0.8,
                "Confidence": 0.7,
                "pLDDT": 0.6,
                "Affinity_Prob": 0.95,
                "pIC50": 7.0,
                "IC50_uM": 0.1,
                "MolWt": 46.07,
                "LogP": 0.2,
                "QED": 0.5,
                "SAS": 3.1,
                "TPSA": 20.2,
                "H_Acceptors": 1,
                "H_Donors": 1,
                "Rotatable_Bonds": 0,
            }
        ]
    )

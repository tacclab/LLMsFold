"""Tests for CLI orchestration in main.py."""

import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import main


@pytest.fixture
def fake_settings() -> SimpleNamespace:
    """Settings object mirroring fields used by `main.main`."""

    def secret(value: str) -> SimpleNamespace:
        return SimpleNamespace(get_secret_value=lambda: value)

    return SimpleNamespace(
        pdb_file="data/target.pdb",
        few_shot_csv="data/few_shot.csv",
        output_dir="results",
        max_iterations=2,
        max_samples=3,
        llm_model="llama-3.3-70b-versatile",
        llm_temperature=0.8,
        log_level="INFO",
        random_seed=None,
        nvidia_api_key=secret("n-key"),
        groq_api_key=secret("g-key"),
    )


def test_build_parser_defaults_and_no_pocket_flag() -> None:
    """Parser should use provided defaults and support disabling pocket mode."""

    parser = main._build_parser(
        {"pdb": "p.pdb", "csv": "few.csv", "out": "out", "iters": 4, "samples": 6}
    )

    defaults = parser.parse_args([])
    assert defaults.pdb == "p.pdb"
    assert defaults.csv == "few.csv"
    assert defaults.out == "out"
    assert defaults.iters == 4
    assert defaults.samples == 6
    assert defaults.use_pocket is True

    no_pocket = parser.parse_args(["--no-pocket"])
    assert no_pocket.use_pocket is False


def test_build_launch_banner_includes_authors_and_group_links() -> None:
    """Launch banner should surface the project credits and research-group URLs."""

    banner = main._build_launch_banner()

    assert "LLMsFold" in banner
    assert "Fabio Bove" in banner
    assert "Cristian Taccioli" in banner
    assert "https://tacclab.org/" in banner
    assert "https://www.neoralab.com/" in banner


def test_main_returns_early_when_pdb_missing(
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing PDB path exits before sequence extraction and generation."""

    parser = MagicMock()
    parser.parse_args.return_value = argparse.Namespace(
        pdb="missing.pdb", csv="few.csv", out="out", iters=1, samples=1, use_pocket=True
    )

    monkeypatch.setattr(main, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(main, "_build_parser", lambda _defaults: parser)
    monkeypatch.setattr(main, "_show_launch_banner", lambda: None)
    monkeypatch.setattr(main.os.path, "exists", lambda _path: False)

    close_mock = AsyncMock()
    monkeypatch.setattr(main, "close_cached_clients", close_mock)

    asyncio.run(main.main())

    close_mock.assert_not_awaited()


def test_main_happy_path_runs_generator_and_closes_clients(
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful run wires dependencies and always closes cached clients."""

    parser = MagicMock()
    parser.parse_args.return_value = argparse.Namespace(
        pdb="protein.pdb",
        csv="few.csv",
        out="results",
        iters=2,
        samples=3,
        use_pocket=False,
    )

    generator_instance = MagicMock()
    generator_instance.run = AsyncMock(return_value="results/unified_report.csv")
    show_banner = MagicMock()

    monkeypatch.setattr(main, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(main, "_build_parser", lambda _defaults: parser)
    monkeypatch.setattr(main, "_show_launch_banner", show_banner)
    monkeypatch.setattr(main.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(main, "extract_sequence_from_pdb", lambda _path: "MKT")
    monkeypatch.setattr(main, "extract_target_chain_id", lambda _path: "A")
    monkeypatch.setattr(main, "extract_residue_position_map", lambda _path: {})
    monkeypatch.setattr(main, "BoltzClient", lambda **kwargs: MagicMock(api_key=kwargs["api_key"]))
    monkeypatch.setattr(main, "MoleculeGenerator", lambda **kwargs: generator_instance)

    close_mock = AsyncMock()
    monkeypatch.setattr(main, "close_cached_clients", close_mock)

    asyncio.run(main.main())

    generator_instance.run.assert_awaited_once()
    show_banner.assert_called_once_with()
    options = generator_instance.run.await_args.args[0]
    assert options.pdb_path == Path("protein.pdb")
    assert options.use_pocket_data is False
    assert options.target_chain_id == "A"
    close_mock.assert_awaited_once()


def test_main_closes_clients_on_generator_failure(
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`close_cached_clients` should execute even when generation fails."""

    parser = MagicMock()
    parser.parse_args.return_value = argparse.Namespace(
        pdb="protein.pdb",
        csv="few.csv",
        out="results",
        iters=2,
        samples=3,
        use_pocket=True,
    )

    generator_instance = MagicMock()
    generator_instance.run = AsyncMock(side_effect=RuntimeError("generation failed"))

    monkeypatch.setattr(main, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(main, "_build_parser", lambda _defaults: parser)
    monkeypatch.setattr(main, "_show_launch_banner", lambda: None)
    monkeypatch.setattr(main.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(main, "extract_sequence_from_pdb", lambda _path: "MKT")
    monkeypatch.setattr(main, "extract_target_chain_id", lambda _path: "A")
    monkeypatch.setattr(main, "extract_residue_position_map", lambda _path: {})
    monkeypatch.setattr(main, "BoltzClient", lambda **kwargs: MagicMock(api_key=kwargs["api_key"]))
    monkeypatch.setattr(main, "MoleculeGenerator", lambda **kwargs: generator_instance)

    close_mock = AsyncMock()
    monkeypatch.setattr(main, "close_cached_clients", close_mock)

    with pytest.raises(RuntimeError, match="generation failed"):
        asyncio.run(main.main())

    close_mock.assert_awaited_once()

"""CLI entry point for the molecular generation pipeline."""

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from time import perf_counter

from pydantic import ValidationError
from tqdm.auto import tqdm

from src.chemistry import (
    extract_residue_position_map,
    extract_sequence_from_pdb,
    extract_target_chain_id,
)
from src.clients import close_cached_clients
from src.core.config import get_settings
from src.core.constants import LOG_DIR_NAME
from src.core.exceptions import SequenceExtractionError
from src.core.logging import configure_logging, get_logger
from src.core.messages import (
    invalid_runtime_options,
    missing_environment_configuration,
    pdb_file_not_found,
)
from src.generator import MoleculeGenerator
from src.nvidia_client import BoltzClient
from src.schemas import PipelineOptions
from src.services import PubChemService

logger = get_logger(__name__)
BANNER_ENV_VAR = "LLMSFOLD_BANNER_SHOWN"


def _build_run_log_path(output_dir: str) -> Path:
    """Builds a unique per-run log file path under `<output_dir>/logs/`."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / LOG_DIR_NAME / f"run_{timestamp}.log"


def _format_duration(seconds: float) -> str:
    """Formats elapsed seconds as `HhMMmSSs` for human-readable summaries."""

    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _build_launch_banner() -> str:
    """Returns the startup banner shown at the beginning of pipeline launches."""

    return dedent(
        r"""
        +------------------------------------------------------------------------------+
        |  _      _     __  __      ______    _     _                                 |
        | | |    | |   |  \/  |    |  ____|  | |   | |                                |
        | | |    | |   | \  / |___ | |__ ___ | | __| |                                |
        | | |    | |   | |\/| / __||  __/ _ \| |/ _` |                                |
        | | |____| |___| |  | \__ \| | | (_) | | (_| |                                |
        | |______|______|_|  |_|___/|_|  \___/|_|\__,_|                                |
        |                                                                              |
        |                                  LLMsFold                                    |
        +------------------------------------------------------------------------------+
        Authors: Wageesha Widuranga, Davide Rigoni, Fabio Bove, Dario Righelli
                 Salvatore Romano, Rosa Visone, Marilena V. Iorio, Marco Grassia
                 Giuseppe Mangioni, Pietro Lio, Cristian Taccioli
        Groups : TaccLab   https://tacclab.org/
                 NeoraLab  https://www.neoralab.com/
        """
    ).strip("\n")


def _show_launch_banner() -> None:
    """Writes the startup banner once per launched process tree."""

    if os.environ.get(BANNER_ENV_VAR) == "1":
        return

    tqdm.write(_build_launch_banner())
    os.environ[BANNER_ENV_VAR] = "1"


def _build_parser(defaults: dict[str, str | int]) -> argparse.ArgumentParser:
    """Builds command line parser with settings-derived defaults."""

    parser = argparse.ArgumentParser(description="AI Molecular Generation Framework")
    parser.add_argument("--pdb", type=str, default=defaults["pdb"], help="Path to PDB file")
    parser.add_argument("--csv", type=str, default=defaults["csv"], help="Path to few-shot CSV")
    parser.add_argument("--out", type=str, default=defaults["out"], help="Output directory")
    parser.add_argument(
        "--iters", type=int, default=defaults["iters"], help="Number of RL iterations"
    )
    parser.add_argument(
        "--samples", type=int, default=defaults["samples"], help="Samples per iteration"
    )
    parser.add_argument(
        "--no-pocket",
        action="store_false",
        dest="use_pocket",
        help="Disable pocket-aware generation",
    )
    parser.set_defaults(use_pocket=True)
    return parser


async def main() -> None:
    """Runs CLI flow from argument parsing to report generation."""

    _show_launch_banner()
    configure_logging()

    try:
        settings = get_settings()
    except ValidationError as exc:
        logger.error(missing_environment_configuration())
        for error in exc.errors():
            logger.error("  - %s", ".".join(str(item) for item in error["loc"]))
        return

    configure_logging(settings.log_level)

    parser = _build_parser(
        {
            "pdb": str(settings.pdb_file),
            "csv": str(settings.few_shot_csv),
            "out": str(settings.output_dir),
            "iters": settings.max_iterations,
            "samples": settings.max_samples,
        }
    )
    args = parser.parse_args()

    run_log_path = _build_run_log_path(args.out)
    configure_logging(settings.log_level, log_file=run_log_path)
    logger.info("Logging run details to %s", run_log_path)

    if not os.path.exists(args.pdb):
        logger.error(pdb_file_not_found(args.pdb))
        return

    logger.info("Extracting sequence from %s...", args.pdb)
    try:
        protein_sequence = extract_sequence_from_pdb(args.pdb)
        target_chain_id = extract_target_chain_id(args.pdb)
        residue_position_map = extract_residue_position_map(args.pdb)
    except SequenceExtractionError as exc:
        logger.error("%s", exc)
        return
    logger.info(
        "Target chain for Boltz submission: %s (%s residue(s) mapped)",
        target_chain_id,
        len(residue_position_map),
    )

    try:
        options = PipelineOptions(
            pdb_path=args.pdb,
            few_shot_csv=args.csv,
            output_dir=args.out,
            protein_sequence=protein_sequence,
            max_iterations=args.iters,
            max_samples=args.samples,
            use_pocket_data=args.use_pocket,
            target_chain_id=target_chain_id,
            residue_position_map=residue_position_map,
        )
    except ValidationError as exc:
        logger.error(invalid_runtime_options())
        for error in exc.errors():
            logger.error("  - %s", ".".join(str(item) for item in error["loc"]))
        return

    boltz_client = BoltzClient(
        api_key=settings.nvidia_api_key.get_secret_value(),
        settings=settings,
    )
    generator = MoleculeGenerator(
        groq_api_key=settings.groq_api_key.get_secret_value(),
        boltz_client=boltz_client,
        pubchem_service=PubChemService(settings=settings),
        llm_model=settings.llm_model,
        llm_temperature=settings.llm_temperature,
        settings=settings,
    )

    run_started_at = perf_counter()
    run_started_wall = datetime.now()
    logger.info(
        "Run started at %s (pdb=%s, iters=%s, samples=%s, pocket=%s, out=%s, seed=%s)",
        run_started_wall.strftime("%Y-%m-%d %H:%M:%S"),
        args.pdb,
        args.iters,
        args.samples,
        args.use_pocket,
        args.out,
        settings.random_seed,
    )
    try:
        await generator.run(options)
    finally:
        elapsed_seconds = perf_counter() - run_started_at
        logger.info(
            "Run finished at %s, total elapsed %.2fs (%s)",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_seconds,
            _format_duration(elapsed_seconds),
        )
        await close_cached_clients()


if __name__ == "__main__":
    asyncio.run(main())

"""CLI entry point for the molecular generation pipeline."""

import argparse
import asyncio
import os
from textwrap import dedent

from pydantic import ValidationError
from tqdm.auto import tqdm

from src.chemistry import extract_sequence_from_pdb
from src.clients import close_cached_clients
from src.core.config import get_settings
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
        Authors: W. W. Waththe Liyanage, Fabio Bove, Dario Righelli, Salvatore Romano
                 Rosa Visone, Marilena V. Iorio, Pietro Lio, Cristian Taccioli
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

    if not os.path.exists(args.pdb):
        logger.error(pdb_file_not_found(args.pdb))
        return

    logger.info("Extracting sequence from %s...", args.pdb)
    try:
        protein_sequence = extract_sequence_from_pdb(args.pdb)
    except SequenceExtractionError as exc:
        logger.error("%s", exc)
        return

    try:
        options = PipelineOptions(
            pdb_path=args.pdb,
            few_shot_csv=args.csv,
            output_dir=args.out,
            protein_sequence=protein_sequence,
            max_iterations=args.iters,
            max_samples=args.samples,
            use_pocket_data=args.use_pocket,
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

    try:
        await generator.run(options)
    finally:
        await close_cached_clients()


if __name__ == "__main__":
    asyncio.run(main())

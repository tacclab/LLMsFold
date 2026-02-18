"""CLI entry point for the molecular generation pipeline."""

import argparse
import asyncio
import os

from pydantic import ValidationError

from src.chemistry import extract_sequence_from_pdb
from src.clients import close_cached_clients
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.generator import MoleculeGenerator
from src.nvidia_client import BoltzClient
from src.schemas import PipelineOptions

logger = get_logger(__name__)


def _build_parser(defaults: dict[str, str | int]) -> argparse.ArgumentParser:
    """Builds command line parser with settings-derived defaults."""

    parser = argparse.ArgumentParser(description="AI Molecular Generation Framework")
    parser.add_argument("--pdb", type=str, default=defaults["pdb"], help="Path to PDB file")
    parser.add_argument("--csv", type=str, default=defaults["csv"], help="Path to few-shot CSV")
    parser.add_argument("--out", type=str, default=defaults["out"], help="Output directory")
    parser.add_argument("--iters", type=int, default=defaults["iters"], help="Number of RL iterations")
    parser.add_argument("--samples", type=int, default=defaults["samples"], help="Samples per iteration")
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

    configure_logging()

    try:
        settings = get_settings()
    except ValidationError as exc:
        logger.error("Error: Missing required environment variables in `.env` or shell:")
        for error in exc.errors():
            logger.error("  - %s", ".".join(str(item) for item in error["loc"]))
        return

    configure_logging(settings.log_level)

    parser = _build_parser(
        {
            "pdb": settings.pdb_file,
            "csv": settings.few_shot_csv,
            "out": settings.output_dir,
            "iters": settings.max_iterations,
            "samples": settings.max_samples,
        }
    )
    args = parser.parse_args()

    if not os.path.exists(args.pdb):
        logger.error("Error: PDB file not found at %s", args.pdb)
        return

    logger.info("Extracting sequence from %s...", args.pdb)
    protein_sequence = extract_sequence_from_pdb(args.pdb)

    options = PipelineOptions(
        pdb_path=args.pdb,
        few_shot_csv=args.csv,
        output_dir=args.out,
        protein_sequence=protein_sequence,
        max_iterations=args.iters,
        max_samples=args.samples,
        use_pocket_data=args.use_pocket,
    )

    boltz_client = BoltzClient(api_key=settings.nvidia_api_key.get_secret_value())
    generator = MoleculeGenerator(
        groq_api_key=settings.groq_api_key.get_secret_value(),
        boltz_client=boltz_client,
        llm_model=settings.llm_model,
        llm_temperature=settings.llm_temperature,
    )

    try:
        await generator.run(options)
    finally:
        await close_cached_clients()


if __name__ == "__main__":
    asyncio.run(main())

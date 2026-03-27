"""Command-line interface for ASFAMProcessor."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from asfam import __version__
from asfam.config import ProcessingConfig
from asfam.pipeline.orchestrator import PipelineOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="asfam",
        description="ASFAMProcessor - ASFAM mass spectrometry data processor",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "mzml_files", nargs="+",
        help="Input mzML files (one per segment x replicate)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--library", default=None,
        help="Spectral library file (MGF/MSP) for low-response m/z inference",
    )
    parser.add_argument(
        "--config", default=None,
        help="Processing config JSON file (overrides defaults)",
    )
    parser.add_argument(
        "--mode", choices=["positive", "negative"], default="positive",
        help="Ionization mode (default: positive)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load or create config
    if args.config:
        config = ProcessingConfig.load(args.config)
    else:
        config = ProcessingConfig()

    config.ionization_mode = args.mode
    config.n_workers = args.workers

    # Validate inputs
    mzml_paths = []
    for p in args.mzml_files:
        path = Path(p)
        if not path.exists():
            print(f"Error: File not found: {p}", file=sys.stderr)
            sys.exit(1)
        mzml_paths.append(str(path))

    # Run pipeline
    def progress_cb(stage, current, total, msg):
        print(f"  [{stage}] {current}/{total} - {msg}")

    orchestrator = PipelineOrchestrator(config)
    orchestrator.add_progress_callback(progress_cb)

    try:
        features = orchestrator.run(mzml_paths, args.output, args.library)
        print(f"\nDone! {len(features)} features exported to {args.output}")
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as e:
        logging.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

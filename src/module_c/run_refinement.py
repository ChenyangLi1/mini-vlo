from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .refinement import (
    load_config,
    load_samples,
    refine_samples,
    save_results,
    save_results_pretty,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "refinement_output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module C refinement.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--input", required=True, help="Input JSONL samples.")
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output JSONL path. Defaults to "
            "results/refinement_output/refined_<timestamp>.jsonl"
        ),
    )
    parser.add_argument(
        "--motion-aggregation",
        choices=["min", "mean"],
        default="",
        help="Optional override for motion_quality.aggregation.",
    )
    parser.add_argument(
        "--pretty-output",
        default="",
        help="Optional pretty JSON path for human-readable inspection.",
    )
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"refined_{timestamp}.jsonl"
    )

    cfg = load_config(args.config)
    if args.motion_aggregation:
        cfg.motion_cfg.aggregation = args.motion_aggregation
    samples = load_samples(args.input)
    results = refine_samples(samples, cfg)
    save_results(results, output_path)
    if args.pretty_output:
        save_results_pretty(results, args.pretty_output)
    print(f"Processed {len(results)} samples -> {output_path}")
    if args.pretty_output:
        print(f"Pretty JSON saved to: {args.pretty_output}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Build a video-task manifest from paired Module D render outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


FIXED_VIDEO_NAME = "Fixed_View.mp4"
EGO_VIDEO_NAME = "Ego_View.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manifest for paired-view Module D render outputs."
    )
    parser.add_argument(
        "--input-dir",
        default="data/module_d_output",
        help="Directory containing one subdirectory per rendered action",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Manifest output path (defaults to <input-dir>/manifest.json)",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip sample directories with missing or ambiguous files",
    )
    return parser.parse_args()


def _manifest_path(path: Path, manifest_dir: Path) -> str:
    """Return a portable path relative to the generated manifest."""
    return Path(os.path.relpath(path, manifest_dir)).as_posix()


def _find_trajectory(sample_dir: Path) -> Path:
    expected = sample_dir / f"{sample_dir.name}_trajectory.json"
    if expected.is_file():
        return expected

    candidates = sorted(sample_dir.glob("*_trajectory.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("missing *_trajectory.json")
    raise ValueError(
        "multiple trajectory files found: "
        + ", ".join(path.name for path in candidates)
    )


def _build_sample(sample_dir: Path, manifest_dir: Path) -> dict[str, object]:
    fixed_video = sample_dir / FIXED_VIDEO_NAME
    ego_video = sample_dir / EGO_VIDEO_NAME
    missing = [
        path.name for path in (fixed_video, ego_video) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing " + ", ".join(missing))

    trajectory = _find_trajectory(sample_dir)
    return {
        "sample_id": sample_dir.name,
        "views": {
            "fixed": _manifest_path(fixed_video, manifest_dir),
            "ego": _manifest_path(ego_video, manifest_dir),
        },
        "trajectory": _manifest_path(trajectory, manifest_dir),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Module D output directory not found: {input_dir}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_dir / "manifest.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(
        (path for path in input_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {input_dir}")

    samples: list[dict[str, object]] = []
    invalid: list[str] = []
    for sample_dir in sample_dirs:
        try:
            samples.append(_build_sample(sample_dir, output_path.parent))
        except (FileNotFoundError, ValueError) as exc:
            invalid.append(f"{sample_dir.name}: {exc}")

    if invalid and not args.skip_invalid:
        details = "\n- ".join(invalid)
        raise RuntimeError(
            "Invalid Module D sample directories:\n"
            f"- {details}\n"
            "Fix these directories or pass --skip-invalid."
        )
    for diagnostic in invalid:
        print(f"Skipped {diagnostic}")
    if not samples:
        raise RuntimeError("No valid Module D samples found")

    manifest = {
        "dataset": "Module D",
        "sample_count": len(samples),
        "samples": samples,
    }
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(samples)} paired-view sample(s): {output_path}")


if __name__ == "__main__":
    main()

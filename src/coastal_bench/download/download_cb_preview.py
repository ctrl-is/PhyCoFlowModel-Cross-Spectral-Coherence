#!/usr/bin/env python3
"""Download a date-range subset of the CoastalBench Hugging Face preview.

Default: 18 days (~20 GB at ~1.14 GB/day). Bump --num-days toward 26 for ~30 GB.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


REPO_ID = "YupuZ/CoastalBench-preview"
DATE_FMT = "%Y%m%d"
AVG_GB_PER_DAY = 1.14


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_out = repo_root / "data" / "CoastalBench-preview"

    parser = argparse.ArgumentParser(
        description="Download a CoastalBench HF preview subset by date range."
    )
    parser.add_argument(
        "--start",
        default="20080102",
        help="Start date YYYYMMDD (default: 20080102)",
    )
    parser.add_argument(
        "--num-days",
        type=int,
        default=18,
        help="Number of consecutive days to download (default: 18 ≈ 20 GB)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=f"Local output directory (default: {default_out})",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Hugging Face dataset repo id (default: {REPO_ID})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Calendar-day gap between downloaded days (default: 1 = consecutive). "
            "Use --stride 3 to download every third day.",
    )
    return parser.parse_args()


def date_range(start: str, num_days: int, stride: int = 1) -> list[str]:
    if num_days < 1:
        raise ValueError("--num-days must be >= 1")
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    start_dt = datetime.strptime(start, DATE_FMT)
    return [(start_dt + timedelta(days=i * stride)).strftime(DATE_FMT) for i in range(num_days)]


def build_allow_patterns(days: list[str]) -> list[str]:
    patterns = ["README.md"]
    for day in days:
        patterns.append(f"ocean_met/{day}.zarr/**")
        patterns.append(f"river/{day}.zarr/**")
    return patterns


def print_du(path: Path) -> None:
    try:
        result = subprocess.run(
            ["du", "-sh", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"On-disk size: {result.stdout.strip()}")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Could not compute du -sh for {path}: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    days = date_range(args.start, args.num_days, args.stride)
    calendar_span = (args.num_days - 1) * args.stride + 1
    patterns = build_allow_patterns(days)
    est_gb = args.num_days * AVG_GB_PER_DAY

    print(f"Repo:          {args.repo_id}")
    print(f"Output:        {args.out_dir.resolve()}")
    print(f"Days:          {days[0]} .. {days[-1]} ({len(days)} days, stride={args.stride}, spanning {calendar_span} calendar days)")
    print(f"Est. size:     ~{est_gb:.1f} GB (avg ~{AVG_GB_PER_DAY} GB/day)")
    print(f"Allow patterns ({len(patterns)}):")
    for p in patterns:
        print(f"  {p}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.out_dir),
        allow_patterns=patterns,
    )

    print("\nDownload complete.")
    print_du(args.out_dir)
    ocean_count = sum(1 for d in days if (args.out_dir / "ocean_met" / f"{d}.zarr").exists())
    river_count = sum(1 for d in days if (args.out_dir / "river" / f"{d}.zarr").exists())
    print(f"ocean_met days present: {ocean_count}/{len(days)}")
    print(f"river days present:     {river_count}/{len(days)}")
    if shutil.which("du") is None:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

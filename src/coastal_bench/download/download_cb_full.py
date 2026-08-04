"""
CoastalBench Full-Precision Dataset Downloader (OSN S3 bucket).

Downloads specific daily NetCDF files from the CoastalBench Open Storage Network
bucket (udel1.osn.mghpcc.org/coastalbench-data).  Anonymous access — no
credentials required.

Usage
-----
  # Download four sample days (one per season) for structure inspection:
  python download_CB_heavy.py \\
      --output-dir /orcd/scratch/.../CoastalBench-heavy \\
      --dates 20080115 20080415 20080715 20081015

  # Download one day per month (full 2008 seasonal spread):
  python download_CB_heavy.py \\
      --output-dir /orcd/scratch/.../CoastalBench-heavy \\
      --dates 20080115 20080215 20080315 20080415 \\
              20080515 20080615 20080715 20080815 \\
              20080915 20081015 20081115 20081215

  # List all files in the bucket without downloading:
  python download_CB_heavy.py --list-only

  # Fallback: filter by glob pattern (downloads entire bucket first):
  python download_CB_heavy.py --file-pattern "2008*.nc" \\
      --output-dir /orcd/scratch/.../CoastalBench-heavy

Notes
-----
  * static.nc (~34 MB) is always downloaded — it contains the fixed mesh
    geometry and land-sea mask required by the NetCDF loader.
  * Each daily .nc file is ~4.3 GB.  Estimate ~4.4 GB × N + 0.04 GB for
    storage planning.
  * Requires boto3: conda install -n PhyCoEnv -c conda-forge boto3
"""

from __future__ import annotations

import argparse
import os
import sys
import fnmatch

import boto3
from botocore import UNSIGNED
from botocore.client import Config

BUCKET      = "coastalbench-data"
ENDPOINT    = "https://udel1.osn.mghpcc.org"
STATIC_KEY  = "static.nc"
AVG_GB_DAY  = 4.35


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def create_s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        config=Config(signature_version=UNSIGNED),
    )


def list_all_files(s3_client, bucket: str = BUCKET) -> list[str]:
    """Return every key in *bucket* (handles S3 pagination automatically)."""
    keys: list[str] = []
    kwargs: dict = {"Bucket": bucket}
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]
    return keys


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_keys(
    s3_client,
    keys: list[str],
    output_dir: str,
    bucket: str = BUCKET,
) -> None:
    """Download *keys* from *bucket* preserving their relative paths."""
    os.makedirs(output_dir, exist_ok=True)
    successful = failed = 0
    n = len(keys)
    print(f"\nDownloading {n} file(s) → {os.path.abspath(output_dir)}")
    print("-" * 60)
    for i, key in enumerate(keys, 1):
        local = os.path.join(output_dir, key)
        os.makedirs(os.path.dirname(local) or output_dir, exist_ok=True)
        try:
            s3_client.download_file(bucket, key, local)
            successful += 1
            print(f"  [{i}/{n}] ✓  {key}")
        except Exception as exc:
            failed += 1
            print(f"  [{i}/{n}] ✗  {key}  —  {exc}", file=sys.stderr)
    print("-" * 60)
    print(f"Done.  successful={successful}  failed={failed}  total={n}")
    print(f"Location: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download CoastalBench full-precision NetCDF files from OSN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output-dir", "-o",
        default="coastalbench_data",
        help="Local directory for downloaded files (default: %(default)s).",
    )
    # --- targeted date selection (recommended) ---
    p.add_argument(
        "--dates",
        nargs="+",
        metavar="YYYYMMDD",
        default=None,
        help=(
            "Explicit list of dates to download (YYYYMMDD format). "
            "Resolves to <date>.nc for each entry. "
            "static.nc is always appended. "
            "Mutually exclusive with --file-pattern."
        ),
    )
    # --- fallback glob selection ---
    p.add_argument(
        "--file-pattern",
        default=None,
        metavar="GLOB",
        help=(
            "Glob pattern to filter files after listing the full bucket "
            '(e.g. "2008*.nc"). Used only when --dates is not set.'
        ),
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Print the resolved file list without downloading.",
    )
    return p


def resolve_keys(args, s3_client) -> list[str]:
    """Return the ordered list of S3 keys to download based on CLI args."""
    if args.dates:
        # Build explicit list from --dates; always prepend static.nc
        keys = [STATIC_KEY]
        for date in args.dates:
            key = f"{date}.nc"
            if key not in keys:
                keys.append(key)
        return keys

    # Fall back to full bucket listing + optional glob filter
    all_keys = list_all_files(s3_client)
    if args.file_pattern:
        filtered = [k for k in all_keys if fnmatch.fnmatch(k, args.file_pattern)]
        # Ensure static.nc is included
        if STATIC_KEY not in filtered:
            filtered.insert(0, STATIC_KEY)
        return filtered
    return all_keys


def main() -> int:
    args = build_parser().parse_args()

    if args.dates and args.file_pattern:
        print("Error: --dates and --file-pattern are mutually exclusive.", file=sys.stderr)
        return 1

    print(f"Connecting to {ENDPOINT} ...")
    s3 = create_s3_client()

    keys = resolve_keys(args, s3)

    n_daily = sum(1 for k in keys if k != STATIC_KEY)
    est_gb  = n_daily * AVG_GB_DAY + 0.04
    print(f"Files selected : {len(keys)}  ({n_daily} daily + static.nc)")
    print(f"Est. size      : ~{est_gb:.1f} GB")
    print(f"Output dir     : {os.path.abspath(args.output_dir)}")

    if args.list_only:
        print("\nFile list (--list-only, no download):")
        for k in keys:
            print(f"  {k}")
        return 0

    download_keys(s3, keys, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
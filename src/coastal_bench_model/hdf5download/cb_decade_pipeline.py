"""
cb_decade_pipeline.py — Full download → carve → HDF5 pipeline for CoastalBench.

Generates a list of dates spread across the CoastalBench decade (2004–2013)
and orchestrates the two-step pipeline:

  Step 1: carve_cb_heavy.py   — download raw .nc from S3 + carve to subset
  Step 2: hdf5convert.py      — convert carved NC subset to HDF5

Date sampling strategies
------------------------
  monthly   — 1 date per month (the 15th) → 120 dates over 2008–2017
  biweekly  — 1st and 15th of every month → 240 dates
  weekly    — every 7 days from Jan 8 of each year → ~520 dates

Both scripts live beside this file in ``personal_files/``.  When run from
any directory, the paths are resolved relative to this file.

Usage
-----
  # Dry-run: print the two commands that would be executed
  python cb_decade_pipeline.py --pool-dir /path/to/pool --dry-run

  # Full monthly pipeline (120 dates, 2008–2017)
  python cb_decade_pipeline.py \\
      --pool-dir /path/to/pool \\
      --vars u v temp zeta \\
      --level 11 \\
      --spacing monthly \\
      --label 2008_2017monthly

  # Biweekly with a Mid-Atlantic bounding box
  python cb_decade_pipeline.py \\
      --pool-dir /path/to/pool \\
      --spacing biweekly \\
      --bbox -76 -70 35 42 \\
      --label 2008_2017biweekly_midatl

  # Only carve (skip HDF5 conversion)
  python cb_decade_pipeline.py --pool-dir /path/to/pool --skip-convert

  # Only convert (carved NC subset already exists)
  python cb_decade_pipeline.py \\
      --pool-dir /path/to/pool \\
      --skip-carve \\
      --input-dir /path/to/pool/cb_tuvz_L11_120d_2004_2013monthly

  # Restrict to a sub-range of years
  python cb_decade_pipeline.py \\
      --pool-dir /path/to/pool \\
      --years 2010 2014

Notes
-----
  * Both sub-scripts are invoked via subprocess with the same Python
    interpreter that runs this file (sys.executable), so they inherit the
    active conda/venv environment.
  * Pass --workers N to parallelise S3 downloads in carve_cb_heavy.py.
  * The auto-generated subset name mirrors carve_cb_heavy.py conventions:
      cb_<abbrevs>_L<level>_<n>d_<label>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths to the sibling scripts (resolved relative to this file)
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent
_CARVE_SCRIPT   = _HERE / "carve_cb_heavy.py"
_CONVERT_SCRIPT = _HERE / "hdf5convert.py"

# ---------------------------------------------------------------------------
# Variable abbreviation map (kept in sync with carve_cb_heavy.py)
# ---------------------------------------------------------------------------
_VAR_ABBREV = {
    "AKs":            "a",
    "lwrad":          "l",
    "Pair":           "P",
    "Qair":           "Q",
    "rain":           "r",
    "salt":           "s",
    "swrad":          "S",
    "Tair":           "T",
    "temp":           "t",
    "u":              "u",
    "Uwind":          "U",
    "v":              "v",
    "Vwind":          "V",
    "w":              "w",
    "wetdry_mask_rho": "d",
    "zeta":           "z",
}

DEFAULT_VARS  = ["u", "v", "temp", "zeta"]
DEFAULT_LEVEL = 11
DEFAULT_YEARS = (2008, 2017)

# CoastalBench S3 date coverage (inclusive)
CB_FIRST_DATE = date(2008, 1, 1)
CB_LAST_DATE  = date(2017, 12, 31)

# ---------------------------------------------------------------------------
# Date generation
# ---------------------------------------------------------------------------

def _monthly_dates(year_start: int, year_end: int) -> List[str]:
    """One date per month (the 15th), inclusive of both endpoint years."""
    dates: List[str] = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            d = date(year, month, 15)
            if CB_FIRST_DATE <= d <= CB_LAST_DATE:
                dates.append(d.strftime("%Y%m%d"))
    return sorted(dates)


def _biweekly_dates(year_start: int, year_end: int) -> List[str]:
    """1st and 15th of every month, inclusive of both endpoint years."""
    dates: List[str] = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            for day in (1, 15):
                d = date(year, month, day)
                if CB_FIRST_DATE <= d <= CB_LAST_DATE:
                    dates.append(d.strftime("%Y%m%d"))
    return sorted(dates)


def _weekly_dates(year_start: int, year_end: int) -> List[str]:
    """
    Every 7 days starting from January 8 of each year, clipped to CB range.
    The anchor day (Jan 8) keeps all dates well within each month.
    """
    dates: List[str] = []
    for year in range(year_start, year_end + 1):
        anchor = date(year, 1, 8)
        d = anchor
        while d.year == year:
            if CB_FIRST_DATE <= d <= CB_LAST_DATE:
                dates.append(d.strftime("%Y%m%d"))
            d += timedelta(days=7)
    return sorted(set(dates))


def _daily_dates(year_start: int, year_end: int) -> List[str]:
    """Every calendar day from Jan 1 of year_start to Dec 31 of year_end, clipped to CB range."""
    dates: List[str] = []
    d = date(year_start, 1, 1)
    end = date(year_end, 12, 31)
    while d <= end:
        if CB_FIRST_DATE <= d <= CB_LAST_DATE:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return dates


def generate_dates(
    spacing: str,
    year_start: int,
    year_end: int,
) -> List[str]:
    """Generate date strings (YYYYMMDD) for the requested spacing and year range."""
    if spacing == "monthly":
        return _monthly_dates(year_start, year_end)
    if spacing == "biweekly":
        return _biweekly_dates(year_start, year_end)
    if spacing == "weekly":
        return _weekly_dates(year_start, year_end)
    if spacing == "daily":
        return _daily_dates(year_start, year_end)
    raise ValueError(f"Unknown spacing {spacing!r}. Choose monthly, biweekly, weekly, or daily.")


# ---------------------------------------------------------------------------
# Subset-name helper (mirrors carve_cb_heavy.py logic)
# ---------------------------------------------------------------------------

def _auto_subset_name(
    var_names: List[str],
    level: int,
    n_dates: int,
    label: str,
) -> str:
    abbrevs = sorted(_VAR_ABBREV.get(v, v[0].lower()) for v in var_names)
    return f"cb_{''.join(abbrevs)}_L{level}_{n_dates}d_{label}"


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _build_carve_cmd(
    dates: List[str],
    var_names: List[str],
    level: int,
    pool_dir: str,
    subset_name: str,
    label: str,
    raw_dir: Optional[str],
    no_download: bool,
    keep_raw: bool,
    force: bool,
    workers: int,
) -> List[str]:
    cmd = [
        sys.executable, str(_CARVE_SCRIPT),
        "--dates", *dates,
        "--vars",  *var_names,
        "--level", str(level),
        "--pool-dir", pool_dir,
        "--subset-name", subset_name,
        "--label", label,
        "--workers", str(workers),
    ]
    if raw_dir:
        cmd += ["--raw-dir", raw_dir]
    if no_download:
        cmd.append("--no-download")
    if keep_raw:
        cmd.append("--keep-raw")
    if force:
        cmd.append("--force")
    return cmd


def _build_convert_cmd(
    input_dir: str,
    pool_dir: str,
    var_names: List[str],
    level: int,
    bbox: Optional[Tuple[float, float, float, float]],
    split_fracs: Tuple[float, float, float],
    force: bool,
    always_wet: bool = False,
    grid_bbox: Optional[Tuple[int, int, int, int]] = None,
    time_stride: int = 1,
    output_name: Optional[str] = None,
) -> List[str]:
    cmd = [
        sys.executable, str(_CONVERT_SCRIPT),
        "--input-dir", input_dir,
        "--pool-dir",  pool_dir,
        "--vars",      *var_names,
        "--level",     str(level),
        "--split-fracs", str(split_fracs[0]), str(split_fracs[1]), str(split_fracs[2]),
        "--time-stride", str(time_stride),
    ]
    if output_name is not None:
        cmd += ["--output-name", output_name]
    if grid_bbox is not None:
        cmd += ["--grid-bbox",
                str(grid_bbox[0]), str(grid_bbox[1]),
                str(grid_bbox[2]), str(grid_bbox[3])]
    elif bbox is not None:
        cmd += ["--bbox", str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]
    if always_wet:
        cmd.append("--always-wet")
    if force:
        cmd.append("--force")
    return cmd


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pool_dir: str,
    var_names: List[str],
    level: int,
    spacing: str,
    year_start: int,
    year_end: int,
    label: str,
    bbox: Optional[Tuple[float, float, float, float]],
    split_fracs: Tuple[float, float, float],
    input_dir: Optional[str],
    raw_dir: Optional[str],
    no_download: bool,
    keep_raw: bool,
    force: bool,
    workers: int,
    skip_carve: bool,
    skip_convert: bool,
    dry_run: bool,
    always_wet: bool = False,
    grid_bbox: Optional[Tuple[int, int, int, int]] = None,
    time_stride: int = 1,
    output_name: Optional[str] = None,
) -> int:
    """Orchestrate carve → convert for the full decade pipeline."""

    # ------------------------------------------------------------------
    # Date generation
    # ------------------------------------------------------------------
    dates = generate_dates(spacing, year_start, year_end)
    n_dates = len(dates)
    print(f"\nDate generation")
    print(f"  Spacing    : {spacing}")
    print(f"  Years      : {year_start}–{year_end}")
    print(f"  Dates      : {n_dates}")
    print(f"  First      : {dates[0]}")
    print(f"  Last       : {dates[-1]}")

    # ------------------------------------------------------------------
    # Subset name
    # ------------------------------------------------------------------
    subset_name = _auto_subset_name(var_names, level, n_dates, label)
    nc_subset_dir = str(Path(pool_dir) / subset_name)
    # hdf5convert.py auto-names: _grid_h5 > _bbox_h5 > _h5 (unless --output-name overrides)
    if output_name is not None:
        hdf5_name = output_name
    elif grid_bbox is not None:
        hdf5_name = subset_name + "_grid_h5"
    elif bbox is not None:
        hdf5_name = subset_name + "_bbox_h5"
    else:
        hdf5_name = subset_name + "_h5"
    hdf5_dir  = str(Path(pool_dir) / hdf5_name)

    print(f"\nSubset name    : {subset_name}")
    print(f"NC subset dir  : {nc_subset_dir}")
    print(f"HDF5 output    : {hdf5_dir}")
    if grid_bbox is not None:
        print(f"Bbox crop      : grid-index  eta[{grid_bbox[0]}..{grid_bbox[1]}]  "
              f"xi[{grid_bbox[2]}..{grid_bbox[3]}]  "
              f"→ {grid_bbox[1]-grid_bbox[0]+1}×{grid_bbox[3]-grid_bbox[2]+1} rectangle")
    elif bbox is not None:
        print(f"Bbox crop      : lon [{bbox[0]:.2f}, {bbox[1]:.2f}]  "
              f"lat [{bbox[2]:.2f}, {bbox[3]:.2f}]")

    # Resolve the NC input for the convert step
    convert_input = input_dir if input_dir else nc_subset_dir

    # ------------------------------------------------------------------
    # Step 1: carve
    # ------------------------------------------------------------------
    carve_cmd = _build_carve_cmd(
        dates       = dates,
        var_names   = var_names,
        level       = level,
        pool_dir    = pool_dir,
        subset_name = subset_name,
        label       = label,
        raw_dir     = raw_dir,
        no_download = no_download,
        keep_raw    = keep_raw,
        force       = force,
        workers     = workers,
    )

    print(f"\n{'='*60}")
    print(f"Step 1 — carve_cb_heavy.py")
    print(f"{'='*60}")
    carve_cmd_str = " \\\n    ".join(carve_cmd)
    print(f"Command:\n  {carve_cmd_str}\n")

    if skip_carve:
        print("  [skip] --skip-carve is set.")
    elif dry_run:
        print("  [dry-run] would execute the above command.")
    else:
        rc = subprocess.run(carve_cmd, check=False).returncode
        if rc != 0:
            print(f"\nError: carve_cb_heavy.py exited with code {rc}.", file=sys.stderr)
            return rc

    # ------------------------------------------------------------------
    # Step 2: convert
    # ------------------------------------------------------------------
    convert_cmd = _build_convert_cmd(
        input_dir   = convert_input,
        pool_dir    = pool_dir,
        var_names   = var_names,
        level       = level,
        bbox        = bbox,
        split_fracs = split_fracs,
        force       = force,
        always_wet  = always_wet,
        grid_bbox   = grid_bbox,
        time_stride = time_stride,
        output_name = output_name,
    )

    print(f"\n{'='*60}")
    print(f"Step 2 — hdf5convert.py")
    print(f"{'='*60}")
    convert_cmd_str = " \\\n    ".join(convert_cmd)
    print(f"Command:\n  {convert_cmd_str}\n")

    if skip_convert:
        print("  [skip] --skip-convert is set.")
    elif dry_run:
        print("  [dry-run] would execute the above command.")
    else:
        rc = subprocess.run(convert_cmd, check=False).returncode
        if rc != 0:
            print(f"\nError: hdf5convert.py exited with code {rc}.", file=sys.stderr)
            return rc

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    if dry_run:
        print(f"Dry-run complete.  Copy the commands above to run manually.")
    else:
        print(f"Pipeline complete.")
        if not skip_carve:
            print(f"  NC subset  : {nc_subset_dir}")
        if not skip_convert:
            print(f"  HDF5 file  : {hdf5_dir}/coastal_bench.h5")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full download→carve→HDF5 pipeline for the CoastalBench decade.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Output ---
    p.add_argument(
        "--pool-dir", "-o", required=True,
        help="Base pool directory.  NC and HDF5 subdirectories are created here.",
    )
    p.add_argument(
        "--output-name", default=None, metavar="NAME",
        help=(
            "Override the auto-generated HDF5 output directory name (relative to "
            "--pool-dir).  Forwarded as --output-name to hdf5convert.py.  "
            "Useful when re-converting from an existing NC subset with a different "
            "grid-bbox without clobbering the original HDF5.  "
            "Default: auto-generated from variable names, level, date count, and label."
        ),
    )

    # --- Variables and level ---
    p.add_argument(
        "--vars", nargs="+", default=DEFAULT_VARS,
        metavar="VAR",
        help=f"Variables to request (default: {DEFAULT_VARS}).",
    )
    p.add_argument(
        "--level", type=int, default=DEFAULT_LEVEL,
        help=f"Sigma level for 3-D variable slicing (default: {DEFAULT_LEVEL} = surface).",
    )

    # --- Date sampling ---
    p.add_argument(
        "--spacing", choices=["monthly", "biweekly", "weekly", "daily"], default="monthly",
        help=(
            "Date sampling strategy over the decade. "
            "monthly=120 dates, biweekly=240 dates, weekly≈520 dates, daily≈3650 dates. "
            "Default: monthly."
        ),
    )
    p.add_argument(
        "--years", nargs=2, type=int, default=list(DEFAULT_YEARS),
        metavar=("START", "END"),
        help=(
            f"Year range (inclusive). Default: {DEFAULT_YEARS[0]} {DEFAULT_YEARS[1]}. "
            f"Must be within {CB_FIRST_DATE.year}–{CB_LAST_DATE.year}."
        ),
    )

    # --- Label ---
    p.add_argument(
        "--label", default=None, metavar="TAG",
        help=(
            "Human-readable tag appended to the auto-generated subset name. "
            "Defaults to '<start>_<end><spacing>' e.g. '2004_2013monthly'."
        ),
    )

    # --- Spatial subregion ---
    p.add_argument(
        "--bbox", nargs=4, type=float, default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help=(
            "Geographic bounding box (decimal degrees) passed to hdf5convert.py. "
            "Example (Mid-Atlantic): --bbox -76 -70 35 42. "
            "Mutually exclusive with --grid-bbox."
        ),
    )
    p.add_argument(
        "--grid-bbox", nargs=4, type=int, default=None,
        metavar=("ETA_MIN", "ETA_MAX", "XI_MIN", "XI_MAX"),
        help=(
            "ROMS grid-index bounding box (0-based, inclusive) passed to hdf5convert.py. "
            "Produces a perfect rectangle in (eta, xi) space — non-skewed in imshow. "
            "Use visualize_cb_domain.py --grid-lines to read off the indices. "
            "Mutually exclusive with --bbox."
        ),
    )

    # --- HDF5 split fractions ---
    p.add_argument(
        "--split-fracs", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test fraction of dates for HDF5 splits. Default: 0.8 0.1 0.1.",
    )

    # --- Download control (forwarded to carve_cb_heavy.py) ---
    p.add_argument(
        "--input-dir", default=None,
        help=(
            "Path to a pre-existing carved NC subset directory. "
            "When set, skips the carve step (equivalent to --skip-carve) "
            "and passes this directory directly to hdf5convert.py."
        ),
    )
    p.add_argument(
        "--raw-dir", default=None,
        help="Directory of pre-downloaded full .nc files (forwarded to carve_cb_heavy.py).",
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Pass --no-download to carve_cb_heavy.py (requires --raw-dir).",
    )
    p.add_argument(
        "--keep-raw", action="store_true",
        help="Pass --keep-raw to carve_cb_heavy.py.",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel download workers forwarded to carve_cb_heavy.py (default: 1).",
    )

    # --- Temporal subsampling ---
    p.add_argument(
        "--time-stride", type=int, default=1, metavar="N",
        help=(
            "Pass --time-stride N to hdf5convert.py: store every Nth snapshot. "
            "T_stored = len(range(0, T_per_day, N)).  "
            "For 48 snapshots/day (30-min output): "
            "1→48 snaps (30-min, default); 2→24 snaps (1-hr); 4→12 snaps (2-hr). "
            "Independent of training-time time_stride in dataset.yaml. "
            "Default: 1 (store all snapshots)."
        ),
    )

    # --- Masking ---
    p.add_argument(
        "--always-wet", action="store_true",
        help=(
            "Pass --always-wet to hdf5convert.py: only retain ocean points that are "
            "wet (wetdry_mask_rho == 1) in EVERY carved daily file.  Requires "
            "'wetdry_mask_rho' to be present in --vars so it gets carved into the "
            "daily NC files.  The field is never written as a training channel."
        ),
    )

    # --- Pipeline control ---
    p.add_argument(
        "--skip-carve", action="store_true",
        help="Skip Step 1 (carve_cb_heavy.py). Use when NC subset already exists.",
    )
    p.add_argument(
        "--skip-convert", action="store_true",
        help="Skip Step 2 (hdf5convert.py). Only carve the NC subset.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Pass --force to both sub-scripts (re-carve / re-convert existing files).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    # Validate scripts exist
    for script, name in [(_CARVE_SCRIPT, "carve_cb_heavy.py"),
                         (_CONVERT_SCRIPT, "hdf5convert.py")]:
        if not script.exists():
            print(f"Error: {name} not found at {script}", file=sys.stderr)
            return 1

    # Validate years
    year_start, year_end = args.years
    if year_start > year_end:
        print("Error: START year must be <= END year.", file=sys.stderr)
        return 1
    if year_start < CB_FIRST_DATE.year or year_end > CB_LAST_DATE.year:
        print(
            f"Error: years must be within {CB_FIRST_DATE.year}–{CB_LAST_DATE.year}.",
            file=sys.stderr,
        )
        return 1

    # Validate bbox / grid_bbox (mutually exclusive)
    bbox: Optional[Tuple[float, float, float, float]] = None
    grid_bbox: Optional[Tuple[int, int, int, int]] = None

    if args.bbox is not None and args.grid_bbox is not None:
        print("Error: --bbox and --grid-bbox are mutually exclusive.", file=sys.stderr)
        return 1

    if args.bbox is not None:
        lon_min, lon_max, lat_min, lat_max = args.bbox
        if lon_min >= lon_max or lat_min >= lat_max:
            print("Error: bbox must have lon_min < lon_max and lat_min < lat_max.",
                  file=sys.stderr)
            return 1
        bbox = (lon_min, lon_max, lat_min, lat_max)

    if args.grid_bbox is not None:
        eta_min, eta_max, xi_min, xi_max = args.grid_bbox
        if eta_min >= eta_max:
            print("Error: --grid-bbox ETA_MIN must be less than ETA_MAX.", file=sys.stderr)
            return 1
        if xi_min >= xi_max:
            print("Error: --grid-bbox XI_MIN must be less than XI_MAX.", file=sys.stderr)
            return 1
        grid_bbox = (eta_min, eta_max, xi_min, xi_max)

    # Validate split fracs
    split_fracs = tuple(args.split_fracs)
    if abs(sum(split_fracs) - 1.0) > 1e-6:
        print(f"Error: --split-fracs must sum to 1.0 (got {split_fracs}).", file=sys.stderr)
        return 1

    # Auto-label
    label = args.label or f"{year_start}_{year_end}{args.spacing}"

    # If --input-dir is given, implicitly skip carve
    skip_carve = args.skip_carve or (args.input_dir is not None)

    return run_pipeline(
        pool_dir     = str(Path(args.pool_dir).resolve()),
        var_names    = list(args.vars),
        level        = args.level,
        spacing      = args.spacing,
        year_start   = year_start,
        year_end     = year_end,
        label        = label,
        bbox         = bbox,
        grid_bbox    = grid_bbox,
        split_fracs  = split_fracs,
        input_dir    = args.input_dir,
        raw_dir      = args.raw_dir,
        no_download  = args.no_download,
        keep_raw     = args.keep_raw,
        force        = args.force,
        workers      = args.workers,
        skip_carve   = skip_carve,
        skip_convert = args.skip_convert,
        dry_run      = args.dry_run,
        always_wet   = args.always_wet,
        time_stride  = args.time_stride,
        output_name  = args.output_name,
    )


if __name__ == "__main__":
    sys.exit(main())

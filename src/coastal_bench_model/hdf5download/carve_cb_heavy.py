"""
carve_cb_heavy.py — Download-and-carve CoastalBench daily NetCDF files.

For each requested date this script:
  1. Downloads the full ~4.3 GB daily .nc file from the OSN S3 bucket.
  2. Extracts only the requested variables.  3-D fields (u, v, temp, …) are
     sliced to a single sigma level (default: s=11, ROMS surface) so they are
     stored as (ocean_time, eta, xi) instead of
     (ocean_time, s, eta, xi).
  3. Writes a compressed NetCDF4 file to the pool directory.
  4. Removes the temporary full file (unless --keep-raw is set).

A carved static.nc is also produced from the full static.nc:
  keeps only lon, lat, mask_rho (the three arrays used by
  CoastalBenchNCDataset).

Typical savings
---------------
  Full daily file  : ~4.3 GB
  Carved (4 vars)  : ~30–80 MB  (deflate level 4, shuffling on)
  Speedup at load  : 10–40× (smaller file → faster seek + read)

Compatibility
-------------
  CoastalBenchNCDataset._read_fields auto-detects carved 3-D vars by
  checking ndim == 3 instead of 4, so no dataset-code change is needed
  beyond the already-merged ndim guard in helpers_cb.py.

Pool layout
----------
  --pool-dir is the BASE directory (e.g. /home/alexis31/orcd/pool/CoastalBench).
  Each invocation creates a named SUBDIRECTORY inside it:

    <pool-dir>/
      cb_tuvz_L11_12d_2008monthly/    ← subset A (4 vars, 12 days)
        static.nc
        20080115.nc  20080215.nc  ...
        carved_manifest.json
        subset_info.yaml
      cb_stuvz_L11_12d_2008monthly/   ← subset B (5 vars, adds salt)
        ...
      cb_tuvz_L11_48d_2008weekly/     ← subset C (4 vars, 48 days)
        ...

  The subdirectory name is auto-generated as
    cb_<sorted_abbrevs>_L<level>_<n>d_<label>
  or supplied explicitly with --subset-name.

Usage
-----
  # 4-field monthly subset for 2008 → pool/CoastalBench/cb_tuvz_L11_12d_2008monthly/
  python carve_cb_heavy.py \\
      --dates 20080115 20080215 20080315 20080415 \\
              20080515 20080615 20080715 20080815 \\
              20080915 20081015 20081115 20081215 \\
      --vars u v temp zeta \\
      --level 11 \\
      --pool-dir /home/alexis31/orcd/pool/CoastalBench \\
      --label 2008monthly

  # Explicit subset name (overrides auto-naming)
  python carve_cb_heavy.py \\
      --dates 20080115 20080415 20080715 20081015 \\
      --vars u v temp zeta \\
      --pool-dir /home/alexis31/orcd/pool/CoastalBench \\
      --subset-name cb_tuvz_L11_4d_seasonal

  # List available files without downloading
  python carve_cb_heavy.py --list-only

  # Carve from pre-downloaded raw files (skip S3)
  python carve_cb_heavy.py \\
      --dates 20080115 20080415 \\
      --vars u v temp zeta \\
      --raw-dir /orcd/scratch/orcd/010/alexis31/CoastalBench-heavy \\
      --pool-dir /home/alexis31/orcd/pool/CoastalBench \\
      --label 2008seasonal \\
      --no-download

Notes
-----
  * Requires boto3, netCDF4, numpy:
      conda install -n PhyCoEnv -c conda-forge boto3 netCDF4 numpy
  * Each subset directory contains carved_manifest.json (tracks carved dates)
    and subset_info.yaml (human-readable summary + YAML snippet for training).
  * Re-running is safe: already-carved dates are skipped unless --force.
  * static.nc is carved once per subset and cached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# S3 constants (identical to download_CB_heavy.py)
# ---------------------------------------------------------------------------
BUCKET    = "coastalbench-data"
ENDPOINT  = "https://udel1.osn.mghpcc.org"
STATIC_KEY = "static.nc"
AVG_GB_DAY = 4.35

# ---------------------------------------------------------------------------
# Variable sets (must match CB_3D_VARS / CB_2D_VARS in helpers_cb.py)
# ---------------------------------------------------------------------------
CB_3D_VARS = frozenset({"u", "v", "w", "temp", "salt", "AKs"})
CB_2D_VARS = frozenset({"zeta", "Pair", "Qair", "Tair", "rain",
                         "swrad", "lwrad", "Uwind", "Vwind",
                         "wetdry_mask_rho"})
ALL_CB_VARS = CB_3D_VARS | CB_2D_VARS

DEFAULT_VARS  = ["u", "v", "temp", "zeta"]
DEFAULT_LEVEL = 11   # ROMS surface: s=11 in a 12-level model (0=bottom)

# Compression parameters for netCDF4 output
_DEFLATE_LEVEL   = 4
_SHUFFLE         = True

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        config=Config(signature_version=UNSIGNED),
    )


def _list_bucket(s3) -> List[str]:
    keys: List[str] = []
    kwargs: dict = {"Bucket": BUCKET}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]
    return keys


def _download_key(s3, key: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    s3.download_file(BUCKET, key, dest)

# ---------------------------------------------------------------------------
# Carving helpers
# ---------------------------------------------------------------------------

def _copy_static(src: str, dst: str) -> None:
    """
    Copy static.nc as-is into the subset directory.

    The file is only ~34 MB so a full copy is far simpler and more robust
    than trying to selectively extract variables with netCDF4 (which can
    hit _FillValue attribute ordering issues).  CoastalBenchNCDataset only
    reads lon, lat, and mask_rho — all present in the original file.
    """
    import shutil
    shutil.copy2(src, dst)
    print(f"  copied static → {dst}  ({os.path.getsize(dst) / 1e6:.1f} MB)")


def _carve_daily(
    src: str,
    dst: str,
    var_names: Sequence[str],
    level: int,
) -> Tuple[int, int]:
    """
    Carve one daily YYYYMMDD.nc file.

    3-D vars (u, v, temp …) are sliced to *level* along the s axis,
    producing (ocean_time, eta, xi) instead of
    (ocean_time, s, eta, xi).

    2-D vars (zeta …) are copied unchanged.

    Returns (size_in_bytes, size_out_bytes).
    """
    import netCDF4 as nc

    size_in = os.path.getsize(src)

    with nc.Dataset(src, "r") as ds_in, \
         nc.Dataset(dst, "w", format="NETCDF4") as ds_out:

        # ------------------------------------------------------------------
        # Dimensions: ocean_time, eta, xi  (s NOT needed — level already extracted)
        # ------------------------------------------------------------------
        needed_dims = {"ocean_time", "eta", "xi"}
        for dim_name in needed_dims:
            if dim_name not in ds_in.dimensions:
                raise KeyError(
                    f"Dimension '{dim_name}' missing from source file {src}."
                )
            dim = ds_in.dimensions[dim_name]
            ds_out.createDimension(dim_name, None if dim.isunlimited() else len(dim))

        # ocean_time coordinate variable (keeps actual datetime values)
        if "ocean_time" in ds_in.variables:
            t_in  = ds_in.variables["ocean_time"]
            t_out = ds_out.createVariable(
                "ocean_time", t_in.dtype, ("ocean_time",),
                zlib=True, complevel=_DEFLATE_LEVEL,
            )
            t_out[:] = t_in[:]
            for attr in t_in.ncattrs():
                t_out.setncattr(attr, t_in.getncattr(attr))

        # ------------------------------------------------------------------
        # Variables
        # ------------------------------------------------------------------
        for vname in var_names:
            if vname not in ds_in.variables:
                print(f"  [warn] variable '{vname}' not found in {src}, skipping.")
                continue

            v_in = ds_in.variables[vname]
            in_dims = v_in.dimensions   # e.g. ('ocean_time', 's', 'eta', 'xi')
            is_3d   = vname in CB_3D_VARS and "s" in in_dims

            if is_3d:
                # Slice to surface level → (ocean_time, eta, xi)
                s_axis   = list(in_dims).index("s")
                out_dims = tuple(d for d in in_dims if d != "s")
                v_out    = ds_out.createVariable(
                    vname, "f4", out_dims,
                    zlib=True, complevel=_DEFLATE_LEVEL, shuffle=_SHUFFLE,
                )
                # Read and slice in one shot (avoids loading full s dim)
                idx  = [slice(None)] * v_in.ndim
                idx[s_axis] = level
                data = v_in[tuple(idx)][:]   # (ocean_time, eta, xi)
                v_out[:] = data.astype("f4")
            else:
                # 2-D var: copy as-is
                v_out = ds_out.createVariable(
                    vname, "f4", v_in.dimensions,
                    zlib=True, complevel=_DEFLATE_LEVEL, shuffle=_SHUFFLE,
                )
                v_out[:] = v_in[:].astype("f4")

            # Preserve metadata (units, long_name, …)
            for attr in v_in.ncattrs():
                if attr == "_FillValue":
                    continue   # already set implicitly by netCDF4
                v_out.setncattr(attr, v_in.getncattr(attr))

        # ------------------------------------------------------------------
        # Global attributes
        # ------------------------------------------------------------------
        ds_out.title = (
            f"CoastalBench carved — vars={list(var_names)} level={level}"
        )
        ds_out.carved_vars  = " ".join(var_names)
        ds_out.carved_level = level

    size_out = os.path.getsize(dst)
    return size_in, size_out

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
_MANIFEST_NAME = "carved_manifest.json"


def _load_manifest(pool_dir: str) -> dict:
    path = os.path.join(pool_dir, _MANIFEST_NAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"carved": {}, "static_carved": False}


def _save_manifest(pool_dir: str, manifest: dict) -> None:
    path = os.path.join(pool_dir, _MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def process_date(
    date: str,
    pool_dir: str,
    var_names: List[str],
    level: int,
    raw_dir: Optional[str],
    no_download: bool,
    keep_raw: bool,
    force: bool,
    manifest: dict,
    s3=None,
) -> None:
    """Download (if needed) and carve one daily file."""
    key       = f"{date}.nc"
    dst_path  = os.path.join(pool_dir, key)

    # Skip already-carved dates
    entry     = manifest["carved"].get(date, {})
    if not force and entry.get("done") and os.path.exists(dst_path):
        print(f"  [skip] {date} already carved — use --force to re-carve")
        return

    # ------------------------------------------------------------------
    # Step 1: obtain raw file path
    # ------------------------------------------------------------------
    if raw_dir:
        raw_path = os.path.join(raw_dir, key)
        if os.path.exists(raw_path):
            # Pre-existing raw file — use it directly, never delete it.
            tmp_path     = None
            delete_after = False
        elif no_download:
            raise FileNotFoundError(
                f"Raw file not found at {raw_path}. "
                "Pass --no-download only when raw files already exist."
            )
        else:
            # Raw file not yet present; download it into raw_dir.
            os.makedirs(raw_dir, exist_ok=True)
            tmp_path     = raw_path + ".partial"
            delete_after = not keep_raw
            print(f"  [download] {key} (~{AVG_GB_DAY:.1f} GB) …", end="", flush=True)
            try:
                _download_key(s3, key, tmp_path)
                os.replace(tmp_path, raw_path)
            except Exception as exc:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                print(f" FAILED: {exc}", file=sys.stderr)
                return
            print(f" done ({os.path.getsize(raw_path)/1e9:.2f} GB)")
            tmp_path = raw_path  # raw_path is now the real file; deleted below if delete_after
    elif no_download:
        raise ValueError(
            f"--no-download set but no --raw-dir provided for {date}."
        )
    else:
        # Download to a temporary file so that a failed download never
        # leaves a partial file in pool_dir.
        fd, tmp_path = tempfile.mkstemp(suffix=f"_{date}.nc",
                                        dir=pool_dir, prefix=".tmp_")
        os.close(fd)
        raw_path     = tmp_path
        delete_after = not keep_raw

        print(f"  [download] {key} (~{AVG_GB_DAY:.1f} GB) …", end="", flush=True)
        try:
            _download_key(s3, key, raw_path)
        except Exception as exc:
            os.unlink(raw_path)
            print(f" FAILED: {exc}", file=sys.stderr)
            return
        print(f" done ({os.path.getsize(raw_path)/1e9:.2f} GB)")

    # ------------------------------------------------------------------
    # Step 2: carve
    # ------------------------------------------------------------------
    print(f"  [carve]    {date} vars={var_names} level={level} …", end="", flush=True)
    tmp_dst = dst_path + ".partial"
    try:
        size_in, size_out = _carve_daily(raw_path, tmp_dst, var_names, level)
        os.replace(tmp_dst, dst_path)
    except Exception as exc:
        if os.path.exists(tmp_dst):
            os.unlink(tmp_dst)
        print(f" FAILED: {exc}", file=sys.stderr)
        return
    finally:
        if delete_after and tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    ratio = size_in / max(size_out, 1)
    print(f" done  {size_in/1e6:.0f} MB → {size_out/1e6:.0f} MB  "
          f"({ratio:.1f}× smaller)")

    manifest["carved"][date] = {
        "done":      True,
        "vars":      var_names,
        "level":     level,
        "size_in_mb":  round(size_in  / 1e6, 1),
        "size_out_mb": round(size_out / 1e6, 1),
    }


def process_static(
    pool_dir: str,
    raw_dir: Optional[str],
    no_download: bool,
    keep_raw: bool,
    force: bool,
    manifest: dict,
    s3=None,
) -> None:
    """Download (if needed) and carve static.nc."""
    dst_path = os.path.join(pool_dir, STATIC_KEY)

    if not force and manifest.get("static_carved") and os.path.exists(dst_path):
        print(f"  [skip] static.nc already carved")
        return

    if raw_dir:
        raw_path = os.path.join(raw_dir, STATIC_KEY)
        if os.path.exists(raw_path):
            delete_after = False
            tmp_path     = None
        elif no_download:
            raise FileNotFoundError(f"static.nc not found at {raw_path}. "
                                    "Pass --no-download only when raw files already exist.")
        else:
            os.makedirs(raw_dir, exist_ok=True)
            tmp_path     = raw_path + ".partial"
            delete_after = not keep_raw
            print(f"  [download] static.nc …", end="", flush=True)
            try:
                _download_key(s3, STATIC_KEY, tmp_path)
                os.replace(tmp_path, raw_path)
            except Exception as exc:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                print(f" FAILED: {exc}", file=sys.stderr)
                return
            print(f" done ({os.path.getsize(raw_path)/1e9:.2f} GB)")
            tmp_path = raw_path  # raw_path is now the real file; deleted below if delete_after
    elif no_download:
        raise ValueError("--no-download set but --raw-dir not provided for static.nc")
    else:
        fd, tmp_path = tempfile.mkstemp(suffix="_static.nc",
                                        dir=pool_dir, prefix=".tmp_")
        os.close(fd)
        raw_path     = tmp_path
        delete_after = not keep_raw

        print(f"  [download] static.nc …", end="", flush=True)
        try:
            _download_key(s3, STATIC_KEY, raw_path)
        except Exception as exc:
            os.unlink(raw_path)
            print(f" FAILED: {exc}", file=sys.stderr)
            return
        print(f" done ({os.path.getsize(raw_path)/1e6:.1f} MB)")

    print(f"  [copy]     static.nc …", end="", flush=True)
    tmp_dst = dst_path + ".partial"
    try:
        _copy_static(raw_path, tmp_dst)
        os.replace(tmp_dst, dst_path)
    except Exception as exc:
        if os.path.exists(tmp_dst):
            os.unlink(tmp_dst)
        print(f" FAILED: {exc}", file=sys.stderr)
        return
    finally:
        if delete_after and tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    manifest["static_carved"] = True


# ---------------------------------------------------------------------------
# Subset naming and info helpers
# ---------------------------------------------------------------------------

# One-letter abbreviations used in auto-generated subset names.
# Sorted alphabetically by abbreviation → canonical regardless of arg order.
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


def _auto_subset_name(var_names: List[str], level: int, n_dates: int, label: str) -> str:
    """
    Build a canonical subset directory name from carving parameters.

    Format:  cb_<sorted_abbrevs>_L<level>_<n>d_<label>
    Example: cb_tuvz_L11_12d_2008monthly

    Abbreviations are sorted so the name is identical regardless of the order
    in which --vars are supplied.
    """
    abbrevs = sorted(_VAR_ABBREV.get(v, v[0].lower()) for v in var_names)
    abbrev_str = "".join(abbrevs)
    return f"cb_{abbrev_str}_L{level}_{n_dates}d_{label}"


def _write_subset_info(
    subset_dir: str,
    subset_name: str,
    base_pool_dir: str,
    var_names: List[str],
    level: int,
    dates: List[str],
    total_size_mb: float,
) -> None:
    """
    Write subset_info.yaml into the subset directory.

    Contains all parameters needed to reproduce the subset, plus the exact
    YAML snippet to paste into a training config.
    """
    from datetime import datetime as _dt
    import yaml as _yaml

    subset_path = os.path.abspath(subset_dir)
    info = {
        "subset_name":  subset_name,
        "vars":         list(var_names),
        "level":        int(level),
        "n_dates":      len(dates),
        "dates":        sorted(dates),
        "created":      _dt.now().isoformat(timespec="seconds"),
        "base_pool_dir": os.path.abspath(base_pool_dir),
        "subset_dir":   subset_path,
        "size_mb_total": round(total_size_mb, 1),
        "yaml_snippet": {
            "data":           subset_path,
            "static_nc":      os.path.join(subset_path, "static.nc"),
            "cb_var_names":   list(var_names),
            "cb_var_level":   int(level),
        },
    }

    out_path = os.path.join(subset_dir, "subset_info.yaml")
    with open(out_path, "w") as f:
        _yaml.dump(info, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  wrote subset_info.yaml → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download and carve CoastalBench daily NetCDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Selection ---
    p.add_argument(
        "--dates", nargs="+", metavar="YYYYMMDD", default=None,
        help="Dates to process (YYYYMMDD). static.nc is always included.",
    )
    p.add_argument(
        "--list-only", action="store_true",
        help="List available bucket files without downloading.",
    )

    # --- Variables and level ---
    p.add_argument(
        "--vars", nargs="+", default=DEFAULT_VARS,
        choices=sorted(ALL_CB_VARS),
        metavar="VAR",
        help=(
            f"Variables to keep (default: {DEFAULT_VARS}). "
            "Choices: " + " ".join(sorted(ALL_CB_VARS))
        ),
    )
    p.add_argument(
        "--level", type=int, default=DEFAULT_LEVEL,
        help=(
            f"Sigma level to extract for 3-D variables (default: {DEFAULT_LEVEL} = ROMS surface "
            "in a 12-level model where s=0 is bottom)."
        ),
    )

    # --- Paths ---
    p.add_argument(
        "--pool-dir", "-o", default="coastalbench_carved",
        help=(
            "Base pool directory. Carved files are written into a named "
            "subdirectory inside this directory (see --subset-name / --label). "
            "Default: %(default)s."
        ),
    )
    p.add_argument(
        "--subset-name", default=None, metavar="NAME",
        help=(
            "Explicit name for the subset subdirectory inside --pool-dir. "
            "Auto-generated as cb_<abbrevs>_L<level>_<n>d_<label> if omitted."
        ),
    )
    p.add_argument(
        "--label", default="custom", metavar="TAG",
        help=(
            'Human-readable tag appended to the auto-generated subset name '
            '(e.g. "2008monthly", "2008weekly"). Ignored when --subset-name is '
            'set. Default: "%(default)s".'
        ),
    )
    p.add_argument(
        "--raw-dir", default=None,
        help=(
            "Directory containing pre-downloaded full .nc files. "
            "When set, skips S3 download and carves from these files directly."
        ),
    )

    # --- Download control ---
    p.add_argument(
        "--no-download", action="store_true",
        help="Do not contact S3 at all. Requires --raw-dir.",
    )
    p.add_argument(
        "--keep-raw", action="store_true",
        help=(
            "Keep the temporary full download in pool-dir after carving "
            "(useful for debugging; files are named .tmp_YYYYMMDD.nc)."
        ),
    )

    # --- Misc ---
    p.add_argument(
        "--force", action="store_true",
        help="Re-carve even if the manifest says a date is already done.",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel download+carve workers (default: 1). "
            "Each worker downloads and carves one date at a time. "
            "Set > 1 only if your network is the bottleneck; carving itself "
            "is CPU-bound and benefits from parallelism."
        ),
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.list_only:
        print(f"Connecting to {ENDPOINT} …")
        s3   = _s3_client()
        keys = _list_bucket(s3)
        print(f"\nFiles in bucket ({len(keys)} total):")
        for k in keys:
            print(f"  {k}")
        return 0

    if not args.dates:
        print("Error: --dates is required (or --list-only).", file=sys.stderr)
        return 1

    # Validate variable names
    unknown = set(args.vars) - ALL_CB_VARS
    if unknown:
        print(f"Error: unknown variable(s): {unknown}", file=sys.stderr)
        print(f"  Valid: {sorted(ALL_CB_VARS)}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Resolve the subset subdirectory
    # ------------------------------------------------------------------
    base_pool_dir = os.path.abspath(args.pool_dir)
    n_dates       = len(args.dates)

    subset_name = (
        args.subset_name
        if args.subset_name
        else _auto_subset_name(args.vars, args.level, n_dates, args.label)
    )
    pool_dir = os.path.join(base_pool_dir, subset_name)
    os.makedirs(pool_dir, exist_ok=True)
    manifest = _load_manifest(pool_dir)

    s3 = None
    if not args.no_download:
        print(f"Connecting to {ENDPOINT} …")
        s3 = _s3_client()

    est_raw    = n_dates * AVG_GB_DAY
    est_carved = n_dates * AVG_GB_DAY * 0.015

    print(f"\nSubset name        : {subset_name}")
    print(f"Dates to process   : {n_dates}")
    print(f"Variables          : {args.vars}")
    print(f"Sigma level        : {args.level}  (0=bottom, {DEFAULT_LEVEL}=surface)")
    print(f"Base pool dir      : {base_pool_dir}")
    print(f"Subset directory   : {pool_dir}")
    if not args.no_download:
        print(f"Est. raw download  : ~{est_raw:.1f} GB  (temporary, deleted after carving)")
    print(f"Est. carved output : ~{est_carved:.1f} GB")
    print()

    # ------------------------------------------------------------------
    # Static.nc
    # ------------------------------------------------------------------
    print("[static.nc]")
    process_static(
        pool_dir    = pool_dir,
        raw_dir     = args.raw_dir,
        no_download = args.no_download,
        keep_raw    = args.keep_raw,
        force       = args.force,
        manifest    = manifest,
        s3          = s3,
    )
    _save_manifest(pool_dir, manifest)

    # ------------------------------------------------------------------
    # Daily files (sequential or parallel)
    # ------------------------------------------------------------------
    print(f"\n[daily files — {n_dates} date(s)]")

    if args.workers <= 1:
        for date in args.dates:
            process_date(
                date        = date,
                pool_dir    = pool_dir,
                var_names   = list(args.vars),
                level       = args.level,
                raw_dir     = args.raw_dir,
                no_download = args.no_download,
                keep_raw    = args.keep_raw,
                force       = args.force,
                manifest    = manifest,
                s3          = s3,
            )
            _save_manifest(pool_dir, manifest)
    else:
        # Parallel workers — each worker gets its own S3 client
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _worker(date: str) -> str:
            worker_s3 = None if args.no_download else _s3_client()
            m = {"carved": {}, "static_carved": False}
            process_date(
                date        = date,
                pool_dir    = pool_dir,
                var_names   = list(args.vars),
                level       = args.level,
                raw_dir     = args.raw_dir,
                no_download = args.no_download,
                keep_raw    = args.keep_raw,
                force       = args.force,
                manifest    = m,
                s3          = worker_s3,
            )
            return date, m.get("carved", {}).get(date, {})

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, d): d for d in args.dates}
            for fut in as_completed(futures):
                date_done, entry = fut.result()
                if entry:
                    manifest["carved"][date_done] = entry
                _save_manifest(pool_dir, manifest)

    # ------------------------------------------------------------------
    # Write subset_info.yaml
    # ------------------------------------------------------------------
    done       = [d for d, e in manifest["carved"].items() if e.get("done")]
    total_in   = sum(e.get("size_in_mb",  0) for e in manifest["carved"].values())
    total_out  = sum(e.get("size_out_mb", 0) for e in manifest["carved"].values())

    print(f"\n[subset_info.yaml]")
    _write_subset_info(
        subset_dir    = pool_dir,
        subset_name   = subset_name,
        base_pool_dir = base_pool_dir,
        var_names     = list(args.vars),
        level         = args.level,
        dates         = done,
        total_size_mb = total_out,
    )

    # ------------------------------------------------------------------
    # Summary + YAML snippet
    # ------------------------------------------------------------------
    var_list_yaml = str(list(args.vars))

    print(f"\n{'='*60}")
    print(f"Carved {len(done)}/{n_dates} date(s) successfully.")
    if total_in > 0:
        print(f"Total: {total_in:.0f} MB raw → {total_out:.0f} MB carved  "
              f"({total_in/max(total_out,1):.1f}× compression)")
    print(f"Subset directory : {pool_dir}")
    print(f"Manifest         : {os.path.join(pool_dir, _MANIFEST_NAME)}")
    print(f"Subset info      : {os.path.join(pool_dir, 'subset_info.yaml')}")
    print(
        f"\n{'─'*60}\n"
        f"To use this subset, copy these lines into your training YAML:\n"
        f"{'─'*60}\n"
        f"data:           \"{pool_dir}\"\n"
        f"static_nc:      \"{os.path.join(pool_dir, 'static.nc')}\"\n"
        f"cb_var_names:   {var_list_yaml}\n"
        f"cb_var_level:   {args.level}\n"
        f"{'─'*60}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

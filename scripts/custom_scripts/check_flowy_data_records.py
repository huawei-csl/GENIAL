#!/usr/bin/env python3
"""
Scan synth_out/res_* folders, read flowy_data_record.parquet, and report:
- number of unique run_identifier values (per file + histogram)
- maximum step value (per file + global max)
- number of res_* folders found

Usage:
  python scan_flowy_data_record.py
  python scan_flowy_data_record.py --base /path/to/run
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from collections import Counter

import pandas as pd

# get env var
DATA_DIR = os.getenv("DATA_DIR")

DEFAULT_BASE = (
    f"{DATA_DIR}/output/"
    "multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0"
)


def text_hist(counter: Counter[int], *, title: str, bar_width: int = 40) -> str:
    if not counter:
        return f"{title}\n  (empty)\n"

    items = sorted(counter.items(), key=lambda kv: kv[0])
    max_count = max(counter.values())

    lines = [title]
    for k, c in items:
        bar_len = int(round((c / max_count) * bar_width)) if max_count > 0 else 0
        bar = "#" * bar_len
        bar = ''
        lines.append(f"  {k:>6}: {c:>6} occurrences {bar}")
    return "\n".join(lines) + "\n"


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=str,
        default=DEFAULT_BASE,
        help="Base folder containing synth_out/ (default is your GENIAL run path).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="synth_out/res_*",
        help="Glob pattern under base to find result folders.",
    )
    ap.add_argument(
        "--parquet-name",
        type=str,
        default="flowy_data_record.parquet",
        help="Parquet filename to read inside each res_* folder.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Only print aggregate statistics (no per-folder lines).",
    )
    args = ap.parse_args()

    base = Path(args.base).expanduser()
    res_glob = str(base / args.pattern)
    res_dirs = [Path(p) for p in sorted(glob.glob(res_glob)) if Path(p).is_dir()]

    # Always print number of folders found (as requested)
    print(f"Found {len(res_dirs)} res_* folders matching: {res_glob}")

    if not res_dirs:
        print(f"[ERROR] No directories matched: {res_glob}")
        return 2

    uniq_hist: Counter[int] = Counter()
    maxstep_hist: Counter[int] = Counter()
    rowcount_hist: Counter[int] = Counter()
    global_max_step: int | None = None

    missing_files = 0
    bad_files = 0
    processed = 0

    if not args.quiet:
        print(f"Base: {base}")
        print("-" * 80)
        
    # Aggregate accumulators across all rows / all files
    all_aig = []
    all_tr = []

    for d in res_dirs:
        pq = d / args.parquet_name
        if not pq.exists():
            missing_files += 1
            if not args.quiet:
                print(f"{d.name}: MISSING {args.parquet_name}")
            continue

        try:
            # read only needed columns (fast)
                        # read only needed columns (fast)
            df = pd.read_parquet(
                pq,
                columns=["run_identifier", "step", "aig_count", "nb_transistors"],
            )
        except Exception as e:
            bad_files += 1
            if not args.quiet:
                print(f"{d.name}: ERROR reading parquet: {e}")
            continue

        if "run_identifier" not in df.columns or "step" not in df.columns or "aig_count" not in df.columns or "nb_transistors" not in df.columns:
            bad_files += 1
            if not args.quiet:
                print(f"{d.name}: ERROR missing required columns in parquet")
            continue

        rowcount = int(len(df))
        uniq = int(df["run_identifier"].nunique(dropna=True))

        step_series = pd.to_numeric(df["step"], errors="coerce")
        if step_series.notna().any():
            max_step = int(step_series.max())
        else:
            max_step = None

        uniq_hist[uniq] += 1
        rowcount_hist[rowcount] += 1
        if max_step is not None:
            maxstep_hist[max_step] += 1
            global_max_step = max_step if global_max_step is None else max(global_max_step, max_step)
            
        aig = _to_num(df["aig_count"])
        tr = _to_num(df["nb_transistors"])
        
        # accumulate for global stats (ignore NaNs)
        
        all_aig.extend(aig.dropna().tolist())        
        all_tr.extend(tr.dropna().tolist())

        processed += 1

        if not args.quiet:
            ms = "NA" if max_step is None else str(max_step)
            print(f"{d.name}: unique(run_identifier)={uniq:4d} | max(step)={ms}")

    if not args.quiet:
        print("-" * 80)

    print(f"Processed: {processed}/{len(res_dirs)}")
    if missing_files:
        print(f"Missing parquet files: {missing_files}")
    if bad_files:
        print(f"Unreadable/invalid parquet files: {bad_files}")

    print()
    print(text_hist(uniq_hist, title="Unique run_identifier per flowy_data_record.parquet"))

    # if global_max_step is None:
    #     print("Global max(step): NA (no valid step values found)")
    # else:
    #     print(f"Global max(step): {global_max_step}")

    print()
    print(text_hist(rowcount_hist, title="Row count per flowy_data_record.parquet"))

    print()
    print(text_hist(maxstep_hist, title="Max step per flowy_data_record.parquet"))
    
    # Global aig/transistor summary
    def safe_mean_min(vals: list[float]):
        if not vals:
            return None, None
        s = pd.Series(vals, dtype="float64")
        return float(s.mean()), float(s.min())
    
    aig_g_mean, aig_g_min = safe_mean_min(all_aig)
    tr_g_mean, tr_g_min = safe_mean_min(all_tr)
    
    print("Aggregate over ALL rows in ALL files:")
    if aig_g_mean is None:
        print("  aig_count     : mean=NA, min=NA")
    else:
        print(f"  aig_count     : mean={aig_g_mean:.3f}, min={aig_g_min:.3f}")
    
    if tr_g_mean is None:
        print("  nb_transistors : mean=NA, min=NA")
    else:
        print(f"  nb_transistors : mean={tr_g_mean:.3f}, min={tr_g_min:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

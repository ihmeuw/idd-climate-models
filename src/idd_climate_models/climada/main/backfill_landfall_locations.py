"""
Backfill stage 1 landfall-location parquets that were written empty by mistake.

The bug: when a storm's land-clipped intensity is a single row or column of
pixels, `compute_storm_landfall_locations` builds a raster whose pixel size is
NaN, every polygon clip fails, and the storm's landfall parquet is written with
no rows, even though its intensity and exposure zarr groups were saved (the
landfall check pads the grid with `ensure_min_grid`; this function didn't).
The zarr data is correct; only the per-location landfall records are missing.

Stage 1 has padded the grid with `ensure_min_grid` inside
`compute_storm_landfall_locations` since 2026-10-12. (A single-row footprint
used to crash there instead, failing the draw until the fix; those draws are
recomputed by the workflow and need no backfill.) This script repairs the
parquets written empty before the fix, using stage 1's own code, loaded from
01_climada_intensity_main.py; the explicit padding below is a no-op now.

Affected storms: a zarr intensity group exists but the landfall parquet has
0 rows. Only draws with a completion marker are scanned, and by default the
script refuses to run until every draw of the run has one (i.e. the workflow
is finished), so it never races a running stage 1 task.

Dry run (default) lists the affected storms and the records they would get;
--apply rewrites their parquets (atomically: written beside, then renamed).

    python backfill_landfall_locations.py --num_cores 64                   # dry run
    python backfill_landfall_locations.py --num_cores 64 --apply           # write
    python backfill_landfall_locations.py --allow_incomplete --num_cores 32   # dry run mid-run
"""

import argparse
import importlib.util
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import xarray as xr  # type: ignore
from rra_tools.parallel import run_parallel  # type: ignore

MAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN_DIR))
from stage1_run_config import DEFAULT_SAVE_ROOT, log_dir_for  # noqa: E402

TASKS_CSV = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/level_4_task_assignments.csv")
N_DRAWS = 100
# An empty landfall parquet is ~5.3 KB and one with a row ~6.7 KB, so only
# footers of files under this size are read.
SMALL_PARQUET_BYTES = 6000


def _load_stage1():
    spec = importlib.util.spec_from_file_location("climada_intensity_main", MAIN_DIR / "01_climada_intensity_main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage1 = _load_stage1()
_LAND = {}  # basin -> land GeoDataFrame, cached per worker process


def all_draws() -> list[tuple]:
    df = pd.read_csv(TASKS_CSV)
    df["basin"] = df["basin"].fillna("NA")
    combos = df[["model", "variant", "scenario", "time_period", "basin"]].drop_duplicates()
    return [(*c, d) for c in combos.itertuples(index=False, name=None) for d in range(N_DRAWS)]


def _intensity_store(save_root: Path, s, v, e, b, basin, draw) -> Path:
    y0, y1 = b.split("-")
    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    return save_root / s / v / e / b / basin / "intensity" / f"intensity_{basin}_{s}_{e}_{v}_{y0}01_{y1}12{draw_text}.zarr"


def scan_draw(args) -> dict:
    """Storms in one finished draw that have an intensity group but an empty landfall parquet."""
    key, save_root = args
    out = {"key": key, "finished": False, "affected": [], "anomalies": []}
    if not stage1.is_draw_completed(log_dir_for(save_root), *key):
        return out
    out["finished"] = True
    store = _intensity_store(save_root, *key)
    if not store.exists():
        return out
    landfall_dir = stage1._landfall_parquet_path(*key, 0, save_root).parent
    for entry in os.scandir(store):
        if not (entry.is_dir() and entry.name.startswith("storm_")):
            continue
        storm = int(entry.name.split("_")[1])
        parquet = landfall_dir / f"storm_{storm:04d}.parquet"
        try:
            size = parquet.stat().st_size
        except FileNotFoundError:
            out["anomalies"].append((storm, "intensity group but no landfall parquet"))
            continue
        if size < SMALL_PARQUET_BYTES and pq.ParquetFile(parquet).metadata.num_rows == 0:
            out["affected"].append(storm)
    return out


def fix_storm(args) -> dict:
    """Recompute one storm's landfall records with the padded grid; write them if `apply`."""
    key, storm, save_root, apply = args
    s, v, e, b, basin, draw = key
    row = {"source_id": s, "variant_label": v, "experiment_id": e, "batch_year": b, "basin": basin,
           "draw": draw, "storm_id": storm}
    try:
        da = xr.open_zarr(_intensity_store(save_root, *key), group=f"storm_{storm:04d}", consolidated=False)["intensity"].load()
        if basin not in _LAND:
            _LAND[basin] = stage1.load_global_land_gdf(basin)
        land_sub = stage1.subset_land_for_storm(_LAND[basin], da, basin, buffer=0.25)
        records = stage1.compute_storm_landfall_locations(stage1.ensure_min_grid(da), land_sub, wind_threshold=17.0)
        row.update(grid=f"{da.sizes['lat']}x{da.sizes['lon']}", peak_ms=float(da.max()), n_records=len(records),
                   locations="; ".join(f"{r['ADM0_NAME']} ({r['loc_id']})" for r in records))
        if not records:
            row["status"] = "still empty after padding"
            return row
        if apply:
            # Write beside the target, then rename over it, so the parquet is never half-written.
            tmp_root = save_root / "_landfall_backfill_tmp" / f"{os.getpid()}"
            stage1.save_storm_landfall_locations(records, *key, storm, da.attrs.get("start_date"), save_root=tmp_root)
            os.replace(stage1._landfall_parquet_path(*key, storm, tmp_root),
                       stage1._landfall_parquet_path(*key, storm, save_root))
            row["status"] = "rewritten"
        else:
            row["status"] = "would rewrite"
    except Exception as ex:
        row["status"] = f"error: {type(ex).__name__}: {ex}"
    return row


def main(save_root: Path, apply: bool, allow_incomplete: bool, num_cores: int, report: Path) -> None:
    draws = all_draws()
    print(f"{'APPLY' if apply else 'DRY RUN'} | output {save_root} | {len(draws):,} draws in the run")

    scans = run_parallel(runner=scan_draw, arg_list=[(k, save_root) for k in draws],
                         num_cores=num_cores, progress_bar=True)
    finished = sum(r["finished"] for r in scans)
    print(f"finished draws: {finished:,} of {len(draws):,}")
    if finished < len(draws) and not allow_incomplete:
        print("❌ Stage 1 is not finished; run after the workflow completes (or pass --allow_incomplete for a dry run).")
        raise SystemExit(1)
    if apply and finished < len(draws):
        print("❌ --apply needs a finished stage 1 (no running task may rewrite these parquets).")
        raise SystemExit(1)

    anomalies = [(r["key"], s, msg) for r in scans for s, msg in r["anomalies"]]
    affected = [(r["key"], s) for r in scans for s in r["affected"]]
    print(f"affected storms (intensity group, empty landfall parquet): {len(affected):,}")
    if anomalies:
        print(f"⚠️ {len(anomalies)} other anomalies, e.g. {anomalies[:3]}")

    rows = run_parallel(runner=fix_storm, arg_list=[(k, s, save_root, apply) for k, s in affected],
                        num_cores=num_cores, progress_bar=True) if affected else []
    shutil.rmtree(save_root / "_landfall_backfill_tmp", ignore_errors=True)

    df = pd.DataFrame(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report, index=False)
    if len(df):
        print(df["status"].str.split(":").str[0].value_counts().to_string())
        print("grid sizes:", df["grid"].value_counts().head(5).to_dict())
    print(f"report: {report}")
    if len(df) and df["status"].str.startswith("error").any():
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_path", type=Path, default=DEFAULT_SAVE_ROOT, help="stage 1 output root")
    parser.add_argument("--apply", action="store_true", help="rewrite the affected parquets (default: dry run)")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="dry run on the draws finished so far, while stage 1 is still running")
    parser.add_argument("--num_cores", type=int, default=32)
    parser.add_argument("--report", type=Path, default=None,
                        help="CSV of affected storms (default: <output>_log/landfall_backfill_<mode>_<timestamp>.csv "
                             "with --apply, ./landfall_backfill_dry_run_<timestamp>.csv otherwise)")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = args.report or (log_dir_for(args.output_path) / f"landfall_backfill_apply_{stamp}.csv" if args.apply
                             else Path.cwd() / f"landfall_backfill_dry_run_{stamp}.csv")
    main(args.output_path, args.apply, args.allow_incomplete, args.num_cores, report)

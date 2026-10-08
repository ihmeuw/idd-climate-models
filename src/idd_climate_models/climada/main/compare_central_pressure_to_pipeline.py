"""
Check stage 1's re-estimated central pressure against the pipeline's
non-ensemble estimate for one draw.

Runs add_parameters_to_dataset(..., use_ensemble=False) on the raw TC-risk
draw and compares it with estimate_central_pressure_with_climada applied to
the CLIMADA input file for the same draw. They must agree within float
tolerance (both fit on TC-risk's 0..360 longitude).

Usage (CLIMADA env):
    python compare_central_pressure_to_pipeline.py \
        --source_id EC-Earth3 --variant_label r1i1p1f1 --experiment_id ssp585 \
        --batch_year 2020-2024 --basin EP --draw 1
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import xarray as xr  # type: ignore

MAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN_DIR))
sys.path.insert(0, str(MAIN_DIR.parents[2]))  # src/, for idd_climate_models

from idd_climate_models.add_tc_params import add_parameters_to_dataset  # noqa: E402

TC_RISK_OUTPUT_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/tc_risk/output/cmip6/")


def load_stage1_main():
    spec = importlib.util.spec_from_file_location(
        "climada_intensity_main", MAIN_DIR / "01_climada_intensity_main.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_draw(source_id, variant_label, experiment_id, batch_year, basin, draw, atol=1e-6) -> dict:
    stage1 = load_stage1_main()

    input_file = stage1.read_custom_tracks_nc(
        source_id, variant_label, experiment_id, batch_year, basin, draw,
    )
    # TC-risk draws use the same file name as the CLIMADA input files.
    raw_file = (
        TC_RISK_OUTPUT_ROOT / source_id / variant_label / experiment_id / batch_year / basin
        / input_file.name
    )

    with xr.open_dataset(input_file) as ds_in, xr.open_dataset(raw_file) as ds_raw:
        ds_in, ds_raw = ds_in.load(), ds_raw.load()

    reestimated = stage1.estimate_central_pressure_with_climada(ds_in)["central_pressure"].values

    pipeline = add_parameters_to_dataset(
        ds_raw, types.SimpleNamespace(use_ensemble=False, verbose=False),
    )
    # Input files keep only valid storms; n_trk holds their raw track index.
    trk = ds_in["n_trk"].values
    pipeline_cp = pipeline["central_pressure"].values[trk]

    same_nan = np.array_equal(np.isnan(reestimated), np.isnan(pipeline_cp))
    max_abs_diff = float(np.nanmax(np.abs(reestimated - pipeline_cp)))
    return {
        "input_file": str(input_file),
        "raw_file": str(raw_file),
        "n_points": int(np.isfinite(reestimated).sum()),
        "same_nan_pattern": same_nan,
        "max_abs_diff": max_abs_diff,
        "agree": same_nan and max_abs_diff <= atol,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source_id", required=True)
    parser.add_argument("--variant_label", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--batch_year", required=True)
    parser.add_argument("--basin", required=True)
    parser.add_argument("--draw", type=int, required=True, help="0 = base file, N = _e{N-1}")
    args = parser.parse_args()

    result = compare_draw(
        args.source_id, args.variant_label, args.experiment_id,
        args.batch_year, args.basin, args.draw,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    if not result["agree"]:
        print("❌ Re-estimated central pressure does NOT match the pipeline's non-ensemble estimate.")
        raise SystemExit(1)
    print("✅ Re-estimated central pressure matches the pipeline's non-ensemble estimate.")

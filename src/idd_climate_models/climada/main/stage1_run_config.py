"""
Stage 1 run options shared by 01_climada_intensity_main.py and its launcher,
plus the stage 1 completeness check used by downstream launchers.

The run config records the processing choices that change stage 1 results. It
is written with every draw's outputs and compared on resume, so one output root
never mixes results computed under different choices.

Kept free of heavy imports so launchers can use it without CLIMADA.
"""

import argparse
from pathlib import Path

DEFAULT_SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v3/")

# m/s -> knots, same factor add_tc_params uses before estimating pressure.
MS_TO_KN = 1.943844


def build_run_config(reestimate_central_pressure: bool, convert_wind_to_knots: bool) -> dict:
    return {
        "central_pressure_source": (
            "climada_fit: climada.hazard.tc_tracks._estimate_pressure"
            "(lat, lon in 0..360, v_max in kn)"
            if reestimate_central_pressure
            else "input_file (ensemble estimate)"
        ),
        "max_sustained_wind_to_climada": (
            "kn (converted from m/s)"
            if convert_wind_to_knots
            else "m/s labelled kn (as stored in input file)"
        ),
    }


# Results written before run configs were recorded used the default choices.
LEGACY_RUN_CONFIG = build_run_config(False, False)


def log_dir_for(save_root: Path) -> Path:
    """Logs and markers live beside the outputs: stage1_v3 -> stage1_v3_log."""
    return save_root.parent / f"{save_root.name}_log"


def completed_stage1_draws(save_root: Path = DEFAULT_SAVE_ROOT) -> set[tuple]:
    """
    (source_id, variant_label, experiment_id, batch_year, basin, draw) for
    every stage 1 draw with a completion marker under <save_root>_log.
    """
    marker_root = log_dir_for(save_root) / "draw_completion_markers"
    completed: set[tuple] = set()
    if not marker_root.exists():
        return completed
    for marker in marker_root.rglob("draw_*.json"):
        parts = marker.relative_to(marker_root).parts
        # expected: (source_id, variant_label, experiment_id, batch_year, basin, "draw_NNNN.json")
        if len(parts) != 6:
            continue
        try:
            draw = int(parts[5].removeprefix("draw_").removesuffix(".json"))
        except ValueError:
            continue
        completed.add((*parts[:5], draw))
    return completed


def require_stage1_complete(required: set[tuple], save_root: Path = DEFAULT_SAVE_ROOT) -> None:
    """
    Exit non-zero unless every required (source_id, variant_label,
    experiment_id, batch_year, basin, draw) has a stage 1 completion marker.

    Downstream stages read a missing stage 1 store as "no landfalling storms",
    so running them before stage 1 finishes writes zero or partial results
    that are then treated as complete and never recomputed.
    """
    if not required:
        return
    print(f"Checking stage 1 completion markers under {log_dir_for(save_root)} ...")
    missing = sorted(required - completed_stage1_draws(save_root))
    if missing:
        print(
            f"❌ Stage 1 is incomplete: {len(missing):,} of {len(required):,} required draws "
            f"have no completion marker, e.g. {missing[:3]}.\n"
            f"   Finish stage 1 first (its launcher should report 0 batches to submit)."
        )
        raise SystemExit(1)
    print(f"✅ Stage 1 complete for all {len(required):,} required draws.")


def add_run_option_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output_path", type=Path, default=DEFAULT_SAVE_ROOT,
        help="Root for stage 1 results; logs go to <output_path>_log",
    )
    parser.add_argument(
        "--reestimate_central_pressure", action="store_true",
        help="Replace central_pressure with CLIMADA's statistical fit (non-ensemble)",
    )
    parser.add_argument(
        "--convert_wind_to_knots", action="store_true",
        help="Convert max_sustained_wind (stored in m/s, labelled kn) to knots before CLIMADA",
    )
    parser.add_argument(
        "--overwrite_okay", action="store_true",
        help="Replace existing results written with a different run config",
    )

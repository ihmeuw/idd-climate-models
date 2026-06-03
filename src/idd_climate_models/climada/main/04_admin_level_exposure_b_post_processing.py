"""
Stage 4B post-processing step: consume every per-(storm × location_id)
parquet written by stage 4B and consolidate into one launcher-friendly
DataFrame.  Runs as a downstream Jobmon task in the stage 4B workflow,
with all per-(storm × loc) tasks as upstream dependencies.

Path enumeration uses 4A's `compiled_admin_level_metadata.parquet` (which
already carries the `year` column the 4B launcher loses) instead of
crawling the stage4b_v2 tree.  At 35M+ expected paths we never
materialize a list of `Path` objects — chunks are sliced directly from
the metadata DataFrame and each worker constructs path strings on the
fly while reading.

Two steps:
  1. enumerate_meta()        — load the 4A compiled metadata and project
                               down to the columns needed to build paths.
  2. compile_exposure(meta)  — parallel-read every expected 4B parquet in
                               row-chunks, concat into one DataFrame, and
                               write the consolidated parquet.
"""

import pandas as pd
from pathlib import Path
from rra_tools.parallel import run_parallel  # type: ignore


CLIMADA_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada")
COMPILED_4A_PARQUET = (
    CLIMADA_ROOT / "output" / "stage4a_metadata_admin0"
    / "compiled_admin_level_metadata.parquet"
)

SAVE_ROOT = CLIMADA_ROOT / "output" / "stage4b_v2"
CONSOLIDATED_DIR = SAVE_ROOT / "_consolidated"
CONSOLIDATED_PARQUET = CONSOLIDATED_DIR / "storm_exposure_all.parquet"

# Path-build columns lifted from 4A. `year` is the only one the 4B
# launcher's own meta_df drops (it's derived inside the worker from
# storm.start_date), so loading from the 4A compile is the cheapest way
# to recover it without re-opening every CMIP zarr.
ID_COLS = [
    "source_id", "variant_label", "experiment_id",
    "batch_year", "basin", "draw",
    "storm_id", "location_id", "year",
]

CHUNK_SIZE = 5000   # rows per parallel-read chunk; 35M / 5000 ≈ 7K chunks
NUM_CORES = 40


# ----------------------------------------------------------------------
# Path helper — must match `save_storm_exposure` in
# 04_admin_level_exposure_b_main.py exactly.
# ----------------------------------------------------------------------
def _build_4b_path(row) -> Path:
    start_year, end_year = row.batch_year.split("-")
    draw = int(row.draw)
    draw_text = "" if draw == 0 else f"_e{draw - 1}"
    return (
        SAVE_ROOT
        / row.source_id / row.variant_label / row.experiment_id
        / row.batch_year / row.basin
        / f"tc_risk_draw_{draw}" / "storm_exposure" / str(int(row.year))
        / (
            f"storm_{row.storm_id}_loc_{row.location_id}_{row.basin}_"
            f"{row.source_id}_{row.variant_label}_{row.experiment_id}_"
            f"{start_year}01_{end_year}12{draw_text}.parquet"
        )
    )


# ----------------------------------------------------------------------
# Step 1 — enumerate (storm × admin × draw) inventory
# ----------------------------------------------------------------------
def enumerate_meta() -> pd.DataFrame:
    """Load 4A's compiled (storm × admin × draw) metadata, project to the
    columns needed to construct 4B output paths."""
    meta = pd.read_parquet(COMPILED_4A_PARQUET, columns=ID_COLS)
    print(f"Loaded 4A inventory: {len(meta):,} expected 4B outputs.")
    return meta


# ----------------------------------------------------------------------
# Step 2 — chunked parallel read + concat
# ----------------------------------------------------------------------
def _read_chunk(meta_chunk: pd.DataFrame) -> pd.DataFrame:
    """Build paths for the chunk and read each parquet. Missing files are
    skipped silently (safe against partially-completed 4B); other read
    errors are logged and skipped — the consolidate step shouldn't crash
    on one corrupt parquet."""
    dfs = []
    for row in meta_chunk.itertuples(index=False):
        path = _build_4b_path(row)
        try:
            dfs.append(pd.read_parquet(path))
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"⚠️ {path}: {type(e).__name__}: {e}")
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _chunk_df(df: pd.DataFrame, chunk_size: int):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


def compile_exposure(meta_df: pd.DataFrame) -> pd.DataFrame:
    chunks = list(_chunk_df(meta_df, CHUNK_SIZE))
    print(
        f"Reading {len(meta_df):,} stage-4B parquets in {len(chunks):,} "
        f"chunks of {CHUNK_SIZE} across {NUM_CORES} cores."
    )
    dfs = run_parallel(
        runner=_read_chunk,
        arg_list=chunks,
        num_cores=NUM_CORES,
        progress_bar=True,
    )
    final = pd.concat(dfs, ignore_index=True)
    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(CONSOLIDATED_PARQUET, index=False)
    print(
        f"Saved consolidated 4B output → {CONSOLIDATED_PARQUET} "
        f"({len(final):,} rows)"
    )
    return final


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    meta_df = enumerate_meta()
    compile_exposure(meta_df)

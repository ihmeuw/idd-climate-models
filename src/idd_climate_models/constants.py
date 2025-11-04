from pathlib import Path

MODEL_ROOT = "/mnt/team/rapidresponse/pub/tropical-storms"
REPO_ROOT = Path("/mnt/share/homes/bcreiner/repos")

DATA_PATH = Path(MODEL_ROOT) / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"


repo_name = "idd-climate-models"
package_name = "idd_climate_models"
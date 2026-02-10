#!/usr/bin/env python3
"""
Quick test of vendored TC-risk code.
Run this to verify the vendoring worked correctly.
"""

import sys
import importlib.util
from pathlib import Path

# Add to path
repo_root = Path('/ihme/homes/bcreiner/repos/idd-climate-models')
if str(repo_root / 'src') not in sys.path:
    sys.path.insert(0, str(repo_root / 'src'))

from idd_climate_models import constants as rfc

# Load a test namelist (use an existing one)
test_namelist = rfc.TC_RISK_OUTPUT_PATH / 'cmip6' / 'CMCC-ESM2' / 'r1i1p1f1' / 'historical' / '1986-2014' / 'GL' / 'namelist.py'

if not test_namelist.exists():
    print(f"❌ Test namelist not found: {test_namelist}")
    print("Run create_custom_namelist first to create a namelist")
    sys.exit(1)

# Load namelist into sys.modules
spec = importlib.util.spec_from_file_location("namelist_loader", test_namelist)
namelist_loader = importlib.util.module_from_spec(spec)
sys.modules['namelist_loader'] = namelist_loader
spec.loader.exec_module(namelist_loader)
sys.modules['namelist'] = namelist_loader

print(f"✅ Loaded test namelist from {test_namelist.parent.name}")
print()

# Test imports
print("Testing vendored TC-risk imports...")
print("=" * 80)

try:
    from idd_climate_models.tc_risk_vendored.util import compute
    print("✅ Successfully imported compute")
except Exception as e:
    print(f"❌ Failed to import compute: {e}")
    sys.exit(1)

try:
    from idd_climate_models.tc_risk_vendored.track import env_wind
    print("✅ Successfully imported env_wind")
except Exception as e:
    print(f"❌ Failed to import env_wind: {e}")
    sys.exit(1)

try:
    from idd_climate_models.tc_risk_vendored.intensity import coupled_fast
    print("✅ Successfully imported coupled_fast")
except Exception as e:
    print(f"❌ Failed to import coupled_fast: {e}")
    sys.exit(1)

try:
    from idd_climate_models.tc_risk_vendored.scripts import generate_land_masks
    print("✅ Successfully imported generate_land_masks")
except Exception as e:
    print(f"❌ Failed to import generate_land_masks: {e}")
    sys.exit(1)

print("=" * 80)
print("✅ All imports successful!")
print()
print("Next step: Test with actual TC-risk run")
print("  python src/idd_climate_models/01_run_tc_risk/04_run_global_tc_risk.py \\")
print("    --model CMCC-ESM2 --variant r1i1p1f1 --scenario historical \\")
print("    --time_period 1986-2014 --basin GL")

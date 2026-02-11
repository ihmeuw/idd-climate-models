#!/usr/bin/env python
"""Test draw number extraction with actual filenames."""

from pathlib import Path
import sys
sys.path.insert(0, '/ihme/homes/bcreiner/repos/idd-climate-models/src')

from idd_climate_models.01_process_raw_through_climada_input.00_create_draw_status_file import extract_draw_number

# Test cases from your actual files
test_files = [
    "tracks_EP_EC-Earth3_historical_r1i1p1f1_196501_196912.zarr",  # Should be Draw 0
    "tracks_EP_EC-Earth3_historical_r1i1p1f1_196501_196912_e0.zarr",  # Should be Draw 1
    "tracks_EP_EC-Earth3_historical_r1i1p1f1_196501_196912_e13.zarr",  # Should be Draw 14
    "tracks_EP_EC-Earth3_historical_r1i1p1f1_196501_196912_e224.zarr",  # Should be Draw 225
    "tracks_EP_EC-Earth3_historical_r1i1p1f1_196501_196912_e248.zarr",  # Should be Draw 249
]

expected = [0, 1, 14, 225, 249]

print("Testing draw number extraction:")
print("=" * 80)

all_correct = True
for filename, expected_draw in zip(test_files, expected):
    path = Path(filename)
    result = extract_draw_number(path)
    status = "✓" if result == expected_draw else "✗"
    if result != expected_draw:
        all_correct = False
    print(f"{status} {filename:70s} → Draw {result:3d} (expected {expected_draw})")

print("=" * 80)
if all_correct:
    print("✅ All tests passed!")
else:
    print("❌ Some tests failed!")
    sys.exit(1)

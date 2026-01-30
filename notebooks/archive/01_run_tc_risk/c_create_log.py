import argparse
from pathlib import Path
import idd_climate_models.constants as rfc
from idd_climate_models.model_validation import (
    create_validation_dict,
    validate_all_models_in_source
)

def main():
    parser = argparse.ArgumentParser(description='Validate reorganized climate data in tc_risk')
    parser.add_argument('--data_source', default='cmip6')
    parser.add_argument('--detail_level', default='time_period', 
                       choices=['model', 'variant', 'scenario', 'time_period', 'file'])
    args = parser.parse_args()
    
    print("Validating reorganized data in tc_risk...")
    
    # Create validation dictionary using tc_risk data type
    validation_dict = create_validation_dict(
        DATA_TYPE='tc_risk',
        IO_TYPE_RAW='processed',
        DATA_SOURCE=args.data_source
    )
    
    # Use the existing validation machinery
    # It will automatically use TC_RISK_INPUT_PATH based on data_type='tc_risk'
    validation_dict = validate_all_models_in_source(
        validation_dict,
        detail_level=args.detail_level,
        verbose=True
    )
    
    # Print summary
    results = validation_dict.get('validation_results', {})
    complete_models = [m for m, r in results.items() if r.get('complete')]
    incomplete_models = [m for m, r in results.items() if not r.get('complete')]
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Complete models ({len(complete_models)}):")
    for model in complete_models:
        print(f"  ✓ {model}")
    
    if incomplete_models:
        print(f"\nIncomplete models ({len(incomplete_models)}):")
        for model in incomplete_models:
            print(f"  ✗ {model}")

if __name__ == "__main__":
    main()
"""
Run TC-risk for global basin.

This script is called by the orchestrator as Level 3 of the workflow.
It creates a configuration dictionary (NO namelist files!) and runs TC-risk compute.
"""

import argparse
import idd_climate_models.constants as rfc
from idd_climate_models.tc_risk_functions import (
    create_tc_risk_config_dict,
    execute_tc_risk_with_config
)


def main():
    parser = argparse.ArgumentParser(
        description='Run TC-risk for global basin using dict-based config.'
    )
    parser.add_argument('--data_source', type=str, default='cmip6',
                        help='Data source (e.g., cmip6 or era5)')
    parser.add_argument('--model', type=str, required=True,
                        help='Climate model name')
    parser.add_argument('--variant', type=str, required=True,
                        help='Model variant')
    parser.add_argument('--scenario', type=str, required=True,
                        help='Scenario name')
    parser.add_argument('--time_period', type=str, required=True,
                        help='Time bin (e.g., 1970-1986)')
    args = parser.parse_args()

    # Set basin to global for this run
    args.basin = 'GL'

    print(f"Running global TC-risk for {args.model}/{args.variant}/{args.scenario}/{args.time_period}")

    # Create config dict (NO namelist file!)
    config_dict = create_tc_risk_config_dict(args)
    
    # Execute TC-risk with dict
    success = execute_tc_risk_with_config(config_dict, script_name='compute', args=args)
    
    if success:
        print(f"✅ Completed global TC-risk for {args.model}/{args.variant}/{args.scenario}/{args.time_period}")
    else:
        print(f"❌ FAILED global TC-risk for {args.model}/{args.variant}/{args.scenario}/{args.time_period}")
        exit(1)


if __name__ == '__main__':
    main()

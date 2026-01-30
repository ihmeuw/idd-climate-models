"""
Run TC-risk for global basin.

This script is called by the orchestrator as Level 3 of the workflow.
It creates a namelist and runs the TC-risk compute step for the global basin.
"""

import argparse
import idd_climate_models.constants as rfc
from idd_climate_models.tc_risk_functions import create_custom_namelist, execute_tc_risk


def main():
    parser = argparse.ArgumentParser(
        description='Create and run tc_risk namelists for global basin.'
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

    create_custom_namelist(args)
    execute_tc_risk(args, script_name='compute')

    print(f"Completed global TC-risk for {args.model}/{args.variant}/{args.scenario}/{args.time_period}")


if __name__ == '__main__':
    main()

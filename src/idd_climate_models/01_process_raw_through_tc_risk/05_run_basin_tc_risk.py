"""
Run TC-risk downscaling for a specific basin.

This script is called by the orchestrator as Level 4 of the workflow.
It creates a namelist and runs the TC-risk downscaling step for a specific basin.
"""

import argparse
import idd_climate_models.constants as rfc
from idd_climate_models.tc_risk_functions import create_custom_namelist, execute_tc_risk


def main():
    parser = argparse.ArgumentParser(
        description='Create and run tc_risk namelists for specified basins.'
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
    parser.add_argument('--basin', type=str, required=True,
                        help='Basin to run (EP, NA, NI, SI, SP, WP)')
    parser.add_argument('--num_draws', type=int, default=100,
                        help='Number of draws for downscaling')
    args = parser.parse_args()

    print(f"Running basin TC-risk for {args.model}/{args.variant}/{args.scenario}/{args.time_period}/{args.basin}")
    print(f"Number of draws: {args.num_draws}")

    create_custom_namelist(args)
    execute_tc_risk(args, script_name='run_downscaling')

    print(f"Completed basin TC-risk for {args.model}/{args.variant}/{args.scenario}/{args.time_period}/{args.basin}")


if __name__ == '__main__':
    main()

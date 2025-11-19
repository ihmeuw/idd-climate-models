import os
import shutil
import sys
import argparse
from pathlib import Path
import idd_climate_models.constants as rfc
from idd_climate_models.tc_risk_functions import create_custom_namelist, execute_tc_risk


def main():
    parser = argparse.ArgumentParser(description='Create and run tc_risk namelists for specified basins.')
    parser.add_argument('--data_source', type=str, default='cmip6', help='Data source (e.g., cmip6 or era5)')
    parser.add_argument('--model', type=str, default='ACCESS-CM2', help='Climate model name')
    parser.add_argument('--variant', type=str, default='r1i1p1f1', help='Model variant')
    parser.add_argument('--scenario', type=str, default='historical', help='Scenario name')
    parser.add_argument('--time_bin', type=str, default='1970-1989', help='Time bin (e.g., 1970-1989)')
    parser.add_argument('--basin', type=str, default='GL', help='Basin to run')
    parser.add_argument('--num_draws', type=int, default=1, help='Number of draws for downscaling')
    args = parser.parse_args()

    create_custom_namelist(args)
    execute_tc_risk(args, script_name='run_downscaling')

if __name__ == '__main__':
    main()
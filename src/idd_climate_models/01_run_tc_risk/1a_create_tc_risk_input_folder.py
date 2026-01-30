import argparse
from pathlib import Path
import idd_climate_models.constants as rfc

def main():
    parser = argparse.ArgumentParser(description='Create all required input folders for tc_risk')
    parser.add_argument('--data_source', type=str, default='cmip6')
    parser.add_argument('--model', type=str, default='ACCESS-CM2')
    parser.add_argument('--variant', type = str, default='r1i1p1f1')
    parser.add_argument('--scenario', type=str, default='historical')
    parser.add_argument('--time_period', type=str, default='1970-1989')
    args = parser.parse_args()

    target_base_path = rfc.TC_RISK_INPUT_PATH / args.data_source
    target_dir = target_base_path / args.model / args.variant / args.scenario / args.time_period
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {target_dir}")

if __name__ == "__main__":
    main()
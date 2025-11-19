import argparse
import sys
from pathlib import Path

# NOTE: We assume 'result_functions.py' is a module accessible in the Python path
# and contains all necessary functions (process_model_scenario, calculate_cumulative_convergence)
# and constants (TC_RISK_OUTPUT_PATH, DATA_SOURCE, VARIANT, THRESHOLD_DICT).
# We import all contents using the '*' convention for simplicity here,
# assuming result_functions.py is renamed to result_functions.py
from result_functions import *

def main():
    """
    Orchestrates the processing, convergence calculation, and saving of 
    the final convergence report for a single model, designed to run 
    in parallel via Jobmon.
    """
    parser = argparse.ArgumentParser(
        description="Process one climate model's TC track data, calculate global aggregates and convergence metrics, and save the final report."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="The name of the climate model to process (e.g., ACCESS-CM2)."
    )
    # The Jobmon script will pass the required constants implicitly through the environment 
    # or they are defined in result_functions.py, but we keep DATA_SOURCE here for path construction.
    parser.add_argument(
        "--data_source", 
        type=str, 
        default=DATA_SOURCE, 
        help="The data source identifier (e.g., cmip6)."
    )

    args = parser.parse_args()
    model = args.model
    
    print(f"--- Starting Full Convergence Pipeline for Model: {model} ---")

    # 1. --- EXECUTE ANNUAL TRACK PROCESSING & GLOBAL AGGREGATION ---
    # This runs the loops over all time-bins/scenarios/basins/draws and generates 
    # a DataFrame indexed k=0 to k=24 for annual counts/days, including 'GL' basin.
    try:
        df_annual_metrics = process_model_scenario(
            model=model, 
            threshold_dict=THRESHOLD_DICT
        )
        
        if df_annual_metrics.empty:
            print(f"ERROR: Model {model} produced an empty metrics DataFrame. Halting.")
            return
            
        print(f"[{model}] Finished annual processing. Total rows: {len(df_annual_metrics)}")

    except Exception as e:
        print(f"CRITICAL ERROR in annual processing for {model}: {e}", file=sys.stderr)
        # Re-raise the exception to signal Jobmon that the task failed
        raise

    # 2. --- CALCULATE CUMULATIVE CONVERGENCE METRICS ---
    # This performs the running mean, std, CV, and relative change calculations.
    try:
        df_convergence_report = calculate_cumulative_convergence(df_annual_metrics)
        print(f"[{model}] Finished convergence calculation. Total rows: {len(df_convergence_report)}")

    except Exception as e:
        print(f"CRITICAL ERROR in convergence calculation for {model}: {e}", file=sys.stderr)
        raise

    # 3. --- SAVE FINAL REPORTS ---
    
    # Construct the save directory
    save_dir = TC_RISK_OUTPUT_PATH / args.data_source / model / VARIANT
    save_dir.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    
    # Save 1: Raw Annual Metrics (The expensive data)
    annual_save_path = save_dir / 'annual_metrics.pkl'
    try:
        df_annual_metrics.to_pickle(annual_save_path)
        print(f"[{model}] SUCCESS! Annual metrics saved to: {annual_save_path}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to save annual metrics file for {model}: {e}", file=sys.stderr)
        raise

    # Save 2: Convergence Report (The stability metrics)
    convergence_save_path = save_dir / 'convergence_report.pkl'
    try:
        df_convergence_report.to_pickle(convergence_save_path)
        print(f"[{model}] SUCCESS! Convergence report saved to: {convergence_save_path}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to save convergence report file for {model}: {e}", file=sys.stderr)
        raise
        

if __name__ == "__main__":
    main()
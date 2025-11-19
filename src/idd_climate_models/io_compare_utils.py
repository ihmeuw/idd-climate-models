from tabnanny import verbose
import idd_climate_models.constants as rfc
from idd_climate_models.validation_functions import validate_all_models_in_source, create_validation_dict
from idd_climate_models.dictionary_utils import get_models



def filter_tc_risk_models(validation_dict):
    """
    Filters TC-Risk Output validation results based on three criteria:
    1. Complete models.
    2. Models missing all non-GL basin folders across all time-periods.
    3. Models that have non-GL basins but with insufficient file counts.

    Args:
        validation_dict (dict): The dictionary containing 'validation_results'.

    Returns:
        dict: A dictionary containing the three filtered sets of model names.
    """
    ALL_BASINS = set(rfc.basin_dict.keys()) # ['EP', 'NA', 'NI', 'SI', 'SP', 'WP', 'GL']
    NON_GL_BASINS = ALL_BASINS - {'GL'}
    NUM_DRAWS_EXPECTED = rfc.NUM_DRAWS # e.g., 100

    results = validation_dict.get('validation_results', {})
    
    # 1. Complete models (simplest check)
    complete_models = {
        model: data for model, data in results.items() 
        if data.get('complete', False)
    }
    
    # Initialize containers for incomplete models
    missing_all_other_basins = set()
    incomplete_file_count_models = {} # Store {model: list of specific basin issues}
    
    
    for model_name, model_data in results.items():
        # Skip models already marked complete
        if model_name in complete_models:
            continue
        
        # Flags for the current model's status across all variants/scenarios/time-periods
        only_gl_present = True
        has_file_count_issue = False
        basin_issue_details = []

        # Start traversal from the highest-level child, typically 'variant'
        first_child_key = next(iter(model_data.keys()), None)
        if first_child_key in ['variant', 'scenario', 'time-period']:
            
            def recurse_check(current_node, model_context):
                nonlocal only_gl_present, has_file_count_issue
                
                # Check at the time-period level for missing folders (Criterion 2)
                if 'issues' in current_node:
                    for issue in current_node['issues']:
                        if "Missing required folders" in issue:
                            # If a time-period folder is missing ANY non-GL basin, 'only_gl_present' must be False
                            # as this is a failure of the full set requirement, not just an 'only-GL' model.
                            # We check the actual present folders below for the 'only-GL' criterion.
                            pass

                # --- Basin-Level Checks ---
                if 'basin' in current_node:
                    present_basins = set(current_node['basin'].keys())
                    
                    # CRIERION 2 Check: If any non-GL basin is present, this model fails the 'only_gl_present' status.
                    if not present_basins.issubset({'GL'}):
                        only_gl_present = False
                        
                    # CRIERION 3 Check: Insufficient files in *any* basin
                    for basin_name, basin_data in current_node['basin'].items():
                        
                        # We only check non-GL basins for the NUM_DRAWS file count issue
                        if basin_name in NON_GL_BASINS:
                            files_present = len(basin_data.get('files', []))
                            
                            # The basin handler flags issues where file count < expected
                            if not basin_data.get('complete', True):
                                # If the basin handler already flagged an issue (which usually means file count is low)
                                has_file_count_issue = True
                                # Find the specific issue from the list
                                issue_msg = next((i for i in basin_data['issues'] if 'draw files' in i), "Insufficient files.")
                                
                                basin_issue_details.append(
                                    f"Basin='{basin_name}' in {model_context}: {issue_msg}"
                                )
                                break # Stop checking files for this time-period

                # Recurse: iterate through all dictionaries that are not metadata keys
                for key, value in current_node.items():
                    if isinstance(value, dict) and key not in ['files', 'issues', 'complete']:
                        recurse_check(value, f"{model_context} -> {key}")
            
            # Start recursion
            recurse_check(model_data, f"Model='{model_name}'")


        # --- Aggregate Results ---
        
        # 2. Models missing all basins other than GL
        # This checks if ALL time-periods processed adhered to the 'only GL is present' condition.
        # This will primarily capture models where the recursive folder validation failed (i.e., only 'GL' folder was found).
        if only_gl_present and model_name not in complete_models:
            missing_all_other_basins.add(model_name)

        # 3. Models that have some basins but don't have enough files
        # This captures models where non-GL basins *exist* but their internal file counts failed.
        if has_file_count_issue and model_name not in complete_models:
            incomplete_file_count_models[model_name] = basin_issue_details


    return {
        "complete_models": set(complete_models.keys()),
        "missing_all_other_basins": missing_all_other_basins,
        "incomplete_file_count": incomplete_file_count_models
    }


def compare_model_validation(
    input_data_type,
    input_io_type,
    output_data_type,
    output_io_type,
    data_source,
    verbose=True
):
    """
    Runs validation on input and output sources, compares results.
    The strict grid check is applied only if output_data_type is "tc_risk".
    """
    # CRITICAL FLAG DEFINITION: The strict grid check only runs if the target is 'tc_risk'.
    strict_grid_check_flag = (output_data_type == "tc_risk") 
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"STEP 1.1: Input Data - Validating {input_data_type} {input_io_type} models (Strict Check: {strict_grid_check_flag}).")
        print("=" * 80)

    # Input validation (This is where the filter needs to run)
    input_validation_dict = create_validation_dict(
        input_data_type, 
        input_io_type, 
        data_source,
        strict_grid_check=strict_grid_check_flag # PASS FLAG to dict
    )
    input_validation_dict = validate_all_models_in_source(
        validation_dict=input_validation_dict,
        verbose=verbose,
        strict_grid_check=strict_grid_check_flag # <-- CRITICAL: PASS FLAG to function
    )
    input_complete_models = get_models(input_validation_dict, complete_only=True)

    if verbose:
        print("\n" + "=" * 80)
        print(f"STEP 1.2: Output Data - Validating {output_data_type} {output_io_type} models (Check for already completed work).\n")
        print("=" * 80)

    # Output validation (Never apply strict check to output completion status)
    output_validation_dict = create_validation_dict(
        output_data_type, 
        output_io_type, 
        data_source,
        strict_grid_check=False 
    )
    output_validation_dict = validate_all_models_in_source(
        validation_dict=output_validation_dict,
        verbose=verbose,
        strict_grid_check=False # <-- CRITICAL: PASS FLAG to function
    )
    output_complete_models = get_models(output_validation_dict, complete_only=True)

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: Comparing validation results and generating tasks.")
        print("=" * 80)

    models_to_process = input_complete_models - output_complete_models

    if output_data_type == 'tc_risk' and output_io_type == 'output':
        if verbose:
            print("\nApplying TC-RISK specific filters for Criterion #3 (Insufficient Draw Files)...")
            
        # Run the specialized filtering function
        filtered_output_results = filter_tc_risk_models(output_validation_dict)
        
        # Identify models with fatal, low-level file count errors in existing output
        models_with_file_count_issues = set(filtered_output_results["incomplete_file_count"].keys())
        
        # Exclude models that have existing, but fatally incomplete output (e.g., non-GL basins < 100 files).
        # These need manual intervention and shouldn't be added to the processing queue.
        models_to_process = models_to_process - models_with_file_count_issues
        
        # (Optional: You could log/return filtered_output_results here for diagnostic purposes)
        if verbose and models_with_file_count_issues:
            print(f"⚠️ Excluded {len(models_with_file_count_issues)} models due to file count failures in existing TC-Risk output.")

    input_results_for_tasks = {
        model: data for model, data in input_validation_dict['validation_results'].items()
        if model in models_to_process
    }
    # Re-create dict for models_to_process, preserving the strict flag context
    models_to_process_dict = create_validation_dict(
        input_data_type, input_io_type, data_source, 
        validation_results=input_results_for_tasks,
        strict_grid_check=strict_grid_check_flag
    )

    return {
        "models_to_process_dict": models_to_process_dict,
        "models_to_process": models_to_process,
        "input_complete_models": input_complete_models,
        "output_complete_models": output_complete_models,
        "input_validation_dict": input_validation_dict,
        "output_validation_dict": output_validation_dict
    }
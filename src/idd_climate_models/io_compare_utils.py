from tabnanny import verbose
import idd_climate_models.constants as rfc
from idd_climate_models.validation_functions import validate_all_models_in_source, create_validation_dict
from idd_climate_models.dictionary_utils import get_models

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
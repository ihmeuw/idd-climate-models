from tabnanny import verbose
import idd_climate_models.constants as rfc
from idd_climate_models.validation_functions import validate_all_model_variants_in_source, create_validation_dict, analyze_validation_for_missing_data
from idd_climate_models.dictionary_utils import get_model_variants


def filter_tc_risk_model_variants(validation_dict):
    """
    Filters TC-Risk Output validation results at the model/variant level based on three criteria:
    1. Complete model/variants.
    2. Model/variants missing all non-GL basin folders across all time_periods.
    3. Model/variants that have non-GL basins but with insufficient file counts.

    Args:
        validation_dict (dict): The dictionary containing 'validation_results'.

    Returns:
        dict: A dictionary containing the three filtered sets of (model, variant) tuples.
    """
    ALL_BASINS = set(rfc.basin_dict.keys()) # ['EP', 'NA', 'NI', 'SI', 'SP', 'WP', 'GL']
    NON_GL_BASINS = ALL_BASINS - {'GL'}
    NUM_DRAWS_EXPECTED = rfc.NUM_DRAWS # e.g., 100

    results = validation_dict.get('validation_results', {})
    
    # Initialize containers for model/variant combinations
    complete_model_variants = set()  # Set of (model, variant) tuples
    missing_all_other_basins = set()  # Set of (model, variant) tuples
    incomplete_file_count_variants = {}  # {(model, variant): [list of specific basin issues]}
    
    for model_name, model_data in results.items():
        # Check if model has variant structure
        if 'variant' not in model_data:
            continue
            
        # Process each variant separately
        for variant_name, variant_data in model_data['variant'].items():
            model_variant_key = (model_name, variant_name)
            
            # Check if this variant is complete
            if variant_data.get('complete', False):
                complete_model_variants.add(model_variant_key)
                continue
            
            # Flags for the current variant's status across all scenarios/time_periods
            only_gl_present = True
            has_file_count_issue = False
            basin_issue_details = []

            def recurse_check(current_node, context_path):
                nonlocal only_gl_present, has_file_count_issue
                
                # Check at the time_period level for missing folders (Criterion 2)
                if 'issues' in current_node:
                    for issue in current_node['issues']:
                        if "Missing required folders" in issue:
                            pass

                # --- Basin-Level Checks ---
                if 'basin' in current_node:
                    present_basins = set(current_node['basin'].keys())
                    
                    # CRITERION 2 Check: If any non-GL basin is present, this variant fails the 'only_gl_present' status.
                    if not present_basins.issubset({'GL'}):
                        only_gl_present = False
                        
                    # CRITERION 3 Check: Insufficient files in *any* basin
                    for basin_name, basin_data in current_node['basin'].items():
                        
                        # We only check non-GL basins for the NUM_DRAWS file count issue
                        if basin_name in NON_GL_BASINS:
                            files_present = len(basin_data.get('files', []))
                            
                            # The basin handler flags issues where file count < expected
                            if not basin_data.get('complete', True):
                                has_file_count_issue = True
                                # Find the specific issue from the list
                                issue_msg = next((i for i in basin_data['issues'] if 'draw files' in i), "Insufficient files.")
                                
                                basin_issue_details.append(
                                    f"Basin='{basin_name}' in {context_path}: {issue_msg}"
                                )
                                break # Stop checking files for this time_period

                # Recurse: iterate through all dictionaries that are not metadata keys
                for key, value in current_node.items():
                    if isinstance(value, dict) and key not in ['files', 'issues', 'complete']:
                        recurse_check(value, f"{context_path} -> {key}")
            
            # Start recursion from variant level
            recurse_check(variant_data, f"Model='{model_name}'/Variant='{variant_name}'")

            # --- Aggregate Results for this variant ---
            
            # 2. Variants missing all basins other than GL
            if only_gl_present:
                missing_all_other_basins.add(model_variant_key)

            # 3. Variants that have some basins but don't have enough files
            if has_file_count_issue:
                incomplete_file_count_variants[model_variant_key] = basin_issue_details

    return {
        "complete_model_variants": complete_model_variants,
        "missing_all_other_basins": missing_all_other_basins,
        "incomplete_file_count": incomplete_file_count_variants
    }


def compare_model_validation(
    input_data_type,
    input_io_type,
    output_data_type,
    output_io_type,
    data_source,
    verbose=True,
    rerun_all=False,
    produce_missing_input_report = False,
    strict_grid_check_flag=False
):
    """
    Runs validation on input and output sources, compares results at model/variant level.
    The strict grid check is applied only if output_data_type is "tc_risk".
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"STEP 1.1: Input Data - Validating {input_data_type} {input_io_type} models (Strict Check: {strict_grid_check_flag}).")
        print("=" * 80)

    # Input validation - NOW USING validate_all_model_variants_in_source
    input_validation_dict = create_validation_dict(
        input_data_type, 
        input_io_type, 
        data_source,
        strict_grid_check=strict_grid_check_flag
    )
    input_validation_dict = validate_all_model_variants_in_source(
        validation_dict=input_validation_dict,
        verbose=verbose,
        strict_grid_check=strict_grid_check_flag
    )
    
    # Extract complete model/variant combinations from input
    # The validation dict now has 'variant_status' with complete_variants list
    input_complete_model_variants = set(input_validation_dict['variant_status']['complete_variants'])

    if verbose:
        print("\n" + "=" * 80)
        print(f"STEP 1.2: Output Data - Validating {output_data_type} {output_io_type} models (Check for already completed work).\n")
        print("=" * 80)

    if not rerun_all:
        # Output validation - NOW USING validate_all_model_variants_in_source
        output_validation_dict = create_validation_dict(
            output_data_type, 
            output_io_type, 
            data_source,
            strict_grid_check=False 
        )
        output_validation_dict = validate_all_model_variants_in_source(
            validation_dict=output_validation_dict,
            verbose=verbose,
            strict_grid_check=False
        )
    
        # Extract complete model/variant combinations from output
        output_complete_model_variants = set(output_validation_dict['variant_status']['complete_variants'])

        if verbose:
            print("\n" + "=" * 80)
            print("STEP 2: Comparing validation results and generating tasks.")
            print("=" * 80)

        model_variants_to_process = input_complete_model_variants - output_complete_model_variants

        # Apply TC-RISK specific filtering at model/variant level
        filtered_output_results = None
        if output_data_type == 'tc_risk' and output_io_type == 'output':
            if verbose:
                print("\nApplying TC-RISK specific filters for Criterion #3 (Insufficient Draw Files)...")
                
            # Run the specialized filtering function (now returns model/variant tuples)
            filtered_output_results = filter_tc_risk_model_variants(output_validation_dict)
            
            # Identify model/variants with fatal, low-level file count errors in existing output
            variants_with_file_count_issues = set(filtered_output_results["incomplete_file_count"].keys())
            
            # Exclude model/variants that have existing, but fatally incomplete output
            model_variants_to_process = model_variants_to_process - variants_with_file_count_issues
            
            if verbose and variants_with_file_count_issues:
                print(f"⚠️ Excluded {len(variants_with_file_count_issues)} model/variant combinations due to file count failures in existing TC-Risk output.")
                if verbose:
                    for model_variant in list(variants_with_file_count_issues)[:5]:  # Show first 5
                        print(f"   - {model_variant[0]}/{model_variant[1]}")
                    if len(variants_with_file_count_issues) > 5:
                        print(f"   ... and {len(variants_with_file_count_issues) - 5} more")

    else: 
        print("⚠️ Rerun all flag is set. All complete input model/variants will be reprocessed.")
        model_variants_to_process = input_complete_model_variants
        output_complete_model_variants = set()
        output_validation_dict = None
        filtered_output_results = None
        
    # Filter input validation results to only include model/variants that need processing
    input_results_for_tasks = {}
    for model_name, model_data in input_validation_dict['validation_results'].items():
        if 'variant' in model_data:
            # Check if any variant of this model needs processing
            variants_to_include = {}
            for variant_name, variant_data in model_data['variant'].items():
                if (model_name, variant_name) in model_variants_to_process:
                    variants_to_include[variant_name] = variant_data
            
            # Only include model if it has variants to process
            if variants_to_include:
                filtered_model_data = model_data.copy()
                filtered_model_data['variant'] = variants_to_include
                input_results_for_tasks[model_name] = filtered_model_data
        else:
            # No variant structure - check if model itself needs processing
            if model_name in model_variants_to_process:
                input_results_for_tasks[model_name] = model_data
    
    # Re-create dict for model_variants_to_process, preserving the strict flag context
    model_variants_to_process_dict = create_validation_dict(
        input_data_type, input_io_type, data_source, 
        validation_results=input_results_for_tasks,
        strict_grid_check=strict_grid_check_flag
    )

    validation_info = {
        "model_variants_to_process_dict": model_variants_to_process_dict,
        "model_variants_to_process": model_variants_to_process,  # Set of (model, variant) tuples
        "input_complete_model_variants": input_complete_model_variants,
        "output_complete_model_variants": output_complete_model_variants,
        "input_validation_dict": input_validation_dict,
        "output_validation_dict": output_validation_dict,
        "filtered_output_results": filtered_output_results
    }

    if verbose:
        print(f"\n✓ {len(model_variants_to_process)} model/variant combinations ready for processing.")

    if produce_missing_input_report:
        analyze_validation_for_missing_data(validation_info, save_csv=True)
    return validation_info
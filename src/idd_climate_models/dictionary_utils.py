import copy
import idd_climate_models.constants as rfc

def snip_validation_results(validation_dict, detail_level):
    """
    Recursively copies the nested validation results up to the specified detail_level, 
    preserving the complete/issues metadata at each level.
    """
    
    data_type = validation_dict['data_type']
    full_validation_results = validation_dict.get('validation_results', {}) 
    folder_structure = validation_dict['folder_structure']
    
    try:
        # stop_index is the index of the level we want to *include* in the output 
        stop_index = folder_structure.index(detail_level)
    except ValueError:
        print(f"⚠️ WARNING: Invalid detail level '{detail_level}'. Returning full validation results.")
        return copy.deepcopy(full_validation_results) 

    # --- Recursive Helper Function (Starts processing from index 1: 'variant') ---
    def _recursive_trim(current_results, current_index):
        """
        Recursively processes a dictionary of results from the current_index down.
        """
        trimmed_node = {}

        # Determine the name of the next level's key (e.g., 'scenario' if current is 'variant')
        next_level_name = folder_structure[current_index + 1] if current_index + 1 < len(folder_structure) else None
        
        for name, child_results in current_results.items():
            
            new_node = {
                'complete': child_results.get('complete', False),
                'issues': child_results.get('issues', [])
            }

            # If we are NOT at the final level AND the next level key exists in the data, recurse.
            # Recursion stops when current_index == stop_index.
            if current_index < stop_index and next_level_name and next_level_name in child_results:
                
                # Recursively trim the contents of the next level's key
                new_node[next_level_name] = _recursive_trim(
                    child_results[next_level_name],
                    current_index + 1
                )
            
            trimmed_node[name] = new_node
        
        return trimmed_node

    # --- Start the trimming process from the top level ('model' at index 0) ---
    trimmed_results = {}
    
    # The first level *after* 'model'
    first_child_level_name = folder_structure[1]
    
    for model_name, model_results in full_validation_results.items():
        # Copy the model's top-level metadata first
        trimmed_model_data = {
            'complete': model_results.get('complete', False),
            'issues': model_results.get('issues', [])
        }
        
        # Only start recursion if the detail_level is beyond the 'model' level (i.e., stop_index > 0)
        if stop_index > 0 and first_child_level_name in model_results:
            
            # Start the recursion from the dictionary that holds the variant results,
            # passing index 1 (for the 'variant' level)
            trimmed_model_data[first_child_level_name] = _recursive_trim(
                model_results[first_child_level_name],
                1
            )
            
        trimmed_results[model_name] = trimmed_model_data
            
    return trimmed_results

def parse_results(validation_dict, detail='variant'):
    """
    If detail='all', collect every file with its path and fill_required flag,
    plus all parent keys. Otherwise, collect up to the specified detail level.
    """
    folder_levels = rfc.FOLDER_STRUCTURE[validation_dict['data_type']]
    flat_path_list = []

    def recursive_collect(data, levels, context):
        current_level = levels[0]
        children = data.get(current_level, {})
        for name, child_data in children.items():
            new_context = context.copy()
            new_context[current_level] = name
            if len(levels) == 1:
                # At the leaf level, collect file info if detail='all'
                if detail == 'all' and 'files' in child_data:
                    for file_meta in child_data['files']:
                        flat_path_list.append({
                            **new_context,
                            'file_path': file_meta['path'],
                            'fill_required': file_meta.get('fill_required', False)
                        })
                else:
                    flat_path_list.append(new_context)
            else:
                recursive_collect(child_data, levels[1:], new_context)

    # Determine levels to use
    if detail == 'all':
        levels_to_use = folder_levels[1:]  # skip 'model', handled separately
    else:
        levels_to_use = folder_levels[:folder_levels.index(detail)+1][1:]
    # Start from the top level (model)
    for model, data in validation_dict['validation_results'].items():
        context = {'model': model}
        if not levels_to_use:
            flat_path_list.append(context)
        else:
            recursive_collect(data, levels_to_use, context)
    return flat_path_list

def nest_parsed_results(flat_path_list, data_type):

    if not flat_path_list:
        return {}
    full_levels = ['model'] + rfc.FOLDER_STRUCTURE.get(data_type, [])
    first_item_keys = set(flat_path_list[0].keys())
    nesting_levels = [level for level in full_levels if level in first_item_keys]
    if len(nesting_levels) < 2:
        return {} # Cannot create key-value pair if there aren't two final levels
    terminal_key = nesting_levels[-1]        # e.g., 'grid'
    penultimate_key = nesting_levels[-2]   # e.g., 'variable'
    
    nested_results = {}

    for path_dict in flat_path_list:
        model_name = path_dict['model']
        if model_name not in nested_results:
            nested_results[model_name] = {}
        current_level_ptr = nested_results[model_name]
        for i, level_key in enumerate(nesting_levels):
            name = path_dict[level_key]
            if level_key == 'model':
                continue # Skip model level
            if level_key == penultimate_key:
                current_level_ptr[name] = path_dict[terminal_key]
            else:
                if name not in current_level_ptr:
                    current_level_ptr[name] = {}
                current_level_ptr = current_level_ptr[name]

    return nested_results

def find_first_failure(validation_data, path="Model"):
    if 'complete' in validation_data and validation_data['complete'] is False:
        issues = validation_data.get('issues', [])
        if issues:
            return (
                f"Validation Failed at: **{path}**\n"
                f"Issues ({len(issues)}): {'; '.join(issues)}"
            )
    for key, child_data in validation_data.items():
        if isinstance(child_data, dict) and key not in ['issues', 'complete', 'files']:
            for child_name, nested_result in child_data.items():
                new_path = f"{path} -> {key}={child_name}"
                failure_summary = find_first_failure(nested_result, new_path)
                if failure_summary:
                    return failure_summary   
    return None

def summarize_all_failures(validation_dict):
    """
    Iterates through all models in the validation_dict, uses find_first_failure
    to locate the highest-level issue for each incomplete model, and returns 
    a summary dictionary.
    
    Args:
        validation_dict (dict): The dictionary containing 'validation_results'.
        
    Returns:
        dict: A dictionary of {model_name: failure_summary_string}.
    """
    
    all_results = validation_dict['validation_results']
    failure_summaries = {}
    
    print("\n" + "=" * 80)
    print("ANALYZING FAILURES (Highest-Level Issue Per Incomplete Model)")
    print("=" * 80)

    for model_name, model_data in all_results.items():
        
        # 1. Check if the model is incomplete at the top level
        if not model_data.get('complete', False):
            
            # 2. Use find_first_failure to pinpoint the exact failure location.
            # We start the search one level down, passing the entire nested result.
            failure_message = find_first_failure(model_data, path=f"Model={model_name}")
            
            # The result should always be found if the top level is 'False'
            if failure_message:
                summary = f"✗ INCOMPLETE: {failure_message}"
                failure_summaries[model_name] = summary
                print(summary)
            else:
                # Should not happen if data structure is valid
                failure_summaries[model_name] = "✗ INCOMPLETE: Failed at model root, but no specific issue found deeper."
                print(f"Model={model_name}: {failure_summaries[model_name]}")
        
    print("=" * 80)
    return failure_summaries

def get_models(validation_dict, complete_only=False):
    """
    Extracts a set of model names from the validation_dict.
    
    Args:
        validation_dict (dict): The dictionary containing 'validation_results'.
        complete_only (bool): If True, only include models marked as complete.
        
    Returns:
        set: A set of model names.
    """
    models = set()
    for model_name, model_data in validation_dict['validation_results'].items():
        if complete_only:
            if model_data.get('complete', False):
                models.add(model_name)
        else:
            models.add(model_name)
    return models



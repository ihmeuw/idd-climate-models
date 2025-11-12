import os

# ============================================================================
# MODEL STRUCTURE UTILITY FUNCTIONS
# ============================================================================

def get_complete_models(validation_results):
    """Get only the complete models from validation results."""
    return {model: data for model, data in validation_results.items() 
            if data.get('complete', False)}


def get_incomplete_models(validation_results):
    """Get only the incomplete models from validation results."""
    return {model: data for model, data in validation_results.items() 
            if not data.get('complete', False)}


def iterate_model_files(validation_results, complete_only=True):
    """
    Generator that yields file paths from validation results.
    
    Parameters:
    -----------
    validation_results : dict
        Results from validate_all_models()
    complete_only : bool
        If True, only iterate over complete model structures
        
    Yields:
    -------
    tuple : (model_name, variant_name, scenario_name, variable_name, 
             grid_name, frequency_name, file_info)
    """
    for model_name, model_data in validation_results.items():
        if complete_only and not model_data.get('complete'):
            continue
            
        for variant_name, variant_data in model_data.get('variants', {}).items():
            if complete_only and not variant_data.get('complete'):
                continue
                
            for scenario_name, scenario_data in variant_data.get('scenarios', {}).items():
                if complete_only and not scenario_data.get('complete'):
                    continue
                    
                for variable_name, variable_data in scenario_data.get('variables', {}).items():
                    if complete_only and not variable_data.get('complete'):
                        continue
                        
                    for grid_name, grid_data in variable_data.get('grids', {}).items():
                        if complete_only and not grid_data.get('complete'):
                            continue
                            
                        for frequency_name, frequency_data in grid_data.get('frequencys', {}).items():
                            if complete_only and not frequency_data.get('complete'):
                                continue
                                
                            for file_info in frequency_data.get('files', []):
                                yield (model_name, variant_name, scenario_name, 
                                      variable_name, grid_name, frequency_name, file_info)


def count_files_by_status(validation_results):
    """
    Count files by their completion status.
    
    Returns:
    --------
    dict : Statistics about file counts
    """
    stats = {
        'total_files': 0,
        'complete_files': 0,
        'fill_required_files': 0,
        'models': {},
        'scenarios': {},
        'variables': {}
    }
    
    for model_name, model_data in validation_results.items():
        model_stats = {'total': 0, 'complete': 0, 'fill_required': 0}
        
        for file_tuple in iterate_model_files(validation_results, complete_only=False):
            model, variant, scenario, variable, grid, frequency, file_info = file_tuple
            
            if model != model_name:
                continue
                
            stats['total_files'] += 1
            model_stats['total'] += 1
            
            # Count by scenario
            if scenario not in stats['scenarios']:
                stats['scenarios'][scenario] = {'total': 0, 'complete': 0, 'fill_required': 0}
            stats['scenarios'][scenario]['total'] += 1
            
            # Count by variable
            if variable not in stats['variables']:
                stats['variables'][variable] = {'total': 0, 'complete': 0, 'fill_required': 0}
            stats['variables'][variable]['total'] += 1
            
            # Check if file needs filling
            if isinstance(file_info, dict) and file_info.get('fill_required', False):
                stats['fill_required_files'] += 1
                model_stats['fill_required'] += 1
                stats['scenarios'][scenario]['fill_required'] += 1
                stats['variables'][variable]['fill_required'] += 1
            else:
                stats['complete_files'] += 1
                model_stats['complete'] += 1
                stats['scenarios'][scenario]['complete'] += 1
                stats['variables'][variable]['complete'] += 1
        
        stats['models'][model_name] = model_stats
    
    return stats


def print_model_tree(validation_results, show_files=False, complete_only=True, max_depth=None):
    """
    Print a tree view of the model structure.
    
    Parameters:
    -----------
    validation_results : dict
        Results from validate_all_models()
    show_files : bool
        Whether to show individual files
    complete_only : bool
        Only show complete structures
    max_depth : int or None
        Maximum depth to show (None for unlimited)
    """
    def get_status_symbol(data):
        if data.get('complete', False):
            return "✓"
        elif data.get('issues', []):
            return "✗"
        else:
            return "?"
    
    print("Model Structure:")
    print("=" * 50)
    
    for model_name, model_data in validation_results.items():
        if complete_only and not model_data.get('complete'):
            continue
            
        print(f"{get_status_symbol(model_data)} {model_name}")
        
        if max_depth is not None and max_depth <= 1:
            continue
            
        for variant_name, variant_data in model_data.get('variants', {}).items():
            if complete_only and not variant_data.get('complete'):
                continue
                
            print(f"  {get_status_symbol(variant_data)} {variant_name}")
            
            if max_depth is not None and max_depth <= 2:
                continue
                
            for scenario_name, scenario_data in variant_data.get('scenarios', {}).items():
                if complete_only and not scenario_data.get('complete'):
                    continue
                    
                print(f"    {get_status_symbol(scenario_data)} {scenario_name}")
                
                if max_depth is not None and max_depth <= 3:
                    continue
                    
                for variable_name, variable_data in scenario_data.get('variables', {}).items():
                    if complete_only and not variable_data.get('complete'):
                        continue
                        
                    print(f"      {get_status_symbol(variable_data)} {variable_name}")
                    
                    if max_depth is not None and max_depth <= 4:
                        continue
                        
                    for grid_name, grid_data in variable_data.get('grids', {}).items():
                        if complete_only and not grid_data.get('complete'):
                            continue
                            
                        print(f"        {get_status_symbol(grid_data)} {grid_name}")
                        
                        if max_depth is not None and max_depth <= 5:
                            continue
                            
                        for frequency_name, frequency_data in grid_data.get('frequencys', {}).items():
                            if complete_only and not frequency_data.get('complete'):
                                continue
                                
                            file_count = len(frequency_data.get('files', []))
                            print(f"          {get_status_symbol(frequency_data)} {frequency_name} ({file_count} files)")
                            
                            if show_files and (max_depth is None or max_depth > 6):
                                for file_info in frequency_data.get('files', []):
                                    if isinstance(file_info, dict):
                                        file_path = file_info.get('path', 'Unknown path')
                                        fill_flag = " [FILL]" if file_info.get('fill_required') else ""
                                    else:
                                        file_path = file_info
                                        fill_flag = ""
                                    
                                    filename = os.path.basename(file_path)
                                    print(f"            📄 {filename}{fill_flag}")


def get_model_summary(validation_results):
    """
    Generate a summary of model validation results.
    
    Returns:
    --------
    dict : Summary statistics
    """
    summary = {
        'total_models': len(validation_results),
        'complete_models': 0,
        'total_variants': 0,
        'complete_variants': 0,
        'scenarios': set(),
        'variables': set(),
        'grids': set(),
        'frequencys': set(),
        'issues_summary': {}
    }
    
    all_issues = []
    
    for model_name, model_data in validation_results.items():
        if model_data.get('complete'):
            summary['complete_models'] += 1
            
        # Collect all issues
        all_issues.extend(model_data.get('issues', []))
        
        for variant_name, variant_data in model_data.get('variants', {}).items():
            summary['total_variants'] += 1
            if variant_data.get('complete'):
                summary['complete_variants'] += 1
                
            all_issues.extend(variant_data.get('issues', []))
            
            for scenario_name, scenario_data in variant_data.get('scenarios', {}).items():
                summary['scenarios'].add(scenario_name)
                all_issues.extend(scenario_data.get('issues', []))
                
                for variable_name, variable_data in scenario_data.get('variables', {}).items():
                    summary['variables'].add(variable_name)
                    all_issues.extend(variable_data.get('issues', []))
                    
                    for grid_name, grid_data in variable_data.get('grids', {}).items():
                        summary['grids'].add(grid_name)
                        all_issues.extend(grid_data.get('issues', []))
                        
                        for frequency_name, frequency_data in grid_data.get('frequencys', {}).items():
                            summary['frequencys'].add(frequency_name)
                            all_issues.extend(frequency_data.get('issues', []))
    
    # Summarize issues
    issue_counts = {}
    for issue in all_issues:
        # Extract issue data_type (first part before colon if present)
        issue_data_type = issue.split(':')[0] if ':' in issue else issue[:50]
        issue_counts[issue_data_type] = issue_counts.get(issue_data_type, 0) + 1
    
    summary['issues_summary'] = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Convert sets to sorted lists
    for key in ['scenarios', 'variables', 'grids', 'frequencys']:
        summary[key] = sorted(list(summary[key]))
    
    return summary


def print_model_summary(validation_results):
    """Print a formatted summary of model validation results."""
    summary = get_model_summary(validation_results)
    
    print("Model Validation Summary")
    print("=" * 50)
    print(f"Models: {summary['complete_models']}/{summary['total_models']} complete "
          f"({100*summary['complete_models']/summary['total_models']:.1f}%)")
    print(f"Variants: {summary['complete_variants']}/{summary['total_variants']} complete "
          f"({100*summary['complete_variants']/summary['total_variants']:.1f}%)")
    print()
    
    print(f"Scenarios found: {', '.join(summary['scenarios'])}")
    print(f"Variables found: {', '.join(summary['variables'])}")
    print(f"Grids found: {', '.join(summary['grids'])}")
    print(f"Time periods found: {', '.join(summary['frequencys'])}")
    print()
    
    if summary['issues_summary']:
        print("Most common issues:")
        for issue_data_type, count in list(summary['issues_summary'].items())[:5]:
            print(f"  {issue_data_type}: {count} occurrences")
        if len(summary['issues_summary']) > 5:
            print(f"  ... and {len(summary['issues_summary']) - 5} other issue data_types")
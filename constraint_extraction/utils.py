import json
import numpy as np
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def find_schema_column(schema, table_name, data_col_name):
    """Map a data column name to its schema column name."""
    table_upper = table_name.upper()

    # Handle missing tables
    if table_upper not in schema:
        return None
    
    # Normalize both data and schema columns for comparison
    def normalize(name):
        return name.upper().replace(' ', '').replace('_', '').replace('-', '')
    
    data_normalized = normalize(data_col_name)
    
    # Find matching schema column
    for schema_col in schema[table_upper].keys():
        if normalize(schema_col) == data_normalized:
            return schema_col  # Return schema version
    
    # No match found
    return None

def check_dependency(df, determinant_col, dependent_col):
    """Check if functional dependency exists between determinant_col and dependent_col."""
    valid_df = df[[determinant_col, dependent_col]].dropna()
    
    if len(valid_df) == 0:
        return {'is_dependent': False}
    
    # Group by determinant and check if dependent has only one unique value per group
    grouped = valid_df.groupby(determinant_col)[dependent_col]
    violations = (grouped.nunique() > 1).sum()
    
    # Require perfect dependency (0 violations)
    if violations > 0:
        return {'is_dependent': False}
    
    return {'is_dependent': True}


def check_ordering_dependency(df, col_a, col_b):
    """Check if col_a has ordering relationship with col_b."""
    # Drop rows where either column is null
    valid_df = df[[col_a, col_b]].dropna()
    
    if len(valid_df) == 0:
        return {'is_ordered': False}
    
    # Try to convert to numeric for comparison
    try:
        col_a_numeric = pd.to_numeric(valid_df[col_a], errors='coerce')
        col_b_numeric = pd.to_numeric(valid_df[col_b], errors='coerce')
        
        # Drop rows that couldn't be converted
        valid_mask = col_a_numeric.notna() & col_b_numeric.notna()
        col_a_numeric = col_a_numeric[valid_mask]
        col_b_numeric = col_b_numeric[valid_mask]
        
        if len(col_a_numeric) == 0:
            return {'is_ordered': False}
        
    except:
        return {'is_ordered': False}
    
    # Check if col_a >= col_b for ALL rows (no violations)
    violations_gte = (col_a_numeric < col_b_numeric).sum()
    
    # Check if col_a <= col_b for ALL rows (no violations)
    violations_lte = (col_a_numeric > col_b_numeric).sum()
    
    # Must have ordering AND columns must not be identical
    if violations_gte == 0 and not col_a_numeric.equals(col_b_numeric):
        return {'is_ordered': True, 'operator': '>='}
    elif violations_lte == 0 and not col_a_numeric.equals(col_b_numeric):
        return {'is_ordered': True, 'operator': '<='}
    else:
        return {'is_ordered': False}
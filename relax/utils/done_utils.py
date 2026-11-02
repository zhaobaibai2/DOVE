"""
Utility functions for safe done handling across scalar and array inputs
"""
import numpy as np


def combine_done(terminated, truncated):
    """
    Safely combine terminated and truncated for scalar/array inputs
    
    Args:
        terminated: bool or np.ndarray - termination flag
        truncated: bool or np.ndarray - truncation flag
        
    Returns:
        bool or np.ndarray - combined done flag
    """
    # Handle scalar inputs (most common case for single env)
    if not isinstance(terminated, np.ndarray) and not isinstance(truncated, np.ndarray):
        return bool(terminated) or bool(truncated)
    
    # Handle array inputs (for vector env compatibility, though we're using single env)
    if isinstance(terminated, np.ndarray):
        term_array = terminated
    else:
        term_array = np.array([terminated])
        
    if isinstance(truncated, np.ndarray):
        trunc_array = truncated
    else:
        trunc_array = np.array([truncated])
        
    return np.logical_or(term_array, trunc_array)

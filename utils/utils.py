"""
Very basic utitlity functions

Copyright (c) 2024 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2024-02-01"

from torch import Tensor

def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Example usage
nested_dict = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}, 'f': 4}
flat_dict = flatten_dict(nested_dict)
print(flat_dict)

def make_lists_strings_in_dict(d):
    for k,v in d.items():
        if isinstance(v, list):
            d[k] = str(v)
    return d

def conformity_improvement(conformity_score:float, conformal_p: float) -> float:
    """
    Improve the conformal p-value by using the conformity score.

    The idea is to calculate how much we would need to squish the conformity_score to reach the conformal_p and then apply the same
    squishing to the conformal_p.

    If it is already calibrated then conformity_score = conformal_p and the result will return conformal_p again.

    Parameters
    ----------
    conformity_score : float
        The conformity score on a calibration/validation set.
    conformal_p : float
        The conformal p-value that is intended.
    """
    
    conformal_p_new = 1 - (1 - conformal_p)**2 / (1 - conformity_score)
    #conformal_p_new = 1 - (1 - conformal_p) * (1 - conformal_p_new) / (1 - conformity_score)
    return conformal_p_new

def make_to_pass_precomputed_variables(
    precomputed_variables: dict, num_steps: int, idx: int
):
    to_pass_precomputed_variables = {}
    for key, value in precomputed_variables.items():
        if isinstance(value, tuple):
            new_tuple = tuple(
                tup_element[idx].unsqueeze(0).expand(num_steps, *tup_element[idx].shape)
                for tup_element in value
            )
            to_pass_precomputed_variables[key] = new_tuple
        elif isinstance(value, Tensor):
            to_pass_precomputed_variables[key] = (
                value[idx].unsqueeze(0).expand(num_steps, *value[idx].shape)
            )
        else:
            raise ValueError(
                f"precomputed_variables must be a dict of tensors or tuples. {key} is not."
            )
    return to_pass_precomputed_variables
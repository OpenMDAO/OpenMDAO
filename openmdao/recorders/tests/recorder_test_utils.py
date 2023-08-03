import time

import numpy as np


def run_driver(problem, **kwargs):
    t0 = time.perf_counter()
    problem.run_driver(**kwargs)
    t1 = time.perf_counter()
    return t0, t1


def assert_model_matches_case(case, system):
    """
    Check to see if the values in the case match those in the model.

    Parameters
    ----------
    case: Case object
        Case to be used for the comparison.
    system: System object
        System to be used for the comparison.
    """
    case_inputs = case.inputs
    model_inputs = system._inputs
    for name, model_input in model_inputs._abs_item_iter(flat=False):
        np.testing.assert_almost_equal(case_inputs[name], model_input,
                                       err_msg=f"The value for input '{name}' in the model, "
                                               f"{model_input}, does not match the value "
                                               f"recorded in the case, {case_inputs[name]}")

    case_outputs = case.outputs
    model_outputs = system._outputs
    for name, model_output in model_outputs._abs_item_iter(flat=False):
        np.testing.assert_almost_equal(case_outputs[name], model_output,
                                       err_msg=f"The value for output '{name}' in the model, "
                                               f"{model_output}, does not match the value "
                                               f"recorded in the case, {case_outputs[name]}")

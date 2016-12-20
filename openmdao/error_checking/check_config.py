"""A module containing various configuration checks for an OpenMDAO Problem."""

import logging

from openmdao.devtools.compat import abs_varname_iter


def check_config(problem, logger=None):
    """Perform optional error checks on a Problem.

    Args
    ----
    problem : Problem
        The Problem being checked.

    logger : object
        Logging object.

    """
    if logger is None:
        logger = logging.getLogger()
        console = logging.StreamHandler()
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)
        logger.addHandler(console)

    _check_hanging_inputs(problem, logger)


def _check_hanging_inputs(problem, logger):
    """Warn if any inputs are not connected."""
    input_src_ids = problem._assembler._input_src_ids

    hanging = sorted([
        name for i, name in enumerate(abs_varname_iter(problem.root, 'input',
                                                       local=False)) if
                                                       input_src_ids[i] == -1
    ])

    if hanging:
        logger.warning("The following inputs are not connected: %s." % hanging)

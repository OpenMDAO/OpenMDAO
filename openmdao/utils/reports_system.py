"""
Utility functions related to the reporting system which generates reports by default for all runs.
"""
from mpi4py import MPI

from collections import namedtuple
import pathlib
import sys
import os

from openmdao.utils.coloring import compute_total_coloring
from openmdao.utils.mpi import MPI

_reports_dir = '.'  # the default location for the reports

def set_reports_dir(reports_dir_path):
    """
    Set the path to where the reports should go. Normally, they go into the current directory.

    Parameters
    ----------
    reports_dir_path : str
        Path to where the report directories should go.
    """
    global _reports_dir
    _reports_dir = reports_dir_path

def get_reports_dir(prob):
    reports_dir = os.environ.get('OPENMDAO_REPORTS_DIR', _reports_dir)

    problem_reports_dirname = f'{prob._name}_reports'
    problem_reports_dirpath = pathlib.Path(reports_dir).joinpath(problem_reports_dirname)

    return problem_reports_dirpath

def register_report(func, method, pre_or_post, class_name, inst_id=None, **kwargs):
    """
    Register a report with the reporting system.

    Parameters
    ----------
    func : function
        A function to do the reporting. Expects the first argument to be a Problem instance.
    desc : str
        A description of the report.
    method : str
        In which method of the Problem should this be run.
    pre_or_post : str
        Valid values are 'pre' and 'post'. Indicates when to run the report in the method.
    probname : str or None
        Either the name of a Problem or None. If None, then this report will be run for all
        Problems.
    **kwargs : dict
        Optional args for the reporting function.
    """

    from openmdao.utils.hooks import _register_hook
    if pre_or_post == 'post':
        _register_hook(method, class_name, post=func, ncalls=1, inst_id=inst_id)
    elif pre_or_post == 'pre':
        _register_hook(method, class_name, pre=func, ncalls=1, inst_id=inst_id)
    else:
        raise ValueError(f"The argument 'pre_or_post' can only have values of 'pre' or 'post', but {pre_or_post} was given")

    return

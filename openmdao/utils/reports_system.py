"""
Utility functions related to the reporting system which generates reports by default for all runs.
"""

from collections import namedtuple, defaultdict
from functools import wraps
import pathlib
import sys
import os

from openmdao.utils.mpi import MPI
from openmdao.utils.hooks import _register_hook, _unregister_hook
from openmdao.visualization.n2_viewer.n2_viewer import n2, _default_n2_filename
from openmdao.visualization.scaling_viewer.scaling_report import _default_scaling_filename
from openmdao.core.constants import _UNDEFINED
from openmdao.core.problem import Problem
from openmdao.core.driver import Driver

# Keeping track of the registered reports
_Report = namedtuple('Report', 'func desc class_name inst_id method pre_or_post report_filename')
_reports_registry = {}

_reports_dir = './reports'  # the default location for the reports


def _is_rank_0(prob):
    # Utility function to determine if MPI and on rank0 or not on MPI at all
    return not MPI or prob.comm.rank == 0


def report_function():
    """
    Decorate report functions. Handles getting the file path to where the report is written.

    Returns
    -------
    function
        The wrapper function.
    """
    def decorate(f):
        @wraps(f)
        def _wrapper(inst, **kwargs):
            if isinstance(inst, Problem):
                prob = inst
            elif isinstance(inst, Driver):
                prob = inst._problem()
            else:
                raise ValueError("User defined reports currently can only be registered "
                                 "on Problems and Drivers")

            problem_reports_dirpath = get_reports_dir(prob)
            if _is_rank_0(prob):
                pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

            report_filename = kwargs['report_filename']
            user_defined_report_filepath = \
                str(pathlib.Path(problem_reports_dirpath).joinpath(report_filename))

            f(inst, user_defined_report_filepath)
        return _wrapper
    return decorate


def register_report(name, func, desc, class_name, method, pre_or_post, report_filename,
                    inst_id=None):
    """
    Register a report with the reporting system.

    Parameters
    ----------
    name : str
        Name of report. Report names must be unique across all reports.
    func : function
        A function to do the reporting. Expects the first argument to be an instance of class_name.
    desc : str
        A description of the report.
    class_name : str
        The name of the class owning the method where the report will be run.
    method : str
        In which method of class_name should this be run.
    pre_or_post : str
        Valid values are 'pre' and 'post'. Indicates when to run the report in the method.
    report_filename : str
        Name of file to use when saving the report.
    inst_id : str or None
        Either the instance ID of an OpenMDAO object (e.g. Problem, Driver) or None.
        If None, then this report will be run for all objects of type class_name.
    """
    global _reports_registry

    report = _Report(func, desc, class_name, inst_id, method, pre_or_post, report_filename)
    if name in _reports_registry:
        raise ValueError(f"A report with the name {name} already exists")
    _reports_registry[name] = report

    if pre_or_post == 'pre':
        _register_hook(method, class_name, pre=func, inst_id=inst_id,
                       report_filename=report_filename)
    elif pre_or_post == 'post':
        _register_hook(method, class_name, post=func, inst_id=inst_id,
                       report_filename=report_filename)
    else:
        raise ValueError(
            f"The argument 'pre_or_post' can only have values of 'pre' or 'post', but {pre_or_post}"
            " was given")
    return


def list_reports(out_stream=None):
    """
    Write table of information about reports currently registered in the reporting system.

    Parameters
    ----------
    out_stream : file-like object
        Where to send report info.
    """
    global _reports_registry

    if not out_stream:
        out_stream = sys.stdout

    column_names = ['name', 'desc', 'class_name', 'inst_id', 'method', 'pre_or_post', 'func']
    column_widths = {}
    # Determine the column widths of the data fields by finding the max width for all rows
    # First for the headers
    for column_name in column_names:
        column_widths[column_name] = len(column_name)

    # Now for the values
    for name, report in _reports_registry.items():
        for column_name in column_names:
            if column_name == 'name':
                val = name
            else:
                val = getattr(report, column_name)
                if column_name == 'func':
                    val = val.__name__
                else:
                    val = str(val)
            column_widths[column_name] = max(column_widths[column_name], len(val))

    out_stream.write("\nHere are the reports available:\n\n")

    column_header = ''
    column_dashes = ''
    column_spacing = 2
    for i, column_name in enumerate(column_names):
        column_header += '{:{width}}'.format(column_name, width=column_widths[column_name])
        column_dashes += column_widths[column_name] * '-'
        if i < len(column_names) - 1:
            column_header += column_spacing * ' '
            column_dashes += column_spacing * ' '

    out_stream.write('\n')
    out_stream.write(column_header + '\n')
    out_stream.write(column_dashes + '\n')

    for name, report in _reports_registry.items():
        report_info = ''
        for i, column_name in enumerate(column_names):
            if column_name == 'name':
                val = name
            else:
                val = getattr(report, column_name)
                if column_name == 'func':
                    val = val.__name__
                else:
                    val = str(val)
            val_formatted = f"{val:<{column_widths[column_name]}}"
            report_info += val_formatted
            if i < len(column_names) - 1:
                report_info += column_spacing * ' '

        out_stream.write(report_info + '\n')

    out_stream.write('\n')


def set_default_reports_dir(reports_dir_path):
    """
    Set the path to where the reports should go. By default, they go into the current directory.

    Parameters
    ----------
    reports_dir_path : str
        Path to where the report directories should go.
    """
    global _reports_dir
    _reports_dir = reports_dir_path


def get_reports_dir(prob):
    """
    Get the path to the directory where the report files should go.

    Parameters
    ----------
    prob : OpenMDAO Problem object
        The report will be run on this Problem.

    Returns
    -------
    str
        The path to the directory where reports should be written.
    """
    if prob._reports_dir is not _UNDEFINED:
        reports_dir = prob._reports_dir
    else:
        reports_dir = os.environ.get('OPENMDAO_REPORTS_DIR', _reports_dir)

    # problem_reports_dirname = f'{prob._name}_reports'
    problem_reports_dirname = f'{prob._name}'
    problem_reports_dirpath = pathlib.Path(reports_dir).joinpath(problem_reports_dirname)

    return problem_reports_dirpath


def clear_reports():
    """
    Clear all of the reports from the registry.
    """
    global _reports_registry

    # need to remove the hooks
    for name, report in _reports_registry.items():
        if getattr(report, 'pre_or_post') == "pre":
            _unregister_hook(getattr(report, 'method'), getattr(report, 'class_name'),
                             inst_id=getattr(report, 'inst_id'), pre=getattr(report, 'func'))
        else:
            _unregister_hook(getattr(report, 'method'), getattr(report, 'class_name'),
                             inst_id=getattr(report, 'inst_id'), post=getattr(report, 'func'))
    _reports_registry = {}


def _should_report_run(reports, report_name):
    # Utility function that checks the _reports attribute on Problem
    #   to determine whether the report named "report_name" should be run
    if isinstance(reports, str):
        reports_on = reports.split(',')
        if report_name not in reports_on:
            return False
    elif isinstance(reports, bool):
        if not reports:
            return False
    elif reports is None:
        return False

    return True


# N2 report definition
# Need to create these closures so that functions can keep track of how many times they have
# been called per Problem (or Driver). In the case of the n2, it is Problem
def _run_n2_report_enclosing():
    def run_n2_report_inner(prob, report_filename=None):

        if not _should_report_run(prob._reports, 'n2'):
            return

        run_n2_report_inner.calls[prob] += 1
        if run_n2_report_inner.calls[prob] > 1:  # Only do the report once per Problem
            return

        problem_reports_dirpath = get_reports_dir(prob)
        if _is_rank_0(prob):
            pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

        n2_filepath = str(pathlib.Path(problem_reports_dirpath).joinpath(report_filename))
        try:
            n2(prob, show_browser=False, outfile=n2_filepath, display_in_notebook=False)
        except RuntimeError as err:
            # We ignore this error
            if str(err) != "Can't compute total derivatives unless " \
                           "both 'of' or 'wrt' variables have been specified.":
                raise err

    run_n2_report_inner.calls = defaultdict(int)
    return run_n2_report_inner


run_n2_report = _run_n2_report_enclosing()


# scaling report definition
def _run_scaling_report_enclosing():
    # def run_scaling_report_inner(driver):
    def run_scaling_report_inner(driver, report_filename=None):

        prob = driver._problem()

        if not _should_report_run(prob._reports, 'scaling'):
            return

        run_scaling_report_inner.calls[driver] += 1
        if run_scaling_report_inner.calls[driver] > 1:
            return

        problem_reports_dirpath = get_reports_dir(prob)

        scaling_filepath = str(
            pathlib.Path(problem_reports_dirpath).joinpath(report_filename))
        if _is_rank_0(prob):
            pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

        try:
            prob.driver.scaling_report(outfile=scaling_filepath, show_browser=False)

        # Need to handle the coloring and scaling reports which can fail in this way
        #   because total Jacobian can't be computed
        except RuntimeError as err:
            if str(err) != "Can't compute total derivatives unless " \
                           "both 'of' or 'wrt' variables have been specified.":
                raise err

    run_scaling_report_inner.calls = defaultdict(int)
    return run_scaling_report_inner


run_scaling_report = _run_scaling_report_enclosing()

_default_reports = {
    'n2': (run_n2_report, 'N2 diagram', 'Problem', 'final_setup', 'post', _default_n2_filename,
           None),
    'scaling': (run_scaling_report, 'Driver scaling report', 'Driver', '_compute_totals', 'post',
                _default_scaling_filename, None)
}


def setup_default_reports():
    """
    Set up the default reports for all OpenMDAO runs.
    """
    if 'TESTFLO_RUNNING' in os.environ:
        return

    if 'OPENMDAO_REPORTS' in os.environ:
        if os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off', "none"]:
            return  # do not do any reports

        if os.environ['OPENMDAO_REPORTS'] in ['1', 'true', 'on', "all"]:
            reports_on = _default_reports.keys()
        else:
            reports_on = os.environ['OPENMDAO_REPORTS'].split(',')
    else:  # if no env set, all reports are on
        reports_on = _default_reports.keys()

    for report_name, report_info in _default_reports.items():
        if report_name in reports_on:
            register_report(report_name, *report_info)


setup_default_reports()

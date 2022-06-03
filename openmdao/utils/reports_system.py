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
from openmdao.utils.om_warnings import issue_warning

# Keeping track of the registered reports
_Report = namedtuple('Report',
                     'func desc class_name inst_id condition method pre_or_post report_filename')
_reports_registry = {}
_default_reports = ['n2', 'scaling']
_active_reports = set()  # these reports will actually run (assuming their hook funcs are triggered)
_cmdline_reports = set()  # cmdline reports can be registered here to prevent default reports

_reports_dir = './reports'  # the default location for the reports


def _register_cmdline_report(name):
    global _cmdline_reports
    _cmdline_reports.add(name)


def register_report(name, func, desc, class_name, method, pre_or_post, filename, inst_id=None,
                    condition=None):
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
    filename : str
        Name of file to use when saving the report.
    inst_id : str or None
        Either the instance ID of an OpenMDAO object (e.g. Problem, Driver) or None.
        If None, then this report will be run for all objects of type class_name.
    condition : function or None
        Function taking the instance as an arg that returns True if the report should run for
        that instance.
    """
    global _reports_registry

    if name in _reports_registry:
        raise ValueError(f"A report with the name {name} was already registered.")
    if pre_or_post not in ('pre', 'post'):
        raise ValueError("The argument 'pre_or_post' can only have values of 'pre' or 'post', "
                         f"but {pre_or_post} was given")

    _reports_registry[name] = _Report(func, desc, class_name, inst_id, condition, method,
                                      pre_or_post, filename)


def activate_report(name):
    """
    Activate a report that has been registered with the reporting system.

    Parameters
    ----------
    name : str
        Name of report. Report names must be unique across all reports.
    """
    global _reports_registry, _active_reports

    if name in _active_reports:
        raise ValueError(f"A report with the name {name} is already active.")
    if name not in _reports_registry:
        raise ValueError(f"No report with the name {name} is registered.")
    if name in _cmdline_reports:
        return  # skip it if it's already being run from the command line

    func, _, class_name, inst_id, cond, method, pre_or_post, report_filename = \
        _reports_registry[name]

    _active_reports.add(name)

    if pre_or_post == 'pre':
        _register_hook(method, class_name, pre=func, inst_id=inst_id, ncalls=1, condition=cond,
                       report_filename=report_filename)
    else:  # post
        _register_hook(method, class_name, post=func, inst_id=inst_id, ncalls=1, condition=cond,
                       report_filename=report_filename)


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


def set_reports_dir(reports_dir_path):
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

    If it doesn't exist, it will be created.

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

    problem_reports_dirpath = pathlib.Path(reports_dir).joinpath(f'{prob._name}')

    if prob.comm.rank == 0:
        pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

    return problem_reports_dirpath


def clear_report_registry():
    """
    Clear all of the reports from the registry.
    """
    global _reports_registry

    _reports_registry = {}


def clear_reports():
    """
    Clear all of the currently active reports.
    """
    global _active_reports, _reports_registry

    # need to remove the hooks
    for name in _active_reports:
        func, _, class_name, inst_id, _, method, pre_or_post, _ = _reports_registry[name]
        if pre_or_post == "pre":
            _unregister_hook(method, class_name, inst_id=inst_id, pre=func)
        else:
            _unregister_hook(method, class_name, inst_id=inst_id, post=func)

    _active_reports = set()


def _should_report_run(reports, report_name):
    # Utility function that checks the _reports attribute on Problem
    #   to determine whether the report named "report_name" should be run
    if isinstance(reports, str):
        for r in reports.split(','):
            if report_name == r.strip():
                return True
        else:
            return False
    elif isinstance(reports, bool):
        return reports
    elif reports is None:
        return False

    return True


# N2 report definition
def _run_n2_report(prob, report_filename=None):

    n2_filepath = str(pathlib.Path(get_reports_dir(prob)).joinpath(report_filename))
    try:
        n2(prob, show_browser=False, outfile=n2_filepath, display_in_notebook=False)
    except RuntimeError as err:
        # We ignore this error
        if str(err) != "Can't compute total derivatives unless " \
                       "both 'of' or 'wrt' variables have been specified.":
            raise err


# scaling report definition
def _run_scaling_report(driver, report_filename=None):

    prob = driver._problem()
    scaling_filepath = str(pathlib.Path(get_reports_dir(prob)).joinpath(report_filename))

    try:
        prob.driver.scaling_report(outfile=scaling_filepath, show_browser=False)

    # Need to handle the coloring and scaling reports which can fail in this way
    #   because total Jacobian can't be computed
    except RuntimeError as err:
        if str(err) != "Can't compute total derivatives unless " \
                       "both 'of' or 'wrt' variables have been specified.":
            raise err


def _do_n2(prob):
    return _should_report_run(prob._reports, 'n2')


def _do_scaling(driver):
    return _should_report_run(driver._problem()._reports, 'scaling')


register_report('n2', _run_n2_report, 'N2 diagram', 'Problem', 'final_setup', 'post',
                _default_n2_filename, None, condition=_do_n2)
register_report('scaling', _run_scaling_report, 'Driver scaling report', 'Driver',
                '_compute_totals', 'post', _default_scaling_filename, None, condition=_do_scaling)


def setup_reports():
    """
    Set up the active reports for an OpenMDAO run.
    """
    if 'TESTFLO_RUNNING' in os.environ:
        return

    reps = os.environ.get('OPENMDAO_REPORTS', 'true').lower()
    if reps in ['0', 'false', 'off', "none", ""]:
        return  # do not do any reports

    if reps in ['1', 'true', 'on', "all"]:
        reports_on = _default_reports[:]
    else:
        reports_on = [s.strip() for s in os.environ['OPENMDAO_REPORTS'].split(',') if s.strip()]

    for report_name in reports_on:
        if report_name in _reports_registry:
            activate_report(report_name)
        else:
            issue_warning(f"Report '{report_name}' not found in reports registry.")


setup_reports()

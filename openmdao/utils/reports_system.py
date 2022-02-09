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
from openmdao.utils.hooks import _register_hook, _hooks, _unregister_hook

from openmdao.utils.coloring import _default_coloring_imagefile
from openmdao.visualization.n2_viewer.n2_viewer import _default_n2_filename
from openmdao.visualization.scaling_viewer.scaling_report import _default_scaling_filename


_Report = namedtuple('Report', 'func desc class_name inst_id method pre_or_post kwargs')

_reports_registry = {}
_reports_dir = '.'  # the default location for the reports

# _Reports_Run = namedtuple('_Reports_Run', 'probname method pre_or_post' )
# _reports_run = []

_reports_run = {}

def is_rank_0(prob):
    # If MPI, only save reports on rank 0
    on_rank0 = True
    if MPI:
        rank = prob.comm.rank
        if rank != 0:
            on_rank0 = False


    print(f"on_rank0 = {on_rank0}")

    return on_rank0

def unregister_report(name):
    if name not in _reports_registry:
        raise ValueError(f"Cannot unregister report because report with name '{name}' does not exist")


def register_report(name, func, desc, class_name, method, pre_or_post, inst_id=None, **kwargs):
    """
    Register a report with the reporting system.

    Parameters
    ----------
    name : str
        Name of report. Report names must be unique across all reports.
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
    global _reports_registry

    report = _Report(func, desc, class_name, inst_id, method, pre_or_post, kwargs)
    if name in _reports_registry:
        raise ValueError(f"A report with the name {name} already exists")
    _reports_registry[name] = report

    if pre_or_post == 'pre':
        # _register_hook(method, class_name, pre=func, inst_id=inst_id, ncalls=1)
        _register_hook(method, class_name, pre=func, inst_id=inst_id)
    elif pre_or_post == 'post':
        # _register_hook(method, class_name, post=func, inst_id=inst_id, ncalls=1)
        _register_hook(method, class_name, post=func, inst_id=inst_id)
    else:
        raise ValueError(
            f"The argument 'pre_or_post' can only have values of 'pre' or 'post', but {pre_or_post}"
            " was given")

    return


from openmdao.visualization.n2_viewer.n2_viewer import n2
from collections import defaultdict


def run_n2_report_enclosing():
    def run_n2_report_inner(prob):
        run_n2_report_inner.calls[prob] += 1
        if run_n2_report_inner.calls[prob] > 1:
            return

        if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off']:
            return

        print("run_n2_report")


        problem_reports_dirpath = get_reports_dir(prob)
        if is_rank_0(prob):
            pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

        n2_filepath = str(pathlib.Path(problem_reports_dirpath).joinpath(_default_n2_filename))
        try:
            n2(prob, show_browser=False, outfile=n2_filepath, display_in_notebook=False)
        except RuntimeError as err:
            if str(err) != "Can't compute total derivatives unless " \
                           "both 'of' or 'wrt' variables have been specified.":
                raise err

    run_n2_report_inner.calls = defaultdict(int)
    return run_n2_report_inner

run_n2_report = run_n2_report_enclosing()




def run_n2_report_func(prob):
    if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off']:
        return

    print("run_n2_report")


    problem_reports_dirpath = get_reports_dir(prob)
    if is_rank_0(prob):
        pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

    n2_filepath = str(pathlib.Path(problem_reports_dirpath).joinpath(_default_n2_filename))
    try:
        n2(prob, show_browser=False, outfile=n2_filepath, display_in_notebook=False)
    except RuntimeError as err:
        if str(err) != "Can't compute total derivatives unless " \
                       "both 'of' or 'wrt' variables have been specified.":
            raise err



def run_scaling_report_enclosing():
    def run_scaling_report_inner(driver):


        print("run_scaling_report_inner starting")

        run_scaling_report_inner.calls[driver] += 1
        if run_scaling_report_inner.calls[driver] > 1:
            return


        if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false',
                                                                                   'off']:
            return

        print("run_scaling_report_inner driver._problem")
        prob = driver._problem()
        problem_reports_dirpath = get_reports_dir(prob)
        print("run_scaling_report_inner scaling_filepath = str")




        scaling_filepath = str(
            pathlib.Path(problem_reports_dirpath).joinpath(_default_scaling_filename))
        if is_rank_0(prob):
            pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)


        print("run_scaling_report_inner prob.driver.scaling_report")

        try:
            # prob.driver.scaling_report(outfile=scaling_filepath, show_browser=False,
            #                            run_compute_totals=False)
            prob.driver.scaling_report(outfile=scaling_filepath, show_browser=False)

        # Need to handle the coloring and scaling reports which can fail in this way
        #   because total Jacobian can't be computed
        except RuntimeError as err:
            if str(err) != "Can't compute total derivatives unless " \
                           "both 'of' or 'wrt' variables have been specified.":
                raise err

    run_scaling_report_inner.calls = defaultdict(int)
    return run_scaling_report_inner

run_scaling_report = run_scaling_report_enclosing()





def run_scaling_report_func(driver, **kwargs):
    """
    Run the scaling report.

    Created for the reporting system, which expects the reporting functions to have Problem as
    their first argument.

    Parameters
    ----------
    prob : Problem
        The problem used for the scaling report.
    **kwargs : dict
        Optional args for the scaling report function.
    """
    if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off']:
        return

    # Keep track of what was run so we don't do it again. Prevents issues with recursion
    # report_run = _Reports_Run(prob._name, method, pre_or_post)
    # if report_run in _reports_run:
    #     return
    # _reports_run.append(report_run)

    prob = driver._problem()
    problem_reports_dirpath = get_reports_dir(prob)

    scaling_filepath = str(pathlib.Path(problem_reports_dirpath).joinpath(_default_scaling_filename))
    if is_rank_0(prob):
        pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

    try:
        prob.driver.scaling_report(outfile=scaling_filepath, show_browser=False, run_compute_totals=False)

    # Need to handle the coloring and scaling reports which can fail in this way
    #   because total Jacobian can't be computed
    except RuntimeError as err:
        if str(err) != "Can't compute total derivatives unless " \
                       "both 'of' or 'wrt' variables have been specified.":
            raise err




def run_coloring_report_enclosing():
    def run_coloring_report_inner(driver):
        run_coloring_report_inner.calls[driver] += 1
        if run_coloring_report_inner.calls[driver] > 1:
            return

        if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off']:
            return


        prob = driver._problem()
        # prob = driver

        problem_reports_dirpath = get_reports_dir(prob)

        coloring_filepath = str(pathlib.Path(problem_reports_dirpath).joinpath(_default_coloring_imagefile))
        if is_rank_0(prob):
            pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

        coloring = compute_total_coloring(prob)
        if coloring:
            coloring.display(show=False, fname=coloring_filepath)

    run_coloring_report_inner.calls = defaultdict(int)
    return run_coloring_report_inner


run_coloring_report = run_coloring_report_enclosing()


def run_coloring_report_func(driver, **kwargs):
    """
    Run the coloring report.

    Created for the reporting system, which expects the reporting functions to have Problem as
    their first argument.

    Parameters
    ----------
    prob : Problem
        The problem used for the coloring report.
    **kwargs : dict
        Optional args for the coloring report function.
    """
    if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off']:
        return

    prob = driver._problem()
    problem_reports_dirpath = get_reports_dir(prob)

    coloring_filepath = str(pathlib.Path(problem_reports_dirpath).joinpath(_default_coloring_imagefile))
    if is_rank_0(prob):
        pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

    coloring = compute_total_coloring(prob, **kwargs)
    if coloring:
        coloring.display(show=False, fname=coloring_filepath)


def setup_default_reports():
    """
    Set up the default reports for all OpenMDAO runs.
    """
    # with open(f"rank_{MPI.COMM_WORLD.rank}.log", "a") as f:
    #     f.write("setup_default_reports\n")
    from openmdao.visualization.n2_viewer.n2_viewer import n2
    if 'TESTFLO_RUNNING' in os.environ:
        return

    # def register_report(name, func, desc, class_name, method, pre_or_post, inst_id=None, **kwargs):

    register_report("n2_report", run_n2_report, 'N2 diagram', 'Problem', 'final_setup', 'post', inst_id=None)
    register_report("scaling_report", run_scaling_report, 'Driver scaling report', 'Driver', '_compute_totals', 'post', inst_id=None)
    # register_report("coloring report", run_coloring_report, 'Coloring report', 'Driver', '_compute_totals', 'post', inst_id=None)
    # register_report("coloring report", run_coloring_report, 'Coloring report', 'Problem', 'final_setup', 'post', inst_id=None)
    #

    # register_report(n2, 'N2 diagram', 'final_setup', 'post', probname=None, show_browser=False, display_in_notebook=False)
    # register_report(run_scaling_report, 'Driver scaling report', 'final_setup', 'post',
    #                 probname=None, show_browser=False)
    # register_report(run_coloring_report, 'Coloring report', 'final_setup', 'post', probname=None)
    pass


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

    out_stream.write("Here are the reports registered to run:\n\n")

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

#
# def run_reports(prob, method, pre_or_post):
#     """
#     Run all the registered reports.
#
#     It takes into account the specifics of when and if
#     they should be run at this point. This function is called from various methods of
#     Problem.
#
#     Parameters
#     ----------
#     prob : Problem
#         OpenMDAO Problem instance.
#     method : str
#         Name of the method in Problem that this is called from.
#     pre_or_post : str
#         Where in the Problem method that this was called from. Only valid values are 'pre' and
#         'post'.
#     """
#     global _reports_dir
#     global _reports_registry
#
#     # If MPI, only save reports on rank 0
#     on_rank0 = True
#     if MPI:
#         rank = prob.comm.rank
#         if rank != 0:
#             on_rank0 = False
#
#     # Keep track of what was run so we don't do it again. Prevents issues with recursion
#     report_run = _Reports_Run(prob._name, method, pre_or_post)
#     if report_run in _reports_run:
#         return
#     _reports_run.append(report_run)
#
#     # No running of reports when running under testflo
#     if 'TESTFLO_RUNNING' in os.environ:
#         return
#
#     # The user can define OPENMDAO_REPORTS to turn off reporting
#     if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off']:
#         return
#
#     # The user can define where to put the reports using an environment variables
#     reports_dir = os.environ.get('OPENMDAO_REPORTS_DIR', _reports_dir)
#
#     # loop through reports registry looking for matches
#     for report in _reports_registry:
#         if report.probname and report.probname != prob._name:
#             continue
#         if report.method != method:
#             continue
#         if report.pre_or_post == pre_or_post:
#             # make the problem reports dir
#             problem_reports_dirname = f'{prob._name}_reports'
#             problem_reports_dirpath = pathlib.Path(reports_dir).joinpath(problem_reports_dirname)
#
#             if on_rank0:
#                 # if not os.path.isdir(problem_reports_dirpath):
#                 #     os.mkdir(problem_reports_dirpath)
#                 #
#                 pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)
#
#             current_cwd = pathlib.Path.cwd()
#             os.chdir(problem_reports_dirpath)
#
#             try:
#                 report.func(prob, **report.kwargs)
#             # Need to handle the coloring and scaling reports which can fail in this way
#             #   because total Jacobian can't be computed
#             except RuntimeError as err:
#                 if str(err) != "Can't compute total derivatives unless " \
#                                    "both 'of' or 'wrt' variables have been specified.":
#                     raise err
#             finally:
#                 os.chdir(current_cwd)


def clear_reports():
    """
    Clear all of the reports from the registry.
    """
    global _reports_registry

    # need to remove the hooks
    for name, report in _reports_registry.items():
        #                 val = getattr(report, column_name)
        #    List of (hook_func, ncalls, exit) tuples

        if getattr(report, 'pre_or_post') == "pre":
            _unregister_hook(getattr(report, 'method'), getattr(report, 'class_name'), inst_id=getattr(report, 'inst_id'), pre=getattr(report, 'func'))
        else:
            _unregister_hook(getattr(report, 'method'), getattr(report, 'class_name'), inst_id=getattr(report, 'inst_id'), post=getattr(report, 'func'))

        # _remove_hook(getattr(report, 'func'),
        #              [(getattr(report, 'func'), 1, False)],
        #              getattr(report, 'class_name'),
        #              getattr(report, 'method'),
        #              getattr(report, 'pre_or_post'),
        #              )
    _reports_registry = {}


def clear_reports_run():
    global _reports_run
    _reports_run = []


# if 'TESTFLO_RUNNING' not in os.environ:
#     setup_default_reports()
setup_default_reports()


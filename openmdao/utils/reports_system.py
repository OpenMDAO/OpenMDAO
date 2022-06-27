"""
Utility functions related to the reporting system which generates reports by default for all runs.
"""

from collections import namedtuple
import sys
import os
import inspect
import functools

from openmdao.core.constants import _UNDEFINED
from openmdao.utils.mpi import MPI
from openmdao.utils.hooks import _register_hook, _unregister_hook
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.file_utils import _iter_entry_points
from openmdao.utils.webview import webview

# Keeping track of the registered reports
_Report = namedtuple(
    'Report', 'func description class_name inst_id method pre_or_post report_filename'
)
_reports_registry = {}
_default_reports = ['scaling', 'total_coloring', 'n2']
_active_reports = set()  # these reports will actually run (assuming their hook funcs are triggered)
_cmdline_reports = set()  # cmdline reports can be registered here to prevent default reports

_reports_dir = os.environ.get('OPENMDAO_REPORTS_DIR', './reports')  # top dir for the reports

_truthy = {'1', 'true', 'on', 'yes'}
_falsey = {'0', 'false', 'off', "none", ""}

_plugins_loaded = False  # use this to ensure plugins only loaded once


def _register_cmdline_report(name):
    global _cmdline_reports
    _cmdline_reports.add(name)


def reports_active():
    """
    Return True if reports are active globally.

    Returns
    -------
    bool
        Return True if reports are active.
    """
    return os.environ.get('TESTFLO_RUNNING', '').lower() not in ('1', 'true')


def register_report(name, func, desc, class_name, method, pre_or_post, filename=None, inst_id=None):
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
    filename : str or None
        Name of file to use when saving the report.
    inst_id : str or None
        Either the instance ID of an OpenMDAO object (e.g. Problem, Driver) or None.
        If None, then this report will be run for all objects of type class_name.
    """
    global _reports_registry

    if name in _reports_registry:
        raise ValueError(f"A report with the name {name} was already registered.")
    if pre_or_post not in ('pre', 'post'):
        raise ValueError("The argument 'pre_or_post' can only have values of 'pre' or 'post', "
                         f"but {pre_or_post} was given")

    _reports_registry[name] = _Report(func, desc, class_name, inst_id, method, pre_or_post,
                                      filename)


def activate_report(name, instance=None):
    """
    Activate a report that has been registered with the reporting system.

    Parameters
    ----------
    name : str
        Name of report. Report names must be unique across all reports.
    instance : object or None
        If not None, report will be activated only for this instance, assuming the instance
        in the reports_registry is either None or matching.
    """
    global _reports_registry, _active_reports

    if name not in _reports_registry:
        issue_warning(f"No report with the name {name} is registered.")
        return
    if name in _cmdline_reports:
        return  # skip it if it's already being run from the command line

    if not reports_active():
        return

    inst_id = None if instance is None else instance._get_inst_id()

    func, _, class_name, _inst_id, method, pre_or_post, report_filename = \
        _reports_registry[name]

    # handle case where report was registered for a specific inst_id
    if _inst_id is not None:
        if inst_id is None:
            inst_id = _inst_id
        elif inst_id != _inst_id:  # registered inst_id doesn't match current instance
            return

    if (name, inst_id) in _active_reports:
        raise ValueError(f"A report with the name '{name}' for instance '{inst_id}' is already "
                         "active.")

    if report_filename is not None:
        kwargs = {'report_filename': report_filename}
    else:
        kwargs = {}

    if pre_or_post == 'pre':
        _register_hook(method, class_name, pre=func, inst_id=inst_id, ncalls=1, **kwargs)
    else:  # post
        _register_hook(method, class_name, post=func, inst_id=inst_id, ncalls=1, **kwargs)

    _active_reports.add((name, inst_id))


def activate_reports(reports, instance):
    """
    Activate any matching reports.

    Parameters
    ----------
    reports : list of str
        List of report names that should be active.  These names come down from the Problem
        and some may refer to Problem reports while others may refer to Driver reports.
    instance : object
        The reports will be activated for this instance.
    """
    cnames = {c.__name__ for c in inspect.getmro(instance.__class__)}
    for name in reports:
        try:
            class_name = _reports_registry[name][2]
        except KeyError:
            issue_warning(f"Report with name '{name}' not found in reports registry.")
            continue

        if class_name in cnames:  # report corresponds to our class
            activate_report(name, instance)


def list_reports(default=False, out_stream=None):
    """
    Write table of information about reports currently registered in the reporting system.

    Parameters
    ----------
    default : bool
        If True, list only the default reports.
    out_stream : file-like object
        Where to send report info.
    """
    global _reports_registry

    # if we haven't created any Problem instances, the registry could still be uninitialized
    # if it *has* already been initialized, this call will do nothing.
    _load_report_plugins()

    if not out_stream:
        out_stream = sys.stdout

    column_names = ['name', 'description', 'class name', 'method', 'pre or post']
    column_widths = {}
    # Determine the column widths of the data fields by finding the max width for all rows
    # First for the headers
    for column_name in column_names:
        column_widths[column_name] = len(column_name)

    if default:
        reg = {r: _reports_registry[r] for r in _default_reports}
    else:
        reg = _reports_registry

    # Now for the values
    for name, report in reg.items():
        for column_name in column_names:
            if column_name == 'name':
                val = name
            else:
                val = getattr(report, column_name.replace(' ', '_'))
                if column_name == 'func':
                    val = val.__name__
                else:
                    val = str(val)
            column_widths[column_name] = max(column_widths[column_name], len(val))

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

    for name, report in reg.items():
        report_info = ''
        for i, column_name in enumerate(column_names):
            if column_name == 'name':
                val = name
            else:
                val = str(getattr(report, column_name.replace(' ', '_')))
            val_formatted = f"{val:<{column_widths[column_name]}}"
            report_info += val_formatted
            if i < len(column_names) - 1:
                report_info += column_spacing * ' '

        out_stream.write(report_info + '\n')

    out_stream.write('\n')


def _list_reports_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao list_reports' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('-d', '--default', action='store_true', dest='dflt',
                        help="List only the default reports.")
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        help='Send list of reports to this file.')


def _list_reports_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao list_reports'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if options.outfile is None:
        list_reports(options.dflt)
    else:
        with open(options.outfile, 'w') as f:
            list_reports(options.dflt, f)


def view_reports(probname=None):
    """
    Pop up a browser to view specified reports.

    Parameters
    ----------
    probname : str or None
        If not None, view only reports for the specified Problem, else view all reports.
    """
    if probname is None:
        tdir = _reports_dir
    else:
        tdir = os.path.join(_reports_dir, probname)

    gen_index_file(tdir)

    webview(os.path.join(tdir, 'index.html'))


def _view_reports_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao view_reports' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('-p', '--problem', action='store', dest='problem',
                        help='View reports only for the specified Problem.')


def _view_reports_cmd(options, user_args):
    """
    View completed reports via the 'openmdao view_reports' command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    view_reports(options.problem)


def set_reports_dir(reports_dir_path):
    """
    Set the path to the top level reports directory. Defaults to './reports'.

    Parameters
    ----------
    reports_dir_path : str
        Path to the top level reports directory.
    """
    global _reports_dir
    _reports_dir = reports_dir_path


# -----------------------------------------
# TODO: remove these once dymos has been updated to current reports system API
def _run_n2_report(x):
    pass


def _setup_default_reports():
    pass


report_function = _run_n2_report
run_n2_report = _run_n2_report
setup_default_reports = _setup_default_reports
set_default_reports_dir = set_reports_dir

# -----------------------------------------


def _reset_reports_dir():
    """
    Reset the path to the top level reports directory from the environment or to './reports'.

    This is used during testing, where environment variables are sometimes modified during
    the test.
    """
    global _reports_dir
    _reports_dir = os.environ.get('OPENMDAO_REPORTS_DIR', './reports')


def get_reports_dir():
    """
    Get the path to the top level reports directory. Defaults to './reports'.

    Returns
    -------
    str
        Path to the top level reports directory.
    """
    return _reports_dir


def _reports2list(reports, defaults):
    if reports in [True, _UNDEFINED]:
        return defaults
    if not reports:  # False or None or empty iter
        return []

    if isinstance(reports, str):
        low = reports.lower()
        if low in _falsey:
            return []
        if low in _truthy:
            return defaults
        if low == 'all':
            return list(_reports_registry)  # activate all registered reports

        return [s.strip() for s in reports.split(',') if s.strip()]

    return list(reports)


def get_reports_to_activate(reports=_UNDEFINED):
    """
    Get the list of names of reports that should be activated.

    Parameters
    ----------
    reports : str, list of str, _UNDEFINED, or None
        Indicates which report(s) are to be activated.

    Returns
    -------
    list of str
        List of report names.
    """
    reps_env = os.environ.get('OPENMDAO_REPORTS', 'true')
    env_list = _reports2list(reps_env, _default_reports[:])
    return _reports2list(reports, env_list)


def clear_report_registry():
    """
    Clear all of the reports from the registry.
    """
    global _reports_registry
    _reports_registry = {}


def clear_reports(instance=None):
    """
    Clear all of the currently active reports.

    Parameters
    ----------
    instance : object or None
        If not None, only clear reports for this instance.
    """
    global _active_reports, _reports_registry

    inst_id = None if instance is None else instance._get_inst_id()

    to_remove = set()

    # need to remove the hooks
    for name, active_inst_id in _active_reports:
        if instance is None:
            inst_id = active_inst_id
        elif inst_id != active_inst_id:
            continue
        func, _, class_name, _, method, pre_or_post, _ = _reports_registry[name]
        if pre_or_post == "pre":
            _unregister_hook(method, class_name, inst_id=inst_id, pre=func)
        else:
            _unregister_hook(method, class_name, inst_id=inst_id, post=func)
        to_remove.add((name, active_inst_id))

    _active_reports -= to_remove


def _load_report_plugins():
    """
    Load all 'openmdao_report' entry points and run them to register their plugins.
    """
    global _plugins_loaded
    if _plugins_loaded:  # make sure we only run this once
        return

    _plugins_loaded = True

    for ep in _iter_entry_points('openmdao_report'):
        register_func = ep.load()
        register_func()  # this runs the function that calls register_report


def _add_dir_to_tree(dirpath, lines):
    """
    Create nested lists of directories with links to files.

    Parameters
    ----------
    dirpath : str
        Starting directory.
    lines : list of str
        List of lines in the final html.
    """
    lines.append(f'<li><span class="caret">{os.path.basename(dirpath)}</span>')
    lines.append(f'<ul>')

    for f in os.listdir(dirpath):
        path = os.path.join(dirpath, f)
        if os.path.isdir(path):
            _add_dir_to_tree(path, lines)
        elif f.endswith('.html') and f != 'index.html':
            lines.append(f'<li> <a href="file:///{path}">{f}</a> </li>')

    lines.append('</ul></li>')


def gen_index_file(reports_dir):
    """
    Generate an index.html file that will have links to all of the reports.

    Parameters
    ----------
    reports_dir : str
        The top directory containing the reports.
    """
    reports_dir = os.path.abspath(reports_dir)

    parts = [
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <style>
            /* Remove default bullets */
            ul {
               list-style-type: none;
            }
        </style>
        </head>
        <body>
        """
    ]

    lines = ['<ul>']
    _add_dir_to_tree(reports_dir, lines)

    parts.append('\n'.join(lines))
    parts.append('</body>\n</html>')

    with open(os.path.join(reports_dir, 'index.html'), 'w') as f:
        f.write('\n'.join(parts))

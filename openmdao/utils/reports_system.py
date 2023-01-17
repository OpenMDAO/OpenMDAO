"""
Utility functions related to the reporting system which generates reports by default for all runs.
"""

from collections import namedtuple
import os
import inspect
from itertools import chain

from numpy import isin

from openmdao.core.constants import _UNDEFINED
from openmdao.utils.hooks import _register_hook, _unregister_hook
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.file_utils import _iter_entry_points
from openmdao.utils.webview import webview
from openmdao.utils.general_utils import env_truthy, is_truthy
from openmdao.visualization.tables.table_builder import generate_table

_reports_registry = {}
_default_reports = ['scaling', 'total_coloring', 'n2', 'optimizer', 'inputs']
_active_reports = set()  # these reports will actually run (assuming their hook funcs are triggered)
_cmdline_reports = set()  # cmdline reports registered here so default reports aren't modified
_reports_dir = os.environ.get('OPENMDAO_REPORTS_DIR', './reports')  # top dir for the reports
_plugins_loaded = False  # use this to ensure plugins only loaded once


class Report(object):
    r"""
    A class to manage the resistration of hooks associated with a particular named report.

    Parameters
    ----------
    name : str
        The name of the corresponding report.
    description : str
        The description displayed when reports are listed using the `list_reports` command.

    Attributes
    ----------
    name : str
        The name of the corresponding report.
    description : str
        The description displayed when reports are listed using the `list_reports` command.
    hooks : list
        List of (args, kwargs) to be passed to the `register_hook` function when the report
        corresponding to this instance is activated.
    """

    def __init__(self, name, description=''):
        """
        Initialize attributes.
        """
        self.name = name
        self.description = description
        self.hooks = []

    def register_hook_args(self, *args, **kwargs):
        r"""
        Store positional and named args to be passed to the `register_hook` function.

        This will only happen if the report corresponding to this instance is activated.

        Parameters
        ----------
        *args : list
            The positional args to be passed to `register_hook`.
        **kwargs : dict
            The named args to be passed to `register_hook`.
        """
        self.hooks.append((args, kwargs))

    def register_hooks(self, instance):
        """
        Register the hook(s) associated with the report corresponding with this instance.

        Parameters
        ----------
        instance : object
            The instance where hooks may be registered.
        """
        if hasattr(instance, '_has_active_report') and not instance._has_active_report(self.name):
            return

        for hook_args, hook_kwargs in self.hooks:
            # always register hook with a specific instance id
            kw = hook_kwargs.copy()
            kw['inst_id'] = None if instance is None else instance._get_inst_id()
            _register_hook(*hook_args, **kw)

    def unregister_hooks(self, instance=None):
        """
        Unregister hooks associated with this report.

        Parameters
        ----------
        instance : object or None
            If not None, only unregister reports for this instance.
        """
        keep = {'fname', 'class_name', 'inst_id', 'pre', 'post'}
        for args, kw in self.hooks:
            kwargs = {k: v for k, v in kw.items() if k in keep}
            if instance is not None:
                kwargs['inst_id'] = instance
            _unregister_hook(*args, **kwargs)

    def __getattr__(self, name):
        """
        Return the named attribute from our stored hook args.

        Parameters
        ----------
        name : str
            The name of the attribute.
        """
        for args, kwargs in self.hooks:
            if name in kwargs:
                return kwargs[name]
            elif name == 'fname' and len(args) > 0:
                return args[0]
            elif name == 'class_name' and len(args) > 1:
                return args[1]
        else:
            raise AttributeError(f"Attribute '{name}' not found.")


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
    return not env_truthy('TESTFLO_RUNNING')


def register_report(name, func, desc, class_name, method, pre_or_post, inst_id=None, **kwargs):
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
    inst_id : str or None
        Either the instance ID of an OpenMDAO object (e.g. Problem, Driver) or None.
        If None, then this report will be run for all objects of type class_name.
    **kwargs : dict
        Keyword args passed to the report function.
    """
    global _reports_registry

    if name in _reports_registry:
        raise ValueError(f"A report with the name {name} was already registered.")
    if pre_or_post not in ('pre', 'post'):
        raise ValueError("The argument 'pre_or_post' can only have values of 'pre' or 'post', "
                         f"but {pre_or_post} was given")

    _reports_registry[name] = report = Report(name, desc)

    pre = func if pre_or_post == 'pre' else None
    post = func if pre_or_post == 'post' else None
    report.register_hook_args(fname=method, class_name=class_name, inst_id=inst_id, pre=pre,
                              post=post, ncalls=1, **kwargs)


def unregister_report(name):
    """
    Unregister a report with the reporting system.

    Parameters
    ----------
    name : str
        Name of report. Report names must be unique across all reports.
    """
    global _reports_registry
    del _reports_registry[name]


def register_report_hook(name, fname, class_name, inst_id=None, pre=None, post=None, description='',
                         **kwargs):
    """
    Register a hook with a specific report name in the reporting system.

    By calling this multiple times, multiple hooks can be registered for the same report. This
    is sometimes necessary to get the correct behavior when setup errors occur prior to report
    creation.

    Parameters
    ----------
    name : str
        Name of report. Report names must be unique across all reports.
    fname : str
        The name of the function where the pre and/or post hook will be applied.
    class_name : str
        The name of the class owning the method where the hook will be applied.
    inst_id : str or None
        The name of the instance owning the method where the hook will be applied.
    pre : function (None)
        If not None, this hook will run before the function named by fname runs.
    post : function (None)
        If not None, this hook will run after the function named by fname runs.
    description : str
        A description of the report.
    **kwargs : dict of keyword arguments
        Keyword arguments that will be passed to the hook function.
    """
    global _reports_registry

    if name not in _reports_registry:
        _reports_registry[name] = report = Report(name, description)
    else:
        report = _reports_registry[name]
        if description is not None:
            report.description = description

    report.register_hook_args(fname, class_name, inst_id=inst_id, pre=pre, post=post, ncalls=1,
                              **kwargs)


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
        issue_warning(f"No report with the name '{name}' is registered.")
        return
    if name in _cmdline_reports:
        return  # skip it if it's already being run from the command line

    if not reports_active():
        return

    inst_id = None if instance is None else instance._get_inst_id()

    report = _reports_registry[name]
    for _, hook_kwargs in report.hooks:
        _inst_id = hook_kwargs['inst_id']
        break

    # handle case where report was registered for a specific inst_id
    if _inst_id is not None:
        if inst_id is None:
            inst_id = _inst_id
        elif inst_id != _inst_id:  # registered inst_id doesn't match current instance
            return

    if (name, inst_id) in _active_reports:
        raise ValueError(f"A report with the name '{name}' for instance '{inst_id}' is already "
                         "active.")

    report.register_hooks(instance)
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
    cnames = [c.__name__ for c in inspect.getmro(instance.__class__)][:-1]
    for name in reports:
        try:
            report = _reports_registry[name]
            if report.class_name in cnames:  # report corresponds to our class
                activate_report(name, instance)
        except KeyError:
            issue_warning(f"Report with name '{name}' not found in reports registry.")
            continue


def list_reports(default=False, outfile=None, max_width=80):
    """
    Write table of information about reports currently registered in the reporting system.

    Parameters
    ----------
    default : bool
        If True, list only the default reports.
    outfile : str or None
        Where to send report info.  None will result in output to stdout.
    max_width : int
        Maximum width of the table. Defaults to 80.
    """
    global _reports_registry

    # if we haven't created any Problem instances, the registry could still be uninitialized.
    # if it *has* already been initialized, this call will do nothing.
    _load_report_plugins()

    headers = ['name', 'description', 'class name', 'method', 'pre or post']

    if default:
        reg = {r: _reports_registry[r] for r in _default_reports}
    else:
        reg = _reports_registry

    rows = []
    # Now for the values
    for name, report in sorted(reg.items()):
        rows.append([])
        rows[-1].append(name)
        for attr in ('description', 'class_name', 'fname'):
            rows[-1].append(getattr(report, attr))
        pre = report.pre
        rows[-1].append('pre' if pre else 'post')

    generate_table(rows, tablefmt='box_grid', headers=headers, max_width=max_width).display(outfile)


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
    list_reports(outfile=options.outfile, default=options.dflt)


def view_reports(probnames=None, level=2):
    """
    Pop up a browser to view specified reports.

    Parameters
    ----------
    probnames : str, iter of str, or None
        If not None, view only reports for the specified Problem(s), else view all reports.
    level : int
        Expand the reports directory tree to this level.  Default is 2.
    """
    tdir = _reports_dir
    to_match = set()
    if probnames:
        if isinstance(probnames, str):
            probnames = (probnames,)

        for probname in probnames:
            subdir = os.path.join(_reports_dir, probname)
            if not os.path.isdir(subdir):
                # see if they provided script name instead of problem name
                dname = os.path.splitext(subdir)[0]
                if os.path.isdir(dname):
                    subdir = dname
                else:
                    print(f"Can't find report dir '{subdir}'.")
                    continue

            to_match.add(subdir)
    else:
        to_match = set(os.listdir(tdir))

    if not to_match:
        print("No matching report dirs found.")
        return

    to_match = {os.path.basename(m) for m in to_match}

    gen_index_file(tdir, level, to_match)

    webview(os.path.join(tdir, 'index.html'))


def _view_reports_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao view_reports' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('problem', metavar='problem', nargs='*',
                        help='View reports only for the specified Problem(s).')
    parser.add_argument('-l', '--level', action='store', dest='level', type=int, default=2,
                        help='Expand the reports directory tree to this level. Default is 2.')


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
    view_reports(options.problem, level=options.level)


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
    """
    Return a list of reports based on the value of the reports var and current default report list.

    Parameters
    ----------
    reports : str, list, or _UNDEFINED
        Variable indicating which reports should be active based on the current 'defaults' list.
    defaults : list
        List of current default reports.  This could be either the global report defaults or
        a newer list of defaults based on previous processing.

    Returns
    -------
    list
        The list of reports that should be active.
    """
    if reports in [True, _UNDEFINED]:
        return defaults
    if not reports:
        return []

    if isinstance(reports, str):
        low = reports.lower()
        if low == 'all':
            return list(_reports_registry)  # activate all registered reports

        if is_truthy(low):
            if ',' in low:
                return [s.strip() for s in reports.split(',') if s.strip()]
            elif reports in _reports_registry:
                return [reports]
            else:
                return defaults
        else:
            return []

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
    if not reports or not reports_active():
        return []

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
        if name in _reports_registry:
            _reports_registry[name].unregister_hooks(inst_id)
        else:
            issue_warning(f"No report with the name '{name}' is registered.")

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


def _add_dir_to_tree(dirpath, lines, explevel, level, to_match):
    """
    Create nested lists of directories with links to files.

    Parameters
    ----------
    dirpath : str
        Starting directory.
    lines : list of str
        List of lines in the final html.
    explevel : int
        Expand the tree to this level.
    level : int
        The current level of the tree.
    to_match : set
        Directory names to show.
    """
    dlist = os.listdir(dirpath)

    if not dlist:  # don't include empty dirs in the index
        return

    # split into files and dirs to make page look better
    directories = {f for f in dlist if os.path.isdir(os.path.join(dirpath, f))}

    files = sorted(f for f in dlist if os.path.isfile(os.path.join(dirpath, f)))

    if level == 1 and os.path.basename(dirpath) not in to_match:
        return
    else:
        op = 'open' if level < explevel else ''
        lines.append(f'<li><details {op}><summary>{os.path.basename(dirpath)}</summary>')
        if not dlist:
            lines.append('</li>')
            return

    lines.append(f'<ul>')

    for f in chain(files, sorted(directories)):
        path = os.path.join(dirpath, f)
        if os.path.isdir(path):
            _add_dir_to_tree(path, lines, explevel, level + 1, to_match)
        elif f.endswith('.html') and f != 'index.html':
            lines.append(f'<li> <a href="file:///{path}">{f}</a> </li>')

    lines.append('</ul></details></li>')


def gen_index_file(reports_dir, level, to_match):
    """
    Generate an index.html file that will have links to all of the reports.

    Parameters
    ----------
    reports_dir : str
        The top directory containing the reports.
    level : int
        Expand the reports directory tree to this level.
    to_match : set
        Set of subdirectory names to show.
    """
    reports_dir = os.path.abspath(reports_dir)

    # tree view courtesy of: https://iamkate.com/code/tree-views/

    parts = [
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <style>
            .tree {
                --spacing : 1.5rem;
                --radius  : 8px;
            }

            .tree li {
                display      : block;
                position     : relative;
                padding-left : calc(2 * var(--spacing) - var(--radius) - 2px);
            }

            .tree ul {
                margin-left  : calc(var(--radius) - var(--spacing));
                padding-left : 0;
            }

            .tree ul li {
                border-left : 2px solid #ddd;
            }

            .tree ul li:last-child {
                border-color : transparent;
            }

            .tree ul li::before {
                content      : '';
                display      : block;
                position     : absolute;
                top          : calc(var(--spacing) / -2);
                left         : -2px;
                width        : calc(var(--spacing) + 2px);
                height       : calc(var(--spacing) + 1px);
                border       : solid #ddd;
                border-width : 0 0 2px 2px;
            }

            .tree summary {
                display : block;
                cursor  : pointer;
            }

            .tree summary::marker,
            .tree summary::-webkit-details-marker {
                display : none;
            }

            .tree summary:focus {
                outline : none;
            }

            .tree summary:focus-visible {
                outline : 1px dotted #000;
            }

            .tree li::after,
            .tree summary::before {
                content       : '';
                display       : block;
                position      : absolute;
                top           : calc(var(--spacing) / 2 - var(--radius));
                left          : calc(var(--spacing) - var(--radius) - 1px);
                width         : calc(2 * var(--radius));
                height        : calc(2 * var(--radius));
                border-radius : 50%;
                background    : #ddd;
            }

            .tree summary::before {
                content     : '+';
                z-index     : 1;
                background  : #696;
                color       : #fff;
                line-height : calc(2 * var(--radius) - 2px);
                text-align  : center;
            }

            .tree details[open] > summary::before {
                content : '−';
            }
        </style>
        <script>
        </script>
        </head>
        <body>
        """
    ]
    lines = ['<ul class="tree">']
    _add_dir_to_tree(reports_dir, lines, explevel=level, level=0, to_match=to_match)

    parts.append('\n'.join(lines))
    parts.append('</body>\n</html>')

    with open(os.path.join(reports_dir, 'index.html'), 'w') as f:
        f.write('\n'.join(parts))

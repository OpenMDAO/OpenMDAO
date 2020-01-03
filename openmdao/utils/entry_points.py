"""
A command to list OpenMDAO recognized types that are installed in the current environment.
"""

import sys
from collections import defaultdict
from importlib import import_module
from os.path import join, basename, dirname, isfile, split, splitext, abspath, expanduser
from inspect import getmembers, isclass

from openmdao.utils.file_utils import package_iter, get_module_path

try:
    import pkg_resources
except ImportError:
    pkg_resources = None


_allowed_types = {
    'components': 'openmdao_components',
    'lin_solvers': 'openmdao_lin_solvers',
    'nl_solvers': 'openmdao_nl_solvers',
    'drivers': 'openmdao_drivers',
    'case_recorders': 'openmdao_case_recorders',
    'case_readers': 'openmdao_case_readers',
    'surrogate_models': 'openmdao_surrogate_models',
    'commands': 'openmdao_commands',
}


def split_ep(entry_point):
    """
    Split an entry point string into name, module, target.

    Parameters
    ----------
    entry_point : EntryPoint
        Entry point object.

    Returns
    -------
    tuple
        (entry_point_name, module_name, target_name)
    """
    epstr = str(entry_point)
    name, rest = epstr.split('=', 1)
    name = name.strip()
    module, target = rest.strip().split(':', 1)
    return name, module, target


def _filtered_ep_iter(epgroup, includes=None, excludes=()):
    """
    Yield a filtered list of entry points and their attributes.

    Parameters
    ----------
    epgroup : str
        Entry point group name.
    includes : iter of str or None
        Sequence of package names to include.
    excludes : iter of str or None
        Sequence of package names to exclude.

    Yields
    ------
    tuples
        (EntryPoint, name, module, target)
    """
    for ep in pkg_resources.iter_entry_points(group=epgroup):
        name, module, target = split_ep(ep)
        for ex in excludes:
            if module.startswith(ex + '.'):
                break
        else:
            if includes:
                for inc in includes:
                    if module.startswith(inc + '.'):
                        yield ep, name, module, target
            else:
                yield ep, name, module, target


def compute_entry_points(package, outstream=sys.stdout):
    """
    Display what the entry point dict should be based on classes that exist in package.

    Parameters
    ----------
    package : str
        The package name.
    outstream : file-like
        Output stream.  Defaults to stdout.
    """
    from openmdao.api import ExplicitComponent, ImplicitComponent, Group, SurrogateModel
    from openmdao.core.component import Component
    from openmdao.core.driver import Driver
    from openmdao.solvers.solver import Solver, LinearSolver, NonlinearSolver, BlockLinearSolver
    from openmdao.recorders.base_case_reader import BaseCaseReader
    from openmdao.recorders.case_recorder import CaseRecorder

    epgroup_info = {
        Component: 'openmdao_components',
        Group: 'openmdao_groups',
        SurrogateModel: 'openmdao_surrogate_models',
        LinearSolver: 'openmdao_lin_solvers',
        NonlinearSolver: 'openmdao_nl_solvers',
        Driver: 'openmdao_drivers',
        BaseCaseReader: 'openmdao_case_readers',
        CaseRecorder: 'openmdao_case_recorders',
    }
    check = tuple(epgroup_info)

    seen = set(check)
    seen.update((ImplicitComponent, ExplicitComponent, BlockLinearSolver))

    groups = defaultdict(list)

    pkgpath = package + '.'
    try:
        pkg = import_module(package)
    except Exception:
        raise RuntimeError("Problem during import of package '{}'.  "
                           "package must be an installed python package.".format(package))
    start_dir = abspath(dirname(pkg.__file__))

    for f in package_iter(start_dir, dir_excludes=('test_suite',)):
        modpath = get_module_path(f)

        try:
            mod = import_module(modpath)
        except Exception:
            print("failed to import {} (file {}).".format(modpath, f))
            continue

        for cname, c in getmembers(mod, isclass):
            if not c.__module__.startswith(pkgpath):
                continue
            if issubclass(c, check) and c not in seen:
                seen.add(c)
                for klass, epgroup in epgroup_info.items():
                    if issubclass(c, klass):
                        groups[epgroup].append((modpath, cname))
                        break

    print("entry_points={", file=outstream)
    for g, eps in sorted(groups.items(), key=lambda x: x[0]):
        print("    '{}': [".format(g), file=outstream)
        for modpath, cname in sorted(eps, key=lambda x: x[0] + x[1]):
            print("        '{}={}:{}',".format(cname.lower(), modpath, cname), file=outstream)
        print("    ],", file=outstream)
    print("}", file=outstream)


def _compute_entry_points_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao compute_entry_points' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('package', nargs=1, help='Compute entry points for this package.')
    parser.add_argument('-o', action='store', dest='outfile', help='output file.')


def _compute_entry_points_cmd(options, user_args):
    """
    Run the `openmdao compute_entry_points` command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str  (ignored)
        Args to be passed to the user script.

    Returns
    -------
    function
        The hook function.
    """
    if options.outfile:
        with open(options.outfile, 'w') as f:
            compute_entry_points(options.package[0], outstream=f)
    else:
        compute_entry_points(options.package[0])


def list_installed(types=None, includes=None, excludes=None, show_docs=False):
    """
    Print a table of installed entry points.

    Parameters
    ----------
    types : iter of str or None
        Sequence of entry point type names, e.g., components, groups, drivers, etc.
    includes : iter of str or None
        Sequence of packages to include.
    excludes : iter of str or None
        Sequence of packages to exclude.
    show_docs : bool
        If True, display docstring after each entry.

    Returns
    -------
    dict
        Nested dict of the form  dct[eptype][target] = (name, module, docs)
    """
    if pkg_resources is None:
        raise RuntimeError("You must install pkg_resources in order to list installed types.")

    if not types:
        types = list(_allowed_types)

    typdict = {}
    for type_ in types:
        if type_ not in _allowed_types:
            raise RuntimeError("Type '{}' is not a valid type.  Try one of {}."
                               .format(type_, sorted(_allowed_types)))
        print("Installed {}:\n".format(type_))
        typdict[type_] = epdict = {}
        title1 = 'Entry Point Name'
        title2 = 'Class or Function'
        title3 = 'Module'

        cwid1 = len(title1)
        cwid2 = len(title2)
        for ep, name, module, target in _filtered_ep_iter(_allowed_types[type_]):
            # we need to actually load the entry point if docs are requested
            if show_docs:
                klass = ep.load()
                docs = klass.__doc__
            else:
                docs = ''
            epdict[target] = (name, module, docs)
            if len(name) > cwid1:
                cwid1 = len(name)
            if len(target) > cwid2:
                cwid2 = len(target)

        if epdict:
            print("  {:<{cwid1}}  {:<{cwid2}}  {}".format(title1, title2, title3,
                                                          cwid1=cwid1, cwid2=cwid2))
            print("  {:<{cwid1}}  {:<{cwid2}}  {}".format('-' * len(title1), '-' * len(title2),
                                                          '-' * len(title3),
                                                          cwid1=cwid1, cwid2=cwid2))

        # sort displayed values by module_name + target so that entry points will be grouped
        # by module and sorted by target name within each module.
        for target, (name, module, docs) in sorted(epdict.items(), key=lambda x: x[1][1] + x[0]):
            line = "  {:<{cwid1}}  {:<{cwid2}}  {}".format(name, target, module,
                                                           cwid1=cwid1, cwid2=cwid2)
            print(line)
            if show_docs and docs:
                print(docs)

        print()

    return typdict


def _list_installed_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao list_installed' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('types', nargs='*', help='List these types of installed classes. '
                        'Allowed types are {}.'.format(sorted(_allowed_types)))
    parser.add_argument('-d', '--docs', action='store_true', dest='show_docs',
                        help="Display the class docstrings.")
    parser.add_argument('-x', '--exclude', default=[], action='append', dest='excludes',
                        help='Package to exclude.')
    parser.add_argument('-i', '--include', default=[], action='append', dest='includes',
                        help='Package to include.')


def _list_installed_cmd(options, user_args):
    """
    Run the `openmdao list_installed` command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str  (ignored)
        Args to be passed to the user script.

    Returns
    -------
    function
        The hook function.
    """
    list_installed(options.types, options.includes, options.excludes, options.show_docs)

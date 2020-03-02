"""
Various functions for working with entry points.
"""

import sys
import traceback
from collections import defaultdict
import itertools
from importlib import import_module
from os.path import join, basename, dirname, isfile, split, splitext, abspath, expanduser
from inspect import getmembers, isclass
import textwrap

from openmdao.utils.file_utils import package_iter, get_module_path
from openmdao.core.component import Component
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.group import Group
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver, LinearSolver, NonlinearSolver, BlockLinearSolver
from openmdao.recorders.base_case_reader import BaseCaseReader
from openmdao.recorders.case_recorder import CaseRecorder
from openmdao.surrogate_models.surrogate_model import SurrogateModel
from openmdao.solvers.linesearch.backtracking import LinesearchSolver


try:
    import pkg_resources
except ImportError:
    pkg_resources = None


_epgroup_bases = {
    Component: 'openmdao_component',
    Group: 'openmdao_group',
    SurrogateModel: 'openmdao_surrogate_model',
    LinearSolver: 'openmdao_lin_solver',
    NonlinearSolver: 'openmdao_nl_solver',
    Driver: 'openmdao_driver',
    BaseCaseReader: 'openmdao_case_reader',
    CaseRecorder: 'openmdao_case_recorder',
}

_allowed_types = {g.split('_', 1)[1]: g for g in _epgroup_bases.values()}
_allowed_types['command'] = 'openmdao_command'

_github_topics = {k: v.replace('_', '-') for k, v in _allowed_types.items()}
_github_topics['openmdao'] = 'openmdao'


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
    if excludes is None:
        excludes = ()
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


def compute_entry_points(package, dir_excludes=(), outstream=sys.stdout):
    """
    Display what the entry point dict should be based on classes that exist in package.

    Parameters
    ----------
    package : str
        The package name.
    dir_excludes : iter of str
        Glob patterns for directory exclusion.
    outstream : file-like
        Output stream.  Defaults to stdout.

    Returns
    -------
    dict
        Mapping of entry point groups to entry point strings.
    """
    check = tuple(_epgroup_bases)

    seen = set(check)
    seen.update((ImplicitComponent, ExplicitComponent, BlockLinearSolver, LinesearchSolver))
    # Driver and Group are instantiatable, so we should have entry points for them
    seen.remove(Driver)
    seen.remove(Group)

    groups = defaultdict(list)

    pkgpath = package + '.'
    try:
        pkg = import_module(package)
    except Exception:
        raise RuntimeError("Problem during import of package '{}'.  "
                           "package must be an installed python package.".format(package))
    start_dir = abspath(dirname(pkg.__file__))

    for f in package_iter(start_dir, dir_excludes=dir_excludes):
        modpath = get_module_path(f)

        try:
            mod = import_module(modpath)
        except Exception:
            print("failed to import {} (file {}).\n{}".format(modpath, f, traceback.format_exc()))
            continue

        for cname, c in getmembers(mod, isclass):
            # if class isn't defined in this module, skip it
            if not c.__module__ == modpath:
                continue
            if issubclass(c, check) and c not in seen:
                seen.add(c)
                for klass, epgroup in _epgroup_bases.items():
                    if issubclass(c, klass):
                        groups[epgroup].append((modpath, cname))
                        break

    if outstream is None:
        def printfunc(*args, **kwargs):
            pass
    else:
        def printfunc(*args, **kwargs):
            print(*args, **kwargs)

    # do our own printing here instead of using pprint so we can control sort order
    dct = {}
    printfunc("entry_points={", file=outstream)
    for g, eps in sorted(groups.items(), key=lambda x: x[0]):
        dct[g] = eplist = []
        printfunc("    '{}': [".format(g), file=outstream)
        for modpath, cname in sorted(eps, key=lambda x: x[0] + x[1]):
            eplist.append("{} = {}:{}".format(cname.lower(), modpath, cname))
            printfunc("        '{}',".format(eplist[-1]), file=outstream)
        printfunc("    ],", file=outstream)
    printfunc("}", file=outstream)

    return dct


def _get_epinfo(type_, includes, excludes):
    epinfo = []
    for ep, name, module, target in _filtered_ep_iter(_allowed_types[type_],
                                                      includes=includes, excludes=excludes):
        pkg = module.split('.', 1)[0]
        try:
            mod = import_module(pkg)
            obj = ep.load()
        except ImportError:
            print("Import of %s failed.\n%s" % (pkg, traceback.format_exc()))
            continue
        try:
            version = mod.__version__
        except AttributeError:
            version = '?'
        if type_ != 'command':
            name = target
        epinfo.append((name, pkg, version, module, obj.__doc__))

    return epinfo


def _display_epinfo(type_, epinfo, show_docs, *titles):
    cwids = []
    unders = []
    for i in range(len(titles)):
        cwids.append(max(len(t[i]) for t in epinfo))
        unders.append('-' * len(titles[i]))
        if len(titles[i]) > cwids[-1]:
            cwids[-1] = len(titles[i])

    template = "    " + '  '.join(['{:<{cwids[%d]}}' % i for i in range(len(cwids))])

    # sort displayed values by module_name + target so that entry points will be grouped
    # by module and sorted by target name within each module.
    ordered = sorted(epinfo, key=lambda x: x[1] + x[3] + x[0])

    print("Installed {}s:".format(type_))

    for pkg, group in itertools.groupby(ordered, lambda x: x[1]):
        group = list(group)

        print("\n  Package:", pkg, " Version:", group[0][2], '\n')

        for i, (name, pkg, version, module, docs) in enumerate(group):
            if i == 0:
                print(template.format(*titles, cwids=cwids))
                print(template.format(*unders, cwids=cwids))

            print(template.format(name, module, cwids=cwids))
            if show_docs and docs:
                docs = textwrap.dedent(docs)
                indented = ['        ' + d for d in docs.splitlines()]
                print('\n'.join(indented))
                print('\n   ', '-' * 80, '\n')

    print()


def _compute_entry_points_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao compute_entry_points' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('package', nargs=1,
                        help='Compute entry points for this package.')
    parser.add_argument('-o', action='store',
                        dest='outfile', help='output file.')


def _compute_entry_points_exec(options, user_args):
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


def list_installed(types=None, includes=None, excludes=(), show_docs=False):
    """
    Print a table of installed entry points.

    Parameters
    ----------
    types : iter of str or None
        Sequence of entry point type names, e.g., component, group, driver, etc.
    includes : iter of str or None
        Sequence of packages to include.
    excludes : iter of str
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
        typdict[type_] = epinfo = _get_epinfo(type_, includes, excludes)

        titles = [
            'Class or Function',
            'Module',
        ]

        if type_ == 'command':
            titles[0] = 'Command'

        if epinfo:
            _display_epinfo(type_, epinfo, show_docs, *titles)

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


def find_plugins(types=None):
    """
    Search github for repositories containing OpenMDAO plugins.

    Parameters
    ----------
    types : iter of str or None
        Sequence of entry point type names, e.g., component, group, driver, etc.

    Returns
    -------
    dict
        Nested dict of the form  dct[eptype] = list of URLs
    """
    if not types:
        types = ['openmdao']

    import requests
    allowed_set = set(_github_topics.values())
    wid1 = wid2 = 0
    pkgs = {}
    for type_ in types:
        if type_ not in _github_topics:
            raise RuntimeError("Type '{}' is not a valid type.  Try one of {}."
                               .format(type_, sorted(_github_topics)))

        query = 'topic:{}'.format(_github_topics[type_])

        response = requests.get('https://api.github.com/search/repositories?q={}'.format(query),
                                headers={'Accept': 'application/vnd.github.mercy-preview+json'})

        if response.status_code != 200:
            print("Query failed for topic '{}' with response code {}.".format(_github_topics[type_],
                                                                              response.status_code))

        resdict = response.json()

        items = resdict['items']
        for item in items:
            url = item['html_url']
            name = item['name']
            topics = [t for t in item['topics'] if t in allowed_set]
            if len(name) > wid1:
                wid1 = len(name)
            if len(url) > wid2:
                wid2 = len(url)
            pkgs[url] = (name, topics)

    template = '{:<{wid1}}  {:<{wid2}}  {}'
    if pkgs:
        print(template.format('Pkg Name', 'URL', 'Topics', wid1=wid1, wid2=wid2))
        print(template.format('--------', '___', '______', wid1=wid1, wid2=wid2))
        for url, (name, topics) in sorted(pkgs.items(), key=lambda x: x[1][0]):
            print(template.format(name, url, topics, wid1=wid1, wid2=wid2))
    else:
        print("No matching packages found.")

    if resdict['incomplete_results']:
        print("\nResults are incomplete.\n")

    return pkgs


def _find_plugins_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao find_plugins' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('types', nargs='*', help='Find these types of plugins. '
                        'Allowed types are {}.'.format(sorted(_github_topics)))


def _find_plugins_exec(options, user_args):
    """
    Run the `openmdao find_plugins` command.

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
    find_plugins(options.types)

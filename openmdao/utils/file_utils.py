"""
Utilities for working with files.
"""

import sys
import os
import ast
from collections import defaultdict, OrderedDict
from importlib import import_module
from fnmatch import fnmatch
from os.path import join, basename, dirname, isfile, split, splitext, abspath, expanduser
from inspect import getmembers, isclass

import networkx as nx


def get_module_path(fpath):
    """Given a module filename, return its full Python name including
    enclosing packages. (based on existence of ``__init__.py`` files)
    """
    if basename(fpath).startswith('__init__.'):
        pnames = []
    else:
        pnames = [splitext(basename(fpath))[0]]
    path = dirname(abspath(fpath))
    while isfile(join(path, '__init__.py')):
            path, pname = split(path)
            pnames.append(pname)
    return '.'.join(pnames[::-1])


def get_class_graph(start_dir):
    from openmdao.api import ExplicitComponent, ImplicitComponent, Group, SurrogateModel
    from openmdao.core.component import Component
    from openmdao.core.driver import Driver
    from openmdao.solvers.solver import Solver, LinearSolver, NonlinearSolver
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
    seen.update((ImplicitComponent, ExplicitComponent))

    groups = defaultdict(list)

    for f in package_iter(start_dir, dir_excludes=('test_suite',)):
        modpath = get_module_path(f)
        try:
            mod = import_module(modpath)
        except Exception:
            print("failed to import {} (file {}).".format(modpath, f))
            continue

        for cname, c in getmembers(mod, isclass):
            if issubclass(c, check) and c not in seen:
                seen.add(c)
                for klass, epgroup in epgroup_info.items():
                    if issubclass(c, klass):
                        groups[epgroup].append("{}={}:{}".format(cname.lower(), modpath, cname))

    final = OrderedDict()
    for g, eps in sorted(groups.items(), key=lambda x: x[0]):
        final[g] = sorted(eps)

    import pprint
    pprint.pprint(final)


def package_iter(start_dir='.', dir_includes=None, dir_excludes=(), file_includes=None,
                 file_excludes=()):
    """
    Iterate over python files in packages (recursively) starting in start_dir.

    NOTE: all *_includes and *_excludes are applied to *local* directory and file names.

    Parameters
    ----------
    start_dir : str
        Starting directory.
    dir_includes : iter of str or None
        Glob patterns for directory inclusion. Be careful here because dir names are local,
        so, for example, if includes=('foo',), then directory 'foo' would be included, but
        any subdirectories of 'foo' that were not also named 'foo' would not.
    dir_excludes : iter of str
        Glob patterns for directory exclusion.
    file_includes : iter of str or None
        Glob patterns for file inclusion.
    file_excludes : iter of str
        Glob patterns for file exclusion.

    Yields
    ------
    str
        Filenames (full path from start_dir).
    """
    file_includes = set() if file_includes is None else set(file_includes)
    file_includes.add('*.py')
    file_excludes = set() if file_excludes is None else set(file_excludes)
    file_excludes.update(('_*', 'test_*', 'api.py', 'parallel_api.py'))
    dir_excludes = set() if dir_excludes is None else set(dir_excludes)
    dir_excludes.update(('test', 'tests'))

    yield from files_iter(start_dir, dir_includes=dir_includes, dir_excludes=dir_excludes,
                          file_includes=file_includes, file_excludes=file_excludes,
                          package_only=True)


def files_iter(start_dir='.', dir_includes=None, dir_excludes=(),
               file_includes=None, file_excludes=(), package_only=False):
    """
    Iterate over files (recursively) starting in start_dir.

    NOTE: all *_includes and *_excludes are applied to *local* directory and file names.

    Parameters
    ----------
    start_dir : str
        Starting directory.
    dir_includes : iter of str or None
        Glob patterns for directory inclusion. Be careful here because dir names are local,
        so, for example, if includes=('foo',), then directory 'foo' would be included, but
        any subdirectories of 'foo' that were not also named 'foo' would not.
    dir_excludes : iter of str
        Glob patterns for directory exclusion.
    file_includes : iter of str or None
        Glob patterns for file inclusion.
    file_excludes : iter of str
        Glob patterns for file exclusion.
    package_only : bool
        If True, only yield files that are contained in a python package.

    Yields
    ------
    str
        Filenames (full path from start_dir).
    """
    for root, dirs, files in os.walk(start_dir):
        if package_only and '__init__.py' not in files:
            dirs[:] = []
            continue
        for pat in dir_excludes:
            dirs[:] = sorted([d for d in dirs if not fnmatch(d, pat)])
        if dir_includes:
            incdirs = set()
            for pat in dir_includes:
                incdirs.update(d for d in dirs if fnmatch(d, pat))
            dirs[:] = sorted(incdirs)
        for f in files:
            for pat in file_excludes:
                if fnmatch(f, pat):
                    break
            else:
                if file_includes:
                    for pat in file_includes:
                        if fnmatch(f, pat):
                            yield join(root, f)
                else:
                    yield join(root, f)


def _load_and_exec(script_name, user_args):
    """
    Load and exec the given script as __main__.

    Parameters
    ----------
    script_name : str
        The name of the script to load and exec.
    user_args : list of str
        Args to be passed to the user script.
    """
    sys.path.insert(0, os.path.dirname(script_name))

    sys.argv[:] = [script_name] + user_args

    with open(script_name, 'rb') as fp:
        code = compile(fp.read(), script_name, 'exec')

    globals_dict = {
        '__file__': script_name,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    exec(code, globals_dict)

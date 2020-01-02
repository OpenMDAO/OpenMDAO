"""
Utilities for working with files.
"""

import sys
import os
from fnmatch import fnmatch
from os.path import join


def files_iter(start_dir='.', dir_includes=None, dir_excludes=(),
               file_includes=None, file_excludes=()):
    """
    Iterate over files (recursively) starting in start_dir.

    Parameters
    ----------
    start_dir : str
        Starting directory.
    dir_includes : iter of str or None
        Glob patterns for directory inclusion.
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
    for root, dirs, files in os.walk(start_dir):
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

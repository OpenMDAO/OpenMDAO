"""
Utilities for working with files.
"""
import sys
import os
import importlib
import types
from collections.abc import Iterable
from fnmatch import fnmatch
from os.path import join, basename, dirname, isfile, split, splitext, abspath
import pathlib
import shutil

from openmdao.utils.om_warnings import issue_warning


def get_module_path(fpath):
    """
    Given a module filename, return its full Python module path.

    This includes enclosing packages and is based on existence of ``__init__.py`` files.

    Parameters
    ----------
    fpath : str
        Pathname of file.

    Returns
    -------
    str or None
        Full module path of the given file.  Returns None if the file is not part of a package.
    """
    fpath = abspath(fpath)
    if basename(fpath).startswith('__init__.'):
        pnames = []
    else:
        pnames = [splitext(basename(fpath))[0]]
    path = dirname(fpath)

    initfile = join(path, '__init__.py')
    if not isfile(initfile):
        return None

    while isfile(initfile):
        path, pname = split(path)
        pnames.append(pname)
        initfile = join(path, '__init__.py')

    return '.'.join(pnames[::-1])


def package_iter(start_dir='.', dir_includes=None, dir_excludes=(), file_includes=None,
                 file_excludes=()):
    """
    Iterate over python files in packages (recursively) starting in start_dir.

    NOTE : all *_includes and *_excludes are applied to *local* directory and file names.

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

    for f in files_iter(start_dir, dir_includes=dir_includes, dir_excludes=dir_excludes,
                        file_includes=file_includes, file_excludes=file_excludes,
                        package_only=True):
        yield f


def files_iter(start_dir='.', dir_includes=None, dir_excludes=(),
               file_includes=None, file_excludes=(), package_only=False):
    """
    Iterate over files (recursively) starting in start_dir.

    NOTE : all *_includes and *_excludes are applied to *local* directory and file names.

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


def _to_filename(spec):
    """
    Return the filename part of the given testspec or the full string if the string is a filename.

    Parameters
    ----------
    spec : str
        The filename or testspec.

    Returns
    -------
    str
        The filename.
    """
    if ':' in spec and not os.path.isfile(spec):
        fname, _ = spec.rsplit(':', 1)
        if not fname.endswith('.py'):
            try:
                mod = importlib.import_module(fname)
                return mod.__file__
            except ImportError:
                return spec
        return fname

    return spec


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
    if ':' in script_name and not os.path.isfile(script_name):
        return _load_and_run_test(script_name)

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

    exec(code, globals_dict)  # nosec: private, internal use only


def fname2mod_name(fname):
    """
    Convert a string to a valid python module name.

    Parameters
    ----------
    fname : str
        The filename to convert.

    Returns
    -------
    str
        A valid module name corresponding to the given filename.
    """
    to_replace = ['-', ' ', '.', '(', ')', '[', ']', '{', '}', '=', '+'
                  '!', '@', '#', '$', '%', '^', '&', '*', '~', '`',
                  ';', ':', '"', "'", '<', '>', '?', '/', '\\', '|']

    if not fname.endswith('.py'):
        raise ValueError(f"'{fname}' does not end with '.py'")

    s = os.path.basename(fname).rsplit('.', 1)[0]

    for c in to_replace:
        s = s.replace(c, '_')
    return s


def _load_and_run_test(testspec):
    """
    Load and run an individual test function.

    Parameters
    ----------
    testspec : str
        <fpath_or_modpath>:<testcase>.<method> OR <fpath_or_modpath>:<function>
    """
    syspath_save = sys.path[:]

    modpath, funcpath = testspec.rsplit(':', 1)
    orig_modpath = modpath

    if modpath.endswith('.py'):
        modpath = get_module_path(modpath)
        if modpath is None:
            # create a module dynamically
            modpath = fname2mod_name(orig_modpath)
            mod = types.ModuleType(modpath)
            sys.modules[modpath] = mod
            mod.__file__ = modpath
            mod.__name__ = modpath
            mod.__package__ = None
            mod.__cached__ = None
            with open(orig_modpath, 'rb') as fp:
                code = compile(fp.read(), orig_modpath, 'exec')
            exec(code, mod.__dict__)  # nosec: private, internal use only
        else:
            mod = importlib.import_module(modpath)
    else:
        mod = importlib.import_module(modpath)

    try:
        return _run_test_func(mod, funcpath)
    finally:
        sys.path = syspath_save


def _run_test_func(mod, funcpath):
    """
    Run the given TestCase method or test function in the given module.

    Parameters
    ----------
    mod : module
        The module where the test resides.
    funcpath : str
        Either <testcase>.<method_name> or <func_name>.

    Returns
    -------
    object
        In the case of a module level function call, returns whatever the function returns.
    """
    parts = funcpath.split('.', 1)
    if len(parts) == 2:
        tcase_name, method_name = parts
        testcase = getattr(mod, tcase_name)(methodName=method_name)
        setup = getattr(testcase, 'setUp', None)
        if setup is not None:
            setup()
        getattr(testcase, method_name)()
        teardown = getattr(testcase, 'tearDown', None)
        if teardown:
            teardown()
    else:
        funcname = parts[0]
        return getattr(mod, funcname)()


if sys.version_info >= (3, 8):
    from importlib.metadata import entry_points

    if sys.version_info >= (3, 10):
        def _eps_get(group):
            eps = entry_points().select(group=group)
            for name in eps.names:
                yield eps[name]
    else:
        def _eps_get(group):
            eps = entry_points()
            if group in eps:
                yield from eps[group]

    def _iter_entry_points(group):
        # there seems to be a bug currently where entry points can show up more than
        # once in the iterator, so keep track of the ones we've already seen.
        # TODO: revisit later to see if we can remove the check
        seen = set()
        for ep in _eps_get(group):
            if ep.name not in seen:
                seen.add(ep.name)
                yield ep
else:
    try:
        import pkg_resources
    except ImportError:
        def _iter_entry_points(group):
            issue_warning("Can't retrieve entry points because pkg_resources is not installed. "
                          "Either install it using 'pip install setuptools' or upgrade to python "
                          "3.8 or newer.")
            return ()
    else:
        def _iter_entry_points(group):
            yield from pkg_resources.iter_entry_points(group)


def text2html(text, title='', style=None):
    """
    Wrap the given text for display as an html file.

    Returns an html syntax string that can be written to a file.

    Parameters
    ----------
    text : str
        Text to be displayed.
    title : str
        Title to display above text.
    style : str or None
        If not None, use as the contents of the style block for the enclosing <pre> tag.

    Returns
    -------
    str
        Content string to create an html file.
    """
    if style is None:
        style = """
            display: block;
            font-family: monospace;
            font-size: 1.5em;
            white-space: pre;
            margin: 1em 0;
        """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        .center {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 90%;
        }
        h2 {text-align: center;}
        pre {""" + style + """
        }
    </style>
</head>
<body>
<h2>""" + title + """</h2>
<pre>
""" + text + """
</pre>
</body>
</html>
"""


def image2html(imagefile, title='', alt=''):
    """
    Wrap the given image for display as an html file.

    Returns an html syntax string that can be written to a file.

    Parameters
    ----------
    imagefile : str
        Name of image file to be displayed.
    title : str
        The page title.
    alt : str
        Set the alt text for the image.

    Returns
    -------
    str
        Content string to create an html file.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        h2 {text-align: center;}
        .center {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80%;
        }
    </style>
</head>
<body>
<h2>""" + title + "</h2>" + f"""
<img src="{imagefile}" alt="{alt}" class="center"></img>

</body>
</html>
"""


def _get_outputs_dir(obj=None, *subdirs, mkdir=True):
    """
    Return a pathlib.Path for the outputs directory related to the given problem or system.

    This path is based on the "problem path" in a hierarchy of problems.
    The resulting outputs directory will be nested where each problem's output directory
    contains its own output files and subdirectories as well as any subproblems.

    This directory also will include a .openmdao_outputs hidden file that
    marks this directory as being created by OpenMDAO. This makes identifying the
    directory during cleanup more reliable.

    Parameters
    ----------
    obj : Problem or System or Solver or None
        The problem or system or Solver from which we are opening a file.
    mkdir : bool
        If True, force the creation of this directory.
    subdirs : str
        Additional subdirectories under the top level directory for the relevant problem.
    """
    from openmdao.core.problem import Problem
    from openmdao.core.system import System
    from openmdao.solvers.solver import Solver

    if isinstance(obj, Problem):
        prob_meta = obj._metadata
        comm = obj.comm
    elif isinstance(obj, System):
        prob_meta = obj._problem_meta
        comm = obj.comm
    elif isinstance(obj, Solver):
        system = obj._system
        if system is None:
            raise RuntimeError('The output directory for Solvers cannot be accessed '
                               'before final_setup.')
        prob_meta = system()._problem_meta
        comm = system().comm
    else:
        raise RuntimeError(f'Cannot get problem metadata for object: {obj}')

    if prob_meta is None or prob_meta.get('pathname', None) is None:
        raise RuntimeError('The output directory cannot be accessed before setup.')

    prob_pathname = prob_meta['pathname']

    dirpath = pathlib.Path(*[f'{p}_out'
                             for p in prob_pathname.split('/')]) / pathlib.Path(*subdirs)

    if comm.rank == 0 and mkdir:
        dirpath.mkdir(parents=True, exist_ok=True)

    return dirpath


def _is_openmdao_output_dir(directory):
    """
    Check if a directory is an OpenMDAO output directory.

    Parameters
    ----------
    directory : str or Path
        The directory to check.

    Returns
    -------
    bool
        True if the directory is an OpenMDAO output directory, False otherwise.
    """
    directory = pathlib.Path(directory)
    return directory.is_dir() and directory.name.endswith('_out') \
        and (directory / '.openmdao_out').exists()


def _find_openmdao_output_dirs(paths, recurse):
    """
    Find all OpenMDAO output directories in the given path.

    Parameters
    ----------
    paths : str or Path or Iterable
        The path to search for OpenMDAO output directories.
    recurse : bool
        If True, search recursively.

    Returns
    -------
    list
        A list of OpenMDAO output directories.
    """
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    elif not isinstance(paths, Iterable):
        raise ValueError("The 'paths' parameter must be a string, Path, or an iterable of them.")

    openmdao_dirs = []
    for path in paths:
        path = pathlib.Path(path)
        if not path.is_dir():
            continue

        for root, dirs, _ in os.walk(path):
            # Use a copy of the dirs list to avoid modifying it while iterating
            if _is_openmdao_output_dir(root):
                openmdao_dirs.append(pathlib.Path(root))
            for d in dirs[:]:
                dir_path = pathlib.Path(root) / d
                if _is_openmdao_output_dir(dir_path):
                    openmdao_dirs.append(dir_path)
                    dirs.remove(d)  # Do not recurse into OpenMDAO output directories
            if not recurse:
                break
    return openmdao_dirs


def clean_outputs(obj='.', recurse=False, prompt=True, dryrun=False):
    """
    Remove output directories created by OpenMDAO.

    A directory is determined to be an OpenMDAO output directory if its name
    ends in `_out` and it contains the file `.openmdao_out`.

    Parameters
    ----------
    obj : Problem or System or Solver or str or Path
        The problem or system or solver whose output file should be removed.
    recurse : bool
        If True, and if obj is a string or Path, recurse into it
        finding and removing OpenMDAO output directories along the way.
        This option is ignored if obj is a Problem, System, or Solver.
    prompt : bool
        If True, prompt the user to confirm directories to be removed.
        This option is ignored if obj is a Problem, System, or Solver.
    dryrun : bool
        If True, report which directories would be removed without actually removing them.
    """
    output_dirs = []

    if isinstance(obj, (str, pathlib.Path)):
        output_dirs = _find_openmdao_output_dirs(obj, recurse)
    elif hasattr(obj, 'get_outputs_dir'):
        output_dir = obj.get_outputs_dir()
        prompt = False
        if output_dir and _is_openmdao_output_dir(output_dir):
            output_dirs.append(pathlib.Path(output_dir))

    if not output_dirs:
        print('No OpenMDAO output directories found.')
        return
    else:
        print(f'Found {len(output_dirs)} OpenMDAO output directories:')

    removed_count = 0
    for dir_path in sorted(output_dirs):
        if dryrun:
            print(f'Would remove {dir_path} (dryrun = True).')
        elif prompt:
            response = input(f"Remove {dir_path}? [y/N] ").strip().lower()
            if response == 'y':
                shutil.rmtree(dir_path)
                print(f'Removed {dir_path}')
                removed_count += 1
        else:
            shutil.rmtree(dir_path)
            print(f'Removed {dir_path}')
            removed_count += 1

    print(f'Removed {removed_count} OpenMDAO output directories.')

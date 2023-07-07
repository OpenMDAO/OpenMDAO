"""
Utilities for working with files.
"""

import sys
import os
import importlib
import types
from fnmatch import fnmatch
from os.path import join, basename, dirname, isfile, split, splitext, abspath


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
        raise ValueError(f"'{s}' does not end with '.py'")

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

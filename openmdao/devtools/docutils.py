import json
import os
import sys
import argparse
from openmdao.utils.file_utils import files_iter


def reset_notebook(fname, dryrun=False):
    """
    Empties the output fields and resets execution_count in all code cells in the given notebook.

    Also removes any empty code cells. The specified notebook is overwritten.

    Parameters
    ----------
    fname : str
        Name of the notebook file.
    dryrun : bool
        If True, don't actually update the file.

    Returns
    -------
    bool
        True if the file was updated or would have been updated if not a dry run.
    """

    with open(fname) as f:
        dct = json.load(f)

    changed = False
    newcells = []
    for cell in dct['cells']:
        if cell['cell_type'] == 'code':
            if cell['source']:  # cell is not empty
                if cell['execution_count'] is not None or len(cell['outputs']) > 0:
                    cell['execution_count'] = None
                    cell['outputs'] = []
                    changed = True
                newcells.append(cell)
        else:
            newcells.append(cell)

    changed |= len(dct['cells']) != len(newcells)

    dct['cells'] = newcells

    if changed and not dryrun:
        with open(fname, 'w') as f:
            json.dump(dct, f, indent=1, ensure_ascii=False)

    return changed


def reset_notebook_cmd():
    """
    Run reset_notebook on notebook files.
    """
    parser = argparse.ArgumentParser(description='Empty output cells, reset execution_count, and '
                                     'remove empty cells of jupyter notebook(s).')
    parser.add_argument('file', nargs='*', help='Jupyter notebook file(s).')
    parser.add_argument('-r', '--recurse', action='store_true', dest='recurse',
                        help='Search through all directories at or below the current one for the '
                        'specified file(s).  If no files are specified, reset all jupyter notebook '
                        'files found.')
    parser.add_argument('-i', '--include', action='append', dest='includes',
                        default=[], help='If the --recurse option is active, this specifies a '
                        'local filename or glob pattern to match. This argument may be supplied '
                        'multiple times.')
    parser.add_argument('-d', '--dryrun', action='store_true', dest='dryrun',
                        help="Report which notebooks would be updated but don't actually update "
                        "them.")
    args = parser.parse_args()

    if args.dryrun:
        updatestr = 'Would have updated file'
    else:
        updatestr = 'Updated file'

    if args.recurse:
        if args.file:
            print("When using the --recurse option, don't specify filenames. Use --include "
                  "instead.")
            sys.exit(-1)

        if not args.includes:
            args.includes = ['*.ipynb']

        for f in files_iter(file_includes=args.includes):
            if not f.endswith('.ipynb'):
                print(f"Ignoring {f} (not a notebook).")
                continue
            if reset_notebook(f, args.dryrun):
                print(updatestr, f)
    else:

        if args.includes:
            print("The --include option only works when also using --recurse.")
            sys.exit(-1)

        for f in sorted(args.file):
            if os.path.isdir(f):
                continue
            if not f.endswith('.ipynb'):
                print(f"Ignoring {f} (not a notebook).")
                continue
            if not os.path.isfile(f):
                print(f"Can't find file '{f}'.")
                sys.exit(-1)
            if reset_notebook(f, args.dryrun):
                print(updatestr, f)


def nb2dict(fname):
    with open(fname) as f:
        return json.load(f)


def notebook_filter(fname, filters):
    """
    Return True is the given notebook satisfies the given filter function.

    Parameters
    ----------
    fname : str
        Name of the notebook file.
    filters : list of functions
        The filter functions.  They take a dictionary as an arg and return True/False.

    Returns
    -------
    bool
        True if the filter returns True.
    """
    dct = nb2dict(fname)

    for f in filters:
        if f(dct):
            return True

    return False


def is_parallel(dct):
    """
    Return True if the notebook containing the dict uses ipyparallel.
    """
    for cell in dct['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if 'ipyparallel' in line:
                    return True
    return False


def section_filter(dct, section):
    """
    Return True if the notebook containing the dict contains the given section string.
    """
    for cell in dct['cells']:
        if cell['cell_type'] == 'markdown':
            for line in cell['source']:
                if section in line and line.startswith('#'):
                    return True
    return False


def string_filter(dct, s):
    """
    Return True if the notebook containing the dict contains the given string.
    """
    for cell in dct['cells']:
        if cell['cell_type'] in ('markdown', 'code'):
            for line in cell['source']:
                if s in line:
                    return True
    return False


def find_notebooks_iter(section=None, string=None):
    filters = []
    if section:
        filters.append(lambda dct: section_filter(dct, section))
    if string:
        filters.append(lambda dct: string_filter(dct, string))

    dexcludes = ['.ipynb_checkpoints', '_*']
    for f in files_iter(file_includes=['*.ipynb'], dir_excludes=dexcludes):
        if not filters or notebook_filter(f, filters):
            yield f


def pick_one(files):
    print("Multiple matches found.")
    while True:
        for i, f in enumerate(files):
            print(f"{i}) {f}")
        try:
            response = int(input("\nSelect the index of the file to view: "))
        except ValueError:
            print("\nBAD index.  Try again.\n")
            continue
        if response < 0 or response > (len(files) + 1):
            print(f"\nIndex {response} is out of range.  Try again.\n")
            continue
        return files[response]


def show_notebook_cmd():
    """
    Display a notebook given a keyword.
    """
    parser = argparse.ArgumentParser(description='Empty output cells, reset execution_count, and '
                                     'remove empty cells of jupyter notebook(s).')
    parser.add_argument('file', nargs='?', help='Look for notebook having the given base filename')
    parser.add_argument('--section', action='store', dest='section',
                        help='Look for notebook(s) having the given section string.')
    parser.add_argument('-s', '--string', action='store', dest='string',
                        help='Look for notebook(s) having the given string in a code or markdown '
                        'cell.')
    args = parser.parse_args()

    if args.file is None:
        fname = None
    elif args.file.endswith('.ipynb'):
        fname = args.file
    else:
        fname = args.file + '.ipynb'

    if fname is not None:
        files = [f for f in find_notebooks_iter() if os.path.basename(f) == fname]
        if not files:
            print(f"Can't find file {fname}.")
            sys.exit(-1)
    else:
        files = list(find_notebooks_iter(section=args.section, string=args.string))
        if not files:
            print(f"No matching notebook files found.")
            sys.exit(-1)

    if len(files) == 1:
        show_notebook(files[0], nb2dict(files[0]))
    else:
        f = pick_one(files)
        show_notebook(f, nb2dict(f))


def show_notebook(f, dct):
    if is_parallel(dct):
        pidfile = os.path.join(os.path.expanduser('~'), '.ipython/profile_mpi/pid/ipcluster.pid')
        if not os.path.isfile(pidfile):
            print("cluster isn't running...")
            sys.exit(-1)
    os.system(f"jupyter notebook {f}")
import json
import os
import sys
import argparse
from openmdao.utils.file_utils import files_iter
from nbformat.validator import normalize


def reset_notebook(fname, dryrun=False):
    """
    Empties the output fields and resets execution_count in all code cells in the given notebook.

    Also removes any empty code cells and normalizes the specified notebook, which is overwritten.

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
    changed = False

    with open(fname) as f:
        dct = json.load(f)

    changes, dct = normalize(dct)
    if changes > 0:
        print(f"nbformat.validator normalize() made {changes} changes.")
        changed = True

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
            print(file=f)  # avoid 'no newline at end of file' message

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

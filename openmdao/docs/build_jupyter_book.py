#!/usr/bin/env python
import argparse
import os
import pathlib
import fnmatch
import shutil


_this_file = pathlib.Path(__file__).resolve()
REPO_ROOT = _this_file.parent
BOOK_DIR = pathlib.Path(REPO_ROOT, 'openmdao_book')


def copy_build_artifacts(book_dir=BOOK_DIR):
    """
    Copy build artifacts (html files, images, etc) to the output _build directory.
    Parameters
    ----------
    book_dir : str
        The directory containing the Jupyter-Book to be created.
    """
    PATTERNS_TO_COPY = ('*.html', '*.png')
    TARGET_DIR = '_build'
    EXCLUDE_DIRS = ('_build', '.ipynb_checkpoints')

    for dirpath, dirs, files in os.walk(book_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel_path = pathlib.PurePath(dirpath).parts[1:]
        target_path = pathlib.PurePath(book_dir, TARGET_DIR, 'html', *rel_path)
        files_to_copy = set()
        for pattern in PATTERNS_TO_COPY:
            files_to_copy |= set(fnmatch.filter(files, pattern))
        for f in files_to_copy:
            src = pathlib.PurePath(dirpath, f)
            dst = pathlib.PurePath(target_path, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)


def build_book(book_dir=BOOK_DIR, clean=True):
    """
    Clean (if requested), build, and copy over necessary files for the JupyterBook to be created.
    Parameters
    ----------
    book_dir
    clean
    """
    save_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    if clean:
        os.system(f'jupyter-book clean {book_dir}')
    os.system(f'jupyter-book build -W {book_dir}')
    copy_build_artifacts(book_dir)
    os.chdir(save_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a JupyterBook and automatically copy over'
                                                 'necessary build artifacts')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Clean the old book out before building (default is False).')
    parser.add_argument('-b', '--book', action='store', default='openmdao_book',
                        help="The name of the book to be built (default is 'openmdao_book').")
    args = parser.parse_args()
    build_book(book_dir=args.book, clean=args.clean)



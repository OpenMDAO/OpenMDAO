#!/usr/bin/env python
import os
import pathlib
import fnmatch
import shutil

def copy_build_artifacts(executed_book_dir='_executed_book'):
    """
    Copy build artifacts (html files, images, etc) to the Sphinx output directory.

    Parameters
    ----------
    executed_book_dir : str
        The directory containing the pre-executed notebooks and Sphinx build output.
        Source files are walked from this directory; build output is written to
        <executed_book_dir>/_build/html/.
    """
    PATTERNS_TO_COPY = ('*.html', '*.png')
    BUILD_HTML = pathlib.PurePath(executed_book_dir, '_build', 'html')
    EXCLUDE_DIRS = ('_build', '.ipynb_checkpoints')

    for dirpath, dirs, files in os.walk(executed_book_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel_path = pathlib.PurePath(dirpath).parts[1:]
        target_path = pathlib.PurePath(BUILD_HTML, *rel_path)

        files_to_copy = set()
        for pattern in PATTERNS_TO_COPY:
            files_to_copy |= set(fnmatch.filter(files, pattern))

        for f in files_to_copy:
            src = pathlib.PurePath(dirpath, f)
            dst = pathlib.PurePath(target_path, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)

    # sphinx-book-theme does not always copy 'sphinx-book-theme.css' to the build
    # directory; om-theme.css extends from it so we copy it explicitly.
    import sphinx_book_theme
    theme_path = pathlib.PurePath(sphinx_book_theme.__file__).parent
    style_path = pathlib.PurePath(theme_path, 'theme/sphinx_book_theme/static/styles')
    css_target = pathlib.PurePath(BUILD_HTML, '_static')
    os.makedirs(css_target, exist_ok=True)
    for f in os.listdir(style_path):
        shutil.copy(pathlib.PurePath(style_path, f), css_target)


if __name__ == '__main__':
    copy_build_artifacts('_executed_book')


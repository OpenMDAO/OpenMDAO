#!/usr/bin/env python
# coding: utf-8
import pathlib
import fnmatch
import json
import os
import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

from parameterized import parameterized


RUN_PATH = '.'
TIMEOUT = 600
KERNEL = 'python3'
EXCLUDE_DIRS = set(['_build', '_srcdocs', '.ipynb_checkpoints'])
BOOK_DIR = pathlib.PurePath(__file__).parent.parent


def collect_filenames(book_dir='openmdao_book'):
    """
    Return filenames of notebooks under book_dir.

    Parameters
    ----------
    book_dir : str
        The directory containing the Jupyter-Book to be created.

    Yields
    ------
    str
        The name of a notebook.

    """
    for dirpath, dirs, files in os.walk(book_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith('.ipynb'):
                yield str(pathlib.PurePath(dirpath, f))


def f2str(func, nparams, params):
    rel_path = str(pathlib.PurePath(params.args[0]).relative_to(BOOK_DIR))
    return 'test_notebook_' + rel_path.replace('/', '_').replace('\\', '_').replace('.', '_')


class TestNotebooks(unittest.TestCase):

    @parameterized.expand(collect_filenames(BOOK_DIR), name_func=f2str)
    def test_notebooks(self, n):
        nb_path = pathlib.PurePath(n)
        nb_rel_path = nb_path.relative_to(BOOK_DIR)
        os.chdir(str(nb_path.parent))
        with open(n) as f:
            try:
                nb = nbformat.read(f, as_version=4)
            except json.read.NotJSONError:
                self.fail(f'Notebook is not valid JSON: {nb_rel_path}.\n')
            except json.decoder.JSONDecodeError:
                self.fail(f'Unable to parse notebook {nb_rel_path}.\n')

            ep = ExecutePreprocessor(timeout=int(TIMEOUT), kernel_name=KERNEL)
            try:
                ep.preprocess(nb, {'metadata': {'path': RUN_PATH}})
            except CellExecutionError as e:
                trb = e.traceback

                # If SNOPT is not available during PR builds, then don't raise an error.
                # SNOPT is only available during merge to main.
                GITHUB_EV = os.environ.get("GITHUB_EVENT_NAME")
                PR = GITHUB_EV and GITHUB_EV == "pull_request"
                if not (PR and 'Optimizer SNOPT is not available' in trb):
                    self.fail(f'{nb_rel_path} failed due to exception.\n{trb}')
            except TimeoutError:
                self.fail(f'Timeout executing the notebook {n}.\n')
            finally:
                # This is where we could optionally write the notebook to an output file
                # with open(n_out + '.ipynb', mode='wt') as f:
                #     nbformat.write(nb, f)
                pass


if __name__ == '__main__':
    unittest.main()

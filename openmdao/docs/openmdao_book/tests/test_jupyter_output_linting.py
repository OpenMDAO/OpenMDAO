import unittest
import os.path
import pathlib
import json
import sys

exclude = [
    'tests',
    'test',
    '_build',
    '.ipynb_checkpoints',
    '_srcdocs',
    '__pycache__'
]

directories = []

top = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for root, dirs, files in os.walk(top, topdown=True):
    # do not bother looking further down in excluded dirs
    dirs[:] = [d for d in dirs if d not in exclude]
    for di in dirs:
        directories.append(os.path.join(root, di))

def _get_files():

    for dir_name in directories:
        dirpath = os.path.join(top, dir_name)

        # Loop over files
        for file_name in os.listdir(dirpath):
            if not file_name.startswith('_') and file_name[-6:] == '.ipynb':
                yield dirpath + "/" + file_name


FILES = list(_get_files())
if len(FILES) < 1:
    raise RuntimeError(f"No notebooks found. Top directory is {top}.")


@unittest.skipIf(sys.platform =='win32', "Tests don't work in Windows")
class LintJupyterOutputsTestCase(unittest.TestCase):
    """
    Check Jupyter Notebooks for outputs through execution count and recommend to remove output.
    """

    def test_output(self):
        """
        Check that output has been cleaned out of all cells.
        """
        for file in FILES:
            with self.subTest(file):
                with open(file) as f:
                    json_data = json.load(f)
                    for i in json_data['cells']:
                        if 'execution_count' in i and i['execution_count'] is not None:
                            msg = "Clear output with 'jupyter nbconvert  --clear-output " \
                                  f"--inplace path_to_notebook.ipynb'"
                            self.fail(f"Output found in {file}.\n{msg}")

    def test_header(self):
        """
        Check Jupyter Notebooks for code cell installing openmdao.
        """
        header = ["try:\n",
                  "    from openmdao.utils.notebook_utils import notebook_mode\n",
                  "except ImportError:\n",
                  "    !python -m pip install openmdao[notebooks]"]

        mpi_header = ['%pylab inline\n',
                      'from ipyparallel import Client, error\n',
                      'cluster=Client(profile="mpi")\n',
                      'view=cluster[:]\n',
                      'view.block=True\n',
                      '\n']
        mpi_header.extend(header)

        for file in FILES:
            with open(file) as f:

                # This one is exempt from these lint rules.
                if 'getting_started.ipynb'  in file:
                    continue

                json_data = json.load(f)

                code_cells = ['code' for cell in json_data['cells'] \
                              if cell['cell_type'] == 'code']
                if len(code_cells) < 1:
                    continue

                first_block = json_data['cells'][0]['source']
                if first_block != header and first_block != mpi_header:
                    header_text = '\n'.join(header)
                    msg = f'required header not found in notebook {file}\n' \
                          f'All notebooks should contain the following block before ' \
                          f'any other code blocks:\n' \
                          f'-----------------------------------------\n' \
                          f'{header_text}\n' \
                          f'-----------------------------------------\n'
                    self.fail(msg)

                correct_tags = ['active-ipynb', 'remove-input', 'remove-output']
                msg = f"Missing metadata tags in header in notebook {file}. Found " \
                      f"headers must contain the following tags.{correct_tags}."
                try:
                    first_cell = json_data['cells'][0]['metadata']['tags']
                except KeyError:
                    msg = f"Missing metadata tags in header in notebook {file}. " \
                          f"Headers must contain the following tags: {correct_tags}."
                    self.fail(msg)

                if sorted(first_cell) != sorted(correct_tags):
                    msg = f"Incorrect header tags in notebook {file}. Found " \
                          f"{sorted(first_cell)}, should be: {sorted(correct_tags)}."
                    self.fail(msg)

    def test_assert(self):
        """
        Make sure any code cells with asserts are hidden.
        """
        for file in FILES:
            with open(file) as f:
                json_data = json.load(f)
                blocks = json_data['cells']
                for block in blocks[1:]:

                    # Don't check markup cells
                    if block['cell_type'] != 'code':
                        continue

                    tags = block['metadata'].get('tags')
                    if tags:

                        # Don't check hidden cells
                        if 'remove-input' in tags and 'remove-output' in tags:
                            continue

                        # We allow an assert in a cell if you tag it.
                        if "allow-assert" in tags:
                            continue

                    code = ''.join(block['source'])
                    if 'assert' in code:

                        msg = f"Assert found in a code block in {file}. "
                        self.fail(msg)


if __name__ == '__main__':
    unittest.main()

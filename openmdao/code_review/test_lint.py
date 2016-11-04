from __future__ import print_function

import unittest
import os
import sys
from subprocess import Popen, PIPE, STDOUT, call

directories = [
    'assemblers',
    'core',
    'jacobians',
    'proc_allocators',
    'solvers',
    'utils',
    'vectors',
]

def _get_files():
    """A generator of files to check for pep8/pep257 violations."""
    topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for dir_name in directories:
        dirpath = os.path.join(topdir, dir_name)

        for file_name in  os.listdir(dirpath):
            if file_name != '__init__.py' and file_name[-3:] == '.py':
                yield os.path.join(dirpath, file_name)


class LintTestCase(unittest.TestCase):
    def test_pep8(self):
        for path in _get_files():
            p = Popen(['pep8', '--show-source',
                       path],
                      stdout=PIPE, stderr=STDOUT)
            output = p.communicate()[0]

            if p.returncode:
                msgs = [line for line in output.split('\n') if ':' in line]
                self.fail('pep8 failure: %s' % '\n'.join(msgs))

    def test_pep257(self):
        for path in _get_files():
            p = Popen(['pep257', path], stdout=PIPE, stderr=STDOUT)
            output = p.communicate()[0]

            if p.returncode:
                msgs = [line for line in output.split('\n') if ':' in line]
                self.fail('pep257 failure: %s' % '\n'.join(msgs))


if __name__ == '__main__':
    for path in _get_files():
        for check in ['pep8', 'pep257']:
            print ('-' * 79)
            print (check, path)
            call([check, path])

    print()
    print()

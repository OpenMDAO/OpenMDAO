from __future__ import print_function

import unittest
import os
import sys
from subprocess import Popen, PIPE, STDOUT, call
from fnmatch import fnmatch, filter as fnfilter

directories = [
    'assemblers',
    'core',
    'jacobians',
    'proc_allocators',
    'solvers',
    'utils',
    'vectors',
]

# fill in patterns to exclude directories here
dir_excludes = [
    'docs',
    '_*',
    'tests',
    'test_suite',  # TODO: this should really be included, but currently has failures
    'devtools',   # TODO: add this back after problem_viewer is fixed
]

file_excludes = [
    'test_*',
    '__init__.py',
]

ignores = {
    'pep8': [
        'E131',  # continuation line unaligned for hanging indent
    ],
    'pep257': [

    ]
}


def _get_files():
    """A generator of files to check for pep8/pep257 violations."""
    topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for dirpath, dnames, fnames in os.walk(topdir):
        for dpattern in dir_excludes:
            newdn = [d for d in dnames if not fnmatch(d, dpattern)]
            if len(newdn) != len(dnames):
                dnames[:] = newdn # replace contents of dnames to cause pruning

        for fpattern in file_excludes:
            fnames = [f for f in fnames if not fnmatch(f, fpattern)]

        for name in fnfilter(fnames, '*.py'):
            yield os.path.join(dirpath, name)


class LintTestCase(unittest.TestCase):
    def test_pep8(self):
        ignore = ','.join(ignores['pep8'])
        if ignore:
            ignore = '--ignore=%s' % ignore

        for path in _get_files():
            p = Popen(['pep8', '--show-source', ignore, path],
                      stdout=PIPE, stderr=STDOUT)
            output = p.communicate()[0]

            if p.returncode:
                msgs = [line for line in output.split('\n') if ':' in line]
                self.fail('pep8 failure: %s' % '\n'.join(msgs))

    def test_pep257(self):
        ignore = ','.join(ignores['pep257'])
        if ignore:
            ignore = '--ignore=%s' % ignore

        for path in _get_files():
            p = Popen(['pep257', ignore, path], stdout=PIPE, stderr=STDOUT)
            output = p.communicate()[0]

            if p.returncode:
                msgs = [line for line in output.split('\n') if ':' in line]
                self.fail('pep257 failure: %s' % '\n'.join(msgs))


if __name__ == '__main__':
    for path in _get_files():
        for check in ['pep8', 'pep257']:
            print ('-' * 79)
            print (check, path)

            ignore = ','.join(ignores[check])
            if ignore:
                ignore = '--ignore=%s' % ignore

            call([check, ignore, path])

    print()
    print()

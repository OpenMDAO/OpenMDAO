import unittest
import os
from fnmatch import fnmatch, filter as fnfilter

try:
    import pycodestyle
except ImportError:
    pycodestyle = None

try:
    import pydocstyle
except ImportError:
    pydocstyle = None

import re

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
    '_*.py',
]

ignores = {
    'pep8': [
        'E131',  # continuation line unaligned for hanging indent
        'W503',  # Line breaks should occur before a binary operator
        'W504',  # Line break occurred after a binary operator
    ],
    'pep257': [
        'D203',  # 1 blank required before class docstrings
        'D212',  # Multi-line doc strings start on second line (see: D213)
        'D200',  # One-line docstring should fit on one line with quotes
        'D413',  # Blank line required after last section of docstrings
    ]
}


def _get_files():
    """
    A generator of files to check for pep8/pep257 violations.
    """
    topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for dirpath, dnames, fnames in os.walk(topdir):
        for dpattern in dir_excludes:
            newdn = [d for d in dnames if not fnmatch(d, dpattern)]
            if len(newdn) != len(dnames):
                dnames[:] = newdn  # replace contents of dnames to cause pruning

        for fpattern in file_excludes:
            fnames = [f for f in fnames if not fnmatch(f, fpattern)]

        for name in fnfilter(fnames, '*.py'):
            yield os.path.join(dirpath, name)

if pycodestyle:
    class StringReport(pycodestyle.StandardReport):

        def get_failures(self):
            """
            Returns the list of failures, including source lines, as a formatted
            strings ready to be printed.
            """
            err_strings = []
            if self.total_errors > 0:
                self._deferred_print.sort()
                for line_number, offset, code, text, doc in self._deferred_print:
                    err_strings.append(self._fmt % {
                        'path': self.filename,
                        'row': self.line_offset + line_number, 'col': offset + 1,
                        'code': code, 'text': text,
                    })
                    if line_number > len(self.lines):
                        line = ''
                    else:
                        line = self.lines[line_number - 1]
                    err_strings.append(line.rstrip())
                    err_strings.append(re.sub(r'\S', ' ', line[:offset]) + '^')
            return err_strings

        def get_file_results(self):
            return self.file_errors


class LintTestCase(unittest.TestCase):

    @unittest.skipUnless(pycodestyle, "requires 'pycodestyle', install openmdao[test]")
    def test_pep8(self):
        pep8opts = pycodestyle.StyleGuide(
            ignore=ignores['pep8'],
            max_line_length=100,
            format='pylint'
        ).options

        report = StringReport(pep8opts)
        failures = []
        for file in _get_files():
            checker = pycodestyle.Checker(file, options=pep8opts, report=report)
            checker.check_all()
            report = checker.report
            if report.get_file_results() > 0:
                failures.extend(report.get_failures())
                failures.append('')

        if failures:
            self.fail('{} PEP 8 Failure(s):\n'.format(report.total_errors) + '\n'.join(failures))

    @unittest.skipUnless(pydocstyle, "requires 'pydocstyle', install openmdao[test]")
    def test_pep257(self):
        failures = [str(fail) for fail in pydocstyle.check(_get_files(), ignore=ignores['pep257']) ]
        if failures:
            self.fail('{} PEP 257 Failure(s):\n'.format(len(failures)) + '\n'.join(failures))


if __name__ == "__main__":
    unittest.main()
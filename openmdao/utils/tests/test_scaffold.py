
import unittest
import os
import importlib
from subprocess import check_call

import openmdao.api as om
from openmdao.utils.scaffold import _camel_case_split, _write_template
from openmdao.utils.testing_utils import use_tempdirs

try:
    import pip
except ImportError:
    pip = None


@use_tempdirs
class TestScaffold(unittest.TestCase):
    def test_explicit_comp(self):
        template = _write_template(None, 'ExplicitComponent', class_name='ExplicitFooBar')
        expected = [
            'class ExplicitFooBar(ExplicitComponent):',
            'def compute(',
            'def compute_partials(',
            'def initialize(',
            'def setup(',
        ]
        for exp in expected:
            self.assertTrue(exp in template, 'template is missing %s' % exp)

    def test_implicit_comp(self):
        template = _write_template(None, 'ImplicitComponent', class_name='ImplicitFooBar')
        expected = [
            'class ImplicitFooBar(ImplicitComponent):',
            'def apply_nonlinear(',
            'def solve_nonlinear(',
            'def linearize(',
            'def apply_linear(',
            'def solve_linear(',
            'def initialize(',
            'def setup(',
        ]
        for exp in expected:
            self.assertTrue(exp in template, 'template is missing %s' % exp)

    def test_camel_case_splitter(self):
        pairs = [
            ('FooBar', 'foo_bar'),
            ('FooBarBaz', 'foo_bar_baz'),
            ('FooBBar', 'foo_bbar'),
            ('BookendS', 'bookend_s')
        ]

        for ccase, expected in pairs:
            self.assertEqual(expected, _camel_case_split(ccase))

    @unittest.skipIf(pip is None, 'pip must be installed to test scaffolding packages.')
    def test_packages(self):
        bases = [
            ('command', ()),
            ('CaseRecorder', ('../rec.out',)),
            ('BaseCaseReader', ('../rec.out',)),
            ('Driver', ()),
            ('ExplicitComponent', ()),
            ('Group', ()),
            ('ImplicitComponent', ()),
            ('LinearSolver', ()),
            ('NonlinearSolver', ()),
            ('SurrogateModel', ())
        ]

        startdir = os.getcwd()

        for base, args in bases:
            os.chdir(startdir)
            cname = 'My' + base
            pkgname = 'my_' + base.lower() + '999'
            if base == 'command':
                check_call(['openmdao', 'scaffold', '--cmd', cname.lower(), '-p', pkgname])
                tgtname = '_' + cname.lower() + '_setup'
            else:
                check_call(['openmdao', 'scaffold', '-c', cname, '-b', base, '-p', pkgname])
                tgtname = cname

            os.chdir(pkgname)

            # install it
            check_call(['pip', 'install', '-q', '--no-cache-dir', '--no-deps', '.'])

            try:
                modname = _camel_case_split(cname)

                # try to instantiate it
                mod = importlib.import_module('.'.join((pkgname, modname)))
                klass = getattr(mod, tgtname)
                instance = klass(*args)

            finally:

                try:
                    # uninstall it
                    check_call(['pip', 'uninstall', '-q', '-y', pkgname])
                except CalledProcessError:
                    self.fail("Package '{}' failed to uninstall.  "
                              "You'll have to do it manually.".format(pkgname))

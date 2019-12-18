
import unittest

from six import iteritems

import openmdao.api as om
from openmdao.utils.scaffold import _camel_case_split, _write_template


class TestScaffold(unittest.TestCase):
    def test_explicit_comp(self):
        template = _write_template('explicitfoobar', 'ExplicitComponent', class_name='ExplicitFooBar')
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
        template = _write_template('implicitfoobar', 'ImplicitComponent', class_name='ImplicitFooBar')
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

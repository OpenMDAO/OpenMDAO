"""Component unittests."""
from __future__ import division

import numpy
import unittest

from openmdao.core.problem import Problem
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimple
from openmdao.test_suite.components.expl_comp_array import TestExplCompArray
from openmdao.test_suite.components.impl_comp_simple import TestImplCompSimple
from openmdao.test_suite.components.impl_comp_array import TestImplCompArray


class TestExplicitComponent(unittest.TestCase):

    def test___init___simple(self):
        """Test a simple explicit component."""
        comp = TestExplCompSimple()
        prob = Problem(comp).setup(check=False)

        prob['length'] = 3.
        prob['width'] = 2.
        prob.run()
        self.assertEqual(prob['area'], 6.)

    def test___init___array(self):
        """Test an explicit component with array inputs/outputs."""
        comp = TestExplCompArray(thickness=1.)
        prob = Problem(comp).setup(check=False)

        prob['lengths'] = 3.
        prob['widths'] = 2.
        prob.run()
        self.assertEqual(prob['total_volume'], 24.)


class TestImplicitComponent(unittest.TestCase):

    def assertEqualArrays(self, a, b):
        self.assertTrue(numpy.linalg.norm(a-b) < 1e-15)

    def test___init___simple(self):
        """Test a simple implicit component."""
        x = -0.5
        a = numpy.abs(numpy.exp(0.5 * x) / x)

        comp = TestImplCompSimple()
        prob = Problem(comp).setup(check=False)

        prob['a'] = a
        prob.run()
        self.assertEqual(prob['x'], x)

    def test___init___array(self):
        """Test an implicit component with array inputs/outputs."""
        comp = TestImplCompArray()
        prob = Problem(comp).setup(check=False)

        prob['rhs'] = numpy.ones(2)
        prob.run()
        self.assertEqualArrays(prob['x'], numpy.ones(2))


class TestIndepVarComp(unittest.TestCase):

    def assertEqualArrays(self, a, b):
        self.assertTrue(numpy.linalg.norm(a-b) < 1e-15)

    def test___init___1var(self):
        """Define one independent variable and set its value."""
        comp = IndepVarComp('indep_var')
        prob = Problem(comp).setup(check=False)

        self.assertEqual(prob['indep_var'], 1.0)

        prob['indep_var'] = 2.0
        self.assertEqual(prob['indep_var'], 2.0)

    def test___init___1var_val(self):
        """Define one independent variable with a default value."""
        comp = IndepVarComp('indep_var', val=2.0)
        prob = Problem(comp).setup(check=False)

        self.assertEqual(prob['indep_var'], 2.0)

    def test___init___1var_array(self):
        """Define one independent array variable."""
        array = numpy.array([
            [1., 2.],
            [3., 4.],
        ])

        comp = IndepVarComp('indep_var', val=array)
        prob = Problem(comp).setup(check=False)

        self.assertEqualArrays(prob['indep_var'], array)

    def test___init___2vars(self):
        """Define two independent variables at once."""
        comp = IndepVarComp((
            ('indep_var_1', 1.0),
            ('indep_var_2', 2.0),
        ))

        prob = Problem(comp).setup(check=False)

        self.assertEqual(prob['indep_var_1'], 1.0)
        self.assertEqual(prob['indep_var_2'], 2.0)


if __name__ == '__main__':
    unittest.main()

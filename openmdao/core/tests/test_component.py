"""Component unittests."""
from __future__ import division

import numpy
import unittest

from openmdao.api import Problem, IndepVarComp
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimple
from openmdao.test_suite.components.expl_comp_array import TestExplCompArray
from openmdao.test_suite.components.impl_comp_simple import TestImplCompSimple
from openmdao.test_suite.components.impl_comp_array import TestImplCompArray
from openmdao.devtools.testutil import assert_rel_error


class TestExplicitComponent(unittest.TestCase):

    def test___init___simple(self):
        """Test a simple explicit component."""
        comp = TestExplCompSimple()
        prob = Problem(comp).setup(check=False)

        # check optional metadata (desc)
        self.assertEqual(comp._var2meta['length']['desc'],
                         'length of rectangle')
        self.assertEqual(comp._var2meta['width']['desc'],
                         'width of rectangle')
        self.assertEqual(comp._var2meta['area']['desc'],
                         'area of rectangle')

        prob['length'] = 3.
        prob['width'] = 2.
        prob.run_model()
        assert_rel_error(self, prob['area'], 6.)

    def test___init___array(self):
        """Test an explicit component with array inputs/outputs."""
        comp = TestExplCompArray(thickness=1.)
        prob = Problem(comp).setup(check=False)

        prob['lengths'] = 3.*numpy.ones((2, 2))
        prob['widths'] = 2.*numpy.ones((2, 2))
        prob.run_model()
        assert_rel_error(self, prob['total_volume'], 24.)


class TestImplicitComponent(unittest.TestCase):

    def test___init___simple(self):
        """Test a simple implicit component."""
        x = -0.5
        a = numpy.abs(numpy.exp(0.5 * x) / x)

        comp = TestImplCompSimple()
        prob = Problem(comp).setup(check=False)

        prob['a'] = a
        prob.run_model()
        assert_rel_error(self, prob['x'], x)

    def test___init___array(self):
        """Test an implicit component with array inputs/outputs."""
        comp = TestImplCompArray()
        prob = Problem(comp).setup(check=False)

        prob['rhs'] = numpy.ones(2)
        prob.run_model()
        assert_rel_error(self, prob['x'], numpy.ones(2))


class TestIndepVarComp(unittest.TestCase):

    def test_simple(self):
        """Define one independent variable and set its value."""
        comp = IndepVarComp('indep_var')
        prob = Problem(comp).setup(check=False)

        assert_rel_error(self, prob['indep_var'], 1.0)

        prob['indep_var'] = 2.0
        assert_rel_error(self, prob['indep_var'], 2.0)

    def test_simple_default(self):
        """Define one independent variable with a default value."""
        comp = IndepVarComp('indep_var', val=2.0)
        prob = Problem(comp).setup(check=False)

        assert_rel_error(self, prob['indep_var'], 2.0)

    def test_simple_kwargs(self):
        """Define one independent variable with a default value and additional options."""
        comp = IndepVarComp('indep_var', 2.0, units='m', lower=0, upper=10)
        prob = Problem(comp).setup(check=False)

        assert_rel_error(self, prob['indep_var'], 2.0)

    def test_simple_array(self):
        """Define one independent array variable."""
        array = numpy.array([
            [1., 2.],
            [3., 4.],
        ])

        comp = IndepVarComp('indep_var', val=array)
        prob = Problem(comp).setup(check=False)

        assert_rel_error(self, prob['indep_var'], array)

    def test_multiple_default(self):
        """Define two independent variables at once."""
        comp = IndepVarComp((
            ('indep_var_1', 1.0),
            ('indep_var_2', 2.0),
        ))

        prob = Problem(comp).setup(check=False)

        assert_rel_error(self, prob['indep_var_1'], 1.0)
        assert_rel_error(self, prob['indep_var_2'], 2.0)

    def test_multiple_kwargs(self):
        """Define two independent variables at once and additional options."""
        comp = IndepVarComp((
            ('indep_var_1', 1.0, {'lower': 0, 'upper': 10}),
            ('indep_var_2', 2.0, {'lower': 1., 'upper': 20}),
        ))

        prob = Problem(comp).setup(check=False)

        assert_rel_error(self, prob['indep_var_1'], 1.0)
        assert_rel_error(self, prob['indep_var_2'], 2.0)

    def test_add_output(self):
        """Define two independent variables using the add_output method."""
        comp = IndepVarComp()
        comp.add_output('indep_var_1', 1.0, lower=0, upper=10)
        comp.add_output('indep_var_2', 2.0, lower=1, upper=20)

        prob = Problem(comp).setup(check=False)

        assert_rel_error(self, prob['indep_var_1'], 1.0)
        assert_rel_error(self, prob['indep_var_2'], 2.0)

    def test_error_novars(self):
        try:
            prob = Problem(IndepVarComp()).setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                "No outputs (independent variables) have been declared for "
                "this component. They must either be declared during "
                "instantiation or by calling add_output afterwards.")
        else:
            self.fail('Exception expected.')

    def test_error_badtup(self):
        try:
            comp = IndepVarComp((
                ('indep_var_1', 1.0, {'lower': 0, 'upper': 10}),
                'indep_var_2',
            ))
            prob = Problem(comp).setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                "IndepVarComp init: arg indep_var_2 must be a tuple of the "
                "form (name, value) or (name, value, keyword_dict).")
        else:
            self.fail('Exception expected.')

    def test_error_bad_arg(self):
        try:
            comp = IndepVarComp(1.0)
            prob = Problem(comp).setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                "first argument to IndepVarComp init must be either of type "
                "`str` or an iterable of tuples of the form (name, value) or "
                "(name, value, keyword_dict).")
        else:
            self.fail('Exception expected.')


if __name__ == '__main__':
    unittest.main()

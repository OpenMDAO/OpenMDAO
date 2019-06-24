"""IndepVarComp tests used in the IndepVarComp feature doc."""
from __future__ import division

import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error


class TestIndepVarComp(unittest.TestCase):

    def test_simple(self):
        """Define one independent variable and set its value."""
        import openmdao.api as om

        comp = om.IndepVarComp('indep_var')
        prob = om.Problem(comp).setup()

        assert_rel_error(self, prob['indep_var'], 1.0)

        prob['indep_var'] = 2.0
        assert_rel_error(self, prob['indep_var'], 2.0)

    def test_simple_default(self):
        """Define one independent variable with a default value."""
        import openmdao.api as om

        comp = om.IndepVarComp('indep_var', val=2.0)
        prob = om.Problem(comp).setup()

        assert_rel_error(self, prob['indep_var'], 2.0)

    def test_simple_kwargs(self):
        """Define one independent variable with a default value and additional options."""
        import openmdao.api as om

        comp = om.IndepVarComp('indep_var', val=2.0, units='m', lower=0, upper=10)
        prob = om.Problem(comp).setup()

        assert_rel_error(self, prob['indep_var'], 2.0)

    def test_simple_array(self):
        """Define one independent array variable."""
        import numpy as np

        import openmdao.api as om

        array = np.array([
            [1., 2.],
            [3., 4.],
        ])

        comp = om.IndepVarComp('indep_var', val=array)
        prob = om.Problem(comp).setup()

        assert_rel_error(self, prob['indep_var'], array)

    def test_multiple_default(self):
        """Define two independent variables at once."""
        import openmdao.api as om

        comp = om.IndepVarComp((
            ('indep_var_1', 1.0),
            ('indep_var_2', 2.0),
        ))

        prob = om.Problem(comp).setup()

        assert_rel_error(self, prob['indep_var_1'], 1.0)
        assert_rel_error(self, prob['indep_var_2'], 2.0)

    def test_multiple_kwargs(self):
        """Define two independent variables at once and additional options."""
        import openmdao.api as om

        comp = om.IndepVarComp((
            ('indep_var_1', 1.0, {'lower': 0, 'upper': 10}),
            ('indep_var_2', 2.0, {'lower': 1., 'upper': 20}),
        ))

        prob = om.Problem(comp).setup()

        assert_rel_error(self, prob['indep_var_1'], 1.0)
        assert_rel_error(self, prob['indep_var_2'], 2.0)

    def test_add_output(self):
        """Define two independent variables using the add_output method."""
        import openmdao.api as om

        comp = om.IndepVarComp()
        comp.add_output('indep_var_1', val=1.0, lower=0, upper=10)
        comp.add_output('indep_var_2', val=2.0, lower=1, upper=20)

        prob = om.Problem(comp).setup()

        assert_rel_error(self, prob['indep_var_1'], 1.0)
        assert_rel_error(self, prob['indep_var_2'], 2.0)

    def test_simple_with_tags(self):
        """Define one independent variable and set its value. Try filtering with tag"""
        from openmdao.api import Problem, IndepVarComp

        comp = IndepVarComp('indep_var', tags='tag1')
        prob = Problem(comp).setup(check=False)
        prob.run_model()

        # Outputs no tags
        outputs = prob.model.list_outputs(out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('indep_var', {'value': [1.]}),
        ])

        # Outputs with tag
        outputs = prob.model.list_outputs(out_stream=None, tags="tag1")
        self.assertEqual(sorted(outputs), [
            ('indep_var', {'value': [1.]}),
        ])

        # Outputs with wrong tag
        outputs = prob.model.list_outputs(out_stream=None, tags="tag_wrong")
        self.assertEqual(sorted(outputs), [])

    def test_add_output_with_tags(self):
        """Define two independent variables using the add_output method.
        Add tags to them and see if we can filter them with list_outputs"""
        from openmdao.api import Problem, IndepVarComp

        comp = IndepVarComp()
        comp.add_output('indep_var_1', val=1.0, lower=0, upper=10, tags="tag1")
        comp.add_output('indep_var_2', val=2.0, lower=1, upper=20, tags="tag2")

        prob = Problem(comp).setup(check=False)
        prob.run_model()

        # Outputs no tags
        outputs = prob.model.list_outputs(out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('indep_var_1', {'value': [1.]}),
            ('indep_var_2', {'value': [2.]}),
        ])

        # Outputs with tags
        outputs = prob.model.list_outputs(out_stream=None, tags="tag1")
        self.assertEqual(sorted(outputs), [
            ('indep_var_1', {'value': [1.]}),
        ])

        # Outputs with the indep_var_comp tags
        outputs = prob.model.list_outputs(out_stream=None, tags="indep_var_comp")
        self.assertEqual(sorted(outputs), [
            ('indep_var_1', {'value': [1.]}),
            ('indep_var_2', {'value': [2.]}),
        ])

        # Outputs with multiple tags
        outputs = prob.model.list_outputs(out_stream=None, tags=["tag1", "tag2"])
        self.assertEqual(sorted(outputs), [
            ('indep_var_1', {'value': [1.]}),
            ('indep_var_2', {'value': [2.]}),
        ])

        # Outputs with tag that does not match
        outputs = prob.model.list_outputs(out_stream=None, tags="tag3")
        self.assertEqual(sorted(outputs), [])

    def test_error_novars(self):
        try:
            prob = om.Problem(om.IndepVarComp()).setup()
        except Exception as err:
            self.assertEqual(str(err),
                "No outputs (independent variables) have been declared for "
                "component ''. They must either be declared during "
                "instantiation or by calling add_output or add_discrete_output afterwards.")
        else:
            self.fail('Exception expected.')

    def test_error_badtup(self):
        try:
            comp = om.IndepVarComp((
                ('indep_var_1', 1.0, {'lower': 0, 'upper': 10}),
                'indep_var_2',
            ))
            prob = om.Problem(comp).setup()
        except Exception as err:
            self.assertEqual(str(err),
                "IndepVarComp init: arg indep_var_2 must be a tuple of the "
                "form (name, value) or (name, value, keyword_dict).")
        else:
            self.fail('Exception expected.')

    def test_error_bad_arg(self):
        try:
            comp = om.IndepVarComp(1.0)
            prob = om.Problem(comp).setup()
        except Exception as err:
            self.assertEqual(str(err),
                "first argument to IndepVarComp init must be either of type "
                "`str` or an iterable of tuples of the form (name, value) or "
                "(name, value, keyword_dict).")
        else:
            self.fail('Exception expected.')

    def test_add_output_type_bug(self):
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x1', val=[1, 2, 3], lower=0, upper=10)

        model.add_subsystem('p', ivc)

        prob.setup()

        prob['p.x1'][0] = 0.5
        prob.run_model()

        assert_rel_error(self, prob['p.x1'][0], 0.5)


if __name__ == '__main__':
    unittest.main()

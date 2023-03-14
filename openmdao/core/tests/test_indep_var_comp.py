"""IndepVarComp tests used in the IndepVarComp feature doc."""
import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_warnings
from openmdao.utils.om_warnings import OMDeprecationWarning


class TestIndepVarComp(unittest.TestCase):

    def test_add_output_retval(self):
        # check basic metadata expected in return value
        expected = {
            'val': 3,
            'shape': (1,),
            'size': 1,
            'units': 'ft',
            'desc': '',
            'tags': {'openmdao:indep_var', 'openmdao:allow_desvar'},
        }
        expected_discrete = {
            'val': 3,
            'type': int,
            'desc': '',
            'tags': {'openmdao:indep_var'},
        }

        class IDVComp(om.IndepVarComp):
            def setup(self):
                meta = self.add_output('y', val=3.0, units='ft')
                for key, val in expected.items():
                    assert meta[key] == val, f'Expected {key}: {val} but got {key}: {meta[key]}'

                meta = self.add_discrete_output('disc', val=3)
                for key, val in expected_discrete.items():
                    assert meta[key] == val, f'Expected {key}: {val} but got {key}: {meta[key]}'

        prob = om.Problem()
        prob.model.add_subsystem('idv', IDVComp())
        prob.setup()

    def test_simple(self):
        """Define one independent variable and set its value."""

        comp = om.IndepVarComp('openmdao:indep_var')
        prob = om.Problem(comp).setup()

        assert_near_equal(prob.get_val('openmdao:indep_var'), 1.0)

        prob.set_val('openmdao:indep_var', 2.0)
        assert_near_equal(prob.get_val('openmdao:indep_var'), 2.0)

    def test_simple_default(self):
        """Define one independent variable with a default value."""

        comp = om.IndepVarComp('openmdao:indep_var', val=2.0)
        prob = om.Problem(comp).setup()

        assert_near_equal(prob.get_val('openmdao:indep_var'), 2.0)

    def test_simple_kwargs(self):
        """Define one independent variable with a default value and additional options."""

        comp = om.IndepVarComp('openmdao:indep_var', val=2.0, units='m', lower=0, upper=10)
        prob = om.Problem(comp).setup()

        assert_near_equal(prob.get_val('openmdao:indep_var'), 2.0)

    def test_simple_array(self):
        """Define one independent array variable."""

        array = np.array([
            [1., 2.],
            [3., 4.],
        ])

        comp = om.IndepVarComp('openmdao:indep_var', val=array)
        prob = om.Problem(comp).setup()

        assert_near_equal(prob.get_val('openmdao:indep_var'), array)

    def test_add_output(self):
        """Define two independent variables using the add_output method."""

        comp = om.IndepVarComp()
        comp.add_output('indep_var_1', val=1.0)
        comp.add_output('indep_var_2', val=2.0)

        prob = om.Problem(comp).setup()

        assert_near_equal(prob.get_val('indep_var_1'), 1.0)
        assert_near_equal(prob.get_val('indep_var_2'), 2.0)

    def test_promote_glob_no_inputs(self):
        p = om.Problem()
        p.model.add_subsystem('indep',
                              om.IndepVarComp('x', 2.0),
                              promotes_inputs=['*'],
                              promotes_outputs=['x'])
        p.model.add_subsystem('C1', om.ExecComp('y=x'), promotes_inputs=['x'], promotes_outputs=['y'])
        p.setup()
        p.run_model()
        self.assertEqual(p.get_val('x'), p.get_val('y'))

    def test_invalid_tags(self):
        with self.assertRaises(TypeError) as cm:
            comp = om.IndepVarComp('openmdao:indep_var', tags=99)

        self.assertEqual(str(cm.exception),
            "IndepVarComp: Value (99) of option 'tags' has type 'int', "
            "but one of types ('str', 'list') was expected.")

    def test_simple_with_tags(self):
        """Define one independent variable and set its value. Try filtering with tag"""

        comp = om.IndepVarComp('openmdao:indep_var', tags='tag1')
        prob = om.Problem(comp).setup(check=False)
        prob.run_model()

        # Outputs no tags
        outputs = prob.model.list_outputs(val=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('openmdao:indep_var', {}),
        ])

        # Outputs with automatically added indep_var_comp tag
        outputs = prob.model.list_outputs(val=False, out_stream=None, tags="openmdao:indep_var")
        self.assertEqual(sorted(outputs), [
            ('openmdao:indep_var', {}),
        ])

        # Outputs with tag
        outputs = prob.model.list_outputs(val=False, out_stream=None, tags="tag1")
        self.assertEqual(sorted(outputs), [
            ('openmdao:indep_var', {}),
        ])

        # Outputs with wrong tag
        outputs = prob.model.list_outputs(val=False, out_stream=None, tags="tag_wrong")
        self.assertEqual(sorted(outputs), [])

    def test_add_output_with_tags(self):
        """Define two independent variables using the add_output method.
        Add tags to them and see if we can filter them with list_outputs"""

        comp = om.IndepVarComp()
        comp.add_output('indep_var_1', val=1.0, tags="tag1")
        comp.add_output('indep_var_2', val=2.0, tags="tag2")

        prob = om.Problem(comp).setup(check=False)
        prob.run_model()

        # Outputs no tags
        outputs = prob.model.list_outputs(out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('indep_var_1', {'val': [1.]}),
            ('indep_var_2', {'val': [2.]}),
        ])

        # Outputs with tags
        outputs = prob.model.list_outputs(out_stream=None, tags="tag1")
        self.assertEqual(sorted(outputs), [
            ('indep_var_1', {'val': [1.]}),
        ])

        # Outputs with the indep_var tags
        outputs = prob.model.list_outputs(out_stream=None, tags="openmdao:indep_var")
        self.assertEqual(sorted(outputs), [
            ('indep_var_1', {'val': [1.]}),
            ('indep_var_2', {'val': [2.]}),
        ])

        # Outputs with multiple tags
        outputs = prob.model.list_outputs(out_stream=None, tags=["tag1", "tag2"])
        self.assertEqual(sorted(outputs), [
            ('indep_var_1', {'val': [1.]}),
            ('indep_var_2', {'val': [2.]}),
        ])

        # Outputs with tag that does not match
        outputs = prob.model.list_outputs(out_stream=None, tags="tag3")
        self.assertEqual(sorted(outputs), [])

    def test_error_novars(self):
        try:
            prob = om.Problem(om.IndepVarComp()).setup()
        except Exception as err:
            self.assertEqual(str(err),
                "<model> <class IndepVarComp>: No outputs (independent variables) have been declared. They must either be declared during "
                "instantiation or by calling add_output or add_discrete_output afterwards.")
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
        ivc.add_output('x1', val=[1, 2, 3])

        model.add_subsystem('p', ivc)

        prob.setup()

        prob['p.x1'][0] = 0.5
        prob.run_model()

        assert_near_equal(prob.get_val('p.x1')[0], 0.5)

    def test_options(self):
        class Parameters(om.IndepVarComp):
            def initialize(self):
                self.options.declare('num_x', default=0)
                self.options.declare('val_y', default=0.)

            def setup(self):
                self.add_discrete_output('num_x', val = np.zeros(self.options['num_x']))
                self.add_output('val_y',val = self.options['val_y'])

        prob = om.Problem(model=Parameters(num_x=4, val_y=2.5))
        prob.setup()
        prob.run_model()

        self.assertEqual(len(prob.get_val('num_x')), 4)
        self.assertEqual(prob.get_val('val_y'), 2.5)


if __name__ == '__main__':
    unittest.main()

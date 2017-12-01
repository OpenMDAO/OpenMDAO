"""Simple example demonstrating how to implement an explicit component."""
from __future__ import division

from six import assertRaisesRegex

import unittest

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp
from openmdao.devtools.testutil import assert_rel_error


# Note: The following class definitions are used in feature docs

class RectangleComp(ExplicitComponent):
    """
    A simple Explicit Component that computes the area of a rectangle.
    """
    def setup(self):
        self.add_input('length', val=1.)
        self.add_input('width', val=1.)
        self.add_output('area', val=1.)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['area'] = inputs['length'] * inputs['width']


class RectanglePartial(RectangleComp):

    def compute_partials(self, inputs, partials):
        partials['area', 'length'] = inputs['width']
        partials['area', 'width'] = inputs['length']


class RectangleJacVec(RectangleComp):

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'area' in d_outputs:
                if 'length' in d_inputs:
                    d_outputs['area'] += inputs['width'] * d_inputs['length']
                if 'width' in d_inputs:
                    d_outputs['area'] += inputs['length'] * d_inputs['width']
        elif mode == 'rev':
            if 'area' in d_outputs:
                if 'length' in d_inputs:
                    d_inputs['length'] += inputs['width'] * d_outputs['area']
                if 'width' in d_inputs:
                    d_inputs['width'] += inputs['length'] * d_outputs['area']


class RectangleGroup(Group):

    def setup(self):
        comp1 = self.add_subsystem('comp1', IndepVarComp())
        comp1.add_output('length', 1.0)
        comp1.add_output('width', 1.0)

        self.add_subsystem('comp2', RectanglePartial())
        self.add_subsystem('comp3', RectangleJacVec())

        self.connect('comp1.length', 'comp2.length')
        self.connect('comp1.length', 'comp3.length')
        self.connect('comp1.width', 'comp2.width')
        self.connect('comp1.width', 'comp3.width')


class ExplCompTestCase(unittest.TestCase):

    def test_simple(self):
        prob = Problem(RectangleComp())
        prob.setup(check=False)
        prob.run_model()

    def test_feature_simple(self):
        from openmdao.api import Problem
        from openmdao.core.tests.test_expl_comp import RectangleComp

        prob = Problem(RectangleComp())
        prob.setup(check=False)
        prob.run_model()

    def test_compute_and_list(self):
        prob = Problem(RectangleGroup())
        prob.setup(check=False)

        msg = "Unable to list inputs until model has been run."
        try:
            prob.model.list_inputs()
        except Exception as err:
            self.assertTrue(msg == str(err))
        else:
            self.fail("Exception expected")

        msg = "Unable to list outputs until model has been run."
        try:
            prob.model.list_outputs()
        except Exception as err:
            self.assertTrue(msg == str(err))
        else:
            self.fail("Exception expected")

        msg = "Unable to list residuals until model has been run."
        try:
            prob.model.list_residuals()
        except Exception as err:
            self.assertTrue(msg == str(err))
        else:
            self.fail("Exception expected")

        prob['comp1.length'] = 3.
        prob['comp1.width'] = 2.
        prob.run_model()
        assert_rel_error(self, prob['comp2.area'], 6.)
        assert_rel_error(self, prob['comp3.area'], 6.)

        # total derivs
        total_derivs = prob.compute_totals(
            wrt=['comp1.length', 'comp1.width'],
            of=['comp2.area', 'comp3.area']
        )
        assert_rel_error(self, total_derivs['comp2.area', 'comp1.length'], [[2.]])
        assert_rel_error(self, total_derivs['comp3.area', 'comp1.length'], [[2.]])
        assert_rel_error(self, total_derivs['comp2.area', 'comp1.width'], [[3.]])
        assert_rel_error(self, total_derivs['comp3.area', 'comp1.width'], [[3.]])

        # list inputs
        inputs = prob.model.list_inputs(out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('comp2.length', [3.]),
            ('comp2.width',  [2.]),
            ('comp3.length', [3.]),
            ('comp3.width',  [2.]),
        ])

        # list explicit outputs
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp1.length', [3.]),
            ('comp1.width',  [2.]),
            ('comp2.area',   [6.]),
            ('comp3.area',   [6.]),
        ])

        # list states
        states = prob.model.list_outputs(explicit=False, out_stream=None)
        self.assertEqual(states, [])

        # list residuals
        resids = prob.model.list_residuals(out_stream=None)
        self.assertEqual(sorted(resids), [
            ('comp1.length', [0.]),
            ('comp1.width',  [0.]),
            ('comp2.area',   [0.]),
            ('comp3.area',   [0.]),
        ])

        # list excluding both explicit and implicit components raises error
        msg = "You have excluded both Explicit and Implicit components."

        with assertRaisesRegex(self, RuntimeError, msg):
            prob.model.list_outputs(explicit=False, implicit=False)

        with assertRaisesRegex(self, RuntimeError, msg):
            prob.model.list_residuals(explicit=False, implicit=False)

    def test_simple_list_vars_options(self):


        from openmdao.api import IndepVarComp, Group, Problem, ExecComp

        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 12.0,
                                               lower=1.0, upper=100.0,
                                               ref = 1.1, ref0 = 2.1,
                                               units='inch',
                                               ))
        model.add_subsystem('p2', IndepVarComp('y', 1.0,
                                               lower=2.0, upper=200.0,
                                               ref = 1.2, res_ref = 2.2,
                                               units='ft',
                                               ))
        model.add_subsystem('comp', ExecComp('z=x+y',
                                             x={'value': 0.0, 'units':'inch'},
                                             y={'value': 0.0, 'units': 'inch'},
                                             z={'value': 0.0, 'units': 'inch'}))
        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')


        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        ###### list_inputs tests #####

        # list inputs
        # Cannot do exact equality here because the units cause comp.y to be slightly different than 12.0
        tol = 1e-7
        inputs = prob.model.list_inputs(out_stream=None)
        for actual, expected in zip(sorted(inputs),
                                    [
                                        ('comp.x', [12.]),
                                        ('comp.y', [12.]),
                                    ]
                                    ):
            self.assertEqual(actual[0], expected[0])
            assert_rel_error(self, actual[1], expected[1], tol)

        # Only other option for list_inputs is units
        inputs = prob.model.list_inputs(values=False, units=True, out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('comp.x', 'inch'),
            ('comp.y', 'inch'),
        ])

        ###### list_outputs tests #####

        # list outputs for implicit comps - should get none
        outputs = prob.model.list_outputs(implicit=True, explicit=False, out_stream=None)
        self.assertEqual(outputs, [])

        # list explicit outputs with values
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', np.array([24.] ), ),
            ('p1.x', np.array([12.]), ),
            ('p2.y', np.array([1.]), ),
        ])

        # list explicit outputs with units
        outputs = prob.model.list_outputs(implicit=False, values=False, units=True, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', 'inch', ),
            ('p1.x', 'inch', ),
            ('p2.y', 'ft', ),
        ])

        # list explicit outputs with bounds
        outputs = prob.model.list_outputs(implicit=False, bounds=True, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', [24.], None, None),
            ('p1.x', [12.], [1.0], [100.0]),
            ('p2.y', [1.], [2.0], [200.0]),
        ])

        # list explicit outputs with scaling
        outputs = prob.model.list_outputs(implicit=False, scaling=True, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', [24.], 1.0, 0.0, 1.0),
            ('p1.x', [12.], 1.1, 2.1, 1.1),
            ('p2.y', [1.], 1.2, 0.0, 2.2),
        ])

        ###### list_residuals tests #####

        # list residuals
        resids = prob.model.list_residuals(implicit=False, out_stream=None)
        self.assertEqual(sorted(resids), [
            ('comp.z', np.array([0.])),
            ('p1.x', np.array([0.])),
            ('p2.y', np.array([0.])),
        ])

        # list explicit residuals with units and values
        resids = prob.model.list_residuals(implicit=False, units=True, out_stream=None)
        self.assertEqual(sorted(resids), [
            ('comp.z', np.array([0.] ), 'inch'),
            ('p1.x', np.array([0.]), 'inch'),
            ('p2.y', np.array([0.]), 'ft'),
        ])

        # list explicit residuals with bounds
        resids = prob.model.list_residuals(implicit=False, bounds=True, out_stream=None)
        self.assertEqual(sorted(resids), [
            ('comp.z', [0.], None, None),
            ('p1.x', [0.], [1.0], [100.0]),
            ('p2.y', [0.], [2.0], [200.0]),
        ])

        # list explicit residuals with bounds
        resids = prob.model.list_residuals(implicit=False, scaling=True, out_stream=None)
        self.assertEqual(sorted(resids), [
            ('comp.z', [0.], 1.0, 0.0, 1.0),
            ('p1.x', [0.], 1.1, 2.1, 1.1),
            ('p2.y', [0.], 1.2, 0.0, 2.2),
        ])


if __name__ == '__main__':
    unittest.main()

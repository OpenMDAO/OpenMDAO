"""Simple example demonstrating how to implement an explicit component."""
from __future__ import division

from six import assertRaisesRegex

from six.moves import cStringIO
import unittest

import numpy as np

from openmdao.api import Problem, ExplicitComponent, NewtonSolver, ScipyKrylov, Group, \
    IndepVarComp, LinearBlockGS
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.double_sellar import SubSellar


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
            ('comp2.length', { 'value' :[3.]}),
            ('comp2.width',  { 'value' :[2.]}),
            ('comp3.length', { 'value' :[3.]}),
            ('comp3.width',  { 'value' :[2.]}),
        ])

        # list explicit outputs
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp1.length', { 'value' :[3.]}),
            ('comp1.width',  { 'value' :[2.]}),
            ('comp2.area',   { 'value' :[6.]}),
            ('comp3.area',   { 'value' :[6.]}),
        ])

        # list states
        states = prob.model.list_outputs(explicit=False, out_stream=None)
        self.assertEqual(states, [])

        # list excluding both explicit and implicit components raises error
        msg = "You have excluded both Explicit and Implicit components."

        with assertRaisesRegex(self, RuntimeError, msg):
            prob.model.list_outputs(explicit=False, implicit=False)

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

        # list inputs
        # Cannot do exact equality here because the units cause comp.y to be slightly different than 12.0
        tol = 1e-7
        inputs = prob.model.list_inputs(out_stream=None)
        for actual, expected in zip(sorted(inputs),
                                    [
                                        ('comp.x', {'value': [12.]}),
                                        ('comp.y', {'value': [12.]}),
                                    ]
                                    ):
            self.assertEqual(actual[0], expected[0])
            assert_rel_error(self, actual[1], expected[1], tol)

        # Only other option for list_inputs is units
        inputs = prob.model.list_inputs(values=False, units=True, out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('comp.x', {'units': 'inch'}),
            ('comp.y', {'units': 'inch'}),
        ])

        ###### list_outputs tests #####

        # list outputs for implicit comps - should get none
        outputs = prob.model.list_outputs(implicit=True, explicit=False, out_stream=None)
        self.assertEqual(outputs, [])

        # list explicit outputs with values
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', { 'value': np.array([24.] ) } ),
            ('p1.x', { 'value': np.array([12.] ) } ),
            ('p2.y', { 'value': np.array([1.] ) } ),
        ])

        # list explicit outputs with residuals
        outputs = prob.model.list_outputs(implicit=False, values=False,
                                          residuals=True, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', { 'resids': np.array([0.] ) } ),
            ('p1.x', { 'resids': np.array([0.] ) } ),
            ('p2.y', { 'resids': np.array([0.] ) } ),
        ])

        # list explicit outputs with units
        outputs = prob.model.list_outputs(implicit=False, values=False, units=True, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', { 'units': 'inch' } ),
            ('p1.x', { 'units': 'inch' } ),
            ('p2.y', { 'units': 'ft' } ),
        ])

        # list explicit outputs with shape
        outputs = prob.model.list_outputs(implicit=False, values=False, shape=True, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', { 'shape': (1,) } ),
            ('p1.x', { 'shape': (1,) } ),
            ('p2.y', { 'shape': (1,) } ),
        ])

        # list explicit outputs with bounds
        outputs = prob.model.list_outputs(implicit=False, values=False, bounds=True, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', { 'lower': None, 'upper': None } ),
            ('p1.x', { 'lower': [1.0], 'upper': [100.0] } ),
            ('p2.y', { 'lower': [2.0], 'upper': [200.0] } ),
        ])

        # list explicit outputs with scaling
        outputs = prob.model.list_outputs(implicit=False, scaling=True, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp.z', { 'value': [24.], 'ref': 1.0, 'ref0': 0.0, 'res_ref': 1.0 } ),
            ('p1.x', { 'value': [12.], 'ref': 1.1, 'ref0': 2.1, 'res_ref': 1.1 } ),
            ('p2.y', { 'value': [1.], 'ref': 1.2, 'ref0': 0.0, 'res_ref': 2.2 } ),
        ])

        # logging inputs
        stream = cStringIO()
        prob.model.list_inputs(values=True, units=True, out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('top'), 1)
        self.assertEqual(text.count('    y'), 1)

        # logging outputs
        stream = cStringIO()
        prob.model.list_outputs(values=True, units=True, out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('top'), 1)
        self.assertEqual(text.count('    z'), 1)



    def test_array_list_vars_options(self):

        class ArrayAdder(ExplicitComponent):
            """
            Just a simple component that has array inputs and outputs
            """

            def __init__(self, size):
                super(ArrayAdder, self).__init__()
                self.size = size

            def setup(self):
                self.add_input('x', val=np.zeros(self.size), units='inch')
                self.add_output('y', val=np.zeros(self.size), units='ft')

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x'] + 10.0

        size = 100 #how many items in the array


        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('des_vars', IndepVarComp('x', np.ones(size), units='inch'), promotes=['x'])
        prob.model.add_subsystem('mult', ArrayAdder(size), promotes=['x', 'y'])

        prob.setup(check=False)

        prob['x'] = np.ones(size)

        prob.run_driver()


        ###### list_outputs tests #####

        # list outputs for implicit comps - should get none
        outputs = prob.model.list_outputs(implicit=True, explicit=False, out_stream=None)
        self.assertEqual(outputs, [])

        # list explicit outputs with values
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        tol = 1e-7
        for actual, expected in zip(sorted(outputs),
                                    [
                                        ('des_vars.x', {'value': np.ones(size)}),
                                        ('mult.y', {'value': np.ones(size) * 11.0}),
                                    ]
                                    ):
            self.assertEqual(actual[0], expected[0])
            assert_rel_error(self, actual[1], expected[1], tol)


        # logging inputs
        stream = cStringIO()

        # np.set_printoptions( linewidth=20, threshold=10)

        # np.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None,
        #                        nanstr=None, infstr=None, formatter=None
        prob.model.list_inputs(values=True, units=True, out_stream=stream, print_arrays=True)
        text = stream.getvalue()
        # self.assertEqual(text.count('  des_vars'), 1)
        # self.assertEqual(text.count('    x'), 1)

        # logging outputs
        stream = cStringIO()
        prob.model.list_outputs(values=True, units=True, hierarchical=False, print_arrays=True, out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('des_vars.x'), 1)
        self.assertEqual(text.count('mult.y'), 1)
        self.assertEqual(text.count('value'), 3)
        self.assertEqual(text.count('units'), 1)


    def test_hierarchy_list_vars_options(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = NewtonSolver()
        g1.linear_solver = LinearBlockGS()

        g2.nonlinear_solver = NewtonSolver()
        g2.linear_solver = ScipyKrylov()
        g2.linear_solver.precon = LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.setup(check=False)
        prob.run_driver()

        # logging outputs

        # Not hierarchical
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          residuals=True,
                                          scaling=True,
                                          hierarchical=False,
                                          out_stream=stream)
        text = stream.getvalue()
        print(text)
        self.assertEqual(text.count('g2.d1.y1'), 1)
        self.assertEqual(text.count('g2.d2.y2'), 1)
        self.assertEqual(text.count('pz.z'), 1)
        self.assertEqual(text.count('sub1.sub2.g1.d1.y1'), 1)
        self.assertEqual(text.count('sub1.sub2.g1.d2.y2'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 11)

        # Hierarchical
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          residuals=True,
                                          scaling=True,
                                          hierarchical=True,
                                          out_stream=stream)
        text = stream.getvalue()
        print(text)
        self.assertEqual(text.count('top'), 1)
        self.assertEqual(text.count('          y1'), 1)
        self.assertEqual(text.count('  g2'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 21)

        # Not hierarchical with printing arrays
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          residuals=True,
                                          scaling=True,
                                          print_arrays=True,
                                          hierarchical=False,
                                          out_stream=stream)
        text = stream.getvalue()
        print(text)
        #######self.assertEqual(text.count('top'), 1)
        # self.assertEqual(text.count('          y1'), 1)
        # self.assertEqual(text.count('  g2'), 1)
        # num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 21)

        # Hierarchical with printing arrays
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          residuals=True,
                                          scaling=True,
                                          print_arrays=True,
                                          hierarchical=True,
                                          out_stream=stream)
        text = stream.getvalue()
        print(text)
        self.assertEqual(text.count('top'), 1)
        self.assertEqual(text.count('          y1'), 1)
        self.assertEqual(text.count('  g2'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 25)



if __name__ == '__main__':
    unittest.main()

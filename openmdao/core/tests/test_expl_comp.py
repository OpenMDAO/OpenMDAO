"""Simple example demonstrating how to implement an explicit component."""
from __future__ import division

from six import assertRaisesRegex

from six.moves import cStringIO
import unittest

import numpy as np

from openmdao.api import Problem, ExplicitComponent, NewtonSolver, ScipyKrylov, Group, \
    IndepVarComp, LinearBlockGS
from openmdao.utils.assert_utils import assert_rel_error
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

        # list_inputs tests
        # Cannot do exact equality here because the units cause comp.y to be slightly different than 12.0
        stream = cStringIO()
        inputs = prob.model.list_inputs(units=True, out_stream=stream)
        tol = 1e-7
        for actual, expected in zip(sorted(inputs),
                                    [
                                        ('comp.x', {'value': [12.], 'units':'inch'}),
                                        ('comp.y', {'value': [12.], 'units':'inch'}),
                                    ]
                                    ):
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(expected[1]['units'], actual[1]['units'])
            assert_rel_error(self, expected[1]['value'], actual[1]['value'], tol)
        text = stream.getvalue()
        self.assertEqual(1, text.count("Input(s) in 'model'"))
        self.assertEqual(1, text.count('varname'))
        self.assertEqual(1, text.count('value'))
        self.assertEqual(1, text.count('top'))
        self.assertEqual(1, text.count('  comp'))
        self.assertEqual(1, text.count('    x'))
        self.assertEqual(1, text.count('    y'))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(8,num_non_empty_lines)

        # list_outputs tests

        # list outputs for implicit comps - should get none
        outputs = prob.model.list_outputs(implicit=True, explicit=False, out_stream=None)
        self.assertEqual(outputs, [])

        # list outputs with out_stream - just check to see if it was logged to
        stream = cStringIO()
        outputs = prob.model.list_outputs(out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(1, text.count('Explicit Output'))
        self.assertEqual(1, text.count('Implicit Output'))

        # list outputs with out_stream and all the optional display values True
        stream = cStringIO()
        outputs = prob.model.list_outputs(values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          residuals=True,
                                          scaling=True,
                                          hierarchical=False,
                                          print_arrays=False,
                                          out_stream=stream)

        self.assertEqual([
            ('comp.z', {'value': [24.], 'resids': [0.], 'units': 'inch', 'shape': (1,),
                        'lower': None, 'upper': None, 'ref': 1.0, 'ref0': 0.0, 'res_ref': 1.0}),
            ('p1.x', {'value': [12.], 'resids': [0.], 'units': 'inch', 'shape': (1,),
                      'lower': [1.], 'upper': [100.], 'ref': 1.1, 'ref0': 2.1, 'res_ref': 1.1}),
            ('p2.y', {'value': [1.], 'resids': [0.], 'units': 'ft', 'shape': (1,),
                      'lower': [2.], 'upper': [200.], 'ref': 1.2, 'ref0': 0.0, 'res_ref': 2.2}),
                         ],
            sorted(outputs))

        text = stream.getvalue()
        self.assertEqual(1, text.count('varname'))
        self.assertEqual(1, text.count('value'))
        self.assertEqual(1, text.count('resids'))
        self.assertEqual(1, text.count('units'))
        self.assertEqual(1, text.count('shape'))
        self.assertEqual(1, text.count('lower'))
        self.assertEqual(1, text.count('upper'))
        self.assertEqual(3, text.count('ref'))
        self.assertEqual(1, text.count('ref0'))
        self.assertEqual(1, text.count('res_ref'))
        self.assertEqual(1, text.count('p1.x'))
        self.assertEqual(1, text.count('p2.y'))
        self.assertEqual(1, text.count('comp.z'))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(9,num_non_empty_lines)

    def test_for_feature_docs_list_vars_options(self):

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
                                             x={'value': 0.0, 'units': 'inch'},
                                             y={'value': 0.0, 'units': 'inch'},
                                             z={'value': 0.0, 'units': 'inch'}))
        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        inputs = prob.model.list_inputs(units=True)
        print(inputs)

        outputs = prob.model.list_outputs(implicit=False,
                                          values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          residuals=True,
                                          scaling=True,
                                          hierarchical=False,
                                          print_arrays=False)

        self.assertEqual(sorted(outputs), [
            ('comp.z', {'value': [ 24.], 'resids': [ 0.], 'units': 'inch', 'shape': (1,),
                        'lower': None, 'upper': None, 'ref': 1.0, 'ref0': 0.0, 'res_ref': 1.0} ),
            ('p1.x', {'value': [ 12.], 'resids': [ 0.], 'units': 'inch', 'shape': (1,),
                      'lower': [ 1.], 'upper': [ 100.], 'ref': 1.1, 'ref0': 2.1, 'res_ref': 1.1} ),
            ('p2.y', {'value': [ 1.], 'resids': [ 0.], 'units': 'ft', 'shape': (1,),
                      'lower': [ 2.], 'upper': [ 200.], 'ref': 1.2, 'ref0': 0.0, 'res_ref': 2.2}),
            ]
            )

        outputs = prob.model.list_outputs(implicit=False,
                                          values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          residuals=True,
                                          scaling=True,
                                          hierarchical=True,
                                          print_arrays=False)

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

        # logging inputs
        # out_stream - not hierarchical - extras - no print_arrays
        stream = cStringIO()
        prob.model.list_inputs(values=True,
                               units=True,
                               hierarchical=False,
                               print_arrays=False,
                               out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(1, text.count("10 Input(s) in 'model'"))
        # make sure they are in the correct order
        self.assertTrue(text.find("sub1.sub2.g1.d1.z") <
                        text.find('sub1.sub2.g1.d1.x') <
                        text.find('sub1.sub2.g1.d1.y2') <
                        text.find('sub1.sub2.g1.d2.z') <
                        text.find('sub1.sub2.g1.d2.y1') <
                        text.find('g2.d1.z') <
                        text.find('g2.d1.x') <
                        text.find('g2.d1.y2') <
                        text.find('g2.d2.z') <
                        text.find('g2.d2.y1'))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(14, num_non_empty_lines)

        # out_stream - hierarchical - extras - no print_arrays
        stream = cStringIO()
        prob.model.list_inputs(values=True,
                               units=True,
                               hierarchical=True,
                               print_arrays=False,
                               out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(1, text.count("10 Input(s) in 'model'"))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(23, num_non_empty_lines)
        self.assertEqual(1, text.count('top'))
        self.assertEqual(1, text.count('  sub1'))
        self.assertEqual(1, text.count('    sub2'))
        self.assertEqual(1, text.count('      g1'))
        self.assertEqual(1, text.count('        d1'))
        self.assertEqual(2, text.count('          z'))

        # logging outputs
        # out_stream - not hierarchical - extras - no print_arrays
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                units=True,
                                shape=True,
                                bounds=True,
                                residuals=True,
                                scaling=True,
                                hierarchical=False,
                                print_arrays=False,
                                out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('5 Explicit Output'), 1)
        # make sure they are in the correct order
        self.assertTrue(text.find("pz.z") < text.find('sub1.sub2.g1.d1.y1') < text.find('sub1.sub2.g1.d2.y2') < \
                        text.find('g2.d1.y1') < text.find('g2.d2.y2'))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(11, num_non_empty_lines)

        # Hierarchical
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                units=True,
                                shape=True,
                                bounds=True,
                                residuals=True,
                                scaling=True,
                                hierarchical=True,
                                print_arrays=False,
                                out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('top'), 1)
        self.assertEqual(text.count('          y1'), 1)
        self.assertEqual(text.count('  g2'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 21)

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

        size = 100 # how many items in the array

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('des_vars', IndepVarComp('x', np.ones(size), units='inch'), promotes=['x'])
        prob.model.add_subsystem('mult', ArrayAdder(size), promotes=['x', 'y'])

        prob.setup(check=False)

        prob['x'] = np.ones(size)

        prob.run_driver()

        # logging inputs
        # out_stream - not hierarchical - extras - no print_arrays
        stream = cStringIO()
        prob.model.list_inputs(values=True,
                               units=True,
                               hierarchical=False,
                               print_arrays=False,
                               out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(1, text.count("1 Input(s) in 'model'"))
        self.assertEqual(1, text.count('mult.x'))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(5, num_non_empty_lines)

        # out_stream - hierarchical - extras - no print_arrays
        stream = cStringIO()
        prob.model.list_inputs(values=True,
                               units=True,
                               hierarchical=True,
                               print_arrays=False,
                               out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(1, text.count("1 Input(s) in 'model'"))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(7, num_non_empty_lines)
        self.assertEqual(1, text.count('top'))
        self.assertEqual(1, text.count('  mult'))
        self.assertEqual(1, text.count('    x'))

        # logging outputs
        # out_stream - not hierarchical - extras - no print_arrays
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                units=True,
                                shape=True,
                                bounds=True,
                                residuals=True,
                                scaling=True,
                                hierarchical=False,
                                print_arrays=False,
                                out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('2 Explicit Output'), 1)
        # make sure they are in the correct order
        self.assertTrue(text.find("des_vars.x") < text.find('mult.y'))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(8, num_non_empty_lines)

        # Hierarchical - no print arrays
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                units=True,
                                shape=True,
                                bounds=True,
                                residuals=True,
                                scaling=True,
                                hierarchical=True,
                                print_arrays=False,
                                out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('top'), 1)
        self.assertEqual(text.count('  des_vars'), 1)
        self.assertEqual(text.count('    x'), 1)
        self.assertEqual(text.count('  mult'), 1)
        self.assertEqual(text.count('    y'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 11)

        # Need to explicitly set this to make sure all ways of running this test
        #   result in the same format of the output. When running this test from the
        #   top level via testflo, the format comes out different than if the test is
        #   run individually
        from distutils.version import LooseVersion
        # formatting has changed in numpy 1.14 and beyond.
        if LooseVersion(np.__version__) >= LooseVersion("1.14"):
            np.set_printoptions(edgeitems=3, infstr='inf',
                                linewidth=75, nanstr='nan', precision=8,
                                suppress=False, threshold=1000, formatter=None, legacy="1.13")
        else:
            np.set_printoptions(edgeitems=3, infstr='inf',
                                linewidth=75, nanstr='nan', precision=8,
                                suppress=False, threshold=1000, formatter=None)
        # logging outputs
        # out_stream - not hierarchical - extras - print_arrays
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                units=True,
                                shape=True,
                                bounds=True,
                                residuals=True,
                                scaling=True,
                                hierarchical=False,
                                print_arrays=True,
                                out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('2 Explicit Output'), 1)
        self.assertEqual(text.count('value:'), 2)
        self.assertEqual(text.count('resids:'), 2)
        self.assertEqual(text.count('['), 4)
        # make sure they are in the correct order
        self.assertTrue(text.find("des_vars.x") < text.find('mult.y'))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(37, num_non_empty_lines)

        # Hierarchical
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                units=True,
                                shape=True,
                                bounds=True,
                                residuals=True,
                                scaling=True,
                                hierarchical=True,
                                print_arrays=True,
                                out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('2 Explicit Output'), 1)
        self.assertEqual(text.count('value:'), 2)
        self.assertEqual(text.count('resids:'), 2)
        self.assertEqual(text.count('['), 4)
        self.assertEqual(text.count('top'), 1)
        self.assertEqual(text.count('  des_vars'), 1)
        self.assertEqual(text.count('    x'), 1)
        self.assertEqual(text.count('  mult'), 1)
        self.assertEqual(text.count('    y'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 40)

    def test_for_docs_array_list_vars_options(self):

        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

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

        size = 30

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('des_vars', IndepVarComp('x', np.ones(size), units='inch'),
                                 promotes=['x'])
        prob.model.add_subsystem('mult', ArrayAdder(size), promotes=['x', 'y'])

        prob.setup(check=False)
        prob['x'] = np.arange(size)
        prob.run_driver()

        prob.model.list_inputs(values=True,
                               units=True,
                               hierarchical=True,
                               print_arrays=True)

        np.set_printoptions(edgeitems=3, infstr='inf',
                            linewidth = 75, nanstr = 'nan', precision = 8,
                            suppress = False, threshold = 1000, formatter = None)

        prob.model.list_outputs(values=True,
                                implicit=False,
                                units=True,
                                shape=True,
                                bounds=True,
                                residuals=True,
                                scaling=True,
                                hierarchical=False,
                                print_arrays=True)

        prob.model.list_outputs(values=True,
                                implicit=False,
                                units=True,
                                shape=True,
                                bounds=True,
                                residuals=True,
                                scaling=True,
                                hierarchical=True,
                                print_arrays=True)


if __name__ == '__main__':
    unittest.main()

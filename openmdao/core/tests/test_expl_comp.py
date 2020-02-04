"""Simple example demonstrating how to implement an explicit component."""
from __future__ import division

from six import assertRaisesRegex

from six.moves import cStringIO
import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.double_sellar import SubSellar
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimple, \
    TestExplCompSimpleDense
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.general_utils import printoptions


# Note: The following class definitions are used in feature docs

class RectangleComp(om.ExplicitComponent):
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


class RectangleCompWithTags(om.ExplicitComponent):
    """
    A simple Explicit Component that also has input and output with tags.
    """

    def setup(self):
        self.add_input('length', val=1., tags=["tag1"])
        self.add_input('width', val=1., tags=["tag2"])
        self.add_output('area', val=1., tags=["tag1"])

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


class RectangleGroup(om.Group):

    def setup(self):
        comp1 = self.add_subsystem('comp1', om.IndepVarComp())
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
        prob = om.Problem(RectangleComp())
        prob.setup()
        prob.run_model()

    def test_feature_simple(self):
        import openmdao.api as om
        from openmdao.core.tests.test_expl_comp import RectangleComp

        prob = om.Problem(RectangleComp())
        prob.setup()
        prob.run_model()

    def test_compute_and_list(self):
        prob = om.Problem(RectangleGroup())
        prob.setup()

        msg = "RectangleGroup (<model>): Unable to list inputs on a Group until model has been run."
        try:
            prob.model.list_inputs()
        except Exception as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail("Exception expected")

        msg = "RectangleGroup (<model>): Unable to list outputs on a Group until model has been run."
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
            ('comp2.length', {'value': [3.]}),
            ('comp2.width',  {'value': [2.]}),
            ('comp3.length', {'value': [3.]}),
            ('comp3.width',  {'value': [2.]}),
        ])

        # list explicit outputs
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp1.length', {'value': [3.]}),
            ('comp1.width',  {'value': [2.]}),
            ('comp2.area',   {'value': [6.]}),
            ('comp3.area',   {'value': [6.]}),
        ])

        # list states
        states = prob.model.list_outputs(explicit=False, out_stream=None)
        self.assertEqual(states, [])

        # list excluding both explicit and implicit components raises error
        msg = "You have excluded both Explicit and Implicit components."

        with assertRaisesRegex(self, RuntimeError, msg):
            prob.model.list_outputs(explicit=False, implicit=False)

    def test_simple_list_vars_options(self):

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 12.0,
                                                  lower=1.0, upper=100.0,
                                                  ref=1.1, ref0=2.1,
                                                  units='inch',
                                                  desc='indep x'))
        model.add_subsystem('p2', om.IndepVarComp('y', 1.0,
                                                  lower=2.0, upper=200.0,
                                                  ref=1.2, res_ref=2.2,
                                                  units='ft',
                                                  desc='indep y'))
        model.add_subsystem('comp', om.ExecComp('z=x+y',
                                                x={'value': 0.0, 'units': 'inch'},
                                                y={'value': 0.0, 'units': 'inch'},
                                                z={'value': 0.0, 'units': 'inch'}))
        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()

        # list outputs before model has been run will raise an exception
        msg = "Group (<model>): Unable to list outputs on a Group until model has been run."
        try:
            prob.model.list_outputs()
        except Exception as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail("Exception expected")

        # list_inputs on a component before run is okay, using relative names
        expl_inputs = prob.model.comp.list_inputs(out_stream=None)
        expected = {
            'x': {'value': 0.},
            'y': {'value': 0.}
        }
        self.assertEqual(dict(expl_inputs), expected)

        expl_inputs = prob.model.comp.list_inputs(includes='x', out_stream=None)
        self.assertEqual(dict(expl_inputs), {'x': {'value': 0.}})

        expl_inputs = prob.model.comp.list_inputs(excludes='x', out_stream=None)
        self.assertEqual(dict(expl_inputs), {'y': {'value': 0.}})

        # specifying prom_name should not cause an error
        expl_inputs = prob.model.comp.list_inputs(prom_name=True, out_stream=None)
        self.assertEqual(dict(expl_inputs), {
            'x': {'value': 0., 'prom_name': 'x'},
            'y': {'value': 0., 'prom_name': 'y'},
        })

        # list_outputs on a component before run is okay, using relative names
        expl_outputs = prob.model.p1.list_outputs(out_stream=None)
        expected = {
            'x': {'value': 12.}
        }
        self.assertEqual(dict(expl_outputs), expected)

        expl_outputs = prob.model.p1.list_outputs(includes='x', out_stream=None)
        self.assertEqual(dict(expl_outputs), expected)

        expl_outputs = prob.model.p1.list_outputs(excludes='x', out_stream=None)
        self.assertEqual(dict(expl_outputs), {})

        # specifying residuals_tol should not cause an error
        expl_outputs = prob.model.p1.list_outputs(residuals_tol=.01, out_stream=None)
        self.assertEqual(dict(expl_outputs), expected)

        # specifying prom_name should not cause an error
        expl_outputs = prob.model.p1.list_outputs(prom_name=True, out_stream=None)
        self.assertEqual(dict(expl_outputs), {
            'x': {'value': 12., 'prom_name': 'x'}
        })

        # run model
        prob.set_solver_print(level=0)
        prob.run_model()

        # list_inputs tests
        # Can't do exact equality here because units cause comp.y to be slightly different than 12.0
        stream = cStringIO()
        inputs = prob.model.list_inputs(units=True, shape=True, out_stream=stream)
        tol = 1e-7
        for actual, expected in zip(sorted(inputs), [
            ('comp.x', {'value': [12.], 'shape': (1,), 'units': 'inch'}),
            ('comp.y', {'value': [12.], 'shape': (1,), 'units': 'inch'})
        ]):
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(expected[1]['units'], actual[1]['units'])
            self.assertEqual(expected[1]['shape'], actual[1]['shape'])
            assert_rel_error(self, expected[1]['value'], actual[1]['value'], tol)

        text = stream.getvalue()

        self.assertEqual(1, text.count("Input(s) in 'model'"))
        self.assertEqual(1, text.count('varname'))
        self.assertEqual(1, text.count('value'))
        self.assertEqual(1, text.count('shape'))
        self.assertEqual(1, text.count('top'))
        self.assertEqual(1, text.count('  comp'))
        self.assertEqual(1, text.count('    x'))
        self.assertEqual(1, text.count('    y'))

        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(8, num_non_empty_lines)

        # list_outputs tests

        # list outputs for implicit comps - should get none
        outputs = prob.model.list_outputs(implicit=True, explicit=False, out_stream=None)
        self.assertEqual(outputs, [])

        # list outputs with out_stream and all the optional display values True
        stream = cStringIO()
        outputs = prob.model.list_outputs(values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          desc=True,
                                          residuals=True,
                                          scaling=True,
                                          hierarchical=False,
                                          print_arrays=False,
                                          out_stream=stream)

        self.assertEqual([
            ('comp.z', {'value': [24.], 'resids': [0.], 'units': 'inch', 'shape': (1,), 'desc': '',
                        'lower': None, 'upper': None, 'ref': 1.0, 'ref0': 0.0, 'res_ref': 1.0}),
            ('p1.x', {'value': [12.], 'resids': [0.], 'units': 'inch', 'shape': (1,), 'desc': 'indep x',
                      'lower': [1.], 'upper': [100.], 'ref': 1.1, 'ref0': 2.1, 'res_ref': 1.1}),
            ('p2.y', {'value': [1.], 'resids': [0.], 'units': 'ft', 'shape': (1,), 'desc': 'indep y',
                      'lower': [2.], 'upper': [200.], 'ref': 1.2, 'ref0': 0.0, 'res_ref': 2.2}),
        ], sorted(outputs))

        text = stream.getvalue()
        self.assertEqual(1, text.count('Explicit Output'))
        self.assertEqual(1, text.count('Implicit Output'))
        self.assertEqual(1, text.count('varname'))
        self.assertEqual(1, text.count('value'))
        self.assertEqual(1, text.count('resids'))
        self.assertEqual(1, text.count('units'))
        self.assertEqual(1, text.count('shape'))
        self.assertEqual(1, text.count('lower'))
        self.assertEqual(1, text.count('upper'))
        self.assertEqual(1, text.count('desc'))
        self.assertEqual(3, text.count('ref'))
        self.assertEqual(1, text.count('ref0'))
        self.assertEqual(1, text.count('res_ref'))
        self.assertEqual(1, text.count('p1.x'))
        self.assertEqual(1, text.count('p2.y'))
        self.assertEqual(1, text.count('comp.z'))

        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(9, num_non_empty_lines)

    def test_for_feature_docs_list_vars_options(self):

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 12.0,
                                                  lower=1.0, upper=100.0,
                                                  ref=1.1, ref0=2.1,
                                                  units='inch',
                                                  ))
        model.add_subsystem('p2', om.IndepVarComp('y', 1.0,
                                                  lower=2.0, upper=200.0,
                                                  ref=1.2, res_ref=2.2,
                                                  units='ft',
                                                  ))
        model.add_subsystem('comp', om.ExecComp('z=x+y',
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
            ('comp.z', {'value': [24.], 'resids': [0.], 'units': 'inch', 'shape': (1,),
                        'lower': None, 'upper': None, 'ref': 1.0, 'ref0': 0.0, 'res_ref': 1.0}),
            ('p1.x', {'value': [12.], 'resids': [0.], 'units': 'inch', 'shape': (1,),
                      'lower': [1.], 'upper': [100.], 'ref': 1.1, 'ref0': 2.1, 'res_ref': 1.1}),
            ('p2.y', {'value': [1.], 'resids': [0.], 'units': 'ft', 'shape': (1,),
                      'lower': [2.], 'upper': [200.], 'ref': 1.2, 'ref0': 0.0, 'res_ref': 2.2}),
        ])

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

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = sub1.add_subsystem('sub2', om.Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = om.NewtonSolver()
        g1.linear_solver = om.LinearBlockGS()

        g2.nonlinear_solver = om.NewtonSolver()
        g2.linear_solver = om.ScipyKrylov()
        g2.linear_solver.precon = om.LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.setup()
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
        self.assertTrue(text.find("pz.z") < text.find('sub1.sub2.g1.d1.y1') <
                        text.find('sub1.sub2.g1.d2.y2') <
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

        class ArrayAdder(om.ExplicitComponent):
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

        size = 100  # how many items in the array

        prob = om.Problem()

        prob.model.add_subsystem('des_vars', om.IndepVarComp('x', np.ones(size), units='inch'),
                                 promotes=['x'])
        prob.model.add_subsystem('mult', ArrayAdder(size), promotes=['x', 'y'])

        prob.setup()

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

        # Promoted names - no print arrays
        stream = cStringIO()
        prob.model.list_outputs(values=True,
                                prom_name=True,
                                print_arrays=False,
                                out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('    x       |10.0|   x'), 1)
        self.assertEqual(text.count('    y       |110.0|  y'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 11)

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
        opts = {
            'edgeitems': 3,
            'infstr': 'inf',
            'linewidth': 75,
            'nanstr': 'nan',
            'precision': 8,
            'suppress': False,
            'threshold': 1000,
        }

        from distutils.version import LooseVersion
        if LooseVersion(np.__version__) >= LooseVersion("1.14"):
            opts['legacy'] = '1.13'

        with printoptions(**opts):
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
            self.assertEqual(46, num_non_empty_lines)

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
            self.assertEqual(num_non_empty_lines, 49)

    def test_for_docs_array_list_vars_options(self):

        import numpy as np

        import openmdao.api as om
        from openmdao.utils.general_utils import printoptions

        class ArrayAdder(om.ExplicitComponent):
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

        prob = om.Problem()
        prob.model.add_subsystem('des_vars', om.IndepVarComp('x', np.ones(size), units='inch'),
                                 promotes=['x'])
        prob.model.add_subsystem('mult', ArrayAdder(size), promotes=['x', 'y'])

        prob.setup()
        prob['x'] = np.arange(size)
        prob.run_driver()

        prob.model.list_inputs(values=True,
                               units=True,
                               hierarchical=True,
                               print_arrays=True)

        with printoptions(edgeitems=3, infstr='inf',
                          linewidth=75, nanstr='nan', precision=8,
                          suppress=False, threshold=1000, formatter=None):

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

    def test_simple_var_tags(self):
        prob = om.Problem(RectangleCompWithTags())
        prob.setup(check=False)
        prob.run_model()

        # Inputs no tags
        inputs = prob.model.list_inputs(out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('length', {'value': [1.]}),
            ('width', {'value': [1.]}),
        ])

        # Inputs with tags
        inputs = prob.model.list_inputs(out_stream=None, tags="tag1")
        self.assertEqual(sorted(inputs), [
            ('length', {'value': [1.]}),
        ])

        # Inputs with multiple tags
        inputs = prob.model.list_inputs(out_stream=None, tags=["tag1", "tag3"])
        self.assertEqual(sorted(inputs), [
            ('length', {'value': [1.]}),
        ])
        inputs = prob.model.list_inputs(out_stream=None, tags=["tag1", "tag2"])
        self.assertEqual(sorted(inputs), [
            ('length', {'value': [1.]}),
            ('width', {'value': [1.]}),
        ])

        # Inputs with tag that does not match
        inputs = prob.model.list_inputs(out_stream=None, tags="tag3")
        self.assertEqual(sorted(inputs), [])

        # Outputs no tags
        outputs = prob.model.list_outputs(out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('area', {'value': [1.]}),
        ])

        # Outputs with tags
        outputs = prob.model.list_outputs(out_stream=None, tags="tag1")
        self.assertEqual(sorted(outputs), [
            ('area', {'value': [1.]}),
        ])

        # Outputs with multiple tags
        outputs = prob.model.list_outputs(out_stream=None, tags=["tag1", "tag3"])
        self.assertEqual(sorted(outputs), [
            ('area', {'value': [1.]}),
        ])

        # Outputs with tag that does not match
        outputs = prob.model.list_outputs(out_stream=None, tags="tag3")
        self.assertEqual(sorted(outputs), [])

    def test_feature_simple_var_tags(self):
        from openmdao.api import Problem, ExplicitComponent

        class RectangleCompWithTags(ExplicitComponent):
            """
            A simple Explicit Component that also has input and output with tags.
            """

            def setup(self):
                self.add_input('length', val=1., tags=["tag1", "tag2"])
                self.add_input('width', val=1., tags=["tag2"])
                self.add_output('area', val=1., tags="tag1")

                self.declare_partials('*', '*')

            def compute(self, inputs, outputs):
                outputs['area'] = inputs['length'] * inputs['width']

        prob = Problem(RectangleCompWithTags())
        prob.setup(check=False)
        prob.run_model()

        # Inputs no tags
        inputs = prob.model.list_inputs(values=False, out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('length', {}),
            ('width', {}),
        ])

        # Inputs with tags
        inputs = prob.model.list_inputs(values=False, out_stream=None, tags="tag1")
        self.assertEqual(sorted(inputs), [
            ('length', {}),
        ])

        # Inputs with multiple tags
        inputs = prob.model.list_inputs(values=False, out_stream=None, tags=["tag1", "tag2"])
        self.assertEqual(sorted(inputs), [
            ('length', {}),
            ('width', {}),
        ])

        # Inputs with tag that does not match
        inputs = prob.model.list_inputs(values=False, out_stream=None, tags="tag3")
        self.assertEqual(sorted(inputs), [])

        # Outputs no tags
        outputs = prob.model.list_outputs(values=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('area', {}),
        ])

        # Outputs with tags
        outputs = prob.model.list_outputs(values=False, out_stream=None, tags="tag1")
        self.assertEqual(sorted(outputs), [
            ('area', {}),
        ])

        # Outputs with multiple tags
        outputs = prob.model.list_outputs(values=False, out_stream=None, tags=["tag1", "tag3"])
        self.assertEqual(sorted(outputs), [
            ('area', {}),
        ])

        # Outputs with tag that does not match
        outputs = prob.model.list_outputs(values=False, out_stream=None, tags="tag3")
        self.assertEqual(sorted(outputs), [])

    def test_compute_inputs_read_only(self):
        class BadComp(TestExplCompSimple):
            def compute(self, inputs, outputs):
                super(BadComp, self).compute(inputs, outputs)
                inputs['length'] = 0.  # should not be allowed

        prob = om.Problem(BadComp())
        prob.setup()

        with self.assertRaises(ValueError) as cm:
            prob.run_model()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'length' in input vector "
                         "when it is read only.")

    def test_compute_inputs_read_only_reset(self):
        class BadComp(TestExplCompSimple):
            def compute(self, inputs, outputs):
                super(BadComp, self).compute(inputs, outputs)
                raise om.AnalysisError("It's just a scratch.")

        prob = om.Problem(BadComp())
        prob.setup()
        with self.assertRaises(om.AnalysisError):
            prob.run_model()

        # verify read_only status is reset after AnalysisError
        prob['length'] = 111.

    def test_compute_partials_inputs_read_only(self):
        class BadComp(TestExplCompSimpleDense):
            def compute_partials(self, inputs, partials):
                super(BadComp, self).compute_partials(inputs, partials)
                inputs['length'] = 0.  # should not be allowed

        prob = om.Problem(BadComp())
        prob.setup()
        prob.run_model()

        with self.assertRaises(ValueError) as cm:
            prob.check_partials()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'length' in input vector "
                         "when it is read only.")

    def test_compute_partials_inputs_read_only_reset(self):
        class BadComp(TestExplCompSimpleDense):
            def compute_partials(self, inputs, partials):
                super(BadComp, self).compute_partials(inputs, partials)
                raise om.AnalysisError("It's just a scratch.")

        prob = om.Problem(BadComp())
        prob.setup()
        prob.run_model()

        with self.assertRaises(om.AnalysisError):
            prob.check_partials()

        # verify read_only status is reset after AnalysisError
        prob['length'] = 111.

    def test_compute_jacvec_product_inputs_read_only(self):
        class BadComp(RectangleJacVec):
            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                super(BadComp, self).compute_jacvec_product(inputs, d_inputs, d_outputs, mode)
                inputs['length'] = 0.  # should not be allowed

        prob = om.Problem(BadComp())
        prob.setup()
        prob.run_model()

        with self.assertRaises(ValueError) as cm:
            prob.check_partials()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'length' in input vector "
                         "when it is read only.")

    def test_compute_jacvec_product_inputs_read_only_reset(self):
        class BadComp(RectangleJacVec):
            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                super(BadComp, self).compute_jacvec_product(inputs, d_inputs, d_outputs, mode)
                raise om.AnalysisError("It's just a scratch.")

        prob = om.Problem(BadComp())
        prob.setup()
        prob.run_model()

        with self.assertRaises(om.AnalysisError):
            prob.check_partials()

        # verify read_only status is reset after AnalysisError
        prob['length'] = 111.


if __name__ == '__main__':
    unittest.main()

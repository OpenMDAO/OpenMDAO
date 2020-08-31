"""Simple example demonstrating how to implement an explicit component."""

from io import StringIO
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import printoptions, remove_whitespace
from openmdao.utils.mpi import MPI
from openmdao.test_suite.components.double_sellar import SubSellar
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimple, \
    TestExplCompSimpleDense
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
     SellarDis2withDerivatives

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
        self.add_subsystem('comp1', RectanglePartial(), promotes_inputs=['width', 'length'])
        self.add_subsystem('comp2', RectangleJacVec(), promotes_inputs=['width', 'length'])


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

        # list explicit outputs
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        expected = {
            'comp1.area': {'value': np.array([1.])},
            'comp2.area': {'value': np.array([1.])}
        }
        self.assertEqual(dict(outputs), expected)

        # list states
        states = prob.model.list_outputs(explicit=False, out_stream=None)
        self.assertEqual(states, [])

        prob.set_val('length', 3.)
        prob.set_val('width', 2.)
        prob.run_model()
        assert_near_equal(prob['comp1.area'], 6.)
        assert_near_equal(prob['comp2.area'], 6.)

        # total derivs
        total_derivs = prob.compute_totals(
            wrt=['length', 'width'],
            of=['comp1.area', 'comp2.area']
        )
        assert_near_equal(total_derivs['comp1.area', 'length'], [[2.]])
        assert_near_equal(total_derivs['comp2.area', 'length'], [[2.]])
        assert_near_equal(total_derivs['comp1.area', 'width'], [[3.]])
        assert_near_equal(total_derivs['comp2.area', 'width'], [[3.]])

        # list inputs
        inputs = prob.model.list_inputs(out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('comp1.length', {'value': [3.]}),
            ('comp1.width',  {'value': [2.]}),
            ('comp2.length', {'value': [3.]}),
            ('comp2.width',  {'value': [2.]}),
        ])

        # list explicit outputs
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp1.area',   {'value': [6.]}),
            ('comp2.area',   {'value': [6.]}),
        ])

        # list states
        states = prob.model.list_outputs(explicit=False, out_stream=None)
        self.assertEqual(states, [])

        # list excluding both explicit and implicit components raises error
        msg = "You have excluded both Explicit and Implicit components."

        with self.assertRaisesRegex(RuntimeError, msg):
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
        outputs = dict(prob.model.list_outputs(out_stream=None))
        expected = {
            'p1.x': {'value': 12.},
            'p2.y': {'value': 1.},
            'comp.z': {'value': 0.},
        }
        self.assertEqual(outputs, expected)

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
        stream = StringIO()
        expl_outputs = prob.model.p1.list_outputs(out_stream=stream)
        expected = {
            'x': {'value': 12.}
        }
        self.assertEqual(dict(expl_outputs), expected)

        text = stream.getvalue().split('\n')
        expected_text = [
            "1 Explicit Output(s) in 'p1'",
            "----------------------------",
            "",
            "varname  value",
            "-------  -----",
            "x        [12.]",
            "",
            "",
            "0 Implicit Output(s) in 'p1'",
            "----------------------------"
        ]
        for i, line in enumerate(expected_text):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

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
        stream = StringIO()
        inputs = prob.model.list_inputs(units=True, shape=True, out_stream=stream)
        tol = 1e-7
        for actual, expected in zip(sorted(inputs), [
            ('comp.x', {'value': [12.], 'shape': (1,), 'units': 'inch'}),
            ('comp.y', {'value': [12.], 'shape': (1,), 'units': 'inch'})
        ]):
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(expected[1]['units'], actual[1]['units'])
            self.assertEqual(expected[1]['shape'], actual[1]['shape'])
            assert_near_equal(expected[1]['value'], actual[1]['value'], tol)

        text = stream.getvalue().split('\n')
        expected_text = [
            "2 Input(s) in 'model'",
            "---------------------",
            "",
            "varname  value  units  shape",
            "-------  -----  -----  -----",
            "comp",
            "  x    [12.]  inch   (1,)",
            "  y    [12.]  inch   (1,)"
        ]
        for i, line in enumerate(expected_text):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]).replace('1L', ''), remove_whitespace(line))

        # list_outputs tests

        # list outputs for implicit comps - should get none
        outputs = prob.model.list_outputs(implicit=True, explicit=False, out_stream=None)
        self.assertEqual(outputs, [])

        # list outputs with out_stream and all the optional display values True
        stream = StringIO()
        outputs = prob.model.list_outputs(values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          desc=True,
                                          residuals=True,
                                          scaling=True,
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

        text = stream.getvalue().split('\n')
        expected_text = [
            "3 Explicit Output(s) in 'model'",
            "-------------------------------",
            "",
            "varname  value  resids  units  shape  lower  upper   ref  ref0  res_ref  desc",
            "-------  -----  ------  -----  -----  -----  ------  ---  ----  -------  -------",
            "p1",
            "  x    [12.]  [0.]    inch   (1,)   [1.]   [100.]  1.1  2.1   1.1      indep x",
            "p2",
            "  y    [1.]   [0.]    ft     (1,)   [2.]   [200.]  1.2  0.0   2.2      indep y",
            "comp",
            "  z    [24.]  [0.]    inch   (1,)   None   None    1.0  0.0   1.0",
            "",
            "",
            "0 Implicit Output(s) in 'model'",
            "-------------------------------",
        ]
        for i, line in enumerate(expected_text):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]).replace('1L', ''), remove_whitespace(line))

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

        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.linear_solver = om.LinearBlockGS()

        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.linear_solver = om.ScipyKrylov()
        g2.linear_solver.precon = om.LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.setup()
        prob.run_driver()

        # logging inputs
        # out_stream - not hierarchical - extras - no print_arrays
        stream = StringIO()
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
        stream = StringIO()
        prob.model.list_inputs(values=True,
                               units=True,
                               hierarchical=True,
                               print_arrays=False,
                               out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(1, text.count("10 Input(s) in 'model'"))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(22, num_non_empty_lines)
        self.assertEqual(1, text.count('\nsub1'))
        self.assertEqual(1, text.count('\n  sub2'))
        self.assertEqual(1, text.count('\n    g1'))
        self.assertEqual(1, text.count('\n      d1'))
        self.assertEqual(2, text.count('\n        z'))

        # logging outputs
        # out_stream - not hierarchical - extras - no print_arrays
        stream = StringIO()
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
        stream = StringIO()
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
        self.assertEqual(text.count('\n        y1'), 1)
        self.assertEqual(text.count('\ng2'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 20)

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
        stream = StringIO()
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
        stream = StringIO()
        prob.model.list_inputs(values=True,
                               units=True,
                               hierarchical=True,
                               print_arrays=False,
                               out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(1, text.count("1 Input(s) in 'model'"))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(6, num_non_empty_lines)
        self.assertEqual(1, text.count('\nmult'))
        self.assertEqual(1, text.count('\n  x'))

        # logging outputs
        # out_stream - not hierarchical - extras - no print_arrays
        stream = StringIO()
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
        stream = StringIO()
        prob.model.list_outputs(values=True,
                                prom_name=True,
                                print_arrays=False,
                                out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('  x       |10.0|   x'), 1)
        self.assertEqual(text.count('  y       |110.0|  y'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 10)

        # Hierarchical - no print arrays
        stream = StringIO()
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
        self.assertEqual(text.count('\ndes_vars'), 1)
        self.assertEqual(text.count('\n  x'), 1)
        self.assertEqual(text.count('\nmult'), 1)
        self.assertEqual(text.count('\n  y'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 10)

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
            stream = StringIO()
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
            stream = StringIO()
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
            self.assertEqual(text.count('\ndes_vars'), 1)
            self.assertEqual(text.count('\n  x'), 1)
            self.assertEqual(text.count('\nmult'), 1)
            self.assertEqual(text.count('\n  y'), 1)
            num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
            self.assertEqual(num_non_empty_lines, 48)

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
                         "BadComp (<model>): Attempt to set value of 'length' in input vector when it is read only.")

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
                         "BadComp (<model>): Attempt to set value of 'length' in input vector "
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
                         "BadComp (<model>): Attempt to set value of 'length' in input vector "
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

    def test_iter_count(self):
        # Make sure we correctly count iters in both _apply_nonlinear and _solve_nonlinear
        class SellarMDF(om.Group):
            def setup(self):
                self.set_input_defaults('x', 1.0)
                self.set_input_defaults('z', np.array([5.0, 2.0]))

                cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
                cycle.add_subsystem('d1', SellarDis1withDerivatives(), promotes_inputs=['x', 'z', 'y2'],
                                    promotes_outputs=['y1'])
                cycle.add_subsystem('d2', SellarDis2withDerivatives(), promotes_inputs=['z', 'y1'],
                                    promotes_outputs=['y2'])

                cycle.linear_solver = om.ScipyKrylov()

                cycle.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)


        prob = om.Problem()
        prob.model = SellarMDF()

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()
        self.assertEqual(prob.model.cycle.d1.iter_count, 0)
        self.assertEqual(prob.model.cycle.d2.iter_count, 0)
        self.assertEqual(prob.model.cycle.d1.iter_count_apply, 10)
        self.assertEqual(prob.model.cycle.d2.iter_count_apply, 10)


@unittest.skipUnless(MPI, "MPI is required.")
class TestMPIExplComp(unittest.TestCase):
    N_PROCS = 3

    def test_list_inputs_outputs_with_parallel_comps(self):
        class TestComp(om.ExplicitComponent):

            def initialize(self):
                self.options['distributed'] = False

            def setup(self):
                self.add_input('x', shape=1)
                self.add_output('y', shape=1)
                self.declare_partials('y', 'x')

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x'] ** 2

            def compute_partials(self, inputs, J):
                J['y', 'x'] = 2 * inputs['x']

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 1.0))
        model.add_subsystem('p2', om.IndepVarComp('x', 1.0))

        parallel = model.add_subsystem('parallel', om.ParallelGroup())
        parallel.add_subsystem('c1', TestComp())
        parallel.add_subsystem('c2', TestComp())

        model.add_subsystem('c3', om.ExecComp(['y=3.0*x1+7.0*x2']))

        model.connect("parallel.c1.y", "c3.x1")
        model.connect("parallel.c2.y", "c3.x2")

        model.connect("p1.x", "parallel.c1.x")
        model.connect("p2.x", "parallel.c2.x")

        prob.setup()
        prob.run_model()

        stream = StringIO()
        prob.model.list_outputs(all_procs=True, out_stream=stream)

        if self.comm.rank == 0:

            text = stream.getvalue().split('\n')
            expected_text = [
                "5 Explicit Output(s) in 'model'",
                "-------------------------------",
                "",
                "varname     value",
                "----------  -----",
                "p1",
                "  x       [1.]",
                "p2",
                "  x       [1.]",
                "parallel",
                "  c1",
                "    y     [1.]",
                "  c2",
                "    y     [1.]",
                "c3",
                "  y       [10.]",
                "",
                "",
                "0 Implicit Output(s) in 'model'",
                "-------------------------------",
            ]
            for i, line in enumerate(expected_text):
                if line and not line.startswith('-'):
                    self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

        stream = StringIO()
        prob.model.list_inputs(all_procs=True, out_stream=stream)

        if self.comm.rank == 0:

            text = stream.getvalue().split('\n')
            expected_text = [
                "4 Input(s) in 'model'",
                "---------------------",
                "",
                "varname     value",
                "----------  -----",
                "parallel",
                "  c1",
                "    x     [1.]",
                "  c2",
                "    x     [1.]",
                "c3",
                "  x1      [1.]",
                "  x2      [1.]",
            ]

            for i, line in enumerate(expected_text):
                if line and not line.startswith('-'):
                    self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

if __name__ == '__main__':
    unittest.main()

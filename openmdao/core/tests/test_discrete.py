""" Unit tests for discrete variables."""

import sys
import unittest
import copy

from io import StringIO
import numpy as np

import openmdao.api as om
from openmdao.core.driver import Driver
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
from openmdao.test_suite.components.sellar import StateConnection, \
     SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import remove_whitespace
from openmdao.utils.logger_utils import TestLogger


class ModCompEx(om.ExplicitComponent):
    def __init__(self, modval, **kwargs):
        super(ModCompEx, self).__init__(**kwargs)
        self.modval = modval

    def setup(self):
        self.add_input('a', val=10.)
        self.add_output('b', val=0.)
        self.add_discrete_input('x', val=10, tags='tagx')
        self.add_discrete_output('y', val=0, tags='tagy')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        outputs['b'] = inputs['a']*2.
        discrete_outputs['y'] = discrete_inputs['x'] % self.modval


class ModCompIm(om.ImplicitComponent):
    def __init__(self, modval, **kwargs):
        super(ModCompIm, self).__init__(**kwargs)
        self.modval = modval

    def setup(self):
        self.add_discrete_input('x', val=10, tags='tagx')
        self.add_discrete_output('y', val=0, tags='tagy')

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs, discrete_outputs):
        discrete_outputs['y'] = discrete_inputs['x'] % self.modval

    def solve_nonlinear(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['y'] = discrete_inputs['x'] % self.modval


class CompDiscWDerivs(om.ExplicitComponent):
    def setup(self):
        self.add_discrete_input('N', 2)
        self.add_discrete_output('Nout', 2)
        self.add_input('x')
        self.add_output('y')
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['Nout'] = discrete_inputs['N'] * 2
        outputs['y'] = inputs['x'] * 3.

    def compute_partials(self, inputs, partials, discrete_inputs):
        partials['y', 'x'] = 3.


class CompDiscWDerivsImplicit(StateConnection):
    def setup(self):
        super(CompDiscWDerivsImplicit, self).setup()
        self.add_discrete_input('N', 2)
        self.add_discrete_output('Nout', 2)

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs, discrete_outputs):
        super(CompDiscWDerivsImplicit, self).apply_nonlinear(inputs, outputs, residuals)
        discrete_outputs['Nout'] = discrete_inputs['N'] * 2

    def solve_nonlinear(self, inputs, outputs, discrete_inputs, discrete_outputs):
        super(CompDiscWDerivsImplicit, self).solve_nonlinear(inputs, outputs)
        discrete_outputs['Nout'] = discrete_inputs['N'] * 2

    def linearize(self, inputs, outputs, J, discrets_inputs, discrete_outputs):
        super(CompDiscWDerivsImplicit, self).linearize(inputs, outputs, J)


class MixedCompDiscIn(om.ExplicitComponent):
    def __init__(self, mult, **kwargs):
        super(MixedCompDiscIn, self).__init__(**kwargs)
        self.mult = mult

    def setup(self):
        self.add_discrete_input('x', val=1)
        self.add_output('y')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        outputs['y'] = discrete_inputs['x'] * self.mult


class MixedCompDiscOut(om.ExplicitComponent):
    def __init__(self, mult, **kwargs):
        super(MixedCompDiscOut, self).__init__(**kwargs)
        self.mult = mult

    def setup(self):
        self.add_input('x')
        self.add_discrete_output('y', val=1)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['y'] = inputs['x'] * self.mult


class InternalDiscreteGroup(om.Group):
    # this group has an internal discrete connection with continuous external vars,
    # so it can be spliced into an existing continuous model to test for discrete
    # var error checking.
    def setup(self):
        self.add_subsystem('C1', MixedCompDiscOut(1), promotes_inputs=['x'])
        self.add_subsystem('C2', MixedCompDiscIn(1), promotes_outputs=['y'])
        self.connect('C1.y', 'C2.x')


class DiscreteDriver(Driver):

    def __init__(self):
        super(DiscreteDriver, self).__init__()
        self.supports.declare('integer_design_vars', types=bool, default=True)

    def run(self):
        self.get_design_var_values()


class _DiscreteVal(object):
    """Generic discrete value to test passing of objects."""
    def __init__(self, val):
        self._val = val

    def getval(self):
        return self._val

    def setval(self, val):
        if isinstance(val, _DiscreteVal):
            val = val.getval()
        self._val = val

    def __iadd__(self, val):
        if isinstance(val, _DiscreteVal):
            val = val.getval()
        self._val += val
        return self

    def __imul__(self, val):
        if isinstance(val, _DiscreteVal):
            val = val.getval()
        self._val *= val
        return self

    def __eq__(self, val):
        if isinstance(val, _DiscreteVal):
            return self._val == val.getval()
        return False

class PathCompEx(om.ExplicitComponent):

    def setup(self):
        self.add_discrete_input('x', val='')
        self.add_discrete_output('y', val='')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['y'] = discrete_inputs['x'] + self.pathname + '/'


class ObjAdderCompEx(om.ExplicitComponent):
    def __init__(self, val, **kwargs):
        super(ObjAdderCompEx, self).__init__(**kwargs)
        self.val = val

    def setup(self):
        self.add_discrete_input('x', val=self.val)
        self.add_discrete_output('y', val=copy.deepcopy(self.val))

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['y'].setval(discrete_inputs['x'].getval() + self.val.getval())


class DiscreteTestCase(unittest.TestCase):

    def test_simple_run_once(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_discrete_output('x', 11)
        model.add_subsystem('comp', ModCompEx(3))

        model.connect('indep.x', 'comp.x')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp.y'], 2)

    def test_simple_run_once_promoted(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
        indep.add_discrete_output('x', 11)
        model.add_subsystem('comp', ModCompEx(3), promotes=['*'])

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['y'], 2)

    def test_simple_run_once_implicit(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_discrete_output('x', 11)
        model.add_subsystem('comp', ModCompIm(3))

        model.connect('indep.x', 'comp.x')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp.y'], 2)

    def test_list_inputs_outputs(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_discrete_output('x', 11)

        model.add_subsystem('expl', ModCompEx(3))
        model.add_subsystem('impl', ModCompIm(3))

        model.connect('indep.x', ['expl.x', 'impl.x'])

        prob.setup()

        #
        # list vars before model has been run (relative names)
        #
        expl_inputs = prob.model.expl.list_inputs(out_stream=None)
        expected = {
            'a': {'value': [10.]},
            'x': {'value': 10}
        }
        self.assertEqual(dict(expl_inputs), expected)

        impl_inputs = prob.model.impl.list_inputs(out_stream=None)
        expected = {
            'x': {'value': 10}
        }
        self.assertEqual(dict(impl_inputs), expected)

        expl_outputs = prob.model.expl.list_outputs(out_stream=None)
        expected = {
            'b': {'value': [0.]},
            'y': {'value': 0}
        }
        self.assertEqual(dict(expl_outputs), expected)

        impl_outputs = prob.model.impl.list_outputs(out_stream=None)
        expected = {
            'y': {'value': 0}
        }
        self.assertEqual(dict(impl_outputs), expected)

        #
        # run model
        #
        prob.run_model()

        #
        # list inputs, not hierarchical
        #
        stream = StringIO()
        prob.model.list_inputs(values=True, hierarchical=False, out_stream=stream)
        text = stream.getvalue()

        self.assertEqual(1, text.count("3 Input(s) in 'model'"))

        # make sure they are in the correct order
        self.assertTrue(text.find('expl.a') < text.find('expl.x') < text.find('impl.x'))

        #
        # list inputs, hierarchical
        #
        stream = StringIO()
        prob.model.list_inputs(values=True, hierarchical=True, out_stream=stream)
        text = stream.getvalue()

        self.assertEqual(1, text.count("3 Input(s) in 'model'"))
        self.assertEqual(1, text.count('\nmodel'))
        self.assertEqual(1, text.count('\n  expl'))
        self.assertEqual(1, text.count('\n    a'))
        self.assertEqual(1, text.count('\n  impl'))
        self.assertEqual(2, text.count('\n    x'))      # both implicit & explicit

        #
        # list outputs, not hierarchical
        #
        stream = StringIO()
        prob.model.list_outputs(values=True, residuals=True, hierarchical=False, out_stream=stream)
        text = stream.getvalue()

        self.assertEqual(text.count('3 Explicit Output'), 1)
        self.assertEqual(text.count('1 Implicit Output'), 1)

        # make sure they are in the correct order
        self.assertTrue(text.find('indep.x') < text.find('expl.b') <
                        text.find('expl.y') < text.find('impl.y'))

        #
        # list outputs, hierarchical
        #
        stream = StringIO()
        prob.model.list_outputs(values=True, residuals=True, hierarchical=True, out_stream=stream)
        text = stream.getvalue()

        self.assertEqual(text.count('\nmodel'), 2)      # both implicit & explicit
        self.assertEqual(text.count('\n  indep'), 1)
        self.assertEqual(text.count('\n    x'), 1)
        self.assertEqual(text.count('\n  expl'), 1)
        self.assertEqual(text.count('\n    b'), 1)
        self.assertEqual(text.count('\n  impl'), 1)
        self.assertEqual(text.count('\n    y'), 2)      # both implicit & explicit

    def test_list_inputs_outputs_promoted(self):
        model = om.Group()

        model.add_subsystem('expl', ModCompEx(3), promotes_inputs=['x'])
        model.add_subsystem('impl', ModCompIm(3), promotes_inputs=['x'])

        prob = om.Problem(model)
        prob.setup()

        prob['x'] = 11

        prob.run_model()

        #
        # list inputs
        #
        stream = StringIO()

        model.list_inputs(hierarchical=False, prom_name=True, out_stream=stream)

        text = stream.getvalue().split('\n')

        expected = [
            "3 Input(s) in 'model'",
            "---------------------",
            "",
            "varname  value  prom_name",
            "-------  -----  ---------",
            "expl.a   [10.]  expl.a",
            "expl.x   11     x",
            "impl.x   11     x",
        ]

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

        #
        # list outputs
        #
        stream = StringIO()

        model.list_outputs(prom_name=True, out_stream=stream, list_autoivcs=True)

        text = stream.getvalue().split('\n')

        expected = [
            "4 Explicit Output(s) in 'model'",
            "-------------------------------",
            "",
            "varname  value  prom_name",
            "-------  -----  ---------",
            "model",
            "  _auto_ivc",
            "    v0    [10.]  _auto_ivc.v0",
            "    v1    11     _auto_ivc.v1",
            "  expl",
            "    b    [20.]  expl.b",
            "    y    2      expl.y",
            "",
            "",
            "1 Implicit Output(s) in 'model'",
            "-------------------------------",
            "",
            "varname  value  prom_name",
            "-------  -----  ---------",
            "model",
            "  impl",
            "    y    2      impl.y",
        ]

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

    def test_list_inputs_outputs_with_tags(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_discrete_output('x', 11)

        model.add_subsystem('expl', ModCompEx(3))
        model.add_subsystem('impl', ModCompIm(3))

        model.connect('indep.x', ['expl.x', 'impl.x'])

        prob.setup()
        prob.run_model()

        # list inputs, no tags
        inputs = prob.model.list_inputs(values=False, out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('expl.a', {}),
            ('expl.x', {}),
            ('impl.x', {}),
        ])

        # list inputs, with tags
        inputs = prob.model.list_inputs(values=False, out_stream=None, tags='tagx')
        self.assertEqual(sorted(inputs), [
            ('expl.x', {}),
            ('impl.x', {}),
        ])

        # list outputs, no tags
        outputs = prob.model.list_outputs(values=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('expl.b', {}),
            ('expl.y', {}),
            ('impl.y', {}),
            ('indep.x', {}),
        ])

        # list outputs, with tags
        outputs = prob.model.list_outputs(values=False, out_stream=None, tags='tagy')
        self.assertEqual(sorted(outputs), [
            ('expl.y', {}),
            ('impl.y', {}),
        ])

    def test_float_to_discrete_error(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', 1.0)
        model.add_subsystem('comp', ModCompEx(3))

        model.connect('indep.x', 'comp.x')

        with self.assertRaises(Exception) as ctx:
            prob.setup()
        self.assertEqual(str(ctx.exception),
                         "Group (<model>): Can't connect continuous output 'indep.x' to discrete input 'comp.x'.")

    def test_discrete_to_float_error(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_discrete_output('x', 1)
        model.add_subsystem('comp', om.ExecComp("y=2.0*x"))

        model.connect('indep.x', 'comp.x')

        with self.assertRaises(Exception) as ctx:
            prob.setup()
        self.assertEqual(str(ctx.exception),
                         "Group (<model>): Can't connect discrete output 'indep.x' to continuous input 'comp.x'.")

    def test_discrete_mismatch_error(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_discrete_output('x', val='foo')
        model.add_subsystem('comp', ModCompEx(3))

        model.connect('indep.x', 'comp.x')

        with self.assertRaises(Exception) as ctx:
            prob.setup()
        self.assertEqual(str(ctx.exception),
                         "Group (<model>): Type 'str' of output 'indep.x' is incompatible with type 'int' of input 'comp.x'.")

    def test_driver_discrete_enforce_int(self):
        # Drivers require discrete vars to be int or ndarrays of int.
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_discrete_output('x', 11)
        model.add_subsystem('comp', ModCompIm(3))

        model.connect('indep.x', 'comp.x')

        model.add_design_var('indep.x', 11)
        prob.driver = DiscreteDriver()
        prob.setup()

        msg = "Only integer scalars or ndarrays are supported as values " + \
              "for discrete variables when used as a design variable. %s " + \
              "was specified."

        # Insert a non integer
        prob['indep.x'] = 3.7

        with self.assertRaises(Exception) as ctx:
            prob.run_driver()

        self.assertEqual(str(ctx.exception), msg % "A value of type 'float'")

        # Insert a float ndarray
        prob['indep.x'] = np.array([3.0])

        with self.assertRaises(Exception) as ctx:
            prob.run_driver()

        self.assertEqual(str(ctx.exception), msg % "An array of type 'float64'")

        # Make sure these work.

        prob['indep.x'] = np.array([3.0], dtype=np.int64)
        prob.run_driver()

        prob['indep.x'] = 5
        prob.run_driver()

    def test_discrete_deriv_explicit(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', 1.0)

        comp = model.add_subsystem('comp', CompDiscWDerivs())
        model.connect('indep.x', 'comp.x')

        model.add_design_var('indep.x')
        model.add_objective('comp.y')

        prob.setup()
        prob.run_model()

        J = prob.compute_totals(return_format='array')

        np.testing.assert_almost_equal(J, np.array([[3.]]))

    def test_discrete_deriv_implicit(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', 1.0, ref=10.)

        comp = model.add_subsystem('comp', CompDiscWDerivsImplicit(), promotes=['N'])
        sink = model.add_subsystem('sink', MixedCompDiscIn(1.0))
        model.connect('indep.x', 'comp.y2_actual')
        model.connect('comp.Nout', 'sink.x')

        model.add_design_var('indep.x')
        model.add_objective('comp.y2_command')

        prob.setup()

        prob['N'] = 1

        prob.run_model()

        J = prob.compute_totals(return_format='array')

        np.testing.assert_almost_equal(J, np.array([[-1]]))

    def test_deriv_err(self):
        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp(), promotes_outputs=['x'])
        indep.add_output('x', 1.0)

        G = model.add_subsystem('G', om.Group(), promotes_inputs=['x'])

        G1 = G.add_subsystem('G1', InternalDiscreteGroup(), promotes_inputs=['x'], promotes_outputs=['y'])

        G2 = G.add_subsystem('G2', om.Group(), promotes_inputs=['x'])
        G2.add_subsystem('C2_1', om.ExecComp('y=3*x'), promotes_inputs=['x'])
        G2.add_subsystem('C2_2', om.ExecComp('y=4*x'), promotes_outputs=['y'])
        G2.connect('C2_1.y', 'C2_2.x')

        model.add_subsystem('C3', om.ExecComp('y=3+x'))
        model.add_subsystem('C4', om.ExecComp('y=4+x'))

        model.connect('G.y', 'C3.x')
        model.connect('G.G2.y', 'C4.x')

        prob.model.add_design_var('x')
        prob.model.add_objective('C3.y')
        prob.model.add_constraint('C4.y')

        prob.setup()
        prob.run_model()

        self.assertEqual(prob['C3.y'], 4.0)
        self.assertEqual(prob['C4.y'], 16.0)

        with self.assertRaises(Exception) as ctx:
            J = prob.compute_totals()
        self.assertEqual(str(ctx.exception),
                         "Total derivative with respect to 'indep.x' depends upon discrete output variables ['G.G1.C1.y'].")

    def test_connection_to_output(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('C1', ModCompEx(modval=2))
        model.add_subsystem('C2', ModCompEx(modval=2))

        model.connect('C1.y', 'C2.y')

        with self.assertRaises(Exception) as cm:
            prob.setup()

        msg = "Group (<model>): Attempted to connect from 'C1.y' to 'C2.y', but 'C2.y' is an output. All connections must be from an output to an input."
        self.assertEqual(str(cm.exception), msg)

    def test_connection_from_input(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('C1', ModCompEx(modval=2))
        model.add_subsystem('C2', ModCompEx(modval=2))

        model.connect('C1.x', 'C2.x')

        with self.assertRaises(Exception) as cm:
            prob.setup()

        msg = "Group (<model>): Attempted to connect from 'C1.x' to 'C2.x', but 'C1.x' is an input. All connections must be from an output to an input."
        self.assertEqual(str(cm.exception), msg)


class SolverDiscreteTestCase(unittest.TestCase):
    def _setup_model(self, solver_class):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        proms = ['x', 'z', 'y1', 'state_eq.y2_actual', 'state_eq.y2_command', 'd1.y2', 'd2.y2']
        sub = model.add_subsystem('sub', om.Group(), promotes=proms)

        subgrp = sub.add_subsystem('state_eq_group', om.Group(),
                                   promotes=['state_eq.y2_actual', 'state_eq.y2_command'])
        subgrp.add_subsystem('state_eq', StateConnection())

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1'])

        model.connect('state_eq.y2_command', 'd1.y2')
        model.connect('d2.y2', 'state_eq.y2_actual')

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                            promotes=['x', 'z', 'y1', 'obj'])
        model.connect('d2.y2', 'obj_cmp.y2')

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2'])

        # splice a group containing discrete vars into the model
        model.add_subsystem('discrete_g', InternalDiscreteGroup())
        model.connect('d2.y2', 'discrete_g.x')
        model.connect('discrete_g.y', 'con_cmp2.y2')

        model.nonlinear_solver = solver_class()

        prob.set_solver_print(level=0)
        prob.setup()

        return prob

    def test_discrete_err_newton(self):
        prob = self._setup_model(om.NewtonSolver)

        with self.assertRaises(Exception) as ctx:
            prob.run_model()

        self.assertEqual(str(ctx.exception),
                         "Group (<model>) has a NewtonSolver solver and contains discrete outputs ['discrete_g.C1.y'].")

    def test_discrete_err_broyden(self):
        prob = self._setup_model(om.BroydenSolver)

        with self.assertRaises(Exception) as ctx:
            prob.run_model()

        self.assertEqual(str(ctx.exception),
                         "Group (<model>) has a BroydenSolver solver and contains discrete outputs ['discrete_g.C1.y'].")


class DiscretePromTestCase(unittest.TestCase):
    def test_str_pass(self):
        prob = om.Problem()
        model = prob.model

        G = model.add_subsystem('G', om.ParallelGroup(), promotes_inputs=['x'])

        G1 = G.add_subsystem('G1', om.Group(), promotes_inputs=['x'], promotes_outputs=['y'])
        G1.add_subsystem('C1_1', PathCompEx(), promotes_inputs=['x'])
        G1.add_subsystem('C1_2', PathCompEx(), promotes_outputs=['y'])
        G1.connect('C1_1.y', 'C1_2.x')

        G2 = G.add_subsystem('G2', om.Group(), promotes_inputs=['x'])
        G2.add_subsystem('C2_1', PathCompEx(), promotes_inputs=['x'])
        G2.add_subsystem('C2_2', PathCompEx(), promotes_outputs=['y'])
        G2.connect('C2_1.y', 'C2_2.x')

        model.add_subsystem('C3', PathCompEx())
        model.add_subsystem('C4', PathCompEx())

        model.connect('G.y', 'C3.x')
        model.connect('G.G2.y', 'C4.x')

        prob.setup()

        prob['x'] = 'indep/'

        prob.run_model()

        self.assertEqual(prob['C3.y'], 'indep/G.G1.C1_1/G.G1.C1_2/C3/')
        self.assertEqual(prob['C4.y'], 'indep/G.G2.C2_1/G.G2.C2_2/C4/')

        prob['x'] = 'foobar/'
        prob.run_model()

        self.assertEqual(prob['C3.y'], 'foobar/G.G1.C1_1/G.G1.C1_2/C3/')
        self.assertEqual(prob['C4.y'], 'foobar/G.G2.C2_1/G.G2.C2_2/C4/')

    def test_obj_pass(self):
        prob = om.Problem()
        model = prob.model

        G = model.add_subsystem('G', om.ParallelGroup(), promotes_inputs=['x'])

        G1 = G.add_subsystem('G1', om.Group(), promotes_inputs=['x'], promotes_outputs=['y'])
        G1.add_subsystem('C1_1', ObjAdderCompEx(_DiscreteVal(5)), promotes_inputs=['x'])
        G1.add_subsystem('C1_2', ObjAdderCompEx(_DiscreteVal(7)), promotes_outputs=['y'])
        G1.connect('C1_1.y', 'C1_2.x')

        G2 = G.add_subsystem('G2', om.Group(), promotes_inputs=['x'])
        G2.add_subsystem('C2_1', ObjAdderCompEx(_DiscreteVal(5)), promotes_inputs=['x'])
        G2.add_subsystem('C2_2', ObjAdderCompEx(_DiscreteVal(11)), promotes_outputs=['y'])
        G2.connect('C2_1.y', 'C2_2.x')

        model.add_subsystem('C3', ObjAdderCompEx(_DiscreteVal(9)))
        model.add_subsystem('C4', ObjAdderCompEx(_DiscreteVal(21)))

        model.connect('G.y', 'C3.x')
        model.connect('G.G2.y', 'C4.x')

        prob.setup()

        prob['x'] =  _DiscreteVal(19)

        prob.run_model()

        self.assertEqual(prob['C3.y'].getval(), 40)
        self.assertEqual(prob['C4.y'].getval(), 56)

        def _var_iter(obj):
            name = obj['name']
            if 'children' in obj:
                for c in obj['children']:
                    for vname in _var_iter(c):
                        if name:
                            yield '.'.join((name, vname))
                        else:
                            yield vname
            else:
                yield name

        # add a test to see if discrete vars show up in n2
        data = _get_viewer_data(prob)
        findvars = [
            '_auto_ivc.v0',
            'G.G1.C1_1.x',
            'G.G1.C1_1.y',
            'G.G1.C1_2.x',
            'G.G1.C1_2.y',
            'G.G2.C2_1.x',
            'G.G2.C2_1.y',
            'G.G2.C2_2.x',
            'G.G2.C2_2.y',
            'C3.x',
            'C3.y',
            'C4.x',
            'C4.y',
        ]
        vnames = list(_var_iter(data['tree']))
        self.assertTrue(sorted(findvars), sorted(vnames))


class DiscreteFeatureTestCase(unittest.TestCase):

    def test_feature_discrete(self):
        import numpy as np
        import openmdao.api as om

        class BladeSolidity(om.ExplicitComponent):
            def setup(self):

                # Continuous Inputs
                self.add_input('r_m', 1.0, units="ft", desc="Mean radius")
                self.add_input('chord', 1.0, units="ft", desc="Chord length")

                # Discrete Inputs
                self.add_discrete_input('num_blades', 2, desc="Number of blades")

                # Continuous Outputs
                self.add_output('blade_solidity', 0.0, desc="Blade solidity")

            def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

                num_blades = discrete_inputs['num_blades']
                chord = inputs['chord']
                r_m = inputs['r_m']

                outputs['blade_solidity'] = chord / (2.0 * np.pi * r_m / num_blades)

        # build the model
        prob = om.Problem()

        prob.model.add_subsystem('SolidityComp', BladeSolidity(),
                                 promotes_inputs=['r_m', 'chord', 'num_blades'])

        prob.setup()

        prob.set_val('num_blades', 2)
        prob.set_val('r_m', 3.2)
        prob.set_val('chord', .3)

        prob.run_model()

        # minimum value
        assert_near_equal(prob['SolidityComp.blade_solidity'], 0.02984155, 1e-4)

    def test_feature_discrete_implicit(self):
        import openmdao.api as om

        class ImpWithInitial(om.ImplicitComponent):
            """
            An implicit component to solve the quadratic equation: x^2 - 4x + 3
            (solutions at x=1 and x=3)
            """
            def setup(self):
                self.add_input('a', val=1.)
                self.add_input('b', val=-4.)
                self.add_discrete_input('c', val=3)
                self.add_output('x', val=5.)

                self.declare_partials(of='*', wrt='*')

            def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs, discrete_outputs):
                a = inputs['a']
                b = inputs['b']
                c = discrete_inputs['c']
                x = outputs['x']
                residuals['x'] = a * x ** 2 + b * x + c

            def linearize(self, inputs, outputs, partials, discrete_inputs, discrete_outputs):
                a = inputs['a']
                b = inputs['b']
                x = outputs['x']

                partials['x', 'a'] = x ** 2
                partials['x', 'b'] = x
                partials['x', 'x'] = 2 * a * x + b

            def guess_nonlinear(self, inputs, outputs, resids, discrete_inputs, discrete_outputs):
                # Default initial state of zero for x takes us to x=1 solution.
                # Here we set it to a value that will take us to the x=3 solution.
                outputs['x'] = 5

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', ImpWithInitial())

        model.comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.comp.linear_solver = om.ScipyKrylov()

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('comp.x'), 3., 1e-4)


if __name__ == "__main__":
    unittest.main()

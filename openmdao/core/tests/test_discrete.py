""" Unit tests for the problem interface."""

import sys
import unittest
import warnings
import copy
from six import assertRaisesRegex, StringIO, assertRegex

import numpy as np

from openmdao.core.group import get_relevant_vars
from openmdao.core.driver import Driver
from openmdao.api import Problem, IndepVarComp, NonlinearBlockGS, ScipyOptimizeDriver, \
    ExecComp, Group, NewtonSolver, ImplicitComponent, ScipyKrylov, ExplicitComponent, \
    ImplicitComponent, ParallelGroup
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives


class ModCompEx(ExplicitComponent):
    def __init__(self, modval, **kwargs):
        super(ModCompEx, self).__init__(**kwargs)
        self.modval = modval

    def setup(self):
        self.add_discrete_input('x', val=10)
        self.add_discrete_output('y', val=0)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['y'] = discrete_inputs['x'] % self.modval


class ModCompIm(ImplicitComponent):
    def __init__(self, modval, **kwargs):
        super(ModCompIm, self).__init__(**kwargs)
        self.modval = modval

    def setup(self):
        self.add_discrete_input('x', val=10)
        self.add_discrete_output('y', val=0)

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs, discrete_outputs):
        discrete_outputs['y'] = discrete_inputs['x'] % self.modval

    def solve_nonlinear(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['y'] = discrete_inputs['x'] % self.modval


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


class PathCompEx(ExplicitComponent):

    def setup(self):
        self.add_discrete_input('x', val=self.pathname)
        self.add_discrete_output('y', val=self.pathname + '/')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['y'] = discrete_inputs['x'] + self.pathname + '/'


class ObjAdderCompEx(ExplicitComponent):
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
        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_discrete_output('x', 11)
        model.add_subsystem('comp', ModCompEx(3))

        model.connect('indep.x', 'comp.x')

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['comp.y'], 2)

    def test_simple_run_once_implicit(self):
        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_discrete_output('x', 11)
        model.add_subsystem('comp', ModCompIm(3))

        model.connect('indep.x', 'comp.x')

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['comp.y'], 2)


class DiscreteStrTestCase(unittest.TestCase):
    def test_str_pass(self):
        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp(), promotes_outputs=['x'])
        indep.add_discrete_output('x', 'indep/')

        G = model.add_subsystem('G', ParallelGroup(), promotes_inputs=['x'])

        G1 = G.add_subsystem('G1', Group(), promotes_inputs=['x'], promotes_outputs=['y'])
        G1.add_subsystem('C1_1', PathCompEx(), promotes_inputs=['x'])
        G1.add_subsystem('C1_2', PathCompEx(), promotes_outputs=['y'])
        G1.connect('C1_1.y', 'C1_2.x')

        G2 = G.add_subsystem('G2', Group(), promotes_inputs=['x'])
        G2.add_subsystem('C2_1', PathCompEx(), promotes_inputs=['x'])
        G2.add_subsystem('C2_2', PathCompEx(), promotes_outputs=['y'])
        G2.connect('C2_1.y', 'C2_2.x')

        model.add_subsystem('C3', PathCompEx())
        model.add_subsystem('C4', PathCompEx())

        model.connect('G.y', 'C3.x')
        model.connect('G.G2.y', 'C4.x')

        prob.setup()
        prob.run_model()

        self.assertEqual(prob['C3.y'], 'indep/G.G1.C1_1/G.G1.C1_2/C3/')
        self.assertEqual(prob['C4.y'], 'indep/G.G2.C2_1/G.G2.C2_2/C4/')

    def test_obj_pass(self):
        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp(), promotes_outputs=['x'])
        indep.add_discrete_output('x', _DiscreteVal(19))

        G = model.add_subsystem('G', ParallelGroup(), promotes_inputs=['x'])

        G1 = G.add_subsystem('G1', Group(), promotes_inputs=['x'], promotes_outputs=['y'])
        G1.add_subsystem('C1_1', ObjAdderCompEx(_DiscreteVal(5)), promotes_inputs=['x'])
        G1.add_subsystem('C1_2', ObjAdderCompEx(_DiscreteVal(7)), promotes_outputs=['y'])
        G1.connect('C1_1.y', 'C1_2.x')

        G2 = G.add_subsystem('G2', Group(), promotes_inputs=['x'])
        G2.add_subsystem('C2_1', ObjAdderCompEx(_DiscreteVal(1)), promotes_inputs=['x'])
        G2.add_subsystem('C2_2', ObjAdderCompEx(_DiscreteVal(11)), promotes_outputs=['y'])
        G2.connect('C2_1.y', 'C2_2.x')

        model.add_subsystem('C3', ObjAdderCompEx(_DiscreteVal(9)))
        model.add_subsystem('C4', ObjAdderCompEx(_DiscreteVal(21)))

        model.connect('G.y', 'C3.x')
        model.connect('G.G2.y', 'C4.x')

        prob.setup()
        prob.run_model()

        self.assertEqual(prob['C3.y'].getval(), 40)
        self.assertEqual(prob['C4.y'].getval(), 52)

if __name__ == "__main__":
    unittest.main()

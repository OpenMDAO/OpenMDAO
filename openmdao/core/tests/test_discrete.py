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

    def test_float_to_discrete_error(self):
        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_output('x', 1.0)
        model.add_subsystem('comp', ModCompEx(3))

        model.connect('indep.x', 'comp.x')

        with self.assertRaises(Exception) as ctx:
            prob.setup()
        self.assertEqual(str(ctx.exception),
                         "Can't connect discrete output 'indep.x' to continuous input 'comp.x'.")

    def test_discrete_to_float_error(self):
        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_discrete_output('x', 1)
        model.add_subsystem('comp', ExecComp("y=2.0*x"))

        model.connect('indep.x', 'comp.x')

        with self.assertRaises(Exception) as ctx:
            prob.setup()
        self.assertEqual(str(ctx.exception),
                         "Can't connect discrete output 'indep.x' to continuous input 'comp.x'.")

    def test_discrete_mismatch_error(self):
        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_discrete_output('x', val='foo')
        model.add_subsystem('comp', ModCompEx(3))

        model.connect('indep.x', 'comp.x')

        with self.assertRaises(Exception) as ctx:
            prob.setup()
        self.assertEqual(str(ctx.exception),
                         "Type 'str' of output 'indep.x' is incompatible with type 'int' of input 'comp.x'.")


class DiscretePromTestCase(unittest.TestCase):
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

        prob['indep.x'] = 'foobar/'
        prob.run_model()

        self.assertEqual(prob['C3.y'], 'foobar/G.G1.C1_1/G.G1.C1_2/C3/')
        self.assertEqual(prob['C4.y'], 'foobar/G.G2.C2_1/G.G2.C2_2/C4/')

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



class DiscreteFeatureTestCase(unittest.TestCase):
    def test_feature_discrete(self):
        from openmdao.api import Problem, IndepVarComp, ExplicitComponent

        class BladeSolidity(ExplicitComponent):
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
        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('r_m', 3.2, units="ft")
        indeps.add_output('chord', .3, units='ft')
        indeps.add_discrete_output('num_blades', 2)

        prob.model.add_subsystem('SolidityComp', BladeSolidity(),
                                 promotes_inputs=['r_m', 'chord', 'num_blades'])

        prob.setup()
        prob.run_model()

        # minimum value
        assert_rel_error(self, prob['SolidityComp.blade_solidity'], 0.02984155, 1e-4)

if __name__ == "__main__":
    unittest.main()

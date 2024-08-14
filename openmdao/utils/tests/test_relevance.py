import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.relevance import _vars2systems
from openmdao.utils.assert_utils import assert_check_totals


class TestRelevance(unittest.TestCase):
    def test_vars2systems(self):
        names = ['abc.def.g', 'xyz.pdq.bbb', 'aaa.xxx', 'foobar.y']
        expected = {'abc', 'abc.def', 'xyz', 'xyz.pdq', 'aaa', 'foobar', ''}
        self.assertEqual(_vars2systems(names), expected)


class TestDerivsWithoutDVs(unittest.TestCase):
    def test_derivs_with_no_dvs(self):
        # this tests github issue #3037

        class DummyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('a', 1.)
                self.add_output('b', 1.)
            def compute(self, inputs, outputs):
                outputs['b'] = 2. * inputs['a']
            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode=='rev':
                    if 'a' in d_inputs:
                        if 'b' in d_outputs:
                            d_inputs['a'] += d_outputs['b'] * 2.

        prob = om.Problem()
        prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        prob.model.ivc.add_output('a', 1.)
        prob.model.add_subsystem('dummy', DummyComp(), promotes=['*'])

        ### when there's a objective/constraint but no design var, derivatives were zero because the
        ### derivative computation was skipped
        #prob.model.add_design_var('a', lower=0., upper=2.)
        #prob.model.add_constraint('b', lower=3.)
        prob.model.add_objective('b')

        prob.setup(mode='rev')
        prob.run_model()
        chk = prob.check_totals(of='b', wrt='a', show_only_incorrect=True)
        assert_check_totals(chk)

class TestRelevanceEmptyGroups(unittest.TestCase):
    def test_emptygroup(self):
        '''Tests that relevance checks do not error if empty groups are present'''
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('empy_group', om.Group(), promotes=['*'])
        grp2: om.Group = model.add_subsystem('non_empty_group', om.Group(), promotes=['*'])
        grp2.add_subsystem('idv', om.IndepVarComp('x', val=1), promotes=['*'])
        grp2.add_subsystem('comp', om.ExecComp('y=2*x**2'), promotes=['*'])
        model.add_design_var('x')
        model.add_objective('y')

        prob.driver = om.ScipyOptimizeDriver()

        prob.setup(force_alloc_complex=True)
        prob.run_driver()

        assert_check_totals(prob.check_totals(method='cs', out_stream=None))


class LinearEquation(om.ExplicitComponent):
    """A linear equation y = Ax - b"""

    def initialize(self):
        self.options.declare("numInputs", types=int, default=3)
        self.options.declare("numOutputs", types=int, default=2)

    def setup(self):
        np.random.seed(0)
        nIn = self.options["numInputs"]
        nOut = self.options["numOutputs"]
        self.A = np.random.rand(nOut, nIn)
        self.b = np.random.rand(nOut)
        self.add_input("x", shape=nIn)
        self.add_output("res", shape=nOut)

        self.declare_partials("res", "x", val=self.A)

    def compute(self, inputs, outputs):
        outputs["res"] = self.A @ inputs["x"] - self.b


class SquaredNorm(om.ExplicitComponent):
    """The square of the L2 norm of a vector"""

    def initialize(self):
        self.options.declare("numInputs", types=int, default=3)

    def setup(self):
        nIn = self.options["numInputs"]
        self.add_input("x", shape=nIn)
        self.add_output("xNorm")

        self.declare_partials("xNorm", "x")

    def compute_partials(self, inputs, partials):
        partials["xNorm", "x"] = 2 * inputs["x"]

    def compute(self, inputs, outputs):
        outputs["xNorm"] = np.dot(inputs["x"], inputs["x"])


class UnderdeterminedSystem(om.Group):
    def initialize(self):
        self.options.declare("numInputs", types=int, default=3)
        self.options.declare("numOutputs", types=int, default=2)

    def setup(self):
        nIn = self.options["numInputs"]
        nOut = self.options["numOutputs"]

        variables = om.IndepVarComp("x", np.random.rand(nIn))
        self.add_subsystem("variables", variables, promotes=["*"])
        self.add_design_var("x")

        self.add_subsystem("linearEquation", LinearEquation(numInputs=nIn, numOutputs=nOut), promotes=["*"])
        self.add_constraint("linearEquation.res", equals=0.0, linear=True)

        self.add_subsystem("norm2", SquaredNorm(numInputs=nIn), promotes=["*"])
        # self.add_objective("xNorm") # Uncomment this line to not have an error


class TestRelevanceNoObjLinearConstraint(unittest.TestCase):
    def test_relevance_no_obj_linear_constraint(self):
        prob = om.Problem(model=UnderdeterminedSystem(numInputs=3, numOutputs=2))
        prob.setup()
        prob.run_model()
        assert_check_totals(prob.check_totals(show_only_incorrect=True))

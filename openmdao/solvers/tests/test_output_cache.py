import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_no_warning
from openmdao.core.tests.test_impl_comp import QuadraticComp

SIZE = 1

class MyTestComp(om.ExplicitComponent):
    def __init__(self, neg_iter=4):
        super().__init__()
        self.neg_iter = neg_iter
        self.count = 0

    def setup(self):
        self.add_input('x', np.zeros(SIZE))
        self.add_output('y', np.ones(SIZE))

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        self.count += 1
        if self.count == self.neg_iter:
            outputs['y'] = 5.
        else:
            outputs['y'] = inputs['x']


class MyLogComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', np.ones(SIZE))
        self.add_output('y', np.ones(SIZE))

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = 3.0*np.log(inputs['x']) + inputs['x']**3


class TestOutputCache(unittest.TestCase):

    def setUp(self):
        p = om.Problem()
        p.model.add_subsystem('C1', MyTestComp(neg_iter=2))
        p.model.add_subsystem('C2', QuadraticComp())
        p.model.connect('C1.y', 'C2.c')
        p.model.nonlinear_solver = om.NonlinearBlockGS()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True

        p.setup()

        p['C2.b'] = 2.

        self.p = p

    def test_simple_caching(self):
        p = self.p
        p.run_model()

        print("iters:", p.model.nonlinear_solver._iter_count)


class SubComp1(om.ExplicitComponent):
    def setup(self):
        self.add_input("x")
        self.add_input("a")
        self.add_output("z")

        self.declare_partials("z", "x", method="cs")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        a = inputs["a"]
        outputs["z"] = (x - 5) ** 5 + a


class SubComp2(om.ExplicitComponent):
    def setup(self):
        self.add_input("x")
        self.add_input("b")
        self.add_output("z")

        self.declare_partials("z", "x", method="cs")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        b = inputs["b"]
        outputs["z"] = np.log(x**3 + 1) + b


class TopComp(om.ImplicitComponent):
    def setup(self):
        self.add_input("x")
        self.add_output("y")

        self.declare_partials("y", ["x", "y"], method="cs")

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs["x"]
        y = outputs["y"]

        residuals["y"] = (2**y) - x


class CoupledGroup(om.Group):
    def setup(self):
        self.add_subsystem("sub_comp1", SubComp1())
        self.add_subsystem("sub_comp2", SubComp2())
        balance = self.add_subsystem("balance", om.BalanceComp())

        balance.add_balance("x")
        self.connect("balance.x", ["sub_comp1.x", "sub_comp2.x"])
        self.connect("sub_comp1.z", "balance.lhs:x")
        self.connect("sub_comp2.z", "balance.rhs:x")

        solver = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        solver.options["maxiter"] = 1000
        solver.options["atol"] = 1e-8
        solver.options["rtol"] = 1e-99
        solver.options["use_cached_states"] = True
        solver.options["err_on_non_converge"] = True
        self.linear_solver = om.DirectSolver(assemble_jac=True)


class NLBGSGroup(om.Group):
    def setup(self):
        self.add_subsystem("coupling", CoupledGroup())
        self.add_subsystem("top_comp", TopComp())

        self.connect("coupling.balance.x", "top_comp.x")

        solver = self.nonlinear_solver = om.NonlinearBlockGS()
        solver.options["maxiter"] = 1000
        solver.options["atol"] = 1e-14
        solver.options["rtol"] = 1e-99
        solver.options["use_cached_states"] = True
        solver.options["err_on_non_converge"] = True
        self.linear_solver = om.DirectSolver(assemble_jac=True)


class TestOutputCache2(unittest.TestCase):
    def test_coupled_system(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem(f"simple", NLBGSGroup())

        prob.setup()

        # --- Set the print levels ---
        # prob.set_solver_print(level=-1)
        # prob.set_solver_print(level=2, depth=1)

        prob.set_val("simple.coupling.sub_comp1.a", val=5.0)
        prob.set_val("simple.coupling.sub_comp2.b", val=10.0)

        prob.run_model()

        prob.model.list_outputs()

        prob.set_val("simple.coupling.sub_comp1.x", val=-1.0)

        try:
            prob.run_model()
        except om.AnalysisError:
            prob.set_val("simple.coupling.sub_comp1.a", val=2.0)
            prob.set_val("simple.coupling.sub_comp2.b", val=3.0)
            prob.run_model()

        prob.model.list_outputs()

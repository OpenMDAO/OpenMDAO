import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_no_warning


class SubComp1(om.ExplicitComponent):
    def setup(self):
        self.add_input("x")
        self.add_input("a")
        self.add_output("z")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        a = inputs["a"]
        outputs["z"] = (x - 5) ** 5 + a


class SubComp2(om.ExplicitComponent):
    def setup(self):
        self.add_input("x")
        self.add_input("b")
        self.add_output("z")

        self.declare_partials("*", "*", method="cs")

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
    def __init__(self, err_on_non_converge=True, **kwargs):
        super().__init__(**kwargs)
        self.err_on_non_converge = err_on_non_converge

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
        solver.options["restart_from_successful"] = True
        solver.options["err_on_non_converge"] = self.err_on_non_converge
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
        solver.options["restart_from_successful"] = True
        solver.options["err_on_non_converge"] = True
        self.linear_solver = om.DirectSolver(assemble_jac=True)

    def initialize(self):
        self.options.declare('use_guess', types=bool, default=False)

    def guess_nonlinear(self, inputs, outputs, residuals):
        if self.options['use_guess']:
            # this guess will cause an AnalysisError
            outputs["coupling.balance.x"] = -1.


class TestOutputCache(unittest.TestCase):
    def test_coupled_system(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem("simple", NLBGSGroup())

        prob.setup()

        # prob.set_solver_print(level=-1)
        # prob.set_solver_print(level=2, depth=9)

        prob.set_val("simple.coupling.sub_comp1.a", val=5.0)
        prob.set_val("simple.coupling.sub_comp2.b", val=10.0)

        prob.run_model()

        assert_near_equal(prob['simple.coupling.sub_comp1.z'][0], 15.66717076361843, 1e-6)
        assert_near_equal(prob['simple.coupling.sub_comp2.z'][0], 15.66717076361843, 1e-6)
        assert_near_equal(prob['simple.coupling.balance.x'][0], 6.6054982979676495, 1e-6)

        # prob.model.list_outputs()

        prob.set_val("simple.coupling.sub_comp1.x", val=-1.0)

        try:
            prob.run_model()
        except om.AnalysisError:
            prob.set_val("simple.coupling.sub_comp1.x", val=2.0)
            prob.set_val("simple.coupling.sub_comp2.b", val=3.0)
            prob.run_model()
        else:
            self.fail("Expected AnalysisError")

        # prob.model.list_outputs()

        assert_near_equal(prob['simple.coupling.sub_comp1.z'][0], 8.51905504, 1e-6)
        assert_near_equal(prob['simple.coupling.sub_comp2.z'][0], 8.51905504, 1e-6)
        assert_near_equal(prob['simple.coupling.balance.x'][0], 6.28613102, 1e-6)

    def test_warning(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('coupling', CoupledGroup(err_on_non_converge=False))

        prob.setup()

        prob.set_val("coupling.sub_comp1.a", val=5.0)
        prob.set_val("coupling.sub_comp2.b", val=10.0)

        expected = "NewtonSolver in 'coupling' <class CoupledGroup>: Option 'restart_from_successful' does nothing unless option 'err_on_non_converge' is set to True."
        with assert_warning(om.SolverWarning, expected):
            prob.run_model()

    def test_coupled_system_with_guess_nonlinear(self):
        prob = om.Problem()
        simple = prob.model.add_subsystem("simple", NLBGSGroup())

        prob.setup()

        #
        # initial run, result will be cached since 'restart_from_successful' is enabled
        #

        prob.set_val("simple.coupling.sub_comp1.a", val=5.0)
        prob.set_val("simple.coupling.sub_comp2.b", val=10.0)

        prob.run_model()

        assert_near_equal(prob['simple.coupling.sub_comp1.z'][0], 15.66717076361843, 1e-6)
        assert_near_equal(prob['simple.coupling.sub_comp2.z'][0], 15.66717076361843, 1e-6)
        assert_near_equal(prob['simple.coupling.balance.x'][0], 6.6054982979676495, 1e-6)

        #
        # use the guess which will cause the run to fail
        #

        simple.options['use_guess'] = True

        try:
            prob.run_model()
        except om.AnalysisError:
            pass
        else:
            self.fail("Expected AnalysisError")

        #
        # run again, should restart from valid point and NOT use the guess
        #

        prob.set_val("simple.coupling.sub_comp1.x", val=2.0)
        prob.set_val("simple.coupling.sub_comp2.b", val=3.0)
        prob.run_model()

        assert_near_equal(prob['simple.coupling.sub_comp1.z'][0], 8.51905504, 1e-6)
        assert_near_equal(prob['simple.coupling.sub_comp2.z'][0], 8.51905504, 1e-6)
        assert_near_equal(prob['simple.coupling.balance.x'][0], 6.28613102, 1e-6)


if __name__ == "__main__":
    unittest.main()

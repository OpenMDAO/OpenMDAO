""" Unit tests for the NLOpt Driver."""

import copy
import sys
import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import (
    SellarDerivativesGrouped,
    SellarDerivatives,
)
from openmdao.test_suite.components.simple_comps import NonSquareArrayComp
from openmdao.test_suite.groups.sin_fitter import SineFitter
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import run_driver
from openmdao.drivers.nlopt_driver import NLoptDriver

try:
    import nlopt
except ImportError:
    nlopt = None


def rastrigin(x):
    a = 10  # constant
    return np.sum(np.square(x) - a * np.cos(2 * np.pi * x)) + a * np.size(x)


@unittest.skipIf(nlopt is None, "only run if NLopt is installed.")
class TestNLoptDriver(unittest.TestCase):
    def test_driver_supports(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver(optimizer="LD_SLSQP", tol=1e-9)

        with self.assertRaises(KeyError) as raises_msg:
            prob.driver.supports["equality_constraints"] = False

        exception = raises_msg.exception

        msg = "NLoptDriver: Tried to set read-only option 'equality_constraints'."

        self.assertEqual(exception.args[0], msg)

    def test_compute_totals_basic_return_array(self):
        # Make sure 'array' return_format works.

        prob = om.Problem()
        model = prob.model
        model.add_subsystem("p1", om.IndepVarComp("x", 0.0), promotes=["x"])
        model.add_subsystem("p2", om.IndepVarComp("y", 0.0), promotes=["y"])
        model.add_subsystem("comp", Paraboloid(), promotes=["x", "y", "f_xy"])

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")

        prob.setup(check=False, mode="fwd")
        prob.set_solver_print(level=0)

        failed = prob.run_driver()

        of = ["f_xy"]
        wrt = ["x", "y"]
        derivs = prob.compute_totals(of=of, wrt=wrt, return_format="array")

        assert_near_equal(derivs[0, 0], -6.0, 1e-6)
        assert_near_equal(derivs[0, 1], 8.0, 1e-6)

        prob.setup(check=False, mode="rev")

        prob.run_model()

        of = ["f_xy"]
        wrt = ["x", "y"]
        derivs = prob.compute_totals(of=of, wrt=wrt, return_format="array")

        assert_near_equal(derivs[0, 0], -6.0, 1e-6)
        assert_near_equal(derivs[0, 1], 8.0, 1e-6)

    def test_compute_totals_return_array_non_square(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("px", om.IndepVarComp(name="x", val=np.ones((2,))))
        comp = model.add_subsystem("comp", NonSquareArrayComp())
        model.connect("px.x", "comp.x1")

        model.add_design_var("px.x")
        model.add_objective("px.x")
        model.add_constraint("comp.y1")
        model.add_constraint("comp.y2")

        prob.setup(check=False, mode="auto")

        failed = prob.run_driver()

        derivs = prob.compute_totals(
            of=["comp.y1"], wrt=["px.x"], return_format="array"
        )

        J = comp.JJ[0:3, 0:2]
        assert_near_equal(J, derivs, 1.0e-3)

        # Support for a name to be in 'of' and 'wrt'

        derivs = prob.compute_totals(
            of=["comp.y2", "px.x", "comp.y1"], wrt=["px.x"], return_format="array"
        )

        assert_near_equal(J, derivs[3:, :], 1.0e-3)
        assert_near_equal(comp.JJ[3:4, 0:2], derivs[0:1, :], 1.0e-3)
        assert_near_equal(np.eye(2), derivs[1:3, :], 1.0e-3)

    def test_deriv_wrt_self(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("px", om.IndepVarComp(name="x", val=np.ones((2,))))

        model.add_design_var("px.x")
        model.add_objective("px.x")

        prob.setup()

        failed = prob.run_driver()

        # Support for a name to be in 'of' and 'wrt'

        J = prob.driver._compute_totals(
            of=["px.x"], wrt=["px.x"], return_format="array"
        )

        assert_near_equal(J, np.eye(2), 1.0e-3)

    def test_optimizer_simple_paraboloid_unconstrained(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver(optimizer="LD_SLSQP", tol=1e-9)

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")

        prob.setup()

        failed = prob.run_driver()

        assert_near_equal(prob["x"], 6.66666667, 1e-6)
        assert_near_equal(prob["y"], -7.3333333, 1e-6)

    def test_simple_paraboloid_unconstrained(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")

        prob.setup()

        failed = prob.run_driver()

        assert_near_equal(prob["x"], 6.66666667, 1e-6)
        assert_near_equal(prob["y"], -7.3333333, 1e-6)

    def test_simple_paraboloid_unconstrained_LN_COBYLA(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LN_COBYLA"
        prob.driver.options["tol"] = 1e-12

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")

        prob.setup()

        failed = prob.run_driver()

        assert_near_equal(prob["x"], 6.66666667, 1e-6)
        assert_near_equal(prob["y"], -7.3333333, 1e-6)

    def test_simple_paraboloid_upper(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob["x"], 7.16667, 1e-6)
        assert_near_equal(prob["y"], -7.833334, 1e-6)

    def test_simple_paraboloid_lower(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = x - y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)

        model.add_objective("f_xy")
        model.add_constraint("c", lower=15.0)

        prob.setup()

        failed = prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob["x"], 7.16667, 1e-6)
        assert_near_equal(prob["y"], -7.833334, 1e-6)

    def test_simple_paraboloid_equality(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0)

        prob.setup()

        failed = prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob["x"], 7.16667, 1e-4)
        assert_near_equal(prob["y"], -7.833334, 1e-4)

    def test_missing_objective(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("x", om.IndepVarComp("x", 2.0), promotes=["*"])
        model.add_subsystem("f_x", Paraboloid(), promotes=["*"])

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"

        prob.model.add_design_var("x", lower=0)

        prob.setup()

        with self.assertRaises(Exception) as raises_msg:
            prob.run_driver()

        exception = raises_msg.exception

        msg = "Driver requires objective to be declared"

        self.assertEqual(exception.args[0], msg)

    def test_simple_paraboloid_double_sided_low(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", lower=-11.0, upper=-10.0)

        prob.setup()

        failed = prob.run_driver()

        assert_near_equal(prob["y"] - prob["x"], -11.0, 1e-6)

    def test_simple_paraboloid_double_sided_high(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = x - y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", lower=10.0, upper=11.0)

        prob.setup()

        failed = prob.run_driver()

        assert_near_equal(prob["x"] - prob["y"], 11.0, 1e-6)

    def test_simple_array_comp2D(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "p1", om.IndepVarComp("widths", np.zeros((2, 2))), promotes=["*"]
        )
        model.add_subsystem("comp", TestExplCompArrayDense(), promotes=["*"])
        model.add_subsystem(
            "con",
            om.ExecComp("c = areas - 20.0", c=np.zeros((2, 2)), areas=np.zeros((2, 2))),
            promotes=["*"],
        )
        model.add_subsystem(
            "obj",
            om.ExecComp("o = areas[0, 0]", areas=np.zeros((2, 2))),
            promotes=["*"],
        )

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("widths", lower=-50.0, upper=50.0)
        model.add_objective("o")
        model.add_constraint("c", equals=0.0)

        prob.setup()

        failed = prob.run_driver()

        obj = prob["o"]
        assert_near_equal(obj, 20.0, 1e-6)

    def test_simple_array_comp2D_eq_con(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "p1", om.IndepVarComp("widths", np.zeros((2, 2))), promotes=["*"]
        )
        model.add_subsystem("comp", TestExplCompArrayDense(), promotes=["*"])
        model.add_subsystem(
            "obj",
            om.ExecComp("o = areas[0, 0] + areas[1, 1]", areas=np.zeros((2, 2))),
            promotes=["*"],
        )

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("widths", lower=-50.0, upper=50.0)
        model.add_objective("o")
        model.add_constraint("areas", equals=np.array([24.0, 21.0, 3.5, 17.5]))

        prob.setup()

        failed = prob.run_driver()

        obj = prob["o"]
        assert_near_equal(obj, 41.5, 1e-6)

    def test_simple_array_comp2D_dbl_sided_con(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "p1", om.IndepVarComp("widths", np.zeros((2, 2))), promotes=["*"]
        )
        model.add_subsystem("comp", TestExplCompArrayDense(), promotes=["*"])
        model.add_subsystem(
            "obj",
            om.ExecComp("o = areas[0, 0]", areas=np.zeros((2, 2))),
            promotes=["*"],
        )

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("widths", lower=-50.0, upper=50.0)
        model.add_objective("o")
        model.add_constraint(
            "areas",
            lower=np.array([24.0, 21.0, 3.5, 17.5]),
            upper=np.array([24.0, 21.0, 3.5, 17.5]),
        )

        prob.setup()

        failed = prob.run_driver()

        con = prob["areas"]
        assert_near_equal(con, np.array([[24.0, 21.0], [3.5, 17.5]]), 1e-6)

    def test_simple_array_comp2D_dbl_sided_con_array(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "p1", om.IndepVarComp("widths", np.zeros((2, 2))), promotes=["*"]
        )
        model.add_subsystem("comp", TestExplCompArrayDense(), promotes=["*"])
        model.add_subsystem(
            "obj",
            om.ExecComp("o = areas[0, 0]", areas=np.zeros((2, 2))),
            promotes=["*"],
        )

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("widths", lower=-50.0, upper=50.0)
        model.add_objective("o")
        model.add_constraint("areas", lower=20.0, upper=20.0)

        prob.setup()

        failed = prob.run_driver()

        obj = prob["o"]
        assert_near_equal(obj, 20.0, 1e-6)

    def test_simple_array_comp2D_array_lo_hi(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "p1", om.IndepVarComp("widths", np.zeros((2, 2))), promotes=["*"]
        )
        model.add_subsystem("comp", TestExplCompArrayDense(), promotes=["*"])
        model.add_subsystem(
            "con",
            om.ExecComp("c = areas - 20.0", c=np.zeros((2, 2)), areas=np.zeros((2, 2))),
            promotes=["*"],
        )
        model.add_subsystem(
            "obj",
            om.ExecComp("o = areas[0, 0]", areas=np.zeros((2, 2))),
            promotes=["*"],
        )

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var(
            "widths", lower=-50.0 * np.ones((2, 2)), upper=50.0 * np.ones((2, 2))
        )
        model.add_objective("o")
        model.add_constraint("c", equals=0.0)

        prob.setup()

        failed = prob.run_driver()

        obj = prob["o"]
        assert_near_equal(obj, 20.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_fwd(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = x - y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0, ref=0.02)
        model.add_design_var("y", lower=-50.0, upper=50.0, ref=0.02)
        model.add_objective("f_xy")
        model.add_constraint("c", lower=10.0, upper=11.0)

        prob.setup(check=False, mode="fwd")

        failed = prob.run_driver()

        assert_near_equal(prob["x"] - prob["y"], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_rev(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = x - y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0, ref=0.02)
        model.add_design_var("y", lower=-50.0, upper=50.0, ref=0.02)
        model.add_objective("f_xy")
        model.add_constraint("c", lower=10.0, upper=11.0)

        prob.setup(check=False, mode="rev")

        failed = prob.run_driver()

        assert_near_equal(prob["x"] - prob["y"], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_fwd(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = x - y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-8

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", lower=10.0, upper=11.0, ref=10.0)

        prob.setup(check=False, mode="fwd")

        failed = prob.run_driver()

        assert_near_equal(prob["x"] - prob["y"], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_fwd(self):

        prob = om.Problem()
        model = prob.model

        prob.set_solver_print(level=0)

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = x - y"), promotes=["*"])

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy", ref=10.0)
        model.add_constraint("c", lower=10.0, upper=11.0)

        prob.setup(check=False, mode="fwd")

        failed = prob.run_driver()

        assert_near_equal(prob["x"] - prob["y"], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_rev(self):

        prob = om.Problem()
        model = prob.model

        prob.set_solver_print(level=0)

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = x - y"), promotes=["*"])

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy", ref=10.0)
        model.add_constraint("c", lower=10.0, upper=11.0)

        prob.setup(check=False, mode="rev")

        failed = prob.run_driver()

        assert_near_equal(prob["x"] - prob["y"], 11.0, 1e-6)

    def test_sellar_mdf(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var(
            "z", lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0])
        )
        model.add_design_var("x", lower=0.0, upper=10.0)
        model.add_objective("obj")
        model.add_constraint("con1", upper=0.0)
        model.add_constraint("con2", upper=0.0)

        prob.setup(check=False, mode="rev")

        failed = prob.run_driver()

        assert_near_equal(prob["z"][0], 1.9776, 1e-3)
        assert_near_equal(prob["z"][1], 0.0, 1e-3)
        assert_near_equal(prob["x"], 0.0, 1e-3)

    def test_bug_in_eq_constraints(self):
        # We were getting extra constraints created because lower and upper are maxfloat instead of
        # None when unused.
        p = om.Problem(model=SineFitter())
        p.driver = NLoptDriver()

        p.setup()
        p.run_driver()

        max_defect = np.max(np.abs(p["defect.defect"]))
        assert_near_equal(max_defect, 0.0, 1e-10)

    def test_reraise_exception_from_callbacks(self):
        class ReducedActuatorDisc(om.ExplicitComponent):
            def setup(self):

                # Inputs
                self.add_input("a", 0.5, desc="Induced Velocity Factor")
                self.add_input(
                    "Vu",
                    10.0,
                    units="m/s",
                    desc="Freestream air velocity, upstream of rotor",
                )

                # Outputs
                self.add_output(
                    "Vd",
                    0.0,
                    units="m/s",
                    desc="Slipstream air velocity, downstream of rotor",
                )

            def compute(self, inputs, outputs):
                a = inputs["a"]
                Vu = inputs["Vu"]

                outputs["Vd"] = Vu * (1 - 2 * a)

            def compute_partials(self, inputs, J):
                Vu = inputs["Vu"]

                J["Vd", "a"] = -2.0 * Vu

        prob = om.Problem()
        indeps = prob.model.add_subsystem("indeps", om.IndepVarComp(), promotes=["*"])
        indeps.add_output("a", 0.5)
        indeps.add_output("Vu", 10.0, units="m/s")

        prob.model.add_subsystem(
            "a_disk", ReducedActuatorDisc(), promotes_inputs=["a", "Vu"]
        )

        # setup the optimization
        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"

        prob.model.add_design_var("a", lower=0.0, upper=1.0)
        # negative one so we maximize the objective
        prob.model.add_objective("a_disk.Vd", scaler=-1)

        prob.setup()

        with self.assertRaises(KeyError) as context:
            prob.run_driver()

        msg = 'Variable name pair ("Vd", "a") must first be declared.'
        self.assertTrue(msg in str(context.exception))

    def test_simple_paraboloid_upper_LN_COBYLA(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LN_COBYLA"
        prob.driver.options["tol"] = 1e-12

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob["x"], 7.16667, 1e-6)
        assert_near_equal(prob["y"], -7.833334, 1e-6)

    def test_sellar_mdf_LN_COBYLA(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LN_COBYLA"
        prob.driver.options["tol"] = 1e-12

        prob.set_solver_print(level=0)

        model.add_design_var(
            "z", lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0])
        )
        model.add_design_var("x", lower=0.0, upper=10.0)
        model.add_objective("obj")
        model.add_constraint("con1", upper=0.0)
        model.add_constraint("con2", upper=0.0)

        prob.setup(check=False, mode="rev")

        failed = prob.run_driver()

        assert_near_equal(prob["z"][0], 1.9776, 1e-3)
        assert_near_equal(prob["z"][1], 0.0, 1e-3)
        assert_near_equal(prob["x"], 0.0, 1e-3)

    def test_simple_paraboloid_lower_linear(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = x - y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", lower=15.0, linear=True)

        prob.setup()

        failed = prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob["x"], 7.16667, 1e-6)
        assert_near_equal(prob["y"], -7.833334, 1e-6)

        self.assertEqual(prob.driver._obj_and_nlcons, ["comp.f_xy"])

    def test_simple_paraboloid_equality_linear(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0, linear=True)

        prob.setup()

        failed = prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob["x"], 7.16667, 1e-6)
        assert_near_equal(prob["y"], -7.833334, 1e-6)

    def test_debug_print_option_totals(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        prob.driver.options["debug_print"] = ["totals"]

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup(check=False, mode="rev")

        failed, output = run_driver(prob)

        self.assertTrue(
            "In mode: rev, Solving variable(s) using simul coloring:" in output
        )
        self.assertTrue("('comp.f_xy', [0])" in output)
        self.assertTrue("Elapsed Time:" in output)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        prob.driver.options["debug_print"] = ["totals"]

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup(check=False, mode="fwd")

        failed, output = run_driver(prob)

        self.assertTrue(
            "In mode: fwd, Solving variable(s) using simul coloring:" in output
        )
        self.assertTrue("('p1.x', [0])" in output)
        self.assertTrue("Elapsed Time:" in output)

    def test_debug_print_option(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs"]

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup()

        failed, output = run_driver(prob)

        output = output.split("\n")

        self.assertTrue(
            output.count("Design Vars") > 1,
            "Should be more than one design vars header printed",
        )
        self.assertTrue(
            output.count("Nonlinear constraints") > 1,
            "Should be more than one nonlinear constraint header printed",
        )
        self.assertTrue(
            output.count("Linear constraints") > 1,
            "Should be more than one linear constraint header printed",
        )
        self.assertTrue(
            output.count("Objectives") > 1,
            "Should be more than one objective header printed",
        )
        self.assertTrue(
            len([s for s in output if s.startswith("{'p1.x")]) > 1,
            "Should be more than one p1.x printed",
        )
        self.assertTrue(
            len([s for s in output if "'p2.y'" in s]) > 1,
            "Should be more than one p2.y printed",
        )
        self.assertTrue(
            len([s for s in output if s.startswith("{'con.c")]) > 1,
            "Should be more than one con.c printed",
        )
        self.assertTrue(
            len([s for s in output if s.startswith("{'comp.f_xy")]) > 1,
            "Should be more than one comp.f_xy printed",
        )

    def test_sellar_mdf_linear_con_directsolver(self):
        # This test makes sure that we call solve_nonlinear first if we have any linear constraints
        # to cache.
        prob = om.Problem()
        model = prob.model = SellarDerivatives()

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-3

        model.add_design_var(
            "z", lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0])
        )
        model.add_design_var("x", lower=0.0, upper=10.0)
        model.add_objective("obj")
        model.add_constraint("con1", upper=0.0)
        model.add_constraint("con2", upper=0.0)
        model.add_constraint("x", upper=11.0, linear=True)

        prob.setup(check=False, mode="rev")
        prob.set_solver_print(level=0)

        failed = prob.run_driver()

        assert_near_equal(prob["z"][0], 1.9776, 1e-3)
        assert_near_equal(prob["z"][1], 0.0, 1e-3)
        assert_near_equal(prob["x"], 0.0, 4e-3)

        self.assertEqual(len(prob.driver._lincongrad_cache), 1)
        # Piggyback test: make sure we can run the driver again as a subdriver without a keyerror.
        prob.driver.run()
        self.assertEqual(len(prob.driver._lincongrad_cache), 1)

    def test_call_final_setup(self):
        # Make sure we call final setup if our model hasn't been setup.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0)

        prob.setup()

        with self.assertRaises(RuntimeError) as cm:
            totals = prob.check_totals(method="fd", out_stream=False)

        expected_msg = (
            "Problem: run_model must be called before total derivatives can be checked."
        )

        self.assertEqual(expected_msg, str(cm.exception))

    def test_LN_COBYLA_linear_constraint(self):
        # Bug where NLoptDriver tried to compute and cache the constraint derivatives for the
        # lower and upper bounds of the desvars even though we were using a non-gradient optimizer.
        # This causd a KeyError.
        prob = om.Problem()
        indeps = prob.model.add_subsystem("indeps", om.IndepVarComp())
        indeps.add_output("x", 3.0)
        indeps.add_output("y", -4.0)

        prob.model.add_subsystem("parab", Paraboloid())

        prob.model.add_subsystem("const", om.ExecComp("g = x + y"))

        prob.model.connect("indeps.x", ["parab.x", "const.x"])
        prob.model.connect("indeps.y", ["parab.y", "const.y"])

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LN_COBYLA"

        prob.model.add_constraint("const.g", lower=0, upper=10.0)
        prob.model.add_design_var(
            "indeps.x", **{"ref0": 0, "ref": 2, "lower": -50, "upper": 50}
        )
        prob.model.add_design_var(
            "indeps.y", **{"ref0": 0, "ref": 2, "lower": -50, "upper": 50}
        )
        prob.model.add_objective("parab.f_xy", scaler=4.0)
        prob.setup()
        prob.run_driver()

        # minimum value
        assert_near_equal(prob["parab.f_xy"], -27, 1e-6)

    def test_multiple_objectives_error(self):

        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        self.assertFalse(prob.driver.supports["multiple_objectives"])
        prob.driver.options["debug_print"] = ["nl_cons", "objs"]

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_objective("c")  # Second objective
        prob.setup()

        with self.assertRaises(RuntimeError):
            prob.run_model()

        with self.assertRaises(RuntimeError):
            prob.run_driver()

    def test_GN_DIRECT(self):
        import openmdao.api as om

        np.random.seed(6)

        size = 2  # size of the design variable

        class Rastrigin(om.ExplicitComponent):
            def setup(self):
                self.add_input("x", 0.5 * np.ones(size))
                self.add_output("f", 0.5)

            def compute(
                self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
            ):
                x = inputs["x"]
                outputs["f"] = rastrigin(x)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "indeps", om.IndepVarComp("x", np.ones(size)), promotes=["*"]
        )
        model.add_subsystem("rastrigin", Rastrigin(), promotes=["*"])

        prob.driver = driver = NLoptDriver()
        driver.options["optimizer"] = "GN_DIRECT"
        driver.options["maxiter"] = 1000

        model.add_design_var(
            "x", lower=-5.12 * np.ones(size), upper=5.12 * np.ones(size)
        )
        model.add_objective("f")
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob["x"], np.zeros(size), 1e-5)
        assert_near_equal(prob["f"], 0.0, 1e-5)

    def test_GN_DIRECT_L(self):
        import openmdao.api as om

        np.random.seed(6)

        size = 2  # size of the design variable

        class Rastrigin(om.ExplicitComponent):
            def setup(self):
                self.add_input("x", 0.5 * np.ones(size))
                self.add_output("f", 0.5)

            def compute(
                self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
            ):
                x = inputs["x"]
                outputs["f"] = rastrigin(x)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "indeps", om.IndepVarComp("x", np.ones(size)), promotes=["*"]
        )
        model.add_subsystem("rastrigin", Rastrigin(), promotes=["*"])

        prob.driver = driver = NLoptDriver()
        driver.options["optimizer"] = "GN_DIRECT_L"
        driver.options["maxiter"] = 500

        model.add_design_var(
            "x", lower=-5.12 * np.ones(size), upper=5.12 * np.ones(size)
        )
        model.add_objective("f")
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob["x"], np.zeros(size), 1e-6)
        assert_near_equal(prob["f"], 0.0, 1e-6)

    def test_GN_DIRECT_L_NOSCAL(self):
        import openmdao.api as om

        np.random.seed(6)

        size = 2  # size of the design variable

        class Rastrigin(om.ExplicitComponent):
            def setup(self):
                self.add_input("x", 0.5 * np.ones(size))
                self.add_output("f", 0.5)

            def compute(
                self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
            ):
                x = inputs["x"]
                outputs["f"] = rastrigin(x)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "indeps", om.IndepVarComp("x", np.ones(size)), promotes=["*"]
        )
        model.add_subsystem("rastrigin", Rastrigin(), promotes=["*"])

        prob.driver = driver = NLoptDriver()
        driver.options["optimizer"] = "GN_DIRECT_L_NOSCAL"
        driver.options["maxiter"] = 500

        model.add_design_var(
            "x", lower=-5.12 * np.ones(size), upper=5.12 * np.ones(size)
        )
        model.add_objective("f")
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob["x"], np.zeros(size), 1e-6)
        assert_near_equal(prob["f"], 0.0, 1e-6)

    def test_GN_ORIG_DIRECT(self):
        import openmdao.api as om

        np.random.seed(6)

        size = 2  # size of the design variable

        class Rastrigin(om.ExplicitComponent):
            def setup(self):
                self.add_input("x", 0.5 * np.ones(size))
                self.add_output("f", 0.5)

            def compute(
                self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
            ):
                x = inputs["x"]
                outputs["f"] = rastrigin(x)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "indeps", om.IndepVarComp("x", np.ones(size)), promotes=["*"]
        )
        model.add_subsystem("rastrigin", Rastrigin(), promotes=["*"])

        prob.driver = driver = NLoptDriver()
        driver.options["optimizer"] = "GN_ORIG_DIRECT"
        driver.options["maxiter"] = 1000

        model.add_design_var(
            "x", lower=-5.12 * np.ones(size), upper=5.12 * np.ones(size)
        )
        model.add_objective("f")
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob["x"], np.zeros(size), 1e-5)
        assert_near_equal(prob["f"], 0.0, 1e-5)

    def test_GN_ORIG_DIRECT_L(self):
        import openmdao.api as om

        np.random.seed(6)

        size = 2  # size of the design variable

        class Rastrigin(om.ExplicitComponent):
            def setup(self):
                self.add_input("x", 0.5 * np.ones(size))
                self.add_output("f", 0.5)

            def compute(
                self, inputs, outputs, discrete_inputs=None, discrete_outputs=None
            ):
                x = inputs["x"]
                outputs["f"] = rastrigin(x)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem(
            "indeps", om.IndepVarComp("x", np.ones(size)), promotes=["*"]
        )
        model.add_subsystem("rastrigin", Rastrigin(), promotes=["*"])

        prob.driver = driver = NLoptDriver()
        driver.options["optimizer"] = "GN_ORIG_DIRECT_L"
        driver.options["maxiter"] = 500

        model.add_design_var(
            "x", lower=-5.12 * np.ones(size), upper=5.12 * np.ones(size)
        )
        model.add_objective("f")
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob["x"], np.zeros(size), 1e-6)
        assert_near_equal(prob["f"], 0.0, 1e-6)

    def test_simple_paraboloid_upper_LD_MMA(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_MMA"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        assert_near_equal(prob["x"], 7.1666666, 1e-6)
        assert_near_equal(prob["y"], -7.83333333, 1e-6)

    def test_simple_paraboloid_upper_LD_CCSAQ(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_CCSAQ"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        assert_near_equal(prob["x"], 7.1666666, 1e-6)
        assert_near_equal(prob["y"], -7.83333333, 1e-6)

    def test_simple_paraboloid_upper_GN_ISRES(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 25.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 25.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "GN_ISRES"
        prob.driver.options["tol"] = 1e-12

        prob.driver.options["maxiter"] = 10000

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        # Just get pretty close to the optimum
        assert_near_equal(prob["x"], 7.1666666, 1e-2)
        assert_near_equal(prob["y"], -7.83333333, 1e-2)

    def test_simple_paraboloid_upper_GN_ORIG_DIRECT(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 7.5), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", -7.2), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "GN_ORIG_DIRECT"
        prob.driver.options["tol"] = 1e-6

        prob.driver.options["maxiter"] = 500

        model.add_design_var("x", lower=5.0, upper=10.0)
        model.add_design_var("y", lower=-10.0, upper=-5.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        # Just get pretty close to the optimum
        assert_near_equal(prob["x"], 7.1666666, 1e-2)
        assert_near_equal(prob["y"], -7.83333333, 1e-2)

    def test_simple_paraboloid_upper_GN_ORIG_DIRECT_L(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 7.5), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", -7.2), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "GN_ORIG_DIRECT_L"
        prob.driver.options["tol"] = 1e-6

        prob.driver.options["maxiter"] = 500

        model.add_design_var("x", lower=5.0, upper=10.0)
        model.add_design_var("y", lower=-10.0, upper=-5.0)
        model.add_objective("f_xy")
        model.add_constraint("c", upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        # Just get pretty close to the optimum
        assert_near_equal(prob["x"], 7.1666666, 1e-2)
        assert_near_equal(prob["y"], -7.83333333, 1e-2)

    def test_simple_paraboloid_equality_COBYLA(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LN_COBYLA"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0)

        prob.setup()

        failed = prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob["x"], 7.16667, 1e-4)
        assert_near_equal(prob["y"], -7.833334, 1e-4)

    def test_simple_paraboloid_equality_ISRES(self):

        prob = om.Problem()
        model = prob.model

        # Start very close to the correct answer with tight bounds to
        # reduce test runtime
        model.add_subsystem("p1", om.IndepVarComp("x", 7.5), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", -7.6), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "GN_ISRES"
        prob.driver.options["tol"] = 1e-6

        prob.driver.options["maxiter"] = 5000

        model.add_design_var("x", lower=7.0, upper=7.5)
        model.add_design_var("y", lower=-8.0, upper=-7.5)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0)

        prob.setup()

        failed = prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        # Loose tolerance
        assert_near_equal(prob["x"], 7.16667, 1e-2)
        assert_near_equal(prob["y"], -7.833334, 1e-2)

    def test_simple_paraboloid_equality_failure_MMA(self):

        prob = om.Problem()
        model = prob.model

        # Start very close to the correct answer with tight bounds to
        # reduce test runtime
        model.add_subsystem("p1", om.IndepVarComp("x", 7.5), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", -7.6), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_MMA"
        prob.driver.options["tol"] = 1e-6

        prob.driver.options["maxiter"] = 5000

        model.add_design_var("x", lower=7.0, upper=7.5)
        model.add_design_var("y", lower=-8.0, upper=-7.5)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0)

        prob.setup()

        with self.assertRaises(NotImplementedError) as raises_msg:
            failed = prob.run_driver()

        exception = raises_msg.exception

        msg = "The selected optimizer, LD_MMA, does not support equality constraints."

        self.assertIn(msg, exception.args[0])

    def test_simple_paraboloid_equality_failure_LD_CCSAQ(self):

        prob = om.Problem()
        model = prob.model

        # Start very close to the correct answer with tight bounds to
        # reduce test runtime
        model.add_subsystem("p1", om.IndepVarComp("x", 7.5), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", -7.6), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "LD_CCSAQ"
        prob.driver.options["tol"] = 1e-6

        prob.driver.options["maxiter"] = 5000

        model.add_design_var("x", lower=7.0, upper=7.5)
        model.add_design_var("y", lower=-8.0, upper=-7.5)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0)

        prob.setup()

        with self.assertRaises(NotImplementedError) as raises_msg:
            failed = prob.run_driver()

        exception = raises_msg.exception

        msg = "The selected optimizer, LD_CCSAQ, does not support equality constraints."

        self.assertIn(msg, exception.args[0])

    def test_simple_paraboloid_equality_failure_GN_ORIG_DIRECT(self):

        prob = om.Problem()
        model = prob.model

        # Start very close to the correct answer with tight bounds to
        # reduce test runtime
        model.add_subsystem("p1", om.IndepVarComp("x", 7.5), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", -7.6), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "GN_ORIG_DIRECT"
        prob.driver.options["tol"] = 1e-6

        prob.driver.options["maxiter"] = 5000

        model.add_design_var("x", lower=7.0, upper=7.5)
        model.add_design_var("y", lower=-8.0, upper=-7.5)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0)

        prob.setup()

        with self.assertRaises(NotImplementedError) as raises_msg:
            failed = prob.run_driver()

        exception = raises_msg.exception

        msg = "The selected optimizer, GN_ORIG_DIRECT, does not support equality constraints."

        self.assertIn(msg, exception.args[0])

    def test_simple_paraboloid_equality_failure_GN_ORIG_DIRECT_L(self):

        prob = om.Problem()
        model = prob.model

        # Start very close to the correct answer with tight bounds to
        # reduce test runtime
        model.add_subsystem("p1", om.IndepVarComp("x", 7.5), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", -7.6), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "GN_ORIG_DIRECT_L"
        prob.driver.options["tol"] = 1e-6

        prob.driver.options["maxiter"] = 5000

        model.add_design_var("x", lower=7.0, upper=7.5)
        model.add_design_var("y", lower=-8.0, upper=-7.5)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0)

        prob.setup()

        with self.assertRaises(NotImplementedError) as raises_msg:
            failed = prob.run_driver()

        exception = raises_msg.exception

        msg = "The selected optimizer, GN_ORIG_DIRECT_L, does not support equality constraints."

        self.assertIn(msg, exception.args[0])

    def test_simple_paraboloid_equality_failure_GN_AGS(self):

        prob = om.Problem()
        model = prob.model

        # Start very close to the correct answer with tight bounds to
        # reduce test runtime
        model.add_subsystem("p1", om.IndepVarComp("x", 7.5), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", -7.6), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "GN_AGS"
        prob.driver.options["tol"] = 1e-6

        prob.driver.options["maxiter"] = 5000

        model.add_design_var("x", lower=7.0, upper=7.5)
        model.add_design_var("y", lower=-8.0, upper=-7.5)
        model.add_objective("f_xy")
        model.add_constraint("c", equals=-15.0)

        prob.setup()

        with self.assertRaises(NotImplementedError) as raises_msg:
            failed = prob.run_driver()

        exception = raises_msg.exception

        msg = "The selected optimizer, GN_AGS, does not support equality constraints."

        self.assertIn(msg, exception.args[0])

    def test_maxtime(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver(optimizer="LD_SLSQP", tol=1e-9, maxtime=0.0001)

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")

        prob.setup()

        failed = prob.run_driver()

        # It shouldn't have time to move from the initial point
        assert_near_equal(prob["x"], 50.0, 1e-6)
        assert_near_equal(prob["y"], 50.0, 1e-6)

    def test_simple_paraboloid_inequalities(self):
        # This test checks that you can set a constraint with the same value
        # for both the lower and upper bounds to effectively create an equality
        # constraint even if the optimization method doesn't allow equality constraints

        prob = om.Problem()
        model = prob.model

        # Start very close to the correct answer with tight bounds to
        # reduce test runtime
        model.add_subsystem("p1", om.IndepVarComp("x", 7.5), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", -7.6), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])
        model.add_subsystem("con", om.ExecComp("c = - x + y"), promotes=["*"])

        prob.set_solver_print(level=0)

        prob.driver = NLoptDriver()
        prob.driver.options["optimizer"] = "GN_ORIG_DIRECT"
        prob.driver.options["tol"] = 1e-6

        prob.driver.options["maxiter"] = 5000

        model.add_design_var("x", lower=7.0, upper=7.5)
        model.add_design_var("y", lower=-8.0, upper=-7.5)
        model.add_objective("f_xy")
        model.add_constraint("c", lower=-15.0, upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        # Loose tolerance
        assert_near_equal(prob["x"], 7.16667, 1e-2)
        assert_near_equal(prob["y"], -7.833334, 1e-2)


@unittest.skipIf(nlopt is None, "only run if NLopt is installed.")
class TestNLoptDriverFeatures(unittest.TestCase):
    def test_feature_basic(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])

        prob.driver = om.NLoptDriver()
        prob.driver.options["optimizer"] = "LD_SLSQP"
        prob.driver.options["tol"] = 1e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob["x"], 6.66666667, 1e-6)
        assert_near_equal(prob["y"], -7.3333333, 1e-6)

    def test_feature_optimizer(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])

        prob.driver = om.NLoptDriver(optimizer="LD_SLSQP")

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob["x"], 6.66666667, 1e-6)
        assert_near_equal(prob["y"], -7.3333333, 1e-6)

    def test_feature_maxiter(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])

        prob.driver = om.NLoptDriver()
        prob.driver.options["maxiter"] = 20

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob["x"], 6.66666667, 1e-6)
        assert_near_equal(prob["y"], -7.3333333, 1e-6)

    def test_feature_tol(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()
        model = prob.model

        model.add_subsystem("p1", om.IndepVarComp("x", 50.0), promotes=["*"])
        model.add_subsystem("p2", om.IndepVarComp("y", 50.0), promotes=["*"])
        model.add_subsystem("comp", Paraboloid(), promotes=["*"])

        prob.driver = om.NLoptDriver()
        prob.driver.options["tol"] = 1.0e-9

        model.add_design_var("x", lower=-50.0, upper=50.0)
        model.add_design_var("y", lower=-50.0, upper=50.0)
        model.add_objective("f_xy")

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob["x"], 6.66666667, 1e-6)
        assert_near_equal(prob["y"], -7.3333333, 1e-6)
        
@unittest.skipUnless(nlopt is None, "only run if NLopt is NOT installed.")
class TestNotInstalled(unittest.TestCase):

    def test_nlopt_not_installed(self):
        # the import should not fail
        import nlopt

        # but we get a RuntimeError if we try to instantiate
        with self.assertRaises(RuntimeError) as ctx:
            NLoptDriver()

        self.assertEqual(str(ctx.exception),
                         'NLoptDriver is not available, NLopt is not installed.')


if __name__ == "__main__":
    unittest.main()

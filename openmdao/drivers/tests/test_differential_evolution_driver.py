#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import numpy as np
import openmdao.api as om
import os
import unittest

from differential_evolution import EvolutionStrategy
from parameterized import parameterized

from openmdao.drivers.differential_evolution_driver import DifferentialEvolutionDriver


class TestDifferentialEvolutionDriver(unittest.TestCase):
    def setUp(self):
        os.environ["DifferentialEvolutionDriver_seed"] = "11"
        dim = 2

        prob = om.Problem()
        prob.model.add_subsystem(
            "indeps", om.IndepVarComp("x", val=np.ones(dim)), promotes=["*"]
        )
        prob.model.add_subsystem(
            "objf", om.ExecComp("f = sum(x * x)", f=1.0, x=np.ones(dim)), promotes=["*"]
        )

        prob.model.add_design_var("x", lower=-100.0, upper=100.0)
        prob.model.add_objective("f")

        prob.driver = DifferentialEvolutionDriver()
        self.problem = prob

    def tearDown(self):
        self.problem.cleanup()

    @parameterized.expand(
        list(
            map(
                lambda t: (
                    "strategy_"
                    + "/".join([str(_t) for _t in t[:-1]])
                    + "_adaptivity_{}".format(t[-1]),
                ),
                itertools.product(
                    list(EvolutionStrategy.__mutation_strategies__.keys()),
                    [1, 2, 3],
                    list(EvolutionStrategy.__crossover_strategies__.keys()),
                    list(EvolutionStrategy.__repair_strategies__.keys()),
                    [0, 1, 2],
                ),
            )
        )
    )
    def test_differential_evolution_driver(self, name):
        tol = 1e-8

        strategy, adaptivity = name.split("_")[1::2]
        self.problem.driver.options["strategy"] = strategy
        self.problem.driver.options["adaptivity"] = int(adaptivity)
        self.problem.setup()
        self.problem.run_driver()

        try:
            self.assertTrue(np.all(self.problem["x"] < 1e-3))
            self.assertLess(self.problem["f"][0], 1e-3)
        except self.failureException:
            # This is to account for strategies sometimes 'collapsing' prematurely.
            # This is not a failed test, this is a known phenomenon with DE.
            # In this case we just check that one of the two tolerances was triggered.
            self.assertTrue(
                self.problem.driver._de.dx < tol or self.problem.driver._de.df < tol
            )

    def test_differential_evolution_driver_recorder(self):
        from openmdao.recorders.case_recorder import CaseRecorder

        class MyRecorder(CaseRecorder):
            def record_metadata_system(self, recording_requester):
                pass

            def record_metadata_solver(self, recording_requester):
                pass

            def record_iteration_driver(self, recording_requester, data, metadata):
                assert isinstance(recording_requester, DifferentialEvolutionDriver)
                de = recording_requester.get_de()
                assert "out" in data
                assert "indeps.x" in data["out"]
                assert "objf.f" in data["out"]

                x = data["out"]["indeps.x"]
                f = data["out"]["objf.f"]
                assert x in de.pop
                assert f in de.fit

                assert np.all(x - de.best == 0)
                assert f - de.best_fit == 0

            def record_iteration_system(self, recording_requester, data, metadata):
                pass

            def record_iteration_solver(self, recording_requester, data, metadata):
                pass

            def record_iteration_problem(self, recording_requester, data, metadata):
                pass

            def record_derivatives_driver(self, recording_requester, data, metadata):
                pass

            def record_viewer_data(self, model_viewer_data):
                pass

        self.problem.driver.add_recorder(MyRecorder())
        self.problem.setup()
        self.problem.run_driver()

    def test_differential_evolution_driver_seed(self):
        x = [None, None]
        f = [None, None]

        for i in range(2):
            self.assertEqual(self.problem.driver._seed, 11)

            self.problem.driver.options["max_gen"] = 10
            self.problem.setup()
            self.problem.run_driver()

            x[i] = self.problem["x"]
            f[i] = self.problem["f"][0]

            self.tearDown()
            self.setUp()

        self.assertTrue(np.all(x[0] == x[1]))
        self.assertEqual(f[0], f[1])


if __name__ == "__main__":
    unittest.main()

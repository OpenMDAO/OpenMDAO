"""
Benchmarks the parallel scaling for the GA driver.
"""
from time import time
import unittest

import openmdao.api as om
from openmdao.test_suite.components.exec_comp_for_test import ExecComp4Test


# Increase to simulate a slower model
DELAY = 0.05

MAXGEN = 5
POPSIZE = 40


class GAGroup(om.Group):

    def setup(self):

        self.add_subsystem('p1', om.IndepVarComp('x', 1.0))
        self.add_subsystem('p2', om.IndepVarComp('y', 1.0))
        self.add_subsystem('p3', om.IndepVarComp('z', 1.0))

        self.add_subsystem('comp', ExecComp4Test(['f = x + y + z'], nl_delay=DELAY))

        self.add_design_var('p1.x', lower=-100, upper=100)
        self.add_design_var('p2.y', lower=-100, upper=100)
        self.add_design_var('p3.z', lower=-100, upper=100)
        self.add_objective('comp.f')


class BenchParGA1(unittest.TestCase):

    N_PROCS = 1

    def benchmark_genetic_1(self):

        prob = om.Problem()
        prob.model = GAGroup()

        driver = prob.driver = om.SimpleGADriver()
        driver.options['max_gen'] = MAXGEN
        driver.options['pop_size'] = POPSIZE
        driver.options['run_parallel'] = True

        prob.setup()

        t0 = time()
        prob.run_driver()
        print('Elapsed Time', time() - t0)


class BenchParGA2(unittest.TestCase):

    N_PROCS = 2

    def benchmark_genetic_2(self):

        prob = om.Problem()
        prob.model = GAGroup()

        driver = prob.driver = om.SimpleGADriver()
        driver.options['max_gen'] = MAXGEN
        driver.options['pop_size'] = POPSIZE
        driver.options['run_parallel'] = True

        prob.setup()

        t0 = time()
        prob.run_driver()
        print('Elapsed Time', time() - t0)


class BenchParGA4(unittest.TestCase):

    N_PROCS = 4

    def benchmark_genetic_4(self):

        prob = om.Problem()
        prob.model = GAGroup()

        driver = prob.driver = om.SimpleGADriver()
        driver.options['max_gen'] = MAXGEN
        driver.options['pop_size'] = POPSIZE
        driver.options['run_parallel'] = True

        prob.setup()

        t0 = time()
        prob.run_driver()
        print('Elapsed Time', time() - t0)

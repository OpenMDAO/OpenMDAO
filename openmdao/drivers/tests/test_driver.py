"""
Test DOE Driver and Generators.
"""
from __future__ import print_function, division

import unittest

import os
import shutil
import tempfile
import csv
import json

import numpy as np

from openmdao.api import Problem, ExplicitComponent, IndepVarComp, ExecComp, \
    SqliteRecorder, CaseReader, PETScVector

from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.doe_generators import ListGenerator, CSVGenerator, \
    UniformGenerator, FullFactorialGenerator, PlackettBurmanGenerator, \
    BoxBehnkenGenerator, LatinHypercubeGenerator

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.groups.parallel_groups import FanInGrouped

from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.general_utils import run_driver, printoptions

from openmdao.utils.mpi import MPI

from openmdao.utils.general_utils import warn_deprecation, simple_warning, make_serializable

class ParaboloidArray(ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + x*y + (y+4)^2 - 3.

    Where x and y are xy[0] and xy[1] repectively.
    """

    def __init__(self):
        super(ParaboloidArray, self).__init__()

        self.add_input('xy', val=np.array([0., 0.]))
        self.add_output('f_xy', val=0.0)

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        """
        x = inputs['xy'][0]
        y = inputs['xy'][1]
        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

class TestDOEDriver(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='TestDOEDriver-')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass


    def test_list(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.setup()

        # create a list of DOE cases
        case_gen = FullFactorialGenerator(levels=3)
        cases = list(case_gen(model.get_design_vars(recurse=True)))

        # create DOEDriver using provided list of cases
        prob.driver = DOEDriver(cases)
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))
        opts = prob.driver.options
        dopts = {k: opts[k] for k in opts}

        prob.run_driver()
        prob.cleanup()

        expected = {
            0: {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            1: {'x': np.array([.5]), 'y': np.array([0.]), 'f_xy': np.array([19.25])},
            2: {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},

            3: {'x': np.array([0.]), 'y': np.array([.5]), 'f_xy': np.array([26.25])},
            4: {'x': np.array([.5]), 'y': np.array([.5]), 'f_xy': np.array([23.75])},
            5: {'x': np.array([1.]), 'y': np.array([.5]), 'f_xy': np.array([21.75])},

            6: {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            7: {'x': np.array([.5]), 'y': np.array([1.]), 'f_xy': np.array([28.75])},
            8: {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        }

        cr = CaseReader("cases.sql")
        cases = cr.list_cases('driver')

        self.assertEqual(len(cases), 9)

        for n in range(len(cases)):
            outputs = cr.get_case(cases[n]).outputs
            self.assertEqual(outputs['x'], expected[n]['x'])
            self.assertEqual(outputs['y'], expected[n]['y'])
            self.assertEqual(outputs['f_xy'], expected[n]['f_xy'])


if __name__ == "__main__":
    unittest.main()

""" Unit tests for the problem interface."""

import sys
import unittest
import warnings
from six import assertRaisesRegex, StringIO, assertRegex

import numpy as np

from openmdao.core.driver import Driver
from openmdao.api import Problem, IndepVarComp, NonlinearBlockGS, ScipyOptimizeDriver, \
    ExecComp, Group, NewtonSolver, ImplicitComponent, ScipyKrylov, ExplicitComponent, ParallelGroup
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives

from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.core.tests.test_discrete import ModCompEx, ModCompIm, DiscretePromTestCase, PathCompEx


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class DiscreteMPITestCase(unittest.TestCase):

    N_PROCS = 2

    def test_simple_run_once_discrete(self):
        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_discrete_output('x', 37)
        par = model.add_subsystem('par', ParallelGroup())
        par.add_subsystem('comp1', ModCompEx(3))
        par.add_subsystem('comp2', ModCompEx(7))

        model.connect('indep.x', 'par.comp1.x')
        model.connect('indep.x', 'par.comp2.x')

        prob.setup()
        prob.run_model()

        if prob.comm.rank == 0:
            assert_rel_error(self, prob['par.comp1.y'], 1)
        else:
            assert_rel_error(self, prob['par.comp2.y'], 2)

    def test_simple_run_once_discrete_implicit(self):
        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_discrete_output('x', 37)
        par = model.add_subsystem('par', ParallelGroup())
        par.add_subsystem('comp1', ModCompIm(3))
        par.add_subsystem('comp2', ModCompIm(7))

        model.connect('indep.x', 'par.comp1.x')
        model.connect('indep.x', 'par.comp2.x')

        prob.setup()
        prob.run_model()

        if prob.comm.rank == 0:
            assert_rel_error(self, prob['par.comp1.y'], 1)
        else:
            assert_rel_error(self, prob['par.comp2.y'], 2)


# This re-runs all of the DiscretePromTestCase tests under MPI

@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class DiscretePromMPITestCase(DiscretePromTestCase):
    N_PROCS = 2


if __name__ == "__main__":
    unittest.main()

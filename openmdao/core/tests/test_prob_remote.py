import unittest

import numpy as np

from openmdao.api import Problem, ExecComp, Group, ParallelGroup

from openmdao.utils.mpi import MPI


if MPI:
    try:
        from openmdao.vectors.petsc_vector import PETScVector
    except ImportError:
        PETScVector = None

@unittest.skipUnless(MPI and PETScVector, "only run with MPI and PETSc.")
class ProbRemoteTestCase(unittest.TestCase):

    N_PROCS = 2

    def test_remote_var_access(self):
        # build the model
        prob = Problem()

        group = prob.model.add_subsystem('group', ParallelGroup())

        comp = ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3', y=2.0)
        group.add_subsystem('comp1', comp)

        comp = ExecComp('g = x*y', y=2.0)
        group.add_subsystem('comp2', comp)

        prob.setup()

        prob['group.comp1.x'] = 4.
        prob['group.comp2.x'] = 5.

        prob.run_model()

        np.testing.assert_almost_equal(prob['group.comp1.f'], 42., decimal=5)
        np.testing.assert_almost_equal(prob['group.comp2.g'], 10., decimal=5)

    def test_remote_var_access_prom(self):
        # build the model
        prob = Problem()

        group = prob.model.add_subsystem('group', ParallelGroup(), promotes=['f', 'g'])

        group.add_subsystem('comp1', ExecComp('f = 3.0*x**2 + 5', x=1.0), promotes=['x', 'f'])
        group.add_subsystem('comp2', ExecComp('g = 7.0*x', x=1.0), promotes=['x', 'g'])

        prob.model.add_subsystem('summ', ExecComp('z = f + g'), promotes=['f', 'g'])
        prob.model.add_subsystem('prod', ExecComp('z = f * g'), promotes=['f', 'g'])

        prob.setup()

        # both of these values will get overwritten, but the error only happens when you
        # set an output, so...
        prob['f'] = 4.
        prob['g'] = 5.

        prob.run_model()

        np.testing.assert_almost_equal(prob['summ.z'], 15., decimal=5)
        np.testing.assert_almost_equal(prob['prod.z'], 56., decimal=5)


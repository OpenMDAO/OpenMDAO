
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@require_pyoptsparse(optimizer='SLSQP')
@use_tempdirs
class MPISetvalBug(unittest.TestCase):
    N_PROCS = 2

    def _build_model(self):
        p = om.Problem()
        par_group = om.ParallelGroup()

        c1 = om.ExecComp('y1 = x1 ** 2', x1={'shape': (1,)}, y1={'copy_shape': 'x1'})
        g1 = om.Group()
        g1.add_subsystem('c1', c1, promotes=['*'])

        c2 = om.ExecComp('g2 = x2', x2={'shape': (1,)}, g2={'copy_shape': 'x2'})
        g2 = om.Group()
        g2.add_subsystem('c2', c2, promotes=['*'])

        par_group.add_subsystem('g1', g1, promotes=['*'])
        par_group.add_subsystem('g2', g2, promotes=['*'])

        p.model.add_objective('y1')
        p.model.add_design_var('x1', lower=2, upper=5)
        p.model.add_design_var('x2', lower=2, upper=5)
        p.model.add_constraint('g2', lower=3.)

        p.model.add_subsystem('par_group', par_group, promotes=['*'])

        p.driver = om.pyOptSparseDriver(optimizer='SLSQP')

        return p

    def test_set_val_mpi_bug_post_setup(self):
        p = self._build_model()
        p.setup()

        p.model.par_group.g1.set_val('x1', 2.5)
        p.model.par_group.g2.set_val('x2', 2.6)

        p.final_setup()

        assert_near_equal(p.model.get_val(p.model.get_source('x1')), 2.5)
        assert_near_equal(p.model.get_val(p.model.get_source('x2')), 2.6)

    def test_set_val_mpi_bug_post_setup_set_single_proc(self):
        p = self._build_model()
        p.setup()

        if p.model.par_group.g1._is_local:
            p.model.par_group.g1.set_val('x1', 2.5)
        if p.model.par_group.g2._is_local:
            p.model.par_group.g2.set_val('x2', 2.6)

        p.final_setup()

        assert_near_equal(p.model.get_val(p.model.get_source('x1')), 2.5)
        assert_near_equal(p.model.get_val(p.model.get_source('x2')), 2.6)

    def test_set_val_mpi_bug_post_final_setup(self):
        p = self._build_model()
        p.setup()

        p.final_setup()

        p.model.par_group.g1.set_val('x1', 2.5)
        p.model.par_group.g2.set_val('x2', 2.6)

        p.run_model()

        assert_near_equal(p.model.get_val(p.model.get_source('x1')), 2.5)
        assert_near_equal(p.model.get_val(p.model.get_source('x2')), 2.6)

    def test_set_val_mpi_bug_post_final_setup_set_single_proc(self):
        p = self._build_model()
        p.setup()

        p.final_setup()

        if p.model.par_group.g1._is_local:
            p.model.par_group.g1.set_val('x1', 2.5)
        if p.model.par_group.g2._is_local:
            p.model.par_group.g2.set_val('x2', 2.6)

        p.run_model()  # source values attached to remote inputs won't update until the final_setup
                       # called within run_model

        assert_near_equal(p.model.get_val(p.model.get_source('x1')), 2.5)
        assert_near_equal(p.model.get_val(p.model.get_source('x2')), 2.6)

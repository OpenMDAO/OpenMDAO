
import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI
from openmdao.test_suite.components.expl_comp_array import TestExplCompArraySparse, TestExplCompArrayJacVec

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class ExplCompArraySparseCounted(TestExplCompArraySparse):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ncomputes = 0
        self.ncompute_partials = 0

    def compute(self, inputs, outputs):
        super().compute(inputs, outputs)
        self.ncomputes += 1

    def compute_partials(self, inputs, partials):
        super().compute_partials(inputs, partials)
        self.ncompute_partials += 1


class ExplCompArrayJacVecCounted(TestExplCompArrayJacVec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ncomputes = 0
        self.njacvec_products = 0

    def compute(self, inputs, outputs):
        super().compute(inputs, outputs)
        self.ncomputes += 1

    def compute_jacvec_product(self, inputs, dinputs, result, mode):
        super().compute_jacvec_product(inputs, outputs)
        self.njacvec_products += 1


@unittest.skipUnless(MPI is not None and PETScVector is not None, "MPI and PETSc are required.")
class TestSimpleSerialReplication(unittest.TestCase):
    N_PROCS = 3

    def test_serial_replication_False(self):
        # run_root_only is False
        p = om.Problem()
        p.model.add_subsystem('C1', ExplCompArraySparseCounted(), promotes_outputs=['areas', 'total_volume'])
        p.model.add_subsystem('C2', om.ExecComp('y = total_volume * 2.5'), promotes_inputs=['total_volume'])
        p.model.add_subsystem('C3', om.ExecComp('y = areas * 1.5', areas=np.zeros((2,2)), y=np.zeros((2,2))), promotes_inputs=['areas'])

        p.setup()

        p.set_val('C1.widths', np.ones(4) * 3.)

        p.run_model()

        self.assertEqual(p.model.C1.ncomputes, 1)

        np.testing.assert_allclose(p.get_val('C2.y'), 30.)
        np.testing.assert_allclose(p.get_val('C3.y'), np.ones((2,2)) * 4.5)

        J = p.compute_totals(of=['C2.y', 'C3.y'], wrt=['C1.lengths', 'C1.widths'])

        self.assertEqual(p.model.C1.ncompute_partials, 1)

        np.testing.assert_allclose(J['C2.y', 'C1.lengths'], [np.ones(4) * 7.5])
        np.testing.assert_allclose(J['C2.y', 'C1.widths'], [np.ones(4) * 2.5])
        np.testing.assert_allclose(J['C3.y', 'C1.lengths'], np.eye(4) * 4.5)
        np.testing.assert_allclose(J['C3.y', 'C1.widths'], np.eye(4) * 1.5)


    def test_serial_replication_True(self):
        # run_root_only is True
        p = om.Problem()
        p.model.add_subsystem('C1', ExplCompArraySparseCounted(run_root_only=True),
                              promotes_outputs=['areas', 'total_volume'])
        p.model.add_subsystem('C2', om.ExecComp('y = total_volume * 2.5'), promotes_inputs=['total_volume'])
        p.model.add_subsystem('C3', om.ExecComp('y = areas * 1.5', areas=np.zeros((2,2)), y=np.zeros((2,2))),
                              promotes_inputs=['areas'])

        p.setup()

        p.set_val('C1.widths', np.ones(4) * 3.)

        p.run_model()

        if p.comm.rank == 0:
            self.assertEqual(p.model.C1.ncomputes, 1)
        else:
            self.assertEqual(p.model.C1.ncomputes, 0)

        np.testing.assert_allclose(p.get_val('C2.y'), 30.)
        np.testing.assert_allclose(p.get_val('C3.y'), np.ones((2,2)) * 4.5)

        J = p.compute_totals(of=['C2.y', 'C3.y'], wrt=['C1.lengths', 'C1.widths'])

        if p.comm.rank == 0:
            self.assertEqual(p.model.C1.ncompute_partials, 1)
        else:
            self.assertEqual(p.model.C1.ncompute_partials, 0)

        np.testing.assert_allclose(J['C2.y', 'C1.lengths'], [np.ones(4) * 7.5])
        np.testing.assert_allclose(J['C2.y', 'C1.widths'], [np.ones(4) * 2.5])
        np.testing.assert_allclose(J['C3.y', 'C1.lengths'], np.eye(4) * 4.5)
        np.testing.assert_allclose(J['C3.y', 'C1.widths'], np.eye(4) * 1.5)

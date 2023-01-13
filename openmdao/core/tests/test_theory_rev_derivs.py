import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI


try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals


class I1O1Comp(om.ExplicitComponent):
    def __init__(self, mult=1.0, idist=False, odist=False, **kwargs):
        self.idist = idist
        self.odist = odist
        self.mult = mult
        super().__init__(**kwargs)

    def setup(self):
        self.add_input("x", distributed=self.idist, val=1.0)
        self.add_output("y", distributed=self.odist, val=1.0)

    def compute(self, inputs, outputs):
        outputs['y'] = self.mult * inputs['x']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'y' in d_outputs:
                if 'x' in d_inputs:
                    d_outputs['y'] += self.mult * d_inputs['x']
        else:  # rev
            if 'y' in d_outputs:
                if 'x' in d_inputs:
                    d_inputs['x'] += self.mult * d_outputs['y']


class I2O1Comp(om.ExplicitComponent):
    def __init__(self, mult1=1.0, mult2=1.0, idist1=False, idist2=False, odist=False, **kwargs):
        self.idist1 = idist1
        self.idist2 = idist2
        self.odist = odist
        self.mult1 = mult1
        self.mult2 = mult2
        super().__init__(**kwargs)

    def setup(self):
        self.add_input("x1", val=1.0, distributed=self.idist1)
        self.add_input("x2", val=1.0, distributed=self.idist2)
        self.add_output("y", val=1.0, distributed=self.odist)

    def compute(self, inputs, outputs):
        outputs['y'] = self.mult1 * inputs['x1'] + self.mult2 * inputs['x2']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'y' in d_outputs:
                if 'x1' in d_inputs:
                    d_outputs['y'] += self.mult1 * d_inputs['x1']
                if 'x2' in d_inputs:
                    d_outputs['y'] += self.mult2 * d_inputs['x2']
        else:  # rev
            if 'y' in d_outputs:
                if 'x1' in d_inputs:
                    d_inputs['x1'] += self.mult1 * d_outputs['y']
                if 'x2' in d_inputs:
                    d_inputs['x2'] += self.mult2 * d_outputs['y']



@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestTheoryDocExample(unittest.TestCase):

    N_PROCS = 2

    # this mimics the "Reverse-mode Derivatives in Parallel Subsystems" example in the theory manual
    def test_theory_example(self):
        p = om.Problem()
        model = p.model
        model.add_subsystem('C1', I1O1Comp(mult=2.0))
        sub = model.add_subsystem('par', om.ParallelGroup())
        sub.add_subsystem('C2', I1O1Comp(mult=2.0))
        sub.add_subsystem('C3', I1O1Comp(mult=3.0))
        model.add_subsystem('C4', I2O1Comp(mult1=2.0, mult2=3.0))

        model.connect('C1.y', 'par.C2.x')
        model.connect('C1.y', 'par.C3.x')
        model.connect('par.C2.y', 'C4.x1')
        model.connect('par.C3.y', 'C4.x2')

        model.add_design_var('C1.x')
        model.add_objective('C4.y')

        p.setup(mode='rev')
        p.run_model()

        assert_check_partials(p.check_partials(out_stream=None))

        data = p.check_totals(of='C4.y', wrt='C1.x', out_stream=None)
        assert_check_totals(data)

        model._dinputs.set_val(0.)
        model._doutputs.set_val(0.)
        model._doutputs['C4.y'] = -1.0

        model.linear_solver.solve('rev')

        all_dinputs = model.comm.allgather(model._dinputs.asarray())

        assert_near_equal(all_dinputs[0], np.array([16.,  8.,  2.,  3.]))
        assert_near_equal(all_dinputs[1], np.array([36., 18.,  2.,  3.]))


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()


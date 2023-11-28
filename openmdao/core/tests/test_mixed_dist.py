import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_check_totals


if MPI:
    try:
        from openmdao.vectors.petsc_vector import PETScVector
    except ImportError:
        PETScVector = None

    dist_shape = 1 if MPI.COMM_WORLD.rank > 0 else 2
else:
    dist_shape = 2


class SerialComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x')
        self.add_output('y')

    def compute(self, inputs, outputs):
        outputs['y'] = 2.0* inputs['x']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'y' in d_outputs:
                if 'x' in d_inputs:
                    d_outputs['y'] += 2.0 * d_inputs['x']
        if mode == 'rev':
            if 'y' in d_outputs:
                if 'x' in d_inputs:
                    d_inputs['x'] += 2.0 * d_outputs['y']


class MixedSerialInComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x')
        self.add_output('yd', shape = dist_shape, distributed=True)

    def compute(self, inputs, outputs):
        outputs['yd'][:] = 0.5 * inputs['x']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'yd' in d_outputs:
                if 'x' in d_inputs:
                    d_outputs['yd'] += 0.5 * d_inputs['x']
        if mode == 'rev':
            if 'yd' in d_outputs:
                if 'x' in d_inputs:
                    d_inputs['x'] += 0.5 * self.comm.allreduce(np.sum(d_outputs['yd']))


class MixedSerialOutComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x')
        self.add_input('xd', shape = dist_shape, distributed=True)
        self.add_output('y')

    def compute(self, inputs, outputs):
        outputs['y'] = 2.0 * inputs['x'] + self.comm.allreduce(3.0 * np.sum(inputs['xd']))

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'y' in d_outputs:
                if 'x' in d_inputs:
                    d_outputs['y'] += 2.0 * d_inputs['x']
                if 'xd' in d_inputs:
                    d_outputs['y'] += 3.0 * self.comm.allreduce(np.sum(d_inputs['xd']))
        if mode == 'rev':
            if 'y' in d_outputs:
                if 'x' in d_inputs:
                    d_inputs['x'] += 2.0 * d_outputs['y']
                if 'xd' in d_inputs:
                    d_inputs['xd'] += 3.0 * d_outputs['y']


class DistComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('xd', shape = dist_shape, distributed=True)
        self.add_output('yd', shape = dist_shape, distributed=True)

    def compute(self, inputs, outputs):
        outputs['yd'] = 3.0 * inputs['xd']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'yd' in d_outputs:
                if 'xd' in d_inputs:
                    d_outputs['yd'] += 3.0 * d_inputs['xd']
        if mode == 'rev':
            if 'yd' in d_outputs:
                if 'xd' in d_inputs:
                    d_inputs['xd'] += 3.0 * d_outputs['yd']


def create_problem():
    prob = om.Problem()
    model = prob.model
    model.add_subsystem('ivc', om.IndepVarComp('x', val = 1.0))

    model.add_subsystem('S', SerialComp())  # x -> y
    model.add_subsystem('MI', MixedSerialInComp()) # x -> yd
    model.add_subsystem('D', DistComp()) # xd -> yd
    model.add_subsystem('MO', MixedSerialOutComp()) # x, xd -> y
    model.connect('ivc.x', 'S.x')
    model.connect('S.y','MI.x')
    model.connect('MI.yd', 'D.xd')
    model.connect('D.yd', 'MO.xd')
    model.connect('S.y', 'MO.x')
    model.add_design_var("ivc.x")
    model.add_objective("MO.xd", index=0)  # adding objective using input name
    return prob


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestMixedDist2(unittest.TestCase):
    N_PROCS = 2

    def test_mixed_dist(self):
        prob = create_problem()
        prob.setup(mode='rev')
        prob.run_model()
        assert_check_totals(prob.check_totals())


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestMixedDist3(TestMixedDist2):
    N_PROCS = 3


if __name__ == '__main__':
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()

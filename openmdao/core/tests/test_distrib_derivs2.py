import numpy as np
import openmdao.api as om
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

import unittest

from openmdao.utils.assert_utils import assert_check_partials, assert_check_totals
from openmdao.utils.array_utils import evenly_distrib_idxs


class BaseTestComp(om.ExplicitComponent):
    def __init__(self, arr_size, dist_size, dist_slice, **kwargs):
        super().__init__(**kwargs)
        self.arr_size = arr_size
        self.dist_size = dist_size
        self.dist_slice = dist_slice


class SerialInSerialOutComp(BaseTestComp):
    def setup(self):
        self.add_input('sin', shape=self.arr_size)
        self.add_output('sout', shape=self.arr_size)

    def compute(self, inputs, outputs):
        outputs['sout'] = 2.0 * inputs['sin']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'sout' in d_outputs:
                if 'sin' in d_inputs:
                    d_outputs['sout'] += 2.0 * d_inputs['sin']
        else:  # rev
            if 'sout' in d_outputs:
                if 'sin' in d_inputs:
                    d_inputs['sin'] += 2.0 * d_outputs['sout']


class SerialInDistOutComp(BaseTestComp):
    def setup(self):
        self.add_input('sin', shape=self.arr_size)
        self.add_output('dout', shape=self.dist_size, distributed=True)
        self.sizes, self.offsets = evenly_distrib_idxs(self.comm.size, self.arr_size)

    def compute(self, inputs, outputs):
        if self.arr_size == 1:
            outputs['dout'][:] = 0.5 * inputs['sin']
        else:
            outputs['dout'] = 0.5 * inputs['sin'][self.dist_slice]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'dout' in d_outputs:
                if 'sin' in d_inputs:
                    if self.arr_size == 1:
                        d_outputs['dout'][:] += 0.5 * d_inputs['sin']
                    else:
                        d_outputs['dout'] += 0.5 * d_inputs['sin'][self.dist_slice]
        else:  # rev
            if 'dout' in d_outputs:
                if 'sin' in d_inputs:
                    if self.arr_size == 1:
                        d_inputs['sin'] += 0.5 * self.comm.allreduce(np.sum(d_outputs['dout']))
                    else:
                        loc_val = np.ascontiguousarray(d_outputs['dout'])
                        val = np.zeros(np.sum(self.sizes))
                        self.comm.Allgatherv(loc_val, [val, self.sizes, self.offsets, MPI.DOUBLE])
                        d_inputs['sin'] += 0.5 * val


class MixedInSerialOutComp(BaseTestComp):
    def setup(self):
        self.add_input('sin', shape=self.arr_size)
        self.add_input('din', shape=self.dist_size, distributed=True)
        self.add_output('sout')

    def compute(self, inputs, outputs):
        outputs['sout'] = 2.0 * np.sum(inputs['sin']) + self.comm.allreduce(3.0 * np.sum(inputs['din']))

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'sout' in d_outputs:
                if 'sin' in d_inputs:
                    d_outputs['sout'] += 2.0 * np.sum(d_inputs['sin'])
                if 'din' in d_inputs:
                    d_outputs['sout'] += 3.0 * self.comm.allreduce(np.sum(d_inputs['din']))
        else:  # rev
            if 'sout' in d_outputs:
                if 'sin' in d_inputs:
                    d_inputs['sin'] += 2.0 * np.sum(d_outputs['sout'])
                if 'din' in d_inputs:
                    d_inputs['din'] += 3.0 * d_outputs['sout']


class DistInDistOutComp(BaseTestComp):
    def setup(self):
        self.add_input('din', shape = self.dist_size, distributed=True)
        self.add_output('dout', shape = self.dist_size, distributed=True)

    def compute(self, inputs, outputs):
        outputs['dout'] = 3.0 * inputs['din']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'dout' in d_outputs:
                if 'din' in d_inputs:
                    d_outputs['dout'] += 3.0 * d_inputs['din']
        if mode == 'rev':
            if 'dout' in d_outputs:
                if 'din' in d_inputs:
                    d_inputs['din'] += 3.0 * d_outputs['dout']


def create_problem(mode, arr_size, dist_size, dist_slice):
    p = om.Problem()
    model = p.model

    model.add_subsystem('ivc', om.IndepVarComp('dv', val = np.arange(arr_size, dtype=float) + 1.))
    model.add_subsystem('sin_sout_comp', SerialInSerialOutComp(arr_size, dist_size, dist_slice))
    model.add_subsystem('sin_dout_comp', SerialInDistOutComp(arr_size, dist_size, dist_slice))
    model.add_subsystem('din_dout_comp', DistInDistOutComp(arr_size, dist_size, dist_slice))
    model.add_subsystem('mixed_in_sout_comp', MixedInSerialOutComp(arr_size, dist_size, dist_slice))

    model.connect('ivc.dv', 'sin_sout_comp.sin')
    model.connect('sin_sout_comp.sout',['sin_dout_comp.sin', 'mixed_in_sout_comp.sin'])
    model.connect('sin_dout_comp.dout', 'din_dout_comp.din')
    model.connect('din_dout_comp.dout', 'mixed_in_sout_comp.din')

    p.setup(mode=mode, force_alloc_complex=True)

    return p


def create_mixed_only_problem(mode, arr_size, dist_size, dist_slice):
    p = om.Problem()
    model = p.model
    ivc = model.add_subsystem('ivc', om.IndepVarComp())
    ivc.add_output('xs', val = np.arange(arr_size, dtype=float) + 1.)
    ivc.add_output('xd', val = np.arange(dist_size, dtype=float) + 1., distributed=True)

    model.add_subsystem('mixed_in_sout_comp', MixedInSerialOutComp(arr_size, dist_size, dist_slice))

    model.connect('ivc.xs', 'mixed_in_sout_comp.sin')
    model.connect('ivc.xd', 'mixed_in_sout_comp.din')

    p.setup(mode=mode, force_alloc_complex=True)

    return p


def get_sizes(total_size, comm):
    rank = comm.rank
    comm_size = comm.size
    if total_size == 1:  # special handling if serial var is a scalar
        sizes = np.ones(comm_size, dtype=int)
        sizes[0] += 1
        offsets = np.empty(comm_size, dtype=int)
        offsets[0] = 0
        offsets[1:] = np.cumsum(sizes)[:-1]
    else:
        sizes, offsets = evenly_distrib_idxs(comm_size, total_size)

    dist_size = sizes[rank]
    dist_offset = offsets[rank]
    dist_slice = slice(dist_offset, dist_offset + dist_size)

    return dist_size, dist_slice


def mixed_partials_test(mode, arr_size):
    dist_size, dist_slice = get_sizes(arr_size, MPI.COMM_WORLD)
    p = create_mixed_only_problem(mode, arr_size, dist_size, dist_slice)
    p.run_model()
    pdata = p.check_partials(show_only_incorrect=True, method='cs')
    assert_check_partials(pdata)


def mixed_totals_test(mode, arr_size):
    dist_size, dist_slice = get_sizes(arr_size, MPI.COMM_WORLD)
    p = create_mixed_only_problem(mode, arr_size, dist_size, dist_slice)
    p.run_model()
    tdata = p.check_totals(of='mixed_in_sout_comp.sout', wrt=['ivc.xs', 'ivc.xd'], show_only_incorrect=True)
    assert_check_totals(tdata)


def full_partials_test(mode, arr_size):
    dist_size, dist_slice = get_sizes(arr_size, MPI.COMM_WORLD)
    p = create_problem(mode, arr_size, dist_size, dist_slice)
    p.run_model()
    pdata = p.check_partials(show_only_incorrect=True, method='cs')
    assert_check_partials(pdata)


def full_totals_test(mode, arr_size):
    dist_size, dist_slice = get_sizes(arr_size, MPI.COMM_WORLD)
    p = create_problem(mode, arr_size, dist_size, dist_slice)
    p.run_model()
    tdata = p.check_totals(of='mixed_in_sout_comp.sout', wrt='ivc.dv', show_only_incorrect=True)
    assert_check_totals(tdata)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribDerivs(unittest.TestCase):
    N_PROCS = 2

    def test_mixed_comp_only_fwd_partials5(self):
        mixed_partials_test('fwd', 5)

    def test_mixed_comp_only_fwd_partials1(self):
        mixed_partials_test('fwd', 1)

    def test_mixed_comp_only_fwd_totals5(self):
        mixed_totals_test('fwd', 5)

    def test_mixed_comp_only_fwd_totals1(self):
        mixed_totals_test('fwd', 1)

    def test_mixed_comp_only_rev_partials5(self):
        mixed_partials_test('rev', 5)

    def test_mixed_comp_only_rev_partials1(self):
        mixed_partials_test('rev', 1)

    def test_mixed_comp_only_rev_totals5(self):
        mixed_totals_test('rev', 5)

    def test_mixed_comp_only_rev_totals1(self):
        mixed_totals_test('rev', 1)

    def test_full_fwd_partials5(self):
        full_partials_test('fwd', 5)

    def test_full_fwd_totals5(self):
        full_totals_test('fwd', 5)

    def test_full_rev_partials5(self):
        full_partials_test('rev', 5)

    def test_full_rev_totals5(self):
        full_totals_test('rev', 5)

    def test_full_fwd_partials1(self):
        full_partials_test('fwd', 1)

    def test_full_fwd_totals1(self):
        full_totals_test('fwd', 1)

    def test_full_rev_partials1(self):
        full_partials_test('rev', 1)

    def test_full_rev_totals1(self):
        full_totals_test('rev', 1)


if __name__ == '__main__':
    unittest.main()

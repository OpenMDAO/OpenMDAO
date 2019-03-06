from six.moves import range
import unittest
import itertools
from six import iterkeys

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ImplicitComponent, ExecComp, \
    ExplicitComponent
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.mpi import MPI
from openmdao.test_suite.components.impl_comp_array import TestImplCompArray, TestImplCompArrayDense

from openmdao.test_suite.components.simple_comps import DoubleArrayComp, NonSquareArrayComp

try:
    from openmdao.parallel_api import PETScVector
    vector_class = PETScVector
except ImportError:
    vector_class = DefaultVector
    PETScVector = None


class SparseCompImplicit(ImplicitComponent):

    def __init__(self, sparsity, isplit=1, osplit=1, **kwargs):
        super(SparseCompImplicit, self).__init__(**kwargs)
        self.sparsity = sparsity
        self.isplit = isplit
        self.osplit = osplit

    def setup(self):
        shapesz = self.sparsity.shape[1]
        sz = shapesz // self.isplit
        rem = shapesz % self.isplit
        for i in range(self.isplit):
            if rem > 0:
                isz = sz + rem
                rem = 0
            else:
                isz = sz

            self.add_input('x%d' % i, np.zeros(isz))

        shapesz = self.sparsity.shape[0]
        sz = shapesz // self.osplit
        rem = shapesz % self.osplit
        for i in range(self.osplit):
            if rem > 0:
                isz = sz + rem
                rem = 0
            else:
                isz = sz
            self.add_output('y%d' % i, np.zeros(isz))

        self.declare_partials(of='*', wrt='*', method='cs')

    # def compute(self, inputs, outputs):
    #     outputs['y'] = self.sparsity.dot(inputs._data)

    def apply_nonlinear(self, inputs, outputs, residuals):
        prod = self.sparsity.dot(inputs._data)
        start = end = 0
        for i in range(self.osplit):
            outname = 'y%d' % i
            end += outputs[outname].size
            residuals[outname] = prod[start:end]
            start = end


class SparseCompExplicit(ExplicitComponent):

    def __init__(self, sparsity, isplit=1, osplit=1, **kwargs):
        super(SparseCompExplicit, self).__init__(**kwargs)
        self.sparsity = sparsity
        self.isplit = isplit
        self.osplit = osplit

    def setup(self):
        shapesz = self.sparsity.shape[1]
        sz = shapesz // self.isplit
        rem = shapesz % self.isplit
        for i in range(self.isplit):
            if rem > 0:
                isz = sz + rem
                rem = 0
            else:
                isz = sz

            self.add_input('x%d' % i, np.zeros(isz))

        shapesz = self.sparsity.shape[0]
        sz = shapesz // self.osplit
        rem = shapesz % self.osplit
        for i in range(self.osplit):
            if rem > 0:
                isz = sz + rem
                rem = 0
            else:
                isz = sz
            self.add_output('y%d' % i, np.zeros(isz))

        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        prod = self.sparsity.dot(inputs._data)
        start = end = 0
        for i in range(self.osplit):
            outname = 'y%d' % i
            end += outputs[outname].size
            outputs[outname] = prod[start:end]
            start = end


def _check_matrix(system, jac, expected):
    blocks = []
    for of in system._var_allprocs_abs_names['output']:
        cblocks = []
        for wrt in system._var_allprocs_abs_names['input']:
            key = (of, wrt)
            if key in jac:
                cblocks.append(jac[key]['value'])
                print(key, jac[key]['value'])
        blocks.append(np.hstack(cblocks))
    fullJ = np.vstack(blocks)
    np.testing.assert_almost_equal(fullJ, expected)


class TestFDColoring(unittest.TestCase):
    def test_simple_totals(self):
        prob = Problem()
        model = prob.model = Group(dynamic_derivs_repeats=1)

        sparsity = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 0]], dtype=float
        )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(3))
        indeps.add_output('x1', np.ones(2))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, isplit=2, osplit=2))
        model.set_approx_coloring('*', method='cs')
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0')#, indices=[0,2])
        model.comp.add_constraint('y1')#, indices=[0,2])
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')
        #model.approx_totals(method='cs')
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        derivs = prob.compute_totals()
        _check_matrix(model, derivs, sparsity)
        print(derivs)

    def test_simple_partials_implicit(self):
        prob = Problem()
        model = prob.model

        sparsity = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 1]], dtype=float
        )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(3))
        indeps.add_output('x1', np.ones(2))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, isplit=2, osplit=2, dynamic_derivs_repeats=1))
        comp.set_approx_coloring('x*', method='cs')
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        comp._linearize()
        comp._linearize()
        jac = comp._jacobian._subjacs_info
        _check_matrix(comp, jac, sparsity)

    def test_simple_partials_explicit(self):
        prob = Problem()
        model = prob.model

        sparsity = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 1]], dtype=float
        )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(3))
        indeps.add_output('x1', np.ones(2))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, isplit=2, osplit=2, dynamic_derivs_repeats=1))
        comp.set_approx_coloring('x*', method='cs')
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        comp._linearize()
        comp._linearize()
        jac = comp._jacobian._subjacs_info
        _check_matrix(comp, jac, sparsity)

if __name__ == '__main__':
    unitest.main()

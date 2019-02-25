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


class SparseComp(ImplicitComponent):

    def __init__(self, sparsity, isplit=1, osplit=1, **kwargs):
        super(SparseComp, self).__init__(**kwargs)
        self.sparsity = sparsity
        self.isplit = isplit
        self.osplit = osplit

    def setup(self):
        if self.isplit > 1:
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
        else:
            self.add_input('x', np.zeros(self.sparsity.shape[1]))


        if self.osplit > 1:
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
        else:
            self.add_output('y', np.zeros(self.sparsity.shape[0]))

        self.declare_partials(of='*', wrt='*', method='cs')

    # def compute(self, inputs, outputs):
    #     outputs['y'] = self.sparsity.dot(inputs._data)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['y'] = self.sparsity.dot(inputs._data)


class TestFDColoring(unittest.TestCase):
    def test_simple(self):
        prob = Problem()
        model = prob.model = Group()

        sparsity = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 1, 0, 0]],
        )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(3))
        indeps.add_output('x1', np.ones(2))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseComp(sparsity, isplit=2, dynamic_derivs_repeats=1))
        comp.set_approx_coloring('x*', method='cs')
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['comp.y']
        wrt = ['indeps.x0', 'indeps.x1']
        derivs = prob.compute_totals(of=of, wrt=wrt)
        print(derivs)



if __name__ == '__main__':
    unitest.main()

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
        prod = self.sparsity.dot(inputs._data)
        residuals['y0'] = prod[:outputs['y0'].size]
        residuals['y1'] = prod[outputs['y0'].size:]


class TestFDColoring(unittest.TestCase):
    def test_simple_totals(self):
        prob = Problem()
        model = prob.model = Group(dynamic_derivs_repeats=1)

        sparsity = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 0]],
        )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(3))
        indeps.add_output('x1', np.ones(2))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseComp(sparsity, isplit=2, osplit=2))#, dynamic_derivs_repeats=1))
        model.set_approx_coloring('*', method='cs')
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0', indices=[0,2])
        model.add_design_var('indeps.x0')
        model.approx_totals(method='cs')
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['comp.y0', 'comp.y1']
        wrt = ['indeps.x0', 'indeps.x1']
        derivs = prob.compute_totals(of=of, wrt=wrt)
        print(derivs)

    def test_simple_partials(self):
        prob = Problem()
        model = prob.model

        sparsity = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 1]],
        )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(3))
        indeps.add_output('x1', np.ones(2))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseComp(sparsity, isplit=2, osplit=2, dynamic_derivs_repeats=1))
        comp.set_approx_coloring('x*', method='cs')
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        comp._linearize()
        comp._linearize()
        jac = comp._jacobian._subjacs_info
        for of in comp._var_allprocs_abs_names['output']:
            for wrt in comp._var_allprocs_abs_names['input']:
                key = (of, wrt)
                if key in jac:
                    print(key, jac[key]['value'])
        print()


if __name__ == '__main__':
    unitest.main()

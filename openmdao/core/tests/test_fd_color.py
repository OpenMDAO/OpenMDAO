from six.moves import range
import unittest
import itertools
from six import iterkeys

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

import numpy as np
from scipy.sparse import coo_matrix

from openmdao.api import Problem, Group, IndepVarComp, ImplicitComponent, ExecComp, \
    ExplicitComponent, NonlinearBlockGS
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


def setup_vars(self, ofs, wrts):
    matrix = self.sparsity
    isplit = self.isplit
    osplit = self.osplit

    shapesz = matrix.shape[1]
    sz = shapesz // isplit
    rem = shapesz % isplit
    for i in range(isplit):
        if rem > 0:
            isz = sz + rem
            rem = 0
        else:
            isz = sz

        self.add_input('x%d' % i, np.zeros(isz))

    shapesz = matrix.shape[0]
    sz = shapesz // osplit
    rem = shapesz % osplit
    for i in range(osplit):
        if rem > 0:
            isz = sz + rem
            rem = 0
        else:
            isz = sz
        self.add_output('y%d' % i, np.zeros(isz))

    self.declare_partials(of=ofs, wrt=wrts, method=self.method)


class SparseCompImplicit(ImplicitComponent):

    def __init__(self, sparsity, method='fd', isplit=1, osplit=1, **kwargs):
        super(SparseCompImplicit, self).__init__(**kwargs)
        self.sparsity = sparsity
        self.isplit = isplit
        self.osplit = osplit
        self.method = method
        self._nruns = 0


    def setup(self):
        setup_vars(self, ofs='*', wrts='*')

    # this is defined for easier testing of coloring of approx partials
    def apply_nonlinear(self, inputs, outputs, residuals):
        prod = self.sparsity.dot(inputs._data) - outputs._data
        start = end = 0
        for i in range(self.osplit):
            outname = 'y%d' % i
            end += outputs[outname].size
            residuals[outname] = prod[start:end]
            start = end
        self._nruns += 1


    # this is defined so we can more easily test coloring of approx totals in a Group above this comp
    def solve_nonlinear(self, inputs, outputs):
        prod = self.sparsity.dot(inputs._data)
        start = end = 0
        for i in range(self.osplit):
            outname = 'y%d' % i
            end += outputs[outname].size
            outputs[outname] = prod[start:end]
            start = end
        self._nruns += 1


class SparseCompExplicit(ExplicitComponent):

    def __init__(self, sparsity, method='fd', isplit=1, osplit=1, **kwargs):
        super(SparseCompExplicit, self).__init__(**kwargs)
        self.sparsity = sparsity
        self.isplit = isplit
        self.osplit = osplit
        self.method = method
        self._nruns = 0

    def setup(self):
        setup_vars(self, ofs='*', wrts='*')

    def compute(self, inputs, outputs):
        prod = self.sparsity.dot(inputs._data)
        start = end = 0
        for i in range(self.osplit):
            outname = 'y%d' % i
            end += outputs[outname].size
            outputs[outname] = prod[start:end]
            start = end
        self._nruns += 1


def _check_partial_matrix(system, jac, expected):
    blocks = []
    for of in system._var_allprocs_abs_names['output']:
        cblocks = []
        for wrt in system._var_allprocs_abs_names['input']:
            key = (of, wrt)
            if key in jac:
                cblocks.append(jac[key]['value'])
        if cblocks:
            blocks.append(np.hstack(cblocks))
    fullJ = np.vstack(blocks)
    np.testing.assert_almost_equal(fullJ, expected)


def _check_total_matrix(system, jac, expected):
    blocks = []
    for of in system._var_allprocs_abs_names['output']:
        cblocks = []
        for wrt in itertools.chain(system._var_allprocs_abs_names['output'], system._var_allprocs_abs_names['input']):
            key = (of, wrt)
            if key in jac:
                cblocks.append(jac[key])
        if cblocks:
            blocks.append(np.hstack(cblocks))
    fullJ = np.vstack(blocks)
    np.testing.assert_almost_equal(fullJ, expected)


def _check_semitotal_matrix(system, jac, expected):
    blocks = []
    for of in system._var_allprocs_abs_names['output']:
        cblocks = []
        for wrt in itertools.chain(system._var_allprocs_abs_names['output'], system._var_allprocs_abs_names['input']):
            key = (of, wrt)
            if key in jac:
                rows = jac[key]['rows']
                if rows is not None:
                    cols = jac[key]['cols']
                    val = coo_matrix((jac[key]['value'], (rows, cols)), shape=jac[key]['shape']).toarray()
                else:
                    val = jac[key]['value']
                cblocks.append(val)
        if cblocks:
            blocks.append(np.hstack(cblocks))
    fullJ = np.vstack(blocks)
    np.testing.assert_almost_equal(fullJ, expected)


class TestCSColoring(unittest.TestCase):
    FD_METHOD = 'cs'

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
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, self.FD_METHOD, isplit=2, osplit=2))
        model.set_approx_coloring('*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0')
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # uncolored, computing sparsity
        # one run per column of jac
        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, sparsity.shape[1])

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored

        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, 3)
        _check_total_matrix(model, derivs, sparsity)

    def test_simple_semitotals(self):
        prob = Problem()
        model = prob.model = Group()

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
        sub = model.add_subsystem('sub', Group(dynamic_derivs_repeats=1))
        comp = sub.add_subsystem('comp', SparseCompExplicit(sparsity, self.FD_METHOD, isplit=2, osplit=2))
        sub.set_approx_coloring('*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'sub.comp.x0')
        model.connect('indeps.x1', 'sub.comp.x1')

        model.sub.comp.add_constraint('y0')
        model.sub.comp.add_constraint('y1')
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # uncolored, computing sparsity
        # one run per column of jac
        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, sparsity.shape[1])
    
        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored
    
        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, 3)
        _check_partial_matrix(sub, sub._jacobian._subjacs_info, sparsity)

    def test_totals_over_implicit_comp(self):
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

        model.nonlinear_solver = NonlinearBlockGS()
        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, self.FD_METHOD, isplit=2, osplit=2))
        model.set_approx_coloring('*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0')
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # uncolored, computing sparsity
        # one run per column of jac
        nruns = comp._nruns - start_nruns
        # wrts of size 5, 2 runs of the implicit comp per approx point run (1 for solver init and 1 for 1 solver iter)
        self.assertEqual(nruns, 5 * 2)
    
        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored
    
        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, 3 * 2)
        _check_total_matrix(model, derivs, sparsity)

    def test_totals_of_indices(self):
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
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, self.FD_METHOD, isplit=2, osplit=2))
        model.set_approx_coloring('*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0', indices=[0,2])
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # uncolored, computing sparsity
        # one run per column of jac
        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, sparsity.shape[1])
    
        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored
    
        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, 3)
        rows = [0,2,3,4]
        _check_total_matrix(model, derivs, sparsity[rows, :])

    def test_totals_wrt_indices(self):
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
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, self.FD_METHOD,
                                                              isplit=2, osplit=2))
        model.set_approx_coloring('*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0')
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0', indices=[0,2])
        model.add_design_var('indeps.x1')
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # uncolored, computing sparsity
        # one run per column of jac
        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, sparsity.shape[1] - 1)
    
        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored
    
        nruns = comp._nruns - start_nruns
        # only 4 cols to solve for, but we get coloring of [[2],[3],[0,1]] so only 1 better
        self.assertEqual(nruns, 3)  
        cols = [0,2,3,4]
        _check_total_matrix(model, derivs, sparsity[:, cols])

    def test_totals_of_wrt_indices(self):
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
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, self.FD_METHOD,
                                                              isplit=2, osplit=2))
        model.set_approx_coloring('*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0', indices=[0,2])
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0', indices=[0,2])
        model.add_design_var('indeps.x1')
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # uncolored, computing sparsity
        # one run per column of jac
        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, sparsity.shape[1] - 1)
    
        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored
    
        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, 3)
        cols = rows = [0,2,3,4]
        _check_total_matrix(model, derivs, sparsity[rows, :][:, cols])

    def test_simple_partials_implicit(self):
        prob = Problem()
        model = prob.model

        sparsity = np.array(
            [[1, 0, 0, 1, 1, 1, 0],
             [0, 1, 0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1, 1, 0],
             [1, 0, 0, 0, 0, 1, 0],
             [0, 1, 1, 0, 1, 1, 1]], dtype=float
        )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(4))
        indeps.add_output('x1', np.ones(3))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, self.FD_METHOD,
                                                              isplit=2, osplit=2, dynamic_derivs_repeats=1))
        comp.set_approx_coloring('x*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        start_nruns = comp._nruns
        comp._linearize()
        # add 5 to number of runs to cover the 5 uncolored output columns
        self.assertEqual(comp._nruns - start_nruns, np.sum(sparsity.shape))
        start_nruns = comp._nruns
        comp._linearize()
        self.assertEqual(comp._nruns - start_nruns, sparsity.shape[0] + 5)
        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity)

    def test_simple_partials_explicit(self):
        prob = Problem()
        model = prob.model

        sparsity = np.array(
                [[1, 0, 0, 1, 1, 1, 0],
                 [0, 1, 0, 1, 0, 1, 1],
                 [0, 1, 0, 1, 1, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0],
                 [0, 1, 1, 0, 1, 1, 1]], dtype=float
            )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(4))
        indeps.add_output('x1', np.ones(3))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, self.FD_METHOD,
                                                              isplit=2, osplit=2, dynamic_derivs_repeats=1))
        comp.set_approx_coloring('x*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        start_nruns = comp._nruns
        comp._linearize()
        self.assertEqual(comp._nruns - start_nruns, 7)
        start_nruns = comp._nruns
        comp._linearize()
        self.assertEqual(comp._nruns - start_nruns, 5)
        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity)


class TestFDColoring(TestCSColoring):
    FD_METHOD = 'fd'


class TestColoringParallelCS(unittest.TestCase):
    N_PROCS = 4
    FD_METHOD = 'cs'

    def test_simple_totals_all_local_vars(self):
        # in this case, num_par_fd == N_PROCS, so each proc has local versions of all vars
        prob = Problem()
        model = prob.model = Group(dynamic_derivs_repeats=1, num_par_fd=self.N_PROCS)

        sparsity = np.array(
                [[1, 0, 0, 1, 1, 1, 0],
                 [0, 1, 0, 1, 0, 1, 1],
                 [0, 1, 0, 1, 1, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0],
                 [0, 1, 1, 0, 1, 1, 1]], dtype=float
            )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(4))
        indeps.add_output('x1', np.ones(3))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, self.FD_METHOD, isplit=2, osplit=2))
        model.set_approx_coloring('*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0')
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        start_nruns = comp._nruns

        derivs = prob.driver._compute_totals()  # uncolored, computing sparsity
        # one run per column of jac
        nruns = comp._nruns - start_nruns
        if model._full_comm:
            nruns = model._full_comm.allreduce(nruns)
        self.assertEqual(nruns, sparsity.shape[1])

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored

        nruns = comp._nruns - start_nruns
        if model._full_comm:
            nruns = model._full_comm.allreduce(nruns)
        self.assertEqual(nruns, 5)
        _check_total_matrix(model, derivs, sparsity)

    def test_simple_partials_implicit(self):
        prob = Problem()
        model = prob.model

        sparsity = np.array(
            [[1, 0, 0, 1, 1, 1, 0],
             [0, 1, 0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1, 1, 0],
             [1, 0, 0, 0, 0, 1, 0],
             [0, 1, 1, 0, 1, 1, 1]], dtype=float
        )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(4))
        indeps.add_output('x1', np.ones(3))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, self.FD_METHOD,
                                                              isplit=2, osplit=2,
                                                              dynamic_derivs_repeats=1,
                                                              num_par_fd=self.N_PROCS))
        comp.set_approx_coloring('x*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        comp._linearize()  # uncolored
        nruns = comp._nruns - start_nruns
        if comp._full_comm:
            nruns = comp._full_comm.allreduce(nruns)
        # one run per column of jac
        self.assertEqual(nruns, np.sum(sparsity.shape))

        start_nruns = comp._nruns
        comp._linearize()   # colored
        # number of runs = ncolors + number of outputs (only input columns were colored here)
        nruns = comp._nruns - start_nruns
        if comp._full_comm:
            nruns = comp._full_comm.allreduce(nruns)
        self.assertEqual(nruns, 5 + sparsity.shape[0])

        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity)

    def test_simple_partials_explicit(self):
        prob = Problem()
        model = prob.model

        sparsity = np.array(
                [[1, 0, 0, 1, 1, 1, 0],
                 [0, 1, 0, 1, 0, 1, 1],
                 [0, 1, 0, 1, 1, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0],
                 [0, 1, 1, 0, 1, 1, 1]], dtype=float
            )

        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(4))
        indeps.add_output('x1', np.ones(3))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, self.FD_METHOD,
                                                              isplit=2, osplit=2,
                                                              dynamic_derivs_repeats=1,
                                                              num_par_fd=self.N_PROCS))
        comp.set_approx_coloring('x*', method=self.FD_METHOD)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        comp._linearize()  # uncolored
        # one run per column of jac
        nruns = comp._nruns - start_nruns
        if comp._full_comm:
            nruns = comp._full_comm.allreduce(nruns)
        self.assertEqual(nruns, sparsity.shape[1])

        start_nruns = comp._nruns
        comp._linearize()   # colored
        nruns = comp._nruns - start_nruns
        if comp._full_comm:
            nruns = comp._full_comm.allreduce(nruns)
        self.assertEqual(nruns, comp._approx_schemes[self.FD_METHOD].ncolors())

        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity)


class TestColoringParallelFD(TestColoringParallelCS):
    FD_METHOD = 'fd'


if __name__ == '__main__':
    unitest.main()

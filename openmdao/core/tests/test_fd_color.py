import os
import tempfile
import shutil
import unittest
import itertools

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

import numpy as np
from scipy.sparse import coo_matrix

from openmdao.api import Problem, Group, IndepVarComp, ImplicitComponent, ExecComp, \
    ExplicitComponent, NonlinearBlockGS, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI
from openmdao.utils.coloring import compute_total_coloring, Coloring

from openmdao.test_suite.components.impl_comp_array import TestImplCompArray, TestImplCompArrayDense

from openmdao.test_suite.components.simple_comps import DoubleArrayComp, NonSquareArrayComp

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None

from openmdao.utils.general_utils import set_pyoptsparse_opt


try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized


# check that pyoptsparse is installed. if it is, try to use SLSQP.
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
else:
    pyOptSparseDriver = None


def setup_vars(self, ofs, wrts):
    matrix = self.sparsity
    isplit = self.isplit
    osplit = self.osplit

    isizes, _ = evenly_distrib_idxs(isplit, matrix.shape[1])
    for i, sz in enumerate(isizes):
        self.add_input('x%d' % i, np.zeros(sz))

    osizes, _ = evenly_distrib_idxs(osplit, matrix.shape[0])
    for i, sz in enumerate(osizes):
        self.add_output('y%d' % i, np.zeros(sz))

    self.declare_partials(of=ofs, wrt=wrts, method=self.method)


def setup_sparsity(mask):
    sparsity = np.random.random(np.product(mask.shape)).reshape(*mask.shape)
    return sparsity * mask


def setup_indeps(isplit, ninputs, indeps_name, comp_name):
    isizes, _ = evenly_distrib_idxs(isplit, ninputs)
    indeps = IndepVarComp()
    conns = []
    for i, sz in enumerate(isizes):
        indep_var = 'x%d' % i
        indeps.add_output(indep_var, np.random.random(sz))
        conns.append((indeps_name + '.' + indep_var, comp_name + '.' + indep_var))

    return indeps, conns


class CounterGroup(Group):
    def __init__(self, *args, **kwargs):
        super(CounterGroup, self).__init__(*args, **kwargs)
        self._nruns = 0

    def _solve_nonlinear(self, *args, **kwargs):
        super(CounterGroup, self)._solve_nonlinear(*args, **kwargs)
        self._nruns += 1


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


# relative tolerances for jacobian checks
_TOLS = {
    'fd': 1e-6,
    'cs': 1e-12,
}

def _check_partial_matrix(system, jac, expected, method):
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
    np.testing.assert_allclose(fullJ, expected, rtol=_TOLS[method])


def _check_total_matrix(system, jac, expected, method):
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
    np.testing.assert_allclose(fullJ, expected, rtol=_TOLS[method])


def _check_semitotal_matrix(system, jac, expected, method):
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
    np.testing.assert_allclose(fullJ, expected, rtol=_TOLS[method])


_BIGMASK = np.array(
    [[1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
     [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
     [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
     [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
     [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
     [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
     [0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]]
)


def _test_func_name(func, num, param):
    args = []
    for p in param.args:
        try:
            arg = p.__name__
        except:
            arg = str(p)
        args.append(arg)
    return func.__name__ + '_' + '_'.join(args)


class TestColoringExplicit(unittest.TestCase):
    def setUp(self):
        np.random.seed(11)
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix=self.__class__.__name__ + '_')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        [1,2,7,19],
        [1,2,5,11]
        ), name_func=_test_func_name
    )
    def test_partials_explicit(self, method, isplit, osplit):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        sparsity = setup_sparsity(_BIGMASK)
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')
        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=osplit))
        comp.declare_coloring('x*', method=method)

        for conn in conns:
            model.connect(*conn)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        comp.run_linearize()
        prob.run_model()
        start_nruns = comp._nruns
        comp.run_linearize()
        self.assertEqual(comp._nruns - start_nruns, 10)
        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity, method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        [1,2,7,19],
        [1,2,5,11]
        ), name_func=_test_func_name
    )
    def test_partials_explicit_static(self, method, isplit, osplit):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        sparsity = setup_sparsity(_BIGMASK)
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')
        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=osplit))
        for conn in conns:
            model.connect(*conn)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        coloring = comp._compute_approx_coloring(wrt_patterns='x*', method=method)[0]
        comp._save_coloring(coloring)

        # now make a second problem to use the coloring
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=osplit))
        for conn in conns:
            model.connect(*conn)

        comp.declare_coloring(wrt='x*', method=method)
        comp.use_fixed_coloring()
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        comp._linearize()
        self.assertEqual(comp._nruns - start_nruns, 10)
        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity, method)

    @parameterized.expand(itertools.product([1,2,5]), name_func=_test_func_name)
    def test_partials_explicit_reuse(self, num_insts):
        method = 'cs'
        osplit = 5
        isplit = 7
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        sparsity = setup_sparsity(_BIGMASK)
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')
        model.add_subsystem('indeps', indeps)

        comps = []
        for i in range(num_insts):
            cname = 'comp%d' % i
            comp = model.add_subsystem(cname, SparseCompExplicit(sparsity, method,
                                                                  isplit=isplit, osplit=osplit))
            comp.declare_coloring('x*', method=method, per_instance=False)
            comps.append(comp)

            _, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', cname)

            for conn in conns:
                model.connect(*conn)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        prob.model.run_linearize()
        prob.run_model()
        start_nruns = [c._nruns for c in comps]
        for i, comp in enumerate(comps):
            comp.run_linearize()
            self.assertEqual(comp._nruns - start_nruns[i], 10)
            jac = comp._jacobian._subjacs_info
            _check_partial_matrix(comp, jac, sparsity, method)

        orig = comps[0]._coloring_info['coloring']
        for comp in comps:
            self.assertTrue(orig is comp._coloring_info['coloring'],
                            "Instance '{}' is using a different coloring".format(comp.pathname))


class TestColoringImplicit(unittest.TestCase):
    def setUp(self):
        np.random.seed(11)
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix=self.__class__.__name__ + '_')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        [1,2,7,19],
        [1,2,5,11]
        ), name_func=_test_func_name
    )
    def test_partials_implicit(self, method, isplit, osplit):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        sparsity = setup_sparsity(_BIGMASK)
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')
        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, method,
                                                              isplit=isplit, osplit=osplit))
        comp.declare_coloring('x*', method=method)

        for conn in conns:
            model.connect(*conn)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        comp.run_linearize()
        prob.run_model()
        start_nruns = comp._nruns
        comp.run_linearize()
        self.assertEqual(comp._nruns - start_nruns, 10 + sparsity.shape[0])
        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity, method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        [1,2,7,19],
        [1,2,5,11]
        ), name_func=_test_func_name
    )
    def test_simple_partials_implicit_static(self, method, isplit, osplit):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        sparsity = setup_sparsity(_BIGMASK)
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, method,
                                                              isplit=isplit, osplit=osplit))
        for conn in conns:
            model.connect(*conn)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        coloring = comp._compute_approx_coloring(wrt_patterns='x*', method=method)[0]
        comp._save_coloring(coloring)

        # now create a new problem and set the static coloring
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, method,
                                                              isplit=isplit, osplit=osplit))
        for conn in conns:
            model.connect(*conn)

        comp.declare_coloring(wrt='x*', method=method)
        comp.use_fixed_coloring()
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        comp._linearize()
        # add 5 to number of runs to cover the 5 uncolored output columns
        self.assertEqual(comp._nruns - start_nruns, sparsity.shape[0] + 10)
        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity, method)


class TestColoringSemitotals(unittest.TestCase):
    def setUp(self):
        np.random.seed(11)
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix=self.__class__.__name__ + '_')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        [1,2,19],
        [1,2,11]
        ), name_func=_test_func_name
    )
    def test_simple_semitotals(self, method, isplit, osplit):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        sparsity = setup_sparsity(_BIGMASK)
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'sub.comp')

        model.add_subsystem('indeps', indeps)
        sub = model.add_subsystem('sub', CounterGroup())
        sub.declare_coloring('*', method=method)
        comp = sub.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=osplit))

        for conn in conns:
            model.connect(*conn)

        for i in range(isplit):
            model.add_design_var('indeps.x%d' % i)

        for i in range(osplit):
            model.sub.comp.add_constraint('y%d' % i)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        derivs = prob.driver._compute_totals()  # this is when the dynamic coloring update happens

        start_nruns = sub._nruns
        derivs = prob.driver._compute_totals()
        self.assertEqual(sub._nruns - start_nruns, 10)
        _check_partial_matrix(sub, sub._jacobian._subjacs_info, sparsity, method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        [1,2,19],
        [1,2,11]
        ), name_func=_test_func_name
    )
    def test_simple_semitotals_static(self, method, isplit, osplit):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        sparsity = setup_sparsity(_BIGMASK)
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'sub.comp')

        model.add_subsystem('indeps', indeps)
        sub = model.add_subsystem('sub', Group())
        comp = sub.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=osplit))

        for conn in conns:
            model.connect(*conn)

        for i in range(isplit):
            model.add_design_var('indeps.x%d' % i)

        for i in range(osplit):
            model.sub.comp.add_constraint('y%d' % i)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        coloring = sub._compute_approx_coloring(wrt_patterns='comp.x*', method=method)[0]
        sub._save_coloring(coloring)

        # now create a second problem and use the static coloring
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'sub.comp')

        model.add_subsystem('indeps', indeps)
        sub = model.add_subsystem('sub', Group())
        comp = sub.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=osplit))

        for conn in conns:
            model.connect(*conn)

        for i in range(isplit):
            model.add_design_var('indeps.x%d' % i)

        for i in range(osplit):
            model.sub.comp.add_constraint('y%d' % i)

        sub.declare_coloring(wrt='comp.x*', method=method)
        sub.use_fixed_coloring()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()

        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, 10)
        _check_partial_matrix(sub, sub._jacobian._subjacs_info, sparsity, method)


class TestColoring(unittest.TestCase):

    def setUp(self):
        np.random.seed(11)
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix=self.__class__.__name__ + '_')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    def test_partials_explicit_shape_bug(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        # create sparsity with last row and col all zeros.
        # bug caused an exception when we created a COO matrix without supplying shape
        mask = np.array(
                [[1, 0, 0, 1, 1, 1, 0],
                 [0, 1, 0, 1, 0, 1, 0],
                 [0, 1, 0, 1, 1, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0]]
            )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=2))
        comp.declare_coloring('x*', method=method)

        for conn in conns:
            model.connect(*conn)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        comp._linearize()
        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity, method)

    def test_partials_min_improvement(self):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        mask = np.array(
                [[1, 0, 1, 0, 1, 1],
                 [1, 1, 1, 0, 0, 1],
                 [0, 1, 0, 1, 1, 0],
                 [1, 0, 1, 0, 0, 1]]
            )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, 'cs',
                                                              isplit=isplit, osplit=2))
        comp.declare_coloring('x*', method='cs', min_improve_pct=20)

        for conn in conns:
            model.connect(*conn)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        with assert_warning(UserWarning, 'SparseCompExplicit (comp): Coloring was deactivated.  Improvement of 16.7% was less than min allowed (20.0%).'):
            prob.model._linearize(None)

        start_nruns = comp._nruns
        comp._linearize()
        # verify we're doing a solve for each column
        self.assertEqual(6, comp._nruns - start_nruns)

        self.assertEqual(comp._coloring_info['coloring'], None)
        self.assertEqual(comp._coloring_info['static'], None)

        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity, 'cs')

    def test_partials_min_improvement_reuse(self):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        mask = np.array(
                [[1, 0, 1, 0, 1, 1],
                 [1, 1, 1, 0, 0, 1],
                 [0, 1, 0, 1, 1, 0],
                 [1, 0, 1, 0, 0, 1]]
            )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, _ = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)

        comps = []
        for i in range(3):
            cname = 'comp%d' % i
            comp = model.add_subsystem(cname, SparseCompExplicit(sparsity, 'cs',
                                                                isplit=isplit, osplit=2))
            comp.declare_coloring('x*', method='cs', min_improve_pct=20, per_instance=False)
            comps.append(comp)
            _, conns = setup_indeps(isplit, mask.shape[1], 'indeps', cname)

            for conn in conns:
                model.connect(*conn)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        for i, comp in enumerate(comps):
            if i == 0:
                with assert_warning(UserWarning, 'SparseCompExplicit (comp0): Coloring was deactivated.  Improvement of 16.7% was less than min allowed (20.0%).'):
                    comp._linearize()

            start_nruns = comp._nruns
            comp._linearize()
            self.assertEqual(6, comp._nruns - start_nruns)
            self.assertEqual(comp._coloring_info['coloring'], None)
            self.assertEqual(comp._coloring_info['static'], None)

            jac = comp._jacobian._subjacs_info
            _check_partial_matrix(comp, jac, sparsity, 'cs')

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    @unittest.skipUnless(OPTIMIZER, 'requires pyoptsparse SLSQP.')
    def test_simple_totals(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model = CounterGroup()
        prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        prob.driver.declare_coloring()

        mask = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 0]]
        )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')
        model.declare_coloring('*', method=method, step=1e-6 if method=='fd' else None)

        model.comp.add_objective('y0', index=0)  # pyoptsparse SLSQP requires a scalar objective, so pick index 0
        model.comp.add_constraint('y1', lower=[1., 2.])
        model.add_design_var('indeps.x0', lower=np.ones(3), upper=np.ones(3)+.1)
        model.add_design_var('indeps.x1', lower=np.ones(2), upper=np.ones(2)+.1)
        model.approx_totals(method=method)
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_driver()  # need this to trigger the dynamic coloring

        prob.driver._total_jac = None

        start_nruns = model._nruns
        derivs = prob.compute_totals()
        _check_total_matrix(model, derivs, sparsity[[0,3,4],:], method)
        nruns = model._nruns - start_nruns
        self.assertEqual(nruns, 3)

    @parameterized.expand(itertools.product(
        [pyOptSparseDriver, ScipyOptimizeDriver],
        ), name_func=_test_func_name
    )
    def test_simple_totals_min_improvement(self, optim):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model = CounterGroup()
        if optim is None:
            raise unittest.SkipTest('requires pyoptsparse SLSQP.')
        prob.driver = optim(optimizer='SLSQP')

        prob.driver.declare_coloring()

        mask = np.array(
            [[1, 0, 1, 1, 1],
             [1, 1, 0, 1, 1],
             [0, 1, 1, 1, 1],
             [1, 0, 1, 0, 0],
             [0, 1, 1, 0, 1]]
        )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, 'cs', isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_objective('y0', index=0)  # pyoptsparse SLSQP requires a scalar objective, so pick index 0
        model.comp.add_constraint('y1', lower=[1., 2.])
        model.add_design_var('indeps.x0', lower=np.ones(3), upper=np.ones(3)+.1)
        model.add_design_var('indeps.x1', lower=np.ones(2), upper=np.ones(2)+.1)
        model.approx_totals(method='cs')
        model.declare_coloring(min_improve_pct=25., method='cs')
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)

        with assert_warning(UserWarning, "CounterGroup (<model>): Coloring was deactivated.  Improvement of 20.0% was less than min allowed (25.0%)."):
            prob.run_driver()  # need this to trigger the dynamic coloring

        prob.driver._total_jac = None

        start_nruns = model._nruns
        derivs = prob.compute_totals()
        nruns = model._nruns - start_nruns
        _check_total_matrix(model, derivs, sparsity[[0,3,4],:], 'cs')
        self.assertEqual(nruns, 5)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    @unittest.skipUnless(OPTIMIZER, 'requires pyoptsparse SLSQP.')
    def test_totals_over_implicit_comp(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model = CounterGroup()
        prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        prob.driver.declare_coloring()

        mask = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 0]]
        )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.nonlinear_solver = NonlinearBlockGS()
        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, method, isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_objective('y0', index=1)
        model.comp.add_constraint('y1', lower=[1., 2.])
        model.add_design_var('indeps.x0', lower=np.ones(3), upper=np.ones(3)+.1)
        model.add_design_var('indeps.x1', lower=np.ones(2), upper=np.ones(2)+.1)

        model.approx_totals(method=method)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_driver()  # need this to trigger the dynamic coloring

        prob.driver._total_jac = None

        start_nruns = model._nruns
        derivs = prob.driver._compute_totals()
        self.assertEqual(model._nruns - start_nruns, 3)
        rows = [1,3,4]
        _check_total_matrix(model, derivs, sparsity[rows, :], method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    @unittest.skipUnless(OPTIMIZER, 'requires pyoptsparse SLSQP.')
    def test_totals_of_wrt_indices(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model = CounterGroup()
        prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        prob.driver.declare_coloring()

        mask = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 0]]
        )

        isplit=2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=2))

        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_objective('y0', index=1)
        model.comp.add_constraint('y1', lower=[1., 2.])
        model.add_design_var('indeps.x0',  indices=[0,2], lower=np.ones(2), upper=np.ones(2)+.1)
        model.add_design_var('indeps.x1', lower=np.ones(2), upper=np.ones(2)+.1)

        model.approx_totals(method=method)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_driver()  # need this to trigger the dynamic coloring

        prob.driver._total_jac = None

        start_nruns = model._nruns
        derivs = prob.driver._compute_totals()  # colored

        self.assertEqual(model._nruns - start_nruns, 2)
        cols = [0,2,3,4]
        rows = [1,3,4]
        _check_total_matrix(model, derivs, sparsity[rows, :][:, cols], method)


class TestStaticColoring(unittest.TestCase):

    def setUp(self):
        np.random.seed(11)
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix=self.__class__.__name__ + '_')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    def test_partials_explicit_shape_bug(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        # create sparsity with last row and col all zeros.
        # bug happened when we created a COO matrix without supplying shape
        mask = np.array(
                [[1, 0, 0, 1, 1, 1, 0],
                 [0, 1, 0, 1, 0, 1, 0],
                 [0, 1, 0, 1, 1, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0]]
            )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        coloring = comp._compute_approx_coloring(wrt_patterns='x*', method=method)[0]
        comp._save_coloring(coloring)

        # now make a second problem to use the coloring
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model
        indeps = IndepVarComp()
        indeps.add_output('x0', np.ones(4))
        indeps.add_output('x1', np.ones(3))

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        comp.declare_coloring(wrt='x*', method=method)
        comp.use_fixed_coloring()
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        comp._linearize()
        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity, method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        [1,2,19],
        [1,2,11]
        ), name_func=_test_func_name
    )
    def test_simple_totals_static(self, method, isplit, osplit):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        sparsity = setup_sparsity(_BIGMASK)
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=osplit))

        for conn in conns:
            model.connect(*conn)

        for i in range(isplit):
            model.add_design_var('indeps.x%d' % i)

        for i in range(osplit):
            model.comp.add_constraint('y%d' % i)

        model.approx_totals(method=method)
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        # try just manually computing the coloring here instead of using declare_coloring
        coloring = compute_total_coloring(prob)
        model._save_coloring(coloring)

        # new Problem, loading the coloring we just computed
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=osplit))

        for conn in conns:
            model.connect(*conn)

        for i in range(isplit):
            model.add_design_var('indeps.x%d' % i)

        for i in range(osplit):
            model.comp.add_constraint('y%d' % i)

        model.approx_totals(method=method)

        model.declare_coloring(wrt='*', method=method)
        model.use_fixed_coloring()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()

        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, 10)
        _check_total_matrix(model, derivs, sparsity, method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        [1,2,19],
        [1,2,11]
        ), name_func=_test_func_name
    )
    def test_totals_over_implicit_comp(self, method, isplit, osplit):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        sparsity = setup_sparsity(_BIGMASK)
        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')

        model.nonlinear_solver = NonlinearBlockGS()
        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, method, isplit=isplit, osplit=osplit))

        for conn in conns:
            model.connect(*conn)

        for i in range(isplit):
            model.add_design_var('indeps.x%d' % i)

        for i in range(osplit):
            model.comp.add_constraint('y%d' % i)

        model.approx_totals(method=method)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        model._save_coloring(compute_total_coloring(prob))

        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, _BIGMASK.shape[1], 'indeps', 'comp')

        model.nonlinear_solver = NonlinearBlockGS()
        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, method, isplit=isplit, osplit=osplit))
        for conn in conns:
            model.connect(*conn)

        for i in range(isplit):
            model.add_design_var('indeps.x%d' % i)

        for i in range(osplit):
            model.comp.add_constraint('y%d' % i)

        model.approx_totals(method=method)

        model.declare_coloring(wrt='*', method=method)
        model.use_fixed_coloring()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored

        nruns = comp._nruns - start_nruns
        # NLBGS ends up doing a single iteration after initialization, resulting in 2
        # runs per NL solve, so we multiplly the number of colored solvers by 2
        self.assertEqual(nruns, 10 * 2)
        _check_total_matrix(model, derivs, sparsity, method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    def test_totals_of_indices(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        mask = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 0]]
        )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=2))
        for conn in conns:
            model.connect(*conn)

        model.comp.add_constraint('y0', indices=[0,2])
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')
        model.approx_totals(method=method)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        model._save_coloring(compute_total_coloring(prob))


        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0', indices=[0,2])
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')
        model.approx_totals(method=method)

        model.declare_coloring(wrt='*', method=method)
        model.use_fixed_coloring()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored

        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, 3)
        rows = [0,2,3,4]
        _check_total_matrix(model, derivs, sparsity[rows, :], method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    def test_totals_wrt_indices(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        mask = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 0]]
        )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0')
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0', indices=[0,2])
        model.add_design_var('indeps.x1')
        model.approx_totals(method=method)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        model._save_coloring(compute_total_coloring(prob))


        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                                  isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0')
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0', indices=[0,2])
        model.add_design_var('indeps.x1')
        model.approx_totals(method=method)

        model.declare_coloring(wrt='*', method=method)
        model.use_fixed_coloring()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored

        nruns = comp._nruns - start_nruns
        # only 4 cols to solve for, but we get coloring of [[2],[3],[0,1]] so only 1 better
        self.assertEqual(nruns, 3)
        cols = [0,2,3,4]
        _check_total_matrix(model, derivs, sparsity[:, cols], method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    def test_totals_of_wrt_indices(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        mask = np.array(
            [[1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 0, 1, 1],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 0, 0]]
        )

        isplit = 2
        sparsity = setup_sparsity(mask)
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=2))
        # model.declare_coloring('*', method=method)
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0', indices=[0,2])
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0', indices=[0,2])
        model.add_design_var('indeps.x1')

        model.approx_totals(method=method)

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        model._save_coloring(compute_total_coloring(prob))


        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                                  isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        model.comp.add_constraint('y0', indices=[0,2])
        model.comp.add_constraint('y1')
        model.add_design_var('indeps.x0', indices=[0,2])
        model.add_design_var('indeps.x1')
        model.approx_totals(method=method)

        model.declare_coloring(wrt='*', method=method)
        model.use_fixed_coloring()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        derivs = prob.driver._compute_totals()  # colored

        nruns = comp._nruns - start_nruns
        self.assertEqual(nruns, 3)
        cols = rows = [0,2,3,4]
        _check_total_matrix(model, derivs, sparsity[rows, :][:, cols], method)



@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc is required.")
class TestStaticColoringParallelCS(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        np.random.seed(11)
        self.startdir = os.getcwd()
        if MPI.COMM_WORLD.rank == 0:
            self.tempdir = tempfile.mkdtemp(prefix=self.__class__.__name__ + '_')
            MPI.COMM_WORLD.bcast(self.tempdir, root=0)
        else:
            self.tempdir = MPI.COMM_WORLD.bcast(None, root=0)
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        if MPI.COMM_WORLD.rank == 0:
            try:
                shutil.rmtree(self.tempdir)
            except OSError:
                pass

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    def test_simple_semitotals_all_local_vars(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        mask = np.array(
            [[1, 0, 0, 1, 1],
                [0, 1, 0, 1, 1],
                [0, 1, 0, 1, 1],
                [1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0]]
        )
        if MPI.COMM_WORLD.rank == 0:
            sparsity = setup_sparsity(mask)
            MPI.COMM_WORLD.bcast(sparsity, root=0)
        else:
            sparsity = MPI.COMM_WORLD.bcast(None, root=0)

        isplit = 2
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        sub = model.add_subsystem('sub', Group(num_par_fd=self.N_PROCS))
        sub.approx_totals(method=method)
        comp = sub.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'sub.comp.x0')
        model.connect('indeps.x1', 'sub.comp.x1')

        model.sub.comp.add_constraint('y0')
        model.sub.comp.add_constraint('y1')
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        coloring = sub._compute_approx_coloring(wrt_patterns='*', method=method)[0]
        sub._save_coloring(coloring)

        # make sure coloring file exists by the time we try to load the spec
        MPI.COMM_WORLD.barrier()

        # now create a second problem and use the static coloring
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        sub = model.add_subsystem('sub', Group(num_par_fd=self.N_PROCS))
        #sub.approx_totals(method=method)
        comp = sub.add_subsystem('comp', SparseCompExplicit(sparsity, method, isplit=isplit, osplit=2))
        model.connect('indeps.x0', 'sub.comp.x0')
        model.connect('indeps.x1', 'sub.comp.x1')

        model.sub.comp.add_constraint('y0')
        model.sub.comp.add_constraint('y1')
        model.add_design_var('indeps.x0')
        model.add_design_var('indeps.x1')

        sub.declare_coloring(wrt='*', method=method)
        sub.use_fixed_coloring()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        #derivs = prob.driver._compute_totals()
        sub._linearize(sub._jacobian)
        nruns = comp._nruns - start_nruns
        if sub._full_comm is not None:
            nruns = sub._full_comm.allreduce(nruns)

        _check_partial_matrix(sub, sub._jacobian._subjacs_info, sparsity, method)
        self.assertEqual(nruns, 3)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    def test_simple_partials_implicit(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        mask = np.array(
            [[1, 0, 0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0, 1, 1],
                [0, 1, 0, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 1, 1]]
        )

        if MPI.COMM_WORLD.rank == 0:
            sparsity = setup_sparsity(mask)
            MPI.COMM_WORLD.bcast(sparsity, root=0)
        else:
            sparsity = MPI.COMM_WORLD.bcast(None, root=0)

        isplit = 2
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, method,
                                                              isplit=isplit, osplit=2,
                                                              num_par_fd=self.N_PROCS))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        coloring = comp._compute_approx_coloring(wrt_patterns='x*', method=method)[0]
        comp._save_coloring(coloring)

        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompImplicit(sparsity, method,
                                                                  isplit=isplit, osplit=2,
                                                                  num_par_fd=self.N_PROCS))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        # make sure coloring file exists by the time we try to load the spec
        MPI.COMM_WORLD.barrier()

        comp.declare_coloring(wrt='x*', method=method)
        comp.use_fixed_coloring()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        comp._linearize()   # colored
        # number of runs = ncolors + number of outputs (only input columns were colored here)
        nruns = comp._nruns - start_nruns
        if comp._full_comm:
            nruns = comp._full_comm.allreduce(nruns)
        self.assertEqual(nruns, 5 + sparsity.shape[0])

        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity, method)

    @parameterized.expand(itertools.product(
        ['fd', 'cs'],
        ), name_func=_test_func_name
    )
    def test_simple_partials_explicit(self, method):
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        mask = np.array(
                [[1, 0, 0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0, 1, 1],
                    [0, 1, 0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                    [0, 1, 1, 0, 1, 1, 1]]
            )

        if MPI.COMM_WORLD.rank == 0:
            sparsity = setup_sparsity(mask)
            MPI.COMM_WORLD.bcast(sparsity, root=0)
        else:
            sparsity = MPI.COMM_WORLD.bcast(None, root=0)

        isplit = 2
        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                              isplit=isplit, osplit=2,
                                                              num_par_fd=self.N_PROCS))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()
        coloring = comp._compute_approx_coloring(wrt_patterns='x*', method=method)[0]
        comp._save_coloring(coloring)

        # now create a new problem and use the previously generated coloring
        prob = Problem(coloring_dir=self.tempdir)
        model = prob.model

        indeps, conns = setup_indeps(isplit, mask.shape[1], 'indeps', 'comp')

        model.add_subsystem('indeps', indeps)
        comp = model.add_subsystem('comp', SparseCompExplicit(sparsity, method,
                                                                  isplit=isplit, osplit=2,
                                                                  num_par_fd=self.N_PROCS))
        model.connect('indeps.x0', 'comp.x0')
        model.connect('indeps.x1', 'comp.x1')

        # make sure coloring file exists by the time we try to load the spec
        MPI.COMM_WORLD.barrier()

        comp.declare_coloring(wrt='x*', method=method)
        comp.use_fixed_coloring()
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        start_nruns = comp._nruns
        comp._linearize()
        nruns = comp._nruns - start_nruns
        if comp._full_comm:
            nruns = comp._full_comm.allreduce(nruns)
        self.assertEqual(nruns, 5)

        jac = comp._jacobian._subjacs_info
        _check_partial_matrix(comp, jac, sparsity, method)



if __name__ == '__main__':
    unitest.main()

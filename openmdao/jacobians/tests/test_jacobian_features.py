from __future__ import print_function, division

import unittest
import numpy as np
import scipy as sp
import itertools

from six import iteritems
from parameterized import parameterized

from openmdao.api import IndepVarComp, Group, Problem, ExplicitComponent, \
                         COOJacobian, ScipyIterativeSolver

from openmdao.devtools.testutil import assert_rel_error


class SimpleComp(ExplicitComponent):
    def initialize_variables(self):
        self.add_input('x', shape=1)
        self.add_input('y1', shape=2)
        self.add_input('y2', shape=2)
        self.add_input('y3', shape=2)
        self.add_input('z', shape=(2, 2))

        self.add_output('f', shape=1)
        self.add_output('g', shape=(2, 2))

    def compute(self, inputs, outputs):
        outputs['f'] = np.sum(inputs['z']) + inputs['x']
        outputs['g'] = (np.outer(inputs['y1'] + inputs['y3'], np.ones(2))
                        + np.outer(np.ones(2), inputs['y2'])
                        + inputs['x']*np.eye(2))

    def compute_partial_derivs(self, inputs, outputs, partials):
        partials['f', 'x'] = 1.
        partials['f', 'z'] = np.ones((1, 4))

        partials['g', 'y1'] = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        partials['g', 'y2'] = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        partials['g', 'y3'] = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        partials['g', 'x'] = np.eye(2)


class SimpleCompDependence(SimpleComp):
    def initialize_partials(self):
        self.declare_partials('f', 'y1', dependent=False)
        self.declare_partials('f', 'y2', dependent=False)
        self.declare_partials('f', 'y3', dependent=False)
        self.declare_partials('g', 'z', dependent=False)


class SimpleCompGlob(SimpleComp):
    def initialize_partials(self):
        # This matches y1, y2, and y3.
        self.declare_partials('f', 'y*', dependent=False)

        # This matches y1 and y3.
        self.declare_partials('g', 'y[13]', val=[[1, 0], [1, 0], [0, 1], [0, 1]])


class SimpleCompConst(SimpleComp):
    def initialize_partials(self):
        self.declare_partials('f', ['y1', 'y2', 'y3'], dependent=False)
        self.declare_partials('g', 'z', dependent=False)

        self.declare_partials('f', 'x', val=1.)
        self.declare_partials('f', 'z', val=np.ones((1, 4)))
        self.declare_partials('g', 'y[13]', val=[[1, 0], [1, 0], [0, 1], [0, 1]])
        self.declare_partials('g', 'y2', val=[1., 1., 1., 1.], cols=[0, 0, 1, 1], rows=[0, 2, 1, 3])
        self.declare_partials('g', 'x', val=sp.sparse.coo_matrix(((1., 1.), ((0, 3), (0, 0)))))

    def compute(self, inputs, outputs):
        outputs['f'] = np.sum(inputs['z']) + inputs['x']
        outputs['g'] = np.outer(inputs['y1'] + inputs['y3'], inputs['y2']) + inputs['x'] * np.eye(2)

    def compute_partial_derivs(self, inputs, outputs, partials):
        pass

class SimpleCompFD(SimpleComp):
    def __init__(self, **kwargs):
        super(SimpleCompFD, self).__init__()
        self.kwargs = kwargs

    def initialize_partials(self):
        self.declare_partials('f', ['y1', 'y2', 'y3'], dependent=False)
        self.declare_partials('g', 'z', dependent=False)

        self.approx_partials('*', '*', **self.kwargs)

    def compute_partial_derivs(self, inputs, outputs, partials):
        pass


class SimpleCompMixedFD(SimpleComp):
    def __init__(self, **kwargs):
        super(SimpleCompMixedFD, self).__init__()
        self.kwargs = kwargs

    def initialize_partials(self):
        self.declare_partials('f', ['y1', 'y2', 'y3'], dependent=False)
        self.declare_partials('g', 'z', dependent=False)

        self.approx_partials('g', 'x', **self.kwargs)
        self.approx_partials('g', 'y2', **self.kwargs)

    def compute_partial_derivs(self, inputs, outputs, partials):
        partials['f', 'x'] = 1.
        partials['f', 'z'] = np.ones((1, 4))

        partials['g', 'y1'] = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        partials['g', 'y3'] = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        # dg/dx and dg/dy2 are FD'd


class SimpleCompKwarg(SimpleComp):
    def __init__(self, partial_kwargs):
        self.partial_kwargs = partial_kwargs
        super(SimpleCompKwarg, self).__init__()

    def initialize_partials(self):
        self.declare_partials(**self.partial_kwargs)

    def compute_partial_derivs(self, inputs, outputs, partials):
        pass


class TestJacobianFeatures(unittest.TestCase):

    def setUp(self):
        self.model = model = Group()
        comp = IndepVarComp()
        variables = (
            ('x', 1.),
            ('y1', np.ones(2)),
            ('y2', np.ones(2)),
            ('y3', np.ones(2)),
            ('z', np.ones((2, 2))),
        )
        for name, val in variables:
            comp.add_output(name, val)
        model.add_subsystem('input_comp', comp, promotes=['x', 'y1', 'y2', 'y3', 'z'])

        self.problem = Problem(model=model)
        model.suppress_solver_output = True
        model.ln_solver = ScipyIterativeSolver()
        model.jacobian = COOJacobian()

    def test_dependence(self):
        problem = self.problem
        model = problem.model
        model.add_subsystem('simple', SimpleCompConst(),
                            promotes=['x', 'y1', 'y2', 'y3', 'z', 'f', 'g'])
        problem.setup(check=False)
        problem.run_model()

        # Note: since this test is looking for something not user-facing, it is inherently fragile
        # w.r.t. internal implementations.
        model._linearize()
        jac = model._jacobian._int_mtx._matrix

        # Testing dependence by examining the number of entries in the Jacobian. If non-zeros are
        # removed during array creation (e.g. `eliminate_zeros` function on scipy.sparse matrices),
        # then this test will fail since there are zero entries in the sub-Jacobians.

        # 16 for outputs w.r.t. themselves
        # 1 for df/dx
        # 4 for df/dz
        # 8 for dg/dy1
        # 4 for dg/dy2
        # 8 for dg/dy3
        # 2 for dg/dx
        expected_nnz = 16 + 1 + 4 + 8 + 4 + 8 + 2

        self.assertEqual(jac.nnz, expected_nnz)

    @parameterized.expand([
        ({'of': 'f', 'wrt': 'z', 'val': np.ones((1, 5))},
         'simple: d\(f\)/d\(z\): Expected 1x4 but val is 1x5'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, -1, 4], 'cols': [0, 0, 0]},
         'simple: d\(f\)/d\(z\): row indices must be non-negative'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, 0, 0], 'cols': [0, -1, 4]},
         'simple: d\(f\)/d\(z\): col indices must be non-negative'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, 0], 'cols': [0, 4]},
         'simple: d\(f\)/d\(z\): Expected 1x4 but declared at least 1x5'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, 10]},
         'If one of rows/cols is specified, then both must be specified'),
        ({'of': 'f', 'wrt': 'z', 'cols': [0, 10]},
         'If one of rows/cols is specified, then both must be specified'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0], 'cols': [0, 3]},
         'rows and cols must have the same shape, rows: \(1L?,\), cols: \(2L?,\)'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, 0, 0], 'cols': [0, 1, 3], 'val': [0, 1]},
         'If rows and cols are specified, val must be a scalar or have the same shape, '
         'val: \(2L?,\), rows/cols: \(3L?,\)'),
    ])
    def test_bad_sizes(self, partials_kwargs, error_msg):
        comp = SimpleCompKwarg(partials_kwargs)
        problem = self.problem
        model = problem.model
        model.add_subsystem('simple', comp, promotes=['x', 'y1', 'y2', 'y3', 'z', 'f', 'g'])
        with self.assertRaises(ValueError) as ex:
            problem.setup(check=False)
        self.assertRegexpMatches(str(ex.exception), error_msg)

    @parameterized.expand([
        ({'of': 'q', 'wrt': 'z'}, 'No matches were found for of="q"'),
        ({'of': 'f?', 'wrt': 'x'}, 'No matches were found for of="f?"'),
        ({'of': 'f', 'wrt': 'q'}, 'No matches were found for wrt="q"'),
        ({'of': 'f', 'wrt': 'x?'}, 'No matches were found for wrt="x?"'),
    ])
    def test_bad_names(self, partials_kwargs, error_msg):
        comp = SimpleCompKwarg(partials_kwargs)
        problem = self.problem
        model = problem.model
        model.add_subsystem('simple', comp, promotes=['x', 'y1', 'y2', 'y3', 'z', 'f', 'g'])
        with self.assertRaises(ValueError) as ex:
            problem.setup(check=False)
        self.assertEquals(str(ex.exception), error_msg)

    def test_const_jacobian(self):
        model = Group()
        comp = IndepVarComp()
        for name, val in (('x', 1.), ('y1', np.ones(2)), ('y2', np.ones(2)),
                          ('y3', np.ones(2)), ('z', np.ones((2, 2)))):
            comp.add_output(name, val)
        model.add_subsystem('input_comp', comp, promotes=['x', 'y1', 'y2', 'y3', 'z'])

        problem = Problem(model=model)
        model.suppress_solver_output = True
        model.ln_solver = ScipyIterativeSolver()
        model.jacobian = COOJacobian()
        model.add_subsystem('simple', SimpleCompConst(),
                            promotes=['x', 'y1', 'y2', 'y3', 'z', 'f', 'g'])
        problem.setup(check=False)
        problem.run_model()
        totals = problem.compute_total_derivs(['f', 'g'],
                                              ['x', 'y1', 'y2', 'y3', 'z'])

        jacobian = {}
        jacobian['f', 'x'] = 1.
        jacobian['f', 'z'] = np.ones((1, 4))
        jacobian['f', 'y1'] = np.zeros((1, 2))
        jacobian['f', 'y2'] = np.zeros((1, 2))
        jacobian['f', 'y3'] = np.zeros((1, 2))

        jacobian['g', 'y1'] = [[1, 0], [1, 0], [0, 1], [0, 1]]
        jacobian['g', 'y2'] = [[1, 0], [0, 1], [1, 0], [0, 1]]
        jacobian['g', 'y3'] = [[1, 0], [1, 0], [0, 1], [0, 1]]

        jacobian['g', 'x'] = [[1], [0], [0], [1]]
        jacobian['g', 'z'] = np.zeros((4, 4))

        assert_rel_error(self, totals, jacobian)

    @parameterized.expand(
        itertools.product([1e-6, 1e-8],  # Step size
                          ['forward', 'central', 'backward'],  # FD Form
                          ['rel', 'abs'],  # Step calc
                          )
    )
    def test_fd(self, step, form, step_calc):
        comp = SimpleCompFD(step=step, form=form, step_calc=step_calc)
        problem = self.problem
        model = problem.model
        model.add_subsystem('simple', comp, promotes=['x', 'y1', 'y2', 'y3', 'z', 'f', 'g'])

        problem.setup(check=True)
        problem.run_model()
        totals = problem.compute_total_derivs(['f', 'g'],
                                              ['x', 'y1', 'y2', 'y3', 'z'])
        jacobian = {}
        jacobian['f', 'x'] = 1.
        jacobian['f', 'z'] = np.ones((1, 4))
        jacobian['f', 'y1'] = np.zeros((1, 2))
        jacobian['f', 'y2'] = np.zeros((1, 2))
        jacobian['f', 'y3'] = np.zeros((1, 2))

        jacobian['g', 'y1'] = [[1, 0], [1, 0], [0, 1], [0, 1]]
        jacobian['g', 'y2'] = [[1, 0], [0, 1], [1, 0], [0, 1]]
        jacobian['g', 'y3'] = [[1, 0], [1, 0], [0, 1], [0, 1]]

        jacobian['g', 'x'] = [[1], [0], [0], [1]]
        jacobian['g', 'z'] = np.zeros((4, 4))

        assert_rel_error(self, totals, jacobian, 1e-6)

    def test_mixed_fd(self):
        comp = SimpleCompMixedFD()
        problem = self.problem
        model = problem.model
        model.add_subsystem('simple', comp, promotes=['x', 'y1', 'y2', 'y3', 'z', 'f', 'g'])

        problem.setup(check=True)
        problem.run_model()
        totals = problem.compute_total_derivs(['f', 'g'],
                                              ['x', 'y1', 'y2', 'y3', 'z'])
        jacobian = {}
        jacobian['f', 'x'] = 1.
        jacobian['f', 'z'] = np.ones((1, 4))
        jacobian['f', 'y1'] = np.zeros((1, 2))
        jacobian['f', 'y2'] = np.zeros((1, 2))
        jacobian['f', 'y3'] = np.zeros((1, 2))

        jacobian['g', 'y1'] = [[1, 0], [1, 0], [0, 1], [0, 1]]
        jacobian['g', 'y2'] = [[1, 0], [0, 1], [1, 0], [0, 1]]
        jacobian['g', 'y3'] = [[1, 0], [1, 0], [0, 1], [0, 1]]

        jacobian['g', 'x'] = [[1], [0], [0], [1]]
        jacobian['g', 'z'] = np.zeros((4, 4))

        assert_rel_error(self, totals, jacobian, 1e-6)

    def test_units_fd(self):
        class UnitCompBase(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('T', val=284., units="degR", desc="Temperature")
                self.add_input('P', val=1., units='lbf/inch**2', desc="Pressure")

                self.add_output('flow:T', val=284., units="degR", desc="Temperature")
                self.add_output('flow:P', val=1., units='lbf/inch**2', desc="Pressure")

            def initialize_partials(self):
                self.approx_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['flow:T'] = inputs['T']
                outputs['flow:P'] = inputs['P']

        p = Problem()
        model = p.model = Group()
        indep = model.add_subsystem('indep', IndepVarComp(), promotes=['*'])

        indep.add_output('T', val=100., units='degK')
        indep.add_output('P', val=1., units='bar')

        units = model.add_subsystem('units', UnitCompBase(), promotes=['*'])

        p.setup()
        p.run_model()
        totals = p.compute_total_derivs(['flow:T', 'flow:P'], ['T', 'P'])
        expected_totals = {
            ('flow:T', 'T'): 9/5,
            ('flow:P', 'T'): 0.,
            ('flow:T', 'P'): 0.,
            ('flow:P', 'P'): 14.50377,
        }
        assert_rel_error(self, totals, expected_totals, 1e-6)

        expected_subjacs = {
            ('units.flow:T', 'units.T'): -1.,
            ('units.flow:P', 'units.T'): 0.,
            ('units.flow:T', 'units.P'): 0.,
            ('units.flow:P', 'units.P'): -1.,
        }

        jac = units._jacobian._subjacs
        for deriv, val in iteritems(expected_subjacs):
            assert_rel_error(self, jac[deriv], val, 1e-6)

if __name__ == '__main__':
    unittest.main()

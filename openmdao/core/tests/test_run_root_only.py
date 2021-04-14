
import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI
from openmdao.test_suite.components.expl_comp_array import TestExplCompArraySparse, TestExplCompArrayJacVec
from openmdao.test_suite.components.impl_comp_array import TestImplCompArraySparse, TestImplCompArrayMatVec

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
        super().compute_jacvec_product(inputs, dinputs, result, mode)
        self.njacvec_products += 1


class ImplCompArraySparseCounted(TestImplCompArraySparse):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nsolve_nonlinears = 0
        self.napply_nonlinears = 0
        self.nlinearizes = 0

    def apply_nonlinear(self, inputs, outputs, residuals):
        super().apply_nonlinear(inputs, outputs, residuals)
        self.napply_nonlinears += 1

    def solve_nonlinear(self, inputs, outputs):
        super().solve_nonlinear(inputs, outputs)
        self.nsolve_nonlinears += 1

    def linearize(self, inputs, outputs, jacobian):
        super().linearize(inputs, outputs, jacobian)
        self.nlinearizes += 1


class ImplCompArrayMatVecCounted(TestImplCompArrayMatVec):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.napply_linears = 0
        self.nsolve_nonlinears = 0
        self.napply_nonlinears = 0

    def apply_nonlinear(self, inputs, outputs, residuals):
        super().apply_nonlinear(inputs, outputs, residuals)
        self.napply_nonlinears += 1

    def solve_nonlinear(self, inputs, outputs):
        super().solve_nonlinear(inputs, outputs)
        self.nsolve_nonlinears += 1

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals,
                     mode):
        super().apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
        self.napply_linears += 1


@unittest.skipUnless(MPI is not None and PETScVector is not None, "MPI and PETSc are required.")
class TestRunRootOnly(unittest.TestCase):
    N_PROCS = 3

    # these tests just take a serial model and replicate it in 3 procs, verifying that
    # compute/etc. run only on rank 0

    def test_serial_replication_ex(self):
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

    def test_serial_replication_ex_root_only(self):
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

    def test_serial_replication_ex_jacvec_root_only(self):
        p = om.Problem()

        p.model.add_subsystem('C1', ExplCompArrayJacVecCounted(run_root_only=True),
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
            self.assertEqual(p.model.C1.njacvec_products, 8)
        else:
            self.assertEqual(p.model.C1.njacvec_products, 0)

        np.testing.assert_allclose(J['C3.y', 'C1.lengths'], np.eye(4) * 4.5)
        np.testing.assert_allclose(J['C3.y', 'C1.widths'], np.eye(4) * 1.5)
        np.testing.assert_allclose(J['C2.y', 'C1.lengths'], [np.ones(4) * 7.5])
        np.testing.assert_allclose(J['C2.y', 'C1.widths'], [np.ones(4) * 2.5])

    def test_serial_replication_impl(self):
        # run_root_only is False
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(2)))
        comp = model.add_subsystem('comp', ImplCompArraySparseCounted())
        model.connect('indeps.x', 'comp.rhs')

        prob.setup()
        prob.run_model()

        self.assertEqual(comp.nsolve_nonlinears, 1)

        np.testing.assert_allclose(prob['comp.rhs'], np.ones(2))
        np.testing.assert_allclose(prob['comp.x'], np.ones(2))

        model.run_linearize()

        np.testing.assert_allclose(comp._jacobian['comp.x', 'comp.x'], comp.mtx)
        np.testing.assert_allclose(comp._jacobian['comp.x', 'comp.rhs'], -np.ones(2))

    def test_serial_replication_impl_root_only(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(2)))
        comp = model.add_subsystem('comp', ImplCompArraySparseCounted(run_root_only=True))
        model.connect('indeps.x', 'comp.rhs')

        prob.setup()
        prob.run_model()

        if prob.comm.rank == 0:
            self.assertEqual(comp.nsolve_nonlinears, 1)
        else:
            self.assertEqual(comp.nsolve_nonlinears, 0)

        np.testing.assert_allclose(prob['comp.rhs'], np.ones(2))
        np.testing.assert_allclose(prob['comp.x'], np.ones(2))

        model.run_linearize()

        if prob.comm.rank == 0:
            self.assertEqual(comp.nlinearizes, 1)
        else:
            self.assertEqual(comp.nlinearizes, 0)

        np.testing.assert_allclose(comp._jacobian['comp.x', 'comp.x'], comp.mtx)
        np.testing.assert_allclose(comp._jacobian['comp.x', 'comp.rhs'], -np.ones(2))

        comp._outputs['x'] = np.array([1.5, 2.5])
        comp._inputs['rhs'] = np.array([2., 3.])

        comp.run_apply_nonlinear()

        if prob.comm.rank == 0:
            self.assertEqual(comp.napply_nonlinears, 1)
        else:
            self.assertEqual(comp.napply_nonlinears, 0)

        np.testing.assert_allclose(comp._residuals['x'],
                                   comp.mtx.dot(np.array([1.5, 2.5]))-np.array([2., 3.]))

        J = prob.compute_totals(of='comp.x', wrt='indeps.x')

        np.testing.assert_allclose(J['comp.x', 'indeps.x'], np.eye(2))

    def test_serial_replication_impl_apply_linear_root_only(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(2)))
        comp = model.add_subsystem('comp', ImplCompArrayMatVecCounted(run_root_only=True))
        model.connect('indeps.x', 'comp.rhs')

        prob.setup()
        prob.run_model()

        if prob.comm.rank == 0:
            self.assertEqual(comp.nsolve_nonlinears, 1)
        else:
            self.assertEqual(comp.nsolve_nonlinears, 0)

        np.testing.assert_allclose(prob['comp.rhs'], np.ones(2))
        np.testing.assert_allclose(prob['comp.x'], np.ones(2))

        comp._outputs['x'] = np.array([1.5, 2.5])
        comp._inputs['rhs'] = np.array([2., 3.])

        comp.run_apply_nonlinear()

        if prob.comm.rank == 0:
            self.assertEqual(comp.napply_nonlinears, 1)
        else:
            self.assertEqual(comp.napply_nonlinears, 0)

        np.testing.assert_allclose(comp._residuals['x'],
                                   comp.mtx.dot(np.array([1.5, 2.5]))-np.array([2., 3.]))

        J = prob.compute_totals(of='comp.x', wrt='indeps.x')

        if prob.comm.rank == 0:
            self.assertEqual(comp.napply_linears, 2)
        else:
            self.assertEqual(comp.napply_linears, 0)

        np.testing.assert_allclose(J['comp.x', 'indeps.x'], np.eye(2))


@unittest.skipUnless(MPI is not None and PETScVector is not None, "MPI and PETSc are required.")
class TestRunRootOnlyErrors(unittest.TestCase):
    N_PROCS = 2

    def test_not_serial_err(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x')
                self.add_output('y', val=np.ones(2), distributed=True)

            def compute(self, inputs, outputs):
                pass

        p = om.Problem()
        p.model.add_subsystem('comp', MyComp(run_root_only=True))
        p.setup()
        with self.assertRaises(Exception) as cm:
            p.run_model()
        self.assertEqual(cm.exception.args[0], f"'comp' <class MyComp>: Can't set 'run_root_only' option when a component has distributed variables.")

    def test_parallel_fd_err(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x')
                self.add_output('y', val=np.ones(2))

            def setup_partials(self):
                self.declare_partials('y', 'x', method='fd')

            def compute(self, inputs, outputs):
                pass

        p = om.Problem()

        p.model.add_subsystem('comp', MyComp(num_par_fd=2, run_root_only=True))
        p.setup()
        with self.assertRaises(Exception) as cm:
            p.run_model()
        self.assertEqual(cm.exception.args[0], f"'comp' <class MyComp>: Can't set 'run_root_only' option when using parallel FD.")

    def test_parallel_deriv_color(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x')
                self.add_output('y', val=np.ones(2))

            def setup_partials(self):
                self.declare_partials('y', 'x',)

            def compute(self, inputs, outputs):
                pass

        p = om.Problem()

        p.model.add_subsystem('comp', MyComp(run_root_only=True))
        p.model.add_constraint('comp.y', lower=0., upper=10., parallel_deriv_color='foobar')

        p.setup(mode='rev')
        with self.assertRaises(Exception) as cm:
            p.run_model()
        self.assertEqual(cm.exception.args[0], f"'comp' <class MyComp>: Can't set 'run_root_only' option when using parallel_deriv_color.")



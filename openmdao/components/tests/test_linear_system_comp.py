"""Test the LinearSystemComp."""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class TestLinearSystemComp(unittest.TestCase):
    """Test the LinearSystemComp class with a 3x3 linear system."""

    def test_basic(self):
        """Check against the scipy solver."""

        model = om.Group()

        x = np.array([1, 2, -3])
        A = np.array([[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
        b = A.dot(x)

        model.add_subsystem('p1', om.IndepVarComp('A', A))
        model.add_subsystem('p2', om.IndepVarComp('b', b))

        lingrp = model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', om.LinearSystemComp(size=3))

        model.connect('p1.A', 'lin.A')
        model.connect('p2.b', 'lin.b')

        prob = om.Problem(model)
        prob.setup()

        lingrp.linear_solver = om.ScipyKrylov()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['lin.x'], x, .0001)
        assert_near_equal(prob.model._residuals.get_norm(), 0.0, 1e-10)

        model.run_apply_nonlinear()

        with model._scaled_context_all():
            val = model.lingrp.lin._residuals['x']
            assert_near_equal(val, np.zeros((3, )), tolerance=1e-8)

    def test_vectorized(self):
        """Check against the scipy solver."""

        model = om.Group()

        x = np.array([[1, 2, -3], [2, -1, 4]])
        A = np.array([[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
        b = np.einsum('jk,ik->ij', A, x)

        model.add_subsystem('p1', om.IndepVarComp('A', A))
        model.add_subsystem('p2', om.IndepVarComp('b', b))

        lingrp = model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', om.LinearSystemComp(size=3, vec_size=2))

        model.connect('p1.A', 'lin.A')
        model.connect('p2.b', 'lin.b')

        prob = om.Problem(model)
        prob.setup()

        lingrp.linear_solver = om.ScipyKrylov()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['lin.x'], x, .0001)
        assert_near_equal(prob.model._residuals.get_norm(), 0.0, 1e-10)

        model.run_apply_nonlinear()

        with model._scaled_context_all():
            val = model.lingrp.lin._residuals['x']
            assert_near_equal(val, np.zeros((2, 3)), tolerance=1e-8)

    def test_vectorized_A(self):
        """Check against the scipy solver."""

        model = om.Group()

        x = np.array([[1, 2, -3], [2, -1, 4]])
        A = np.array([[[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]],
                      [[2.0, 3.0, 4.0], [1.0, -1.0, -2.0], [3.0, 2.0, -2.0]]])
        b = np.einsum('ijk,ik->ij', A, x)

        model.add_subsystem('p1', om.IndepVarComp('A', A))
        model.add_subsystem('p2', om.IndepVarComp('b', b))

        lingrp = model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', om.LinearSystemComp(size=3, vec_size=2, vectorize_A=True))

        model.connect('p1.A', 'lin.A')
        model.connect('p2.b', 'lin.b')

        prob = om.Problem(model)
        prob.setup()

        lingrp.linear_solver = om.ScipyKrylov()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['lin.x'], x, .0001)
        assert_near_equal(prob.model._residuals.get_norm(), 0.0, 1e-10)

        model.run_apply_nonlinear()

        with model._scaled_context_all():
            val = model.lingrp.lin._residuals['x']
            assert_near_equal(val, np.zeros((2, 3)), tolerance=1e-8)

    def test_solve_linear(self):
        """Check against solve_linear."""

        x = np.array([1, 2, -3])
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = A.dot(x)
        b_T = A.T.dot(x)

        lin_sys_comp = om.LinearSystemComp(size=3)

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('A', A))
        prob.model.add_subsystem('p2', om.IndepVarComp('b', b))

        lingrp = prob.model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', lin_sys_comp)

        prob.model.connect('p1.A', 'lin.A')
        prob.model.connect('p2.b', 'lin.b')

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()
        prob.model.run_linearize()

        # Compare against calculated derivs
        Ainv = np.array([[3., -2., 1.],
                         [-3., 3., -2.],
                         [1., -1., 1.]])

        dx_dA = np.outer(Ainv, -x).reshape(3, 9)
        dx_db = Ainv

        d_inputs, d_outputs, d_residuals = lingrp.get_linear_vectors()

        # Forward mode with RHS of self.b
        d_residuals['lin.x'] = b
        lingrp.run_solve_linear(['linear'], 'fwd')
        sol = d_outputs['lin.x']
        assert_near_equal(sol, x, .0001)

        # Reverse mode with RHS of self.b_T
        d_outputs['lin.x'] = b_T
        lingrp.run_solve_linear(['linear'], 'rev')
        sol = d_residuals['lin.x']
        assert_near_equal(sol, x, .0001)

        prob.model.lingrp.lin._no_check_partials = False  # override skipping of check_partials

        J = prob.compute_totals(['lin.x'], ['p1.A', 'p2.b', 'lin.x'], return_format='flat_dict')
        assert_near_equal(J['lin.x', 'p1.A'], dx_dA, .0001)
        assert_near_equal(J['lin.x', 'p2.b'], dx_db, .0001)

        data = prob.check_partials(out_stream=None)

        abs_errors = data['lingrp.lin'][('x', 'x')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        self.assertTrue(abs_errors[0] < 1.e-6)

    def test_solve_linear_vectorized(self):
        """Check against solve_linear."""

        x = np.array([[1, 2, -3], [2, -1, 4]])
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.einsum('jk,ik->ij', A, x)
        b_T = np.einsum('jk,ik->ij', A.T, x)

        lin_sys_comp = om.LinearSystemComp(size=3, vec_size=2)

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('A', A))
        prob.model.add_subsystem('p2', om.IndepVarComp('b', b))

        lingrp = prob.model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', lin_sys_comp)

        prob.model.connect('p1.A', 'lin.A')
        prob.model.connect('p2.b', 'lin.b')

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()
        prob.model.run_linearize()

        # Compare against calculated derivs
        Ainv = np.array([[3., -2., 1.],
                         [-3., 3., -2.],
                         [1., -1., 1.]])

        dx_dA0 = np.outer(Ainv, -x[0]).reshape(3, 9)
        dx_dA1 = np.outer(Ainv, -x[1]).reshape(3, 9)
        dx_dA = np.vstack((dx_dA0, dx_dA1))
        dx_db = np.kron(np.eye(2), Ainv)

        d_inputs, d_outputs, d_residuals = lingrp.get_linear_vectors()

        # Forward mode with RHS of self.b
        d_residuals['lin.x'] = b
        lingrp.run_solve_linear(['linear'], 'fwd')
        sol = d_outputs['lin.x']
        assert_near_equal(sol, x, .0001)

        # Reverse mode with RHS of self.b_T
        d_outputs['lin.x'] = b_T
        lingrp.run_solve_linear(['linear'], 'rev')
        sol = d_residuals['lin.x']
        assert_near_equal(sol, x, .0001)

        prob.model.lingrp.lin._no_check_partials = False  # override skipping of check_partials

        J = prob.compute_totals(['lin.x'], ['p1.A', 'p2.b'], return_format='flat_dict')
        assert_near_equal(J['lin.x', 'p1.A'], dx_dA, .0001)
        assert_near_equal(J['lin.x', 'p2.b'], dx_db, .0001)

        data = prob.check_partials(out_stream=None)

        abs_errors = data['lingrp.lin'][('x', 'x')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        self.assertTrue(abs_errors[0] < 1.e-6)

    def test_solve_linear_vectorized_A(self):
        """Check against solve_linear."""
        x = np.array([[1, 2, -3], [2, -1, 4]])
        A = np.array([[[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]],
                      [[2.0, 3.0, 4.0], [1.0, -1.0, -2.0], [3.0, 2.0, -2.0]]])
        b = np.einsum('ijk,ik->ij', A, x)
        b_T = np.einsum('ijk,ik->ij', A.transpose(0, 2, 1), x)

        lin_sys_comp = om.LinearSystemComp(size=3, vec_size=2, vectorize_A=True)

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('A', A))
        prob.model.add_subsystem('p2', om.IndepVarComp('b', b))

        lingrp = prob.model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', lin_sys_comp)

        prob.model.connect('p1.A', 'lin.A')
        prob.model.connect('p2.b', 'lin.b')

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()
        prob.model.run_linearize()

        # Compare against calculated derivs
        Ainv1 = np.array([[3., -2., 1.],
                          [-3., 3., -2.],
                          [1., -1., 1.]])
        Ainv2 = np.array([[ 0.3 ,  0.7 , -0.1 ],
                          [-0.2 , -0.8 ,  0.4 ],
                          [ 0.25,  0.25, -0.25]])

        dx_dA0 = np.outer(Ainv1, -x[0]).reshape(3, 9)
        dx_dA1 = np.outer(Ainv2, -x[1]).reshape(3, 9)

        dx_dA = np.zeros((6, 18))
        dx_dA[:3, :9] = dx_dA0
        dx_dA[3:, 9:] = dx_dA1

        dx_db = np.zeros((6, 6))
        dx_db[:3, :3] = Ainv1
        dx_db[3:, 3:] = Ainv2

        d_inputs, d_outputs, d_residuals = lingrp.get_linear_vectors()

        # Forward mode with RHS of self.b
        d_residuals['lin.x'] = b
        lingrp.run_solve_linear(['linear'], 'fwd')
        sol = d_outputs['lin.x']
        assert_near_equal(sol, x, .0001)

        # Reverse mode with RHS of self.b_T
        d_outputs['lin.x'] = b_T
        lingrp.run_solve_linear(['linear'], 'rev')
        sol = d_residuals['lin.x']
        assert_near_equal(sol, x, .0001)

        prob.model.lingrp.lin._no_check_partials = False  # override skipping of check_partials

        J = prob.compute_totals(['lin.x'], ['p1.A', 'p2.b'], return_format='flat_dict')
        assert_near_equal(J['lin.x', 'p1.A'], dx_dA, .0001)
        assert_near_equal(J['lin.x', 'p2.b'], dx_db, .0001)

        data = prob.check_partials(out_stream=None)

        abs_errors = data['lingrp.lin'][('x', 'x')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        self.assertTrue(abs_errors[0] < 1.e-6)

    def test_feature_basic(self):
        import numpy as np

        import openmdao.api as om

        model = om.Group()

        A = np.array([[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
        b = np.array([1.0, 2.0, -3.0])

        lingrp = model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', om.LinearSystemComp(size=3))

        prob = om.Problem(model)
        prob.setup()

        prob.set_val('lin.A', A)
        prob.set_val('lin.b', b)

        lingrp.linear_solver = om.ScipyKrylov()

        prob.run_model()

        assert_near_equal(prob.get_val('lin.x'), np.array([0.36423841, -0.00662252, -0.4205298 ]), .0001)

    def test_feature_vectorized(self):
        import numpy as np

        import openmdao.api as om

        model = om.Group()

        A = np.array([[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
        b = np.array([[2.0, -3.0, 4.0], [1.0, 0.0, -1.0]])

        lingrp = model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', om.LinearSystemComp(size=3, vec_size=2))

        prob = om.Problem(model)
        prob.setup()

        prob.set_val('lin.A', A)
        prob.set_val('lin.b', b)

        lingrp.linear_solver = om.ScipyKrylov()

        prob.run_model()

        assert_near_equal(prob.get_val('lin.x'), np.array([[ 0.10596026, -0.16556291,  0.48675497],
                                                        [ 0.19205298, -0.11258278, -0.14900662]]),
                         .0001)

    def test_feature_vectorized_A(self):
        import numpy as np

        import openmdao.api as om

        model = om.Group()

        A = np.array([[[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]],
                      [[2.0, 3.0, 4.0], [1.0, -1.0, -2.0], [3.0, 2.0, -2.0]]])
        b = np.array([[-5.0, 2.0, 3.0], [-1.0, 1.0, -3.0]])

        lingrp = model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', om.LinearSystemComp(size=3, vec_size=2, vectorize_A=True))

        prob = om.Problem(model)
        prob.setup()

        prob.set_val('lin.A', A)
        prob.set_val('lin.b', b)

        lingrp.linear_solver = om.ScipyKrylov()

        prob.run_model()

        assert_near_equal(prob.get_val('lin.x'), np.array([[-0.78807947,  0.66887417,  0.47350993],
                                                        [ 0.7       , -1.8       ,  0.75      ]]),
                         .0001)

if __name__ == "__main__":
    unittest.main()

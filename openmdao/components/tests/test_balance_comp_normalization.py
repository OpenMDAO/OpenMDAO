"""Regression coverage for elementwise balance normalization."""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import force_check_partials, use_tempdirs


@use_tempdirs
class TestBalanceNormalization(unittest.TestCase):
    """Check mixed normalization branches in shaped balances."""

    def test_mixed_branch_partials(self):
        """Compare every partial with complex step, including multipliers."""
        values = np.array([0., 1., -10., -1., 2., 10.])
        for shape in ((6,), (2, 3), (1, 2, 3)):
            for normalize in (False, True):
                for use_mult in (False, True):
                    with self.subTest(shape=shape, normalize=normalize, use_mult=use_mult):
                        rhs = values.reshape(shape)
                        p = om.Problem(reports=False)
                        balance = om.BalanceComp(
                            'x', val=np.ones(shape), rhs_val=rhs,
                            normalize=normalize, use_mult=use_mult)
                        p.model.add_subsystem('balance', balance)
                        p.setup(force_alloc_complex=True)
                        p.set_val('balance.lhs:x', np.arange(6.).reshape(shape) + 3.)
                        if use_mult:
                            p.set_val('balance.mult:x', np.linspace(.5, 2., 6).reshape(shape))
                        p.run_model()
                        data = force_check_partials(p, method='cs', out_stream=None)
                        try:
                            assert_check_partials(data, atol=1e-12, rtol=1e-12)
                        finally:
                            p.cleanup()

    def test_linear_solve_is_independent_of_shape(self):
        """A normalized linear balance should converge in one Newton step."""
        for shape in ((2,), (1, 2), (1, 1, 2)):
            for normalize in (False, True):
                with self.subTest(shape=shape, normalize=normalize):
                    rhs = np.array([1., 10.]).reshape(shape)
                    p = om.Problem(reports=False)
                    p.model.add_subsystem(
                        'balance', om.BalanceComp(
                            'x', val=np.ones(shape), rhs_val=rhs, normalize=normalize))
                    p.model.add_subsystem(
                        'lhs', om.ExecComp('y=x', x=np.ones(shape), y=np.ones(shape)))
                    p.model.connect('balance.x', 'lhs.x')
                    p.model.connect('lhs.y', 'balance.lhs:x')
                    p.model.nonlinear_solver = om.NewtonSolver(
                        solve_subsystems=False, maxiter=1, iprint=-1,
                        atol=1e-12, rtol=1e-12, err_on_non_converge=True)
                    p.model.linear_solver = om.DirectSolver(iprint=-1)
                    p.setup()
                    try:
                        p.run_model()
                        np.testing.assert_allclose(
                            p.get_val('balance.x'), rhs, atol=1e-12, rtol=0.)
                    finally:
                        p.cleanup()


if __name__ == '__main__':
    unittest.main()

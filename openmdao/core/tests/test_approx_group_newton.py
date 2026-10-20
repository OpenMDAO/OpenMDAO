"""A group that approximates its own derivatives and also owns a Newton or Broyden solver."""
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class _Lin(om.ExplicitComponent):
    """out = sum(c * in) with constant partials."""

    def initialize(self):
        self.options.declare('coefs')

    def setup(self):
        for name in self.options['coefs']:
            self.add_input(name, 1.0)
        self.add_output('out', 1.0)
        for name, c in self.options['coefs'].items():
            self.declare_partials('out', name, val=c)

    def compute(self, inputs, outputs):
        outputs['out'] = sum(c * inputs[n] for n, c in self.options['coefs'].items())


class _State(om.ImplicitComponent):
    """R(w) = 4*w - sum(c * in), which does not solve itself."""

    def initialize(self):
        self.options.declare('coefs')

    def setup(self):
        self.add_output('w', 1.0)
        for name in self.options['coefs']:
            self.add_input(name, 1.0)
        self.declare_partials('w', 'w', val=4.0)
        for name, c in self.options['coefs'].items():
            self.declare_partials('w', name, val=-c)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['w'] = 4.0 * outputs['w'] - sum(c * inputs[n]
                                                  for n, c in self.options['coefs'].items())

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['w'] = d_residuals['w'] / 4.0
        else:
            d_residuals['w'] = d_outputs['w'] / 4.0


_LINEAR = {
    'direct': lambda: om.DirectSolver(assemble_jac=False),
    'direct_asm': lambda: om.DirectSolver(assemble_jac=True),
    'krylov': lambda: om.ScipyKrylov(atol=1e-12, rtol=1e-12, maxiter=100),
    'lnbgs': lambda: om.LinearBlockGS(atol=1e-12, rtol=1e-12, maxiter=100),
    'lnbj': lambda: om.LinearBlockJac(atol=1e-12, rtol=1e-12, maxiter=200),
}


def _chain(linear, method, maxiter=20):
    """Chain model: ivc -> G[c1 (y = 2x) -> state (4w = y)].  Newton needs one iteration."""
    p = om.Problem()
    p.model.add_subsystem('ivc', om.IndepVarComp('x', 1.5))
    G = p.model.add_subsystem('G', om.Group())
    G.add_subsystem('c1', _Lin(coefs={'x': 2.0}))
    G.add_subsystem('state', _State(coefs={'y': 1.0}))
    G.connect('c1.out', 'state.y')
    p.model.connect('ivc.x', 'G.c1.x')
    G.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, atol=1e-12, rtol=1e-12,
                                         maxiter=maxiter, iprint=-1, err_on_non_converge=True)
    G.linear_solver = _LINEAR[linear]()
    if method is not None:
        G.approx_totals(method=method)
    return p, G


def _coupled(linear, method, nonlinear='newton'):
    """Coupled model: ivc -> G[c1 (y1 = x + y2/2) <-> c2 (y2 = y1/4)].  Exact y1 = 8x/7."""
    p = om.Problem()
    p.model.add_subsystem('ivc', om.IndepVarComp('x', 1.0))
    G = p.model.add_subsystem('G', om.Group())
    G.add_subsystem('c1', _Lin(coefs={'x': 1.0, 'y2': 0.5}))
    G.add_subsystem('c2', _Lin(coefs={'y1': 0.25}))
    G.connect('c1.out', 'c2.y1')
    G.connect('c2.out', 'c1.y2')
    p.model.connect('ivc.x', 'G.c1.x')
    if nonlinear == 'newton':
        G.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, atol=1e-12, rtol=1e-12,
                                             maxiter=10, iprint=-1, err_on_non_converge=True)
    elif nonlinear == 'newton_ss':
        G.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, atol=1e-12, rtol=1e-12,
                                             maxiter=10, iprint=-1, err_on_non_converge=True)
    else:
        G.nonlinear_solver = om.BroydenSolver(atol=1e-12, rtol=1e-12, maxiter=20, iprint=-1,
                                              compute_jacobian=True, max_jacobians=20,
                                              err_on_non_converge=True)
    G.linear_solver = _LINEAR[linear]()
    if method is not None:
        G.approx_totals(method=method)
    return p, G


class _CountApplies(object):

    def __init__(self, cls):
        self.cls = cls
        self.n = 0

    def __enter__(self):
        orig = self.orig = self.cls.apply_nonlinear

        def apply_nonlinear(comp, inputs, outputs, residuals):
            self.n += 1
            return orig(comp, inputs, outputs, residuals)

        self.cls.apply_nonlinear = apply_nonlinear
        return self

    def __exit__(self, *args):
        self.cls.apply_nonlinear = self.orig


class TestApproxGroupWithNewton(unittest.TestCase):

    def test_first_newton_iteration_uses_the_residual_jacobian(self):
        # Setting up the approximation during Newton's first linearization must not switch the
        # approximation back on: the first Newton step would then be taken with the wrong
        # operator, and the group would compute its finite differences in the middle of it.
        for linear in _LINEAR:
            for method in ('fd', 'cs'):
                with self.subTest(linear=linear, method=method):
                    p, G = _chain(linear, method, maxiter=1)
                    p.setup(force_alloc_complex=True)
                    with _CountApplies(_State) as count:
                        p.run_model()
                    self.assertEqual(G.nonlinear_solver._iter_count, 1)
                    assert_near_equal(p.get_val('G.state.w'), 0.75, 1e-12)
                    # same number of residual evaluations as without approx_totals()
                    pc, _ = _chain(linear, None, maxiter=1)
                    pc.setup()
                    with _CountApplies(_State) as control:
                        pc.run_model()
                    self.assertEqual(count.n, control.n)

    def test_coupled_group_newton_converges(self):
        # While Newton has switched the approximation off, the group must apply its residual
        # jacobian, not the approximation's semi-totals.
        for nonlinear in ('newton', 'newton_ss'):
            for linear in ('direct', 'direct_asm', 'krylov', 'lnbgs'):
                for method in ('fd', 'cs'):
                    with self.subTest(nonlinear=nonlinear, linear=linear, method=method):
                        p, G = _coupled(linear, method, nonlinear)
                        p.setup(force_alloc_complex=True)
                        p.run_model()
                        assert_near_equal(p.get_val('G.c1.out'), 8.0 / 7.0, 1e-12)
                        self.assertEqual(G.nonlinear_solver._iter_count, 1)

    def test_newton_after_linearize(self):
        # The approximation's jacobian, once it exists, must not be used by a later Newton solve.
        for linear in ('direct', 'krylov', 'lnbgs'):
            for method in ('fd', 'cs'):
                with self.subTest(linear=linear, method=method):
                    p, G = _coupled(linear, method)
                    p.setup(force_alloc_complex=True)
                    p.run_model()
                    p.model.run_linearize()
                    p.set_val('ivc.x', 2.0)
                    p.run_model()
                    assert_near_equal(p.get_val('G.c1.out'), 16.0 / 7.0, 1e-12)
                    self.assertEqual(G.nonlinear_solver._iter_count, 1)

    def test_broyden_inverse_jacobian(self):
        for linear in ('direct', 'direct_asm'):
            for method in ('fd', 'cs'):
                with self.subTest(linear=linear, method=method):
                    pc, Gc = _coupled(linear, None, 'broyden')
                    pc.setup()
                    pc.run_model()
                    p, G = _coupled(linear, method, 'broyden')
                    p.setup(force_alloc_complex=True)
                    p.run_model()
                    assert_near_equal(p.get_val('G.c1.out'), 8.0 / 7.0, 1e-12)
                    self.assertEqual(G.nonlinear_solver._iter_count,
                                     Gc.nonlinear_solver._iter_count)

    def test_totals_through_block_solver_parent(self):
        # The approximation's jacobian must survive the Newton solves that its own finite
        # differences run, or a parent block solver skips the group's semi-totals.
        for linear in ('lnbgs', 'lnbj', 'direct', 'krylov'):
            for parent in ('runonce', 'lnbgs'):
                for mode in ('fwd', 'rev'):
                    with self.subTest(linear=linear, parent=parent, mode=mode):
                        p, G = _chain(linear, 'fd')
                        p.model.linear_solver = (
                            om.LinearRunOnce() if parent == 'runonce' else
                            om.LinearBlockGS(atol=1e-12, rtol=1e-12, maxiter=50))
                        p.setup(mode=mode)
                        p.run_model()
                        J = p.compute_totals(of=['G.state.w'], wrt=['ivc.x'])
                        assert_near_equal(J['G.state.w', 'ivc.x'], [[0.5]], 1e-6)
                        J = p.compute_totals(of=['G.state.w'], wrt=['ivc.x'])
                        assert_near_equal(J['G.state.w', 'ivc.x'], [[0.5]], 1e-6)

    def test_complex_step_with_assembled_solver(self):
        # Under complex step the group's Newton updates its assembled jacobian with complex
        # values; the subjacs it shares with the approximation must be real again afterwards.
        # Complex step and finite differences must agree, on the first and on a later call.
        for parent in ('direct', 'runonce'):
            with self.subTest(parent=parent):
                results = {}
                for method in ('fd', 'cs'):
                    p, G = _coupled('direct_asm', method)
                    p.model.linear_solver = (om.DirectSolver(assemble_jac=False)
                                             if parent == 'direct' else om.LinearRunOnce())
                    p.setup(force_alloc_complex=True)
                    p.run_model()
                    first = p.compute_totals(of=['G.c1.out'], wrt=['ivc.x'])['G.c1.out', 'ivc.x']
                    p.set_val('ivc.x', 2.0)
                    p.run_model()
                    second = p.compute_totals(of=['G.c1.out'], wrt=['ivc.x'])['G.c1.out', 'ivc.x']
                    results[method] = (first.item(), second.item())
                assert_near_equal(results['cs'], results['fd'], 1e-6)

    def test_assembled_jacobian_when_totals_come_first(self):
        # The approximation's jacobian may exist before the first suspension (totals computed
        # before any run_model).  The group's assembled jacobian must still be built for Newton.
        for method in ('fd', 'cs'):
            with self.subTest(method=method):
                p, G = _coupled('direct_asm', method)
                p.setup(force_alloc_complex=True)
                # start from the converged state, so the finite differences have a valid base
                p.set_val('G.c1.out', 8.0 / 7.0)
                p.set_val('G.c2.out', 2.0 / 7.0)
                J = p.compute_totals(of=['G.c1.out'], wrt=['ivc.x'])
                assert_near_equal(J['G.c1.out', 'ivc.x'], [[8.0 / 7.0]], 1e-6)
                self.assertIsNotNone(G._assembled_jac)
                self.assertIsNot(G._jacobian, G._assembled_jac)
                p.set_val('ivc.x', 2.0)
                p.run_model()
                assert_near_equal(p.get_val('G.c1.out'), 16.0 / 7.0, 1e-12)
                self.assertTrue(G.linear_solver._factored_assembled_jac)

    def test_relevance_change_with_assembled_parent(self):
        # A group that does not approximate keeps rebuilding its assembled jacobian when the
        # relevance changes, as before.
        for method in ('fd', 'cs'):
            with self.subTest(method=method):
                p = om.Problem()
                p.model.add_subsystem('ivc', om.IndepVarComp('x', 1.5))
                p.model.add_subsystem('ivc2', om.IndepVarComp('x2', 0.5))
                P = p.model.add_subsystem('P', om.Group())
                P.add_subsystem('a', _Lin(coefs={'x': 2.0}))
                P.add_subsystem('b', _Lin(coefs={'x2': 3.0, 'y': 1.0}))
                P.connect('a.out', 'b.y')
                G = p.model.add_subsystem('G', om.Group())
                G.add_subsystem('c1', _Lin(coefs={'x': 1.0}))
                G.approx_totals(method=method)
                p.model.connect('ivc.x', ['P.a.x', 'G.c1.x'])
                p.model.connect('ivc2.x2', 'P.b.x2')
                P.linear_solver = om.ScipyKrylov(assemble_jac=True, atol=1e-12, rtol=1e-12)
                p.setup(force_alloc_complex=True)
                p.run_model()
                J = p.compute_totals(of=['P.a.out'], wrt=['ivc.x'])
                assert_near_equal(J['P.a.out', 'ivc.x'], [[2.0]], 1e-10)
                J = p.compute_totals(of=['P.b.out'], wrt=['ivc2.x2'])
                assert_near_equal(J['P.b.out', 'ivc2.x2'], [[3.0]], 1e-10)
                J = p.compute_totals(of=['P.b.out'], wrt=['ivc.x'])
                assert_near_equal(J['P.b.out', 'ivc.x'], [[2.0]], 1e-10)

    def test_apply_linear_before_the_first_linearize(self):
        # run_apply_linear reaches an approximating group before its first linearize, so its
        # jacobian exists before the approximation has registered its keys.  The totals
        # afterwards must still use the approximation.
        for mode in ('fwd', 'rev'):
            for method in ('fd', 'cs'):
                with self.subTest(mode=mode, method=method):
                    p = om.Problem()
                    p.model.add_subsystem('ivc', om.IndepVarComp('x', 1.5))
                    G = p.model.add_subsystem('G', om.Group())
                    G.add_subsystem('c1', _Lin(coefs={'x': 2.0}))
                    G.add_subsystem('c2', _Lin(coefs={'y': 0.25}))
                    G.connect('c1.out', 'c2.y')
                    p.model.connect('ivc.x', 'G.c1.x')
                    G.approx_totals(method=method)
                    p.model.linear_solver = om.DirectSolver(assemble_jac=False)
                    p.setup(mode=mode, force_alloc_complex=True)
                    p.run_model()
                    G.run_apply_linear(mode)
                    J = p.compute_totals(of=['G.c2.out'], wrt=['ivc.x'])
                    assert_near_equal(J['G.c2.out', 'ivc.x'], [[0.5]], 1e-6)


if __name__ == '__main__':
    unittest.main()

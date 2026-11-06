"""Totals through a group that approximates its own derivatives and contains implicit states."""
import contextlib
import io
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class _Scale(om.ExplicitComponent):
    """y = a*x."""

    def initialize(self):
        self.options.declare('a', default=2.0)

    def setup(self):
        self.add_input('x', 1.0)
        self.add_output('y', 1.0)
        self.declare_partials('y', 'x', val=self.options['a'])

    def compute(self, inputs, outputs):
        outputs['y'] = self.options['a'] * inputs['x']


class _State(om.ImplicitComponent):
    """R(w, u) = p*w + q*u, so w = -q*u/p."""

    def initialize(self):
        self.options.declare('p', default=4.0)
        self.options.declare('q', default=-1.0)
        self.options.declare('partials', default='linearize', values=('linearize', 'val'))
        self.options.declare('solve', default=True)
        self.options.declare('ref', default=1.0)
        self.options.declare('res_ref', default=None)

    def setup(self):
        self.add_input('u', 1.0)
        self.add_output('w', 1.0, ref=self.options['ref'], res_ref=self.options['res_ref'])
        if self.options['partials'] == 'val':
            self.declare_partials('w', 'w', val=self.options['p'])
            self.declare_partials('w', 'u', val=self.options['q'])
        else:
            self.declare_partials('w', ['w', 'u'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['w'] = self.options['p'] * outputs['w'] + self.options['q'] * inputs['u']

    def solve_nonlinear(self, inputs, outputs):
        if self.options['solve']:
            outputs['w'] = -self.options['q'] * inputs['u'] / self.options['p']

    def linearize(self, inputs, outputs, partials):
        if self.options['partials'] == 'linearize':
            partials['w', 'w'] = self.options['p']
            partials['w', 'u'] = self.options['q']


class _StateWithSolveLinear(_State):
    """_State that can also be inverted by a block linear solver."""

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['w'] = d_residuals['w'] / self.options['p']
        else:
            d_residuals['w'] = d_outputs['w'] / self.options['p']


class _MatrixFreeScale(om.ExplicitComponent):
    """y = a*x with compute_jacvec_product, so it declares no partials."""

    def initialize(self):
        self.options.declare('a', default=0.25)

    def setup(self):
        self.add_input('x', 1.0)
        self.add_output('y', 1.0)

    def compute(self, inputs, outputs):
        outputs['y'] = self.options['a'] * inputs['x']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if 'x' in d_inputs and 'y' in d_outputs:
            if mode == 'fwd':
                d_outputs['y'] += self.options['a'] * d_inputs['x']
            else:
                d_inputs['x'] += self.options['a'] * d_outputs['y']


class _Coupled(om.ImplicitComponent):
    """2a + b - u = 0 and a + 3b - 2u = 0, so a = u/5 and b = 3u/5."""

    def setup(self):
        self.add_input('u', 1.0)
        self.add_output('a', 1.0)
        self.add_output('b', 1.0)
        self.declare_partials('a', 'a', val=2.0)
        self.declare_partials('a', 'b', val=1.0)
        self.declare_partials('a', 'u', val=-1.0)
        self.declare_partials('b', 'a', val=1.0)
        self.declare_partials('b', 'b', val=3.0)
        self.declare_partials('b', 'u', val=-2.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['a'] = 2.0 * outputs['a'] + outputs['b'] - inputs['u']
        residuals['b'] = outputs['a'] + 3.0 * outputs['b'] - 2.0 * inputs['u']

    def solve_nonlinear(self, inputs, outputs):
        outputs['a'] = inputs['u'] / 5.0
        outputs['b'] = 3.0 * inputs['u'] / 5.0


def _linear_solver(name):
    if name == 'direct':
        return om.DirectSolver(assemble_jac=False)
    if name == 'krylov':
        return om.ScipyKrylov(atol=1e-12, rtol=1e-12, err_on_non_converge=True)
    if name == 'lnbgs':
        return om.LinearBlockGS(atol=1e-12, rtol=1e-12, maxiter=20, err_on_non_converge=True)
    return om.LinearRunOnce()


# The solvers that see the approximating group only through its _apply_linear/_solve_linear.
_LINEAR = ('direct', 'krylov', 'lnbgs', 'runonce')


def _model(fill, linear, nesting=0, method='fd'):
    """Return (problem, path of G) with x -> G, where fill(G) adds G's contents."""
    p = om.Problem()
    p.model.add_subsystem('ivc', om.IndepVarComp('x', 1.5))
    parent = p.model
    for i in range(nesting):
        parent = parent.add_subsystem(f'P{i}', om.Group())
    G = parent.add_subsystem('G', om.Group())
    path = '.'.join([f'P{i}' for i in range(nesting)] + ['G'])
    boundary = fill(G)
    p.model.connect('ivc.x', [f'{path}.{b}' for b in boundary])
    G.approx_totals(method=method)
    p.model.linear_solver = _linear_solver(linear)
    return p, path


def _explicit_then_state(**state_opts):
    def fill(G):
        G.add_subsystem('c1', _Scale(a=2.0))
        G.add_subsystem('imp', _State(**state_opts))
        G.connect('c1.y', 'imp.u')
        return ['c1.x']
    return fill


class TestApproxGroupImplicitStates(unittest.TestCase):

    def check(self, fill, expected, nesting=0, method='fd', linear=_LINEAR,
              modes=('fwd', 'rev'), before_totals=None, tol=2e-6):
        for lin in linear:
            for mode in modes:
                with self.subTest(linear=lin, mode=mode):
                    p, path = _model(fill, lin, nesting, method)
                    p.setup(mode=mode, force_alloc_complex=True)
                    p.run_model()
                    if before_totals is not None:
                        before_totals(p)
                    of = [f'{path}.{name}' for name in expected]
                    J = p.compute_totals(of=of, wrt=['ivc.x'])
                    for name, val in expected.items():
                        assert_near_equal(J[f'{path}.{name}', 'ivc.x'], np.atleast_2d(val), tol)

    def test_state_after_explicit_component(self):
        # x -> y = 2x -> 4w - y = 0, so dw/dx = 0.5.
        for partials in ('linearize', 'val'):
            with self.subTest(partials=partials):
                self.check(_explicit_then_state(partials=partials), {'imp.w': 0.5})

    def test_state_on_group_boundary_signs(self):
        # p*w + q*x = 0, so dw/dx = -q/p, for each sign combination.
        for p_, q_ in ((4.0, -1.0), (4.0, 1.0), (1.0, -3.0), (-2.0, 5.0)):
            def fill(G, p_=p_, q_=q_):
                G.add_subsystem('imp', _State(p=p_, q=q_, partials='val'))
                return ['imp.u']
            with self.subTest(p=p_, q=q_):
                self.check(fill, {'imp.w': -q_ / p_}, method='cs')

    def test_matrix_free_explicit_component(self):
        # A matrix-free component has no diagonal subjac either.
        def fill(G):
            G.add_subsystem('c1', _Scale(a=2.0))
            G.add_subsystem('mf', _MatrixFreeScale(a=0.25))
            G.connect('c1.y', 'mf.x')
            return ['c1.x']
        self.check(fill, {'mf.y': 0.5}, method='cs')

    def test_coupled_states(self):
        def fill(G):
            G.add_subsystem('imp', _Coupled())
            return ['imp.u']
        self.check(fill, {'imp.a': 0.2, 'imp.b': 0.6}, method='cs')

    def test_state_with_downstream_explicit_output(self):
        def fill(G):
            G.add_subsystem('c1', _Scale(a=2.0))
            G.add_subsystem('imp', _State(partials='val'))
            G.add_subsystem('c2', _Scale(a=3.0))
            G.connect('c1.y', 'imp.u')
            G.connect('imp.w', 'c2.x')
            return ['c1.x']
        self.check(fill, {'imp.w': 0.5, 'c2.y': 1.5}, method='cs')

    def test_newton_inside_group(self):
        # The group's own Newton solve linearizes the implicit component, so its residual
        # partials hold real values while the group approximates.
        def fill(G):
            G.add_subsystem('c1', _Scale(a=2.0))
            G.add_subsystem('imp', _State(solve=False))
            G.connect('c1.y', 'imp.u')
            G.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, atol=1e-14, rtol=1e-14,
                                                 iprint=-1)
            G.linear_solver = om.DirectSolver()
            return ['c1.x']
        self.check(fill, {'imp.w': 0.5}, method='fd')

    def test_newton_with_block_solver_inside_group(self):
        # A block linear solver used by the group's own Newton leaves values in the d_inputs of
        # inputs connected inside the group.  Partials with respect to those inputs are not part
        # of the group's approximated block and must not be applied with them.
        def fill(G):
            G.add_subsystem('c1', _Scale(a=2.0))
            G.add_subsystem('imp', _StateWithSolveLinear(partials='val', solve=False))
            G.add_subsystem('c2', _Scale(a=3.0))
            G.connect('c1.y', 'imp.u')
            G.connect('imp.w', 'c2.x')
            G.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, atol=1e-14, rtol=1e-14,
                                                 maxiter=50, iprint=-1)
            G.linear_solver = om.LinearBlockGS(atol=1e-14, rtol=1e-14, maxiter=50)
            return ['c1.x']
        self.check(fill, {'imp.w': 0.5, 'c2.y': 1.5}, method='fd', linear=('direct', 'krylov'),
                   tol=1e-8)

    def test_nested_approximating_group(self):
        for nesting in (1, 2):
            with self.subTest(nesting=nesting):
                self.check(_explicit_then_state(partials='val'), {'imp.w': 0.5}, nesting=nesting,
                           method='cs')

    def test_residual_scaling(self):
        self.check(_explicit_then_state(partials='val', res_ref=7.0), {'imp.w': 0.5},
                   method='cs')

    def test_output_and_residual_scaling(self):
        # Matrix-free solve only: the non-assembled DirectSolver in rev mode is not exercised
        # here with output scaling.
        self.check(_explicit_then_state(partials='val', ref=10.0, res_ref=10.0), {'imp.w': 0.5},
                   method='cs', linear=('krylov',))

    def test_coupling_through_parent(self):
        # G <-> c3 is a cycle, so the parent's LinearBlockGS iterates with nonzero group
        # d_outputs.  y = 2x + v, 4w - y = 0, v = w/2, so w = 2x/3.5.
        for mode in ('fwd', 'rev'):
            with self.subTest(mode=mode):
                p = om.Problem()
                p.model.add_subsystem('ivc', om.IndepVarComp('x', 1.5))
                G = p.model.add_subsystem('G', om.Group())
                G.add_subsystem('c1', om.ExecComp('y = 2.0*x + v'))
                G.add_subsystem('imp', _State(partials='val'))
                G.connect('c1.y', 'imp.u')
                p.model.add_subsystem('c3', _Scale(a=0.5))
                p.model.connect('ivc.x', 'G.c1.x')
                p.model.connect('G.imp.w', 'c3.x')
                p.model.connect('c3.y', 'G.c1.v')
                p.model.nonlinear_solver = om.NonlinearBlockGS(atol=1e-14, rtol=1e-14,
                                                               maxiter=200, iprint=-1)
                p.model.linear_solver = om.LinearBlockGS(atol=1e-14, rtol=1e-14, maxiter=200,
                                                         err_on_non_converge=True)
                G.approx_totals(method='cs')
                p.setup(mode=mode, force_alloc_complex=True)
                p.run_model()
                J = p.compute_totals(of=['G.imp.w'], wrt=['ivc.x'])
                assert_near_equal(J['G.imp.w', 'ivc.x'], [[2.0 / 3.5]], 1e-9)

    def test_check_partials_then_repeated_totals(self):
        def before_totals(p):
            p.check_partials(method='cs', out_stream=None)
            p.compute_totals(of=[f'G.imp.w'], wrt=['ivc.x'])
            p.set_val('ivc.x', 3.0)
            p.run_model()
        self.check(_explicit_then_state(), {'imp.w': 0.5}, method='cs',
                   before_totals=before_totals)



class _State2(om.ImplicitComponent):
    """R = w**3 + w - u - 2v."""

    def setup(self):
        self.add_input('u', 1.0)
        self.add_input('v', 1.0)
        self.add_output('w', 1.0)
        self.declare_partials('w', ['w', 'u', 'v'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['w'] = outputs['w'] ** 3 + outputs['w'] - inputs['u'] - 2 * inputs['v']

    def linearize(self, inputs, outputs, partials):
        partials['w', 'w'] = 3 * outputs['w'] ** 2 + 1
        partials['w', 'u'] = -1.0
        partials['w', 'v'] = -2.0


class _Square(om.ExplicitComponent):
    """z = w**2."""

    def setup(self):
        self.add_input('w', 1.0)
        self.add_output('z', 1.0)
        self.declare_partials('z', 'w')

    def compute(self, inputs, outputs):
        outputs['z'] = inputs['w'] ** 2

    def compute_partials(self, inputs, partials):
        partials['z', 'w'] = 2.0 * inputs['w']


class TestApproxGroupImplicitStatesUnderGroupNewton(unittest.TestCase):
    """Cases that also need the group's own Newton to see its residual jacobian."""

    def test_block_solver_under_group_newton(self):
        for g_lin in ('lnbgs', 'lnbj'):
            for method in ('fd', 'cs'):
                for mode in ('fwd', 'rev'):
                    with self.subTest(g_lin=g_lin, method=method, mode=mode):
                        p = om.Problem()
                        p.model.add_subsystem('ivc', om.IndepVarComp('x', 1.5))
                        G = p.model.add_subsystem('G', om.Group())
                        G.add_subsystem('c1', _Scale(a=2.0))
                        G.add_subsystem('imp', _StateWithSolveLinear(partials='val', solve=False))
                        G.add_subsystem('c2', _Scale(a=3.0))
                        G.connect('c1.y', 'imp.u')
                        G.connect('imp.w', 'c2.x')
                        p.model.connect('ivc.x', 'G.c1.x')
                        G.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, atol=1e-14,
                                                             rtol=1e-14, maxiter=50, iprint=-1)
                        cls = om.LinearBlockGS if g_lin == 'lnbgs' else om.LinearBlockJac
                        G.linear_solver = cls(atol=1e-14, rtol=1e-14, maxiter=50,
                                              err_on_non_converge=True)
                        G.approx_totals(method=method)
                        p.model.linear_solver = om.DirectSolver(assemble_jac=False)
                        p.setup(mode=mode, force_alloc_complex=True)
                        with contextlib.redirect_stdout(io.StringIO()):
                            p.run_model()
                        assert_near_equal(p.get_val('G.imp.w'), [0.75], 1e-12)

    def test_two_boundary_inputs_under_group_newton(self):
        for mode in ('fwd', 'rev'):
            with self.subTest(mode=mode):
                p = om.Problem()
                ivc = p.model.add_subsystem('ivc', om.IndepVarComp())
                ivc.add_output('a', 1.5)
                ivc.add_output('b', 0.5)
                G = p.model.add_subsystem('G', om.Group())
                G.add_subsystem('imp', _State2())
                p.model.connect('ivc.a', 'G.imp.u')
                p.model.connect('ivc.b', 'G.imp.v')
                G.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, atol=1e-14,
                                                     rtol=1e-14, maxiter=50, iprint=-1)
                G.linear_solver = om.DirectSolver()
                G.approx_totals(method='fd')
                p.model.linear_solver = om.DirectSolver(assemble_jac=False)
                p.setup(mode=mode)
                with contextlib.redirect_stdout(io.StringIO()):
                    p.run_model()
                    J = p.compute_totals(of=['G.imp.w'], wrt=['ivc.a', 'ivc.b'])
                w = p.get_val('G.imp.w').item()
                d = 3 * w * w + 1
                assert_near_equal(np.array([J['G.imp.w', 'ivc.a'].item(),
                                            J['G.imp.w', 'ivc.b'].item()]),
                                  np.array([1 / d, 2 / d]), 1e-5)

    def test_balance_in_child_group_under_approximation(self):
        # A BalanceComp and its own Newton in a child group H of the approximating group G.  The
        # inputs connected inside H are also differentiated by G's approximation, so the
        # components' own partials with respect to them are approximation keys as well.  The
        # approximation must keep its own values for them (0), not the components' partials.
        for linear in ('direct', 'direct_asm'):
            for method in ('fd', 'cs'):
                with self.subTest(linear=linear, method=method):
                    p = om.Problem()
                    p.model.add_subsystem('ivc', om.IndepVarComp('y', 4.0))
                    G = p.model.add_subsystem('G', om.Group())
                    H = G.add_subsystem('H', om.Group())
                    H.add_subsystem('c', _Square())
                    bal = H.add_subsystem('bal', om.BalanceComp())
                    bal.add_balance('w', lhs_name='z', rhs_name='y', val=1.0)
                    H.connect('bal.w', 'c.w')
                    H.connect('c.z', 'bal.z')
                    p.model.connect('ivc.y', 'G.H.bal.y')
                    H.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, atol=1e-12,
                                                         rtol=1e-12, maxiter=30, iprint=-1,
                                                         err_on_non_converge=True)
                    H.linear_solver = om.DirectSolver(assemble_jac=(linear == 'direct_asm'))
                    G.approx_totals(method=method)
                    p.model.linear_solver = om.DirectSolver(assemble_jac=False)
                    p.setup(force_alloc_complex=True)
                    for y in (4.0, 9.0):
                        p.set_val('ivc.y', y)
                        with contextlib.redirect_stdout(io.StringIO()):
                            p.run_model()
                            J = p.compute_totals(of=['G.H.bal.w'], wrt=['ivc.y'])
                        assert_near_equal(p.get_val('G.H.bal.w'), [np.sqrt(y)], 1e-10)
                        assert_near_equal(J['G.H.bal.w', 'ivc.y'], [[0.5 / np.sqrt(y)]], 1e-5)


if __name__ == '__main__':
    unittest.main()

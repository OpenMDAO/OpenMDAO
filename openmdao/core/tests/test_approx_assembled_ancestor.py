"""Approximating groups below an ancestor that uses an assembled jacobian (issue #3824)."""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

X0 = 0.8


class _Scale(om.ExplicitComponent):
    """out = k * inp, with selectable partial declarations."""

    def initialize(self):
        self.options.declare('k', default=2.0)
        self.options.declare('partials', default='analytic',
                             values=['analytic', 'val', 'wrong_val'])
        self.options.declare('in_units', default=None, allow_none=True)
        self.options.declare('out_units', default=None, allow_none=True)
        self.options.declare('out_kw', default={})

    def setup(self):
        self.add_input('inp', 1.0, units=self.options['in_units'])
        self.add_output('out', 1.0, units=self.options['out_units'], **self.options['out_kw'])
        k = self.options['k']
        if self.options['partials'] == 'val':
            self.declare_partials('out', 'inp', val=k)
        elif self.options['partials'] == 'wrong_val':
            self.declare_partials('out', 'inp', val=k + 7.0)
        else:
            self.declare_partials('out', 'inp')

    def compute(self, inputs, outputs):
        outputs['out'] = self.options['k'] * inputs['inp']

    def compute_partials(self, inputs, partials):
        if self.options['partials'] == 'analytic':
            partials['out', 'inp'] = self.options['k']


class _Quarter(om.ImplicitComponent):
    """Residual 4*w - u = 0."""

    def setup(self):
        self.add_input('u', 1.0)
        self.add_output('w', 1.0)
        self.declare_partials('w', ['u', 'w'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['w'] = 4.0 * outputs['w'] - inputs['u']

    def solve_nonlinear(self, inputs, outputs):
        outputs['w'] = inputs['u'] / 4.0

    def linearize(self, inputs, outputs, partials):
        partials['w', 'u'] = -1.0
        partials['w', 'w'] = 4.0


class _SqrtState(om.ImplicitComponent):
    """Residual y**2 - x = 0 (the configuration from PR #3817)."""

    def setup(self):
        self.add_input('x', 3.0)
        self.add_output('y', 2.0)
        self.declare_partials('y', ['x', 'y'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['y'] = outputs['y'] ** 2 - inputs['x']

    def linearize(self, inputs, outputs, partials):
        partials['y', 'y'] = 2.0 * outputs['y']
        partials['y', 'x'] = -1.0


def _linear_solver(kind):
    if kind in ('csc', 'dense'):
        return om.DirectSolver(assemble_jac=True)
    return {'direct': lambda: om.DirectSolver(assemble_jac=False),
            'krylov': lambda: om.ScipyKrylov(atol=1e-14, rtol=1e-14),
            'lbgs': lambda: om.LinearBlockGS(atol=1e-14, rtol=1e-14),
            'default': lambda: None}[kind]()


def _chain(gains=(2.0, 3.0), solver='csc', approx='G', mode='auto', partials='analytic',
           method='fd', form=None):
    """src.x -> G.c0 -> G.c1 -> ... with exact total prod(gains)."""
    p = om.Problem(reports=False)
    model = p.model
    model.add_subsystem('src', om.IndepVarComp('x', X0))
    G = model.add_subsystem('G', om.Group())
    for i, k in enumerate(gains):
        G.add_subsystem(f'c{i}', _Scale(k=k, partials=partials))
        if i:
            G.connect(f'c{i - 1}.out', f'c{i}.inp')
    model.connect('src.x', 'G.c0.inp')
    of = f'G.c{len(gains) - 1}.out'
    model.add_design_var('src.x')
    model.add_objective(of)
    lin = _linear_solver(solver)
    if lin is not None:
        model.linear_solver = lin
    if solver == 'dense':
        model.options['assembled_jac_type'] = 'dense'
    kw = {'method': method}
    if form:
        kw['form'] = form
    if approx == 'G':
        G.approx_totals(**kw)
    elif approx == 'root':
        model.approx_totals(**kw)
    p.setup(mode=mode, force_alloc_complex=True)
    p.run_model()
    return p, of, float(np.prod(gains))


def _total(p, of, wrt='src.x'):
    return p.compute_totals(of=[of], wrt=[wrt])[of, wrt].item()


class TestApproxGroupUnderAssembledJac(unittest.TestCase):

    def test_minimal_chain(self):
        for solver in ('csc', 'dense'):
            for mode in ('fwd', 'rev'):
                with self.subTest(solver=solver, mode=mode):
                    p, of, exact = _chain(solver=solver, mode=mode)
                    assert_near_equal(_total(p, of), exact, 1e-6)

    def test_controls(self):
        # configurations that were already correct before the fix
        for solver, approx in (('direct', 'G'), ('krylov', 'G'), ('lbgs', 'G'), ('default', 'G'),
                               ('csc', None), ('csc', 'root')):
            with self.subTest(solver=solver, approx=approx):
                p, of, exact = _chain(solver=solver, approx=approx)
                assert_near_equal(_total(p, of), exact, 1e-6)

    def test_methods_and_formats(self):
        for solver in ('csc', 'dense'):
            for method, form in (('fd', 'forward'), ('fd', 'backward'), ('fd', 'central'),
                                 ('cs', None)):
                with self.subTest(solver=solver, method=method, form=form):
                    p, of, exact = _chain(gains=(2.0, 3.0, 5.0), solver=solver, method=method,
                                          form=form)
                    assert_near_equal(_total(p, of), exact, 1e-12 if method == 'cs' else 1e-6)

    def test_internal_partials_not_double_counted(self):
        # The group's block already contains the path through its internal connections, so
        # whatever the component partials hold must not be applied a second time.
        for partials in ('val', 'wrong_val'):
            with self.subTest(partials=partials):
                p, of, exact = _chain(partials=partials)
                assert_near_equal(_total(p, of), exact, 1e-6)
        with self.subTest(partials='computed by check_partials first'):
            p, of, exact = _chain()
            p.check_partials(method='cs', out_stream=None)
            assert_near_equal(_total(p, of), exact, 1e-6)
        with self.subTest(partials='implicit component'):
            p = om.Problem(reports=False)
            model = p.model
            model.add_subsystem('src', om.IndepVarComp('x', X0))
            G = model.add_subsystem('G', om.Group())
            G.add_subsystem('c1', _Scale(k=2.0))
            G.add_subsystem('imp', _Quarter())
            G.connect('c1.out', 'imp.u')
            model.connect('src.x', 'G.c1.inp')
            model.add_design_var('src.x')
            model.add_objective('G.imp.w')
            model.linear_solver = om.DirectSolver(assemble_jac=True)
            G.approx_totals()
            p.setup()
            p.run_model()
            assert_near_equal(_total(p, 'G.imp.w'), 0.5, 1e-6)

    def test_boundary_mapping(self):
        with self.subTest('src_indices'):
            x0 = np.array([0.5, 1.1, -0.7, 1.9, 0.3, 1.4])
            p = om.Problem(reports=False)
            model = p.model
            model.add_subsystem('src', om.IndepVarComp('x', x0))
            G = model.add_subsystem('G', om.Group())
            G.add_subsystem('c1', om.ExecComp('y = 2*a**2', a=np.ones(3), y=np.ones(3)))
            G.add_subsystem('c2', om.ExecComp('z = 3*y*b', y=np.ones(3), b=np.ones(3),
                                              z=np.ones(3)))
            G.connect('c1.y', 'c2.y')
            model.connect('src.x', 'G.c1.a', src_indices=[0, 2, 4])
            model.connect('src.x', 'G.c2.b', src_indices=[5, 3, 1])
            model.add_design_var('src.x')
            model.add_constraint('G.c2.z', upper=1e6)
            model.add_subsystem('f', om.ExecComp('f = sum(z)', z=np.ones(3)))
            model.connect('G.c2.z', 'f.z')
            model.add_objective('f.f')
            model.linear_solver = om.DirectSolver(assemble_jac=True)
            G.approx_totals(method='cs')
            p.setup(force_alloc_complex=True)
            p.run_model()
            J = p.compute_totals(of=['G.c2.z'], wrt=['src.x'])['G.c2.z', 'src.x']
            a, b = x0[[0, 2, 4]], x0[[5, 3, 1]]
            expected = np.zeros((3, 6))
            for i, (ia, ib) in enumerate(zip([0, 2, 4], [5, 3, 1])):
                expected[i, ia] = 12.0 * a[i] * b[i]
                expected[i, ib] = 6.0 * a[i] ** 2
            assert_near_equal(J, expected, 1e-12)

        with self.subTest('units'):
            # x [ft] -> c0 (m) y = 2x -> c1 (cm) z = 3y  =>  dz/dx = 3 * 100 * 2 * 0.3048 cm/ft
            p = om.Problem(reports=False)
            model = p.model
            model.add_subsystem('src', om.IndepVarComp('x', X0, units='ft'))
            G = model.add_subsystem('G', om.Group())
            G.add_subsystem('c0', _Scale(k=2.0, in_units='m', out_units='m'))
            G.add_subsystem('c1', _Scale(k=3.0, in_units='cm', out_units='cm'))
            G.connect('c0.out', 'c1.inp')
            model.connect('src.x', 'G.c0.inp')
            model.add_design_var('src.x')
            model.add_objective('G.c1.out')
            model.linear_solver = om.DirectSolver(assemble_jac=True)
            G.approx_totals(method='cs')
            p.setup(force_alloc_complex=True)
            p.run_model()
            assert_near_equal(_total(p, 'G.c1.out'), 3.0 * 100.0 * 2.0 * 0.3048, 1e-12)

        with self.subTest('scaling'):
            p = om.Problem(reports=False)
            model = p.model
            model.add_subsystem('src', om.IndepVarComp('x', X0, ref=4.0))
            G = model.add_subsystem('G', om.Group())
            G.add_subsystem('c0', _Scale(k=2.0, out_kw={'ref': 10.0, 'ref0': -2.0,
                                                        'res_ref': 7.0}))
            G.add_subsystem('c1', _Scale(k=3.0, out_kw={'ref': 0.1, 'res_ref': 5.0}))
            G.connect('c0.out', 'c1.inp')
            model.connect('src.x', 'G.c0.inp')
            model.add_design_var('src.x')
            model.add_objective('G.c1.out')
            model.linear_solver = om.DirectSolver(assemble_jac=True)
            G.approx_totals(method='cs')
            p.setup(force_alloc_complex=True)
            p.run_model()
            assert_near_equal(_total(p, 'G.c1.out'), 6.0, 1e-12)

    def test_nesting(self):
        with self.subTest('assembled intermediate ancestor'):
            p = om.Problem(reports=False)
            model = p.model
            model.add_subsystem('src', om.IndepVarComp('x', X0))
            M = model.add_subsystem('M', om.Group())
            M.linear_solver = om.DirectSolver(assemble_jac=True)
            G = M.add_subsystem('G', om.Group())
            for i, k in enumerate((2.0, 3.0, 5.0)):
                G.add_subsystem(f'c{i}', _Scale(k=k))
                if i:
                    G.connect(f'c{i - 1}.out', f'c{i}.inp')
            model.connect('src.x', 'M.G.c0.inp')
            model.add_design_var('src.x')
            model.add_objective('M.G.c2.out')
            G.approx_totals()
            p.setup()
            p.run_model()
            assert_near_equal(_total(p, 'M.G.c2.out'), 30.0, 1e-6)

        with self.subTest('assembled root above a plain intermediate group'):
            p = om.Problem(reports=False)
            model = p.model
            model.add_subsystem('src', om.IndepVarComp('x', X0))
            P = model.add_subsystem('P', om.Group())
            G = P.add_subsystem('G', om.Group())
            G.add_subsystem('c0', _Scale(k=2.0))
            G.add_subsystem('c1', _Scale(k=3.0))
            G.connect('c0.out', 'c1.inp')
            model.connect('src.x', 'P.G.c0.inp')
            model.add_design_var('src.x')
            model.add_objective('P.G.c1.out')
            model.linear_solver = om.DirectSolver(assemble_jac=True)
            G.approx_totals()
            p.setup()
            p.run_model()
            assert_near_equal(_total(p, 'P.G.c1.out'), 6.0, 1e-6)

        with self.subTest('approximating group inside an approximating group'):
            # x -> H.d (5x) -> H.G.c0 (2u) -> H.G.c1 (3y) -> H.c3 (7z); responses inside G and H
            p = om.Problem(reports=False)
            model = p.model
            model.add_subsystem('src', om.IndepVarComp('x', X0))
            H = model.add_subsystem('H', om.Group())
            H.add_subsystem('d', _Scale(k=5.0))
            G = H.add_subsystem('G', om.Group())
            G.add_subsystem('c0', _Scale(k=2.0))
            G.add_subsystem('c1', _Scale(k=3.0))
            G.connect('c0.out', 'c1.inp')
            H.add_subsystem('c3', _Scale(k=7.0))
            H.connect('d.out', 'G.c0.inp')
            H.connect('G.c1.out', 'c3.inp')
            model.connect('src.x', 'H.d.inp')
            model.add_design_var('src.x')
            model.add_objective('H.c3.out')
            model.add_constraint('H.G.c1.out', upper=1e6)
            model.linear_solver = om.DirectSolver(assemble_jac=True)
            H.approx_totals()
            G.approx_totals()
            p.setup()
            p.run_model()
            J = p.compute_totals(of=['H.G.c1.out', 'H.c3.out'], wrt=['src.x'])
            assert_near_equal(J['H.G.c1.out', 'src.x'].item(), 30.0, 1e-6)
            assert_near_equal(J['H.c3.out', 'src.x'].item(), 210.0, 1e-6)

        with self.subTest('sibling whose name starts with the approximating group name'):
            p = om.Problem(reports=False)
            model = p.model
            model.add_subsystem('src', om.IndepVarComp('x', X0))
            G = model.add_subsystem('G', om.Group())
            G.add_subsystem('c0', _Scale(k=2.0))
            G.add_subsystem('c1', _Scale(k=3.0))
            G.connect('c0.out', 'c1.inp')
            G2 = model.add_subsystem('G2', om.Group())
            G2.add_subsystem('d0', _Scale(k=5.0))
            G2.add_subsystem('d1', _Scale(k=7.0))
            G2.connect('d0.out', 'd1.inp')
            model.connect('src.x', 'G.c0.inp')
            model.connect('G.c1.out', 'G2.d0.inp')
            model.add_design_var('src.x')
            model.add_objective('G2.d1.out')
            model.linear_solver = om.DirectSolver(assemble_jac=True)
            G.approx_totals()
            p.setup()
            p.run_model()
            assert_near_equal(_total(p, 'G2.d1.out'), 210.0, 1e-6)

        for mode in ('fwd', 'rev'):
            with self.subTest('cycle inside the group', mode=mode):
                # a = x + 0.5 b ; b = 0.3 a + 0.2 a**2
                p = om.Problem(reports=False)
                model = p.model
                model.add_subsystem('src', om.IndepVarComp('x', X0))
                G = model.add_subsystem('G', om.Group())
                G.add_subsystem('c1', om.ExecComp('a = x + 0.5*b'))
                G.add_subsystem('c2', om.ExecComp('b = 0.3*a + 0.2*a**2'))
                G.connect('c1.a', 'c2.a')
                G.connect('c2.b', 'c1.b')
                G.nonlinear_solver = om.NonlinearBlockGS(atol=1e-15, rtol=1e-15, maxiter=500)
                model.connect('src.x', 'G.c1.x')
                model.add_design_var('src.x')
                model.add_objective('G.c2.b')
                model.add_constraint('G.c1.a', upper=100.0)
                model.linear_solver = om.DirectSolver(assemble_jac=True)
                G.approx_totals(form='central')
                p.setup(mode=mode)
                p.run_model()
                J = p.compute_totals()
                a = p.get_val('G.c1.a').item()
                db_da = 0.3 + 0.4 * a
                da_dx = 1.0 / (1.0 - 0.5 * db_da)
                assert_near_equal(J['G.c1.a', 'src.x'].item(), da_dx, 1e-8)
                assert_near_equal(J['G.c2.b', 'src.x'].item(), db_da * da_dx, 1e-8)

    def test_repeated_totals_and_resetup(self):
        # y = x**2, z = 3y: dz/dx = 6x and dy/dx = 2x
        p = om.Problem(reports=False)
        model = p.model
        model.add_subsystem('src', om.IndepVarComp('x', X0))
        G = model.add_subsystem('G', om.Group())
        G.add_subsystem('c1', om.ExecComp('y = x**2'))
        G.add_subsystem('c2', om.ExecComp('z = 3.0*y'))
        G.connect('c1.y', 'c2.y')
        model.connect('src.x', 'G.c1.x')
        model.add_design_var('src.x')
        model.add_objective('G.c2.z')
        model.add_constraint('G.c1.y', upper=1e6)
        model.linear_solver = om.DirectSolver(assemble_jac=True)
        G.approx_totals(method='cs')

        for x in (X0, 1.7, 2.9):
            p.setup(force_alloc_complex=True)
            p.set_val('src.x', x)
            p.run_model()
            assert_near_equal(_total(p, 'G.c2.z'), 6.0 * x, 1e-12)
            assert_near_equal(_total(p, 'G.c1.y'), 2.0 * x, 1e-12)
            p.check_partials(method='cs', out_stream=None)
            J = p.compute_totals()
            assert_near_equal(J['G.c2.z', 'src.x'].item(), 6.0 * x, 1e-12)
            assert_near_equal(J['G.c1.y', 'src.x'].item(), 2.0 * x, 1e-12)

    def test_group_owning_direct_solver(self):
        # The configuration from PR #3817, alone and under an assembled ancestor.
        for method in ('fd', 'cs'):
            for ancestor in ('default', 'csc'):
                with self.subTest(method=method, ancestor=ancestor):
                    p = om.Problem(reports=False)
                    g = p.model.add_subsystem('g', om.Group(), promotes=['*'])
                    g.add_subsystem('comp', _SqrtState(), promotes=['*'])
                    g.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=-1,
                                                         atol=1e-14, rtol=1e-14)
                    g.linear_solver = om.DirectSolver()
                    g.approx_totals(method=method,
                                    **({'form': 'central'} if method == 'fd' else {}))
                    lin = _linear_solver(ancestor)
                    if lin is not None:
                        p.model.linear_solver = lin
                    p.setup(force_alloc_complex=(method == 'cs'))
                    p.set_val('x', 3.0)
                    p.run_model()
                    J = p.compute_totals(of=['y'], wrt=['x'])['y', 'x'].item()
                    assert_near_equal(J, 1.0 / (2.0 * np.sqrt(3.0)),
                                      1e-10 if method == 'cs' else 1e-6)


class _ParallelAssembledMixin(object):
    """Three subgroups in a ParallelGroup, each assembling a jacobian around an approximating
    group.  Total = 3 * (2 + 5 + 7) = 42 on every rank."""

    def _build(self, asm_type, mode):
        p = om.Problem(reports=False)
        model = p.model
        model.add_subsystem('src', om.IndepVarComp('x', X0), promotes=['x'])
        par = model.add_subsystem('par', om.ParallelGroup(), promotes_inputs=['x'])
        for i, k in enumerate((2.0, 5.0, 7.0)):
            S = par.add_subsystem(f'S{i}', om.Group(), promotes_inputs=['x'])
            G = S.add_subsystem('G', om.Group(), promotes_inputs=['x'])
            G.add_subsystem('c0', _Scale(k=k), promotes_inputs=[('inp', 'x')])
            G.add_subsystem('c1', _Scale(k=3.0))
            G.connect('c0.out', 'c1.inp')
            G.approx_totals()
            S.linear_solver = om.DirectSolver(assemble_jac=True)
            S.options['assembled_jac_type'] = asm_type
        model.add_subsystem('obj', om.ExecComp('q = a + b + c'))
        for i, v in enumerate('abc'):
            model.connect(f'par.S{i}.G.c1.out', f'obj.{v}')
        model.add_design_var('x')
        model.add_objective('obj.q')
        p.setup(mode=mode)
        p.run_model()
        return p

    def test_parallel_subgroups_with_assembled_jacs(self):
        for asm_type in ('csc', 'dense'):
            for mode in ('fwd', 'rev'):
                with self.subTest(asm_type=asm_type, mode=mode):
                    p = self._build(asm_type, mode)
                    J = p.compute_totals()['obj.q', 'x'].item()
                    assert_near_equal(J, 42.0, 1e-6)
                    self.assertEqual(len(set(p.comm.allgather(round(J, 8)))), 1)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestApproxGroupUnderAssembledJacMPI2(_ParallelAssembledMixin, unittest.TestCase):

    N_PROCS = 2


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestApproxGroupUnderAssembledJacMPI4(_ParallelAssembledMixin, unittest.TestCase):

    N_PROCS = 4


if __name__ == '__main__':
    unittest.main()

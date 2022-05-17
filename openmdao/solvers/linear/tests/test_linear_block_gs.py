"""Test the LinearBlockGS linear solver class."""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.solvers.linear.tests.linear_test_base import LinearSolverTests
from openmdao.test_suite.components.sellar import SellarImplicitDis1, SellarImplicitDis2, \
    SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals

from openmdao.utils.mpi import MPI
try:
    from openmdao.api import PETScVector
except:
    PETScVector = None


class SimpleImp(om.ImplicitComponent):
    def setup(self):
        self.add_input('a', val=1.)
        self.add_output('x', val=0.)

        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = 3.0*inputs['a'] + 2.0*outputs['x']

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = 2.0
        jacobian['x', 'a'] = 3.0


class TestBGSSolver(LinearSolverTests.LinearSolverTestCase):
    linear_solver_class = om.LinearBlockGS

    def test_globaljac_err(self):
        prob = om.Problem()
        model = prob.model = om.Group(assembled_jac_type='dense')
        model.add_subsystem('x_param', om.IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.linear_solver = self.linear_solver_class(assemble_jac=True)
        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.run_model()

        self.assertEqual(str(context.exception),
                         "Linear solver LinearBlockGS in <model> <class Group> doesn't support assembled jacobians.")

    def test_simple_implicit(self):
        # This verifies that we can perform lgs around an implicit comp and get the right answer
        # as long as we slot a non-lgs linear solver on that component.

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p', om.IndepVarComp('a', 5.0))
        comp = model.add_subsystem('comp', SimpleImp())
        model.connect('p.a', 'comp.a')

        model.linear_solver = self.linear_solver_class()
        comp.linear_solver = om.DirectSolver()

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        deriv = prob.compute_totals(of=['comp.x'], wrt=['p.a'])
        self.assertEqual(deriv['comp.x', 'p.a'], -1.5)

    def test_implicit_cycle(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 1.0))
        model.add_subsystem('d1', SellarImplicitDis1())
        model.add_subsystem('d2', SellarImplicitDis2())
        model.connect('d1.y1', 'd2.y1')
        model.connect('d2.y2', 'd1.y2')

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.nonlinear_solver.options['maxiter'] = 5
        model.linear_solver = self.linear_solver_class()

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()
        res = model._residuals.get_norm()

        # Newton is kinda slow on this for some reason, this is how far it gets with directsolver too.
        self.assertLess(res, 2.0e-2)

    def test_implicit_cycle_precon(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 1.0))
        model.add_subsystem('d1', SellarImplicitDis1())
        model.add_subsystem('d2', SellarImplicitDis2())
        model.connect('d1.y1', 'd2.y1')
        model.connect('d2.y2', 'd1.y2')

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.nonlinear_solver.options['maxiter'] = 5
        model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        model.linear_solver = om.ScipyKrylov()
        model.linear_solver.precon = self.linear_solver_class()

        prob.setup()

        prob['d1.y1'] = 4.0
        prob.set_solver_print()
        prob.run_model()
        res = model._residuals.get_norm()

        # Newton is kinda slow on this for some reason, this is how far it gets with directsolver too.
        self.assertLess(res, 2.0e-2)

    def test_full_desvar_with_index_obj_relevance_bug(self):
        prob = om.Problem()
        sub = prob.model.add_subsystem('sub', SellarDerivatives())
        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.linear_solver = om.LinearBlockGS()
        sub.nonlinear_solver = om.NonlinearBlockGS()
        sub.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('sub.z', lower=-100, upper=100)
        prob.model.add_objective('sub.z', index=1)

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        derivs = prob.compute_totals(of=['sub.z'], wrt=['sub.z'])

        assert_near_equal(derivs[('sub.z', 'sub.z')], [[0., 1.]])

    def test_aitken(self):
        prob = om.Problem()
        model = prob.model

        aitken = om.LinearBlockGS()
        aitken.options['use_aitken'] = True
        aitken.options['err_on_non_converge'] = True

        # It takes 6 iterations without Aitken.
        aitken.options['maxiter'] = 4

        sub = model.add_subsystem('sub', SellarDerivatives(nonlinear_solver=om.NonlinearRunOnce(),
                                                           linear_solver=aitken))
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('sub.y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('sub.y2'), 12.05848819, .00001)

        prob = om.Problem()
        model = prob.model

        aitken = om.LinearBlockGS()
        aitken.options['use_aitken'] = True
        aitken.options['err_on_non_converge'] = True

        # It takes 6 iterations without Aitken.
        aitken.options['maxiter'] = 4

        sub = model.add_subsystem('sub', SellarDerivatives(nonlinear_solver=om.NonlinearRunOnce(),
                                                           linear_solver=aitken))
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

        prob.setup(mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('sub.y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('sub.y2'), 12.05848819, .00001)


class TestBGSSolverFeature(unittest.TestCase):

    def test_specify_solver(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.linear_solver = om.LinearBlockGS()
        model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_near_equal(J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_feature_maxiter(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = om.NonlinearBlockGS()

        model.linear_solver = om.LinearBlockGS()
        model.linear_solver.options['maxiter'] = 2

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['obj', 'z'][0][0], 9.60230118004, .00001)
        assert_near_equal(J['obj', 'z'][0][1], 1.78022500547, .00001)

    def test_feature_atol(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = om.NonlinearBlockGS()

        model.linear_solver = om.LinearBlockGS()
        model.linear_solver.options['atol'] = 1.0e-3

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['obj', 'z'][0][0], 9.61016296175, .00001)
        assert_near_equal(J['obj', 'z'][0][1], 1.78456955704, .00001)

    def test_feature_rtol(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = om.NonlinearBlockGS()

        model.linear_solver = om.LinearBlockGS()
        model.linear_solver.options['rtol'] = 1.0e-3

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['obj', 'z'][0][0], 9.61016296175, .00001)
        assert_near_equal(J['obj', 'z'][0][1], 1.78456955704, .00001)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProcTestCase1(unittest.TestCase):

    N_PROCS = 2

    def test_linear_analysis_error(self):

        # test fwd mode
        prob = om.Problem()
        model = prob.model

        # takes 6 iterations normally
        linear_solver = om.LinearBlockGS(maxiter=2, err_on_non_converge=True)

        model.add_subsystem('sub', SellarDerivatives(nonlinear_solver=om.NonlinearRunOnce(),
                                                     linear_solver=linear_solver))
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

        prob.setup(mode='fwd')
        prob.set_solver_print(level=2)

        # test if the analysis error is raised properly on all procs
        try:
            prob.run_model()
        except om.AnalysisError as err:
            self.assertEqual(str(err), "Solver 'LN: LNBGS' on system 'sub' failed to converge in 2 iterations.")
        else:
            self.fail("expected AnalysisError")

        # test rev mode
        prob = om.Problem()
        model = prob.model

        # takes 6 iterations normally
        linear_solver = om.LinearBlockGS(maxiter=2, err_on_non_converge=True)

        model.add_subsystem('sub', SellarDerivatives(nonlinear_solver=om.NonlinearRunOnce(),
                                                     linear_solver=linear_solver))
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

        prob.setup(mode='rev')
        prob.set_solver_print(level=2)

        # test if the analysis error is raised properly on all procs
        try:
            prob.run_model()
        except om.AnalysisError as err:
            self.assertEqual(str(err), "Solver 'LN: LNBGS' on system 'sub' failed to converge in 2 iterations.")
        else:
            self.fail("expected AnalysisError")


class MyParaboloid(Paraboloid):
    """ Use matrix-vector product."""

    def setup_partials(self):
        pass

    def compute_partials(self, inputs, partials):
        """Analytical derivatives."""
        pass

    def compute_jacvec_product(self, inputs, dinputs, doutputs, mode):
        x = inputs['x'][0]
        y = inputs['y'][0]

        if mode == 'fwd':
            if 'x' in dinputs:
                doutputs['f_xy'] += (2.0*x - 6.0 + y)*dinputs['x']
            if 'y' in dinputs:
                doutputs['f_xy'] += (2.0*y + 8.0 + x)*dinputs['y']

        elif mode == 'rev':
            if 'x' in dinputs:
                dinputs['x'] += (2.0*x - 6.0 + y)*doutputs['f_xy']
            if 'y' in dinputs:
                dinputs['y'] += (2.0*y + 8.0 + x)*doutputs['f_xy']


def execute_model1(mode):
    prob = om.Problem()
    model = prob.model

    model.add_subsystem('indeps', om.IndepVarComp('dv1', val=1.0))

    sub1 = model.add_subsystem('sub1', om.Group())
    sub1.add_subsystem('c1', om.ExecComp(exprs=['y = x']))

    sub2 = sub1.add_subsystem('sub2', om.Group())
    comp = sub2.add_subsystem('comp', MyParaboloid())

    model.connect('indeps.dv1', ['sub1.c1.x', 'sub1.sub2.comp.x'])
    sub1.connect('c1.y', 'sub2.comp.y')

    model.add_design_var('indeps.dv1')
    model.add_constraint('sub1.sub2.comp.f_xy')

    prob.setup(mode=mode, force_alloc_complex=True)

    prob['indeps.dv1'] = 2.

    prob.run_model()
    assert_check_totals(prob.check_totals(method='cs', out_stream=None))
    assert_check_partials(prob.check_partials(method='cs', out_stream=None))


def execute_model2(mode):
    prob = om.Problem()
    model = prob.model

    model.add_subsystem('indeps', om.IndepVarComp('dv1', val=1.0))

    sub1 = model.add_subsystem('sub1', om.Group())
    sub1.add_subsystem('c1', om.ExecComp(exprs=['y = x']))
    sub1.add_subsystem('c2', om.ExecComp(exprs=['y = x']))

    sub2 = sub1.add_subsystem('sub2', om.Group())
    comp = sub2.add_subsystem('comp', MyParaboloid())

    model.connect('indeps.dv1', ['sub1.c1.x', 'sub1.c2.x'])
    sub1.connect('c1.y', 'sub2.comp.x')
    sub1.connect('c2.y', 'sub2.comp.y')

    model.add_design_var('indeps.dv1')
    model.add_constraint('sub1.sub2.comp.f_xy')

    prob.setup(mode=mode, force_alloc_complex=True)

    prob['indeps.dv1'] = 2.

    prob.run_model()
    assert_check_totals(prob.check_totals(method='cs', out_stream=None))
    assert_check_partials(prob.check_partials(method='cs', out_stream=None))


class TestRecursiveApplyFix(unittest.TestCase):

    def test_matrix_free_explicit_fwd(self):
        execute_model1('fwd')

    def test_matrix_free_explicit_rev(self):
        execute_model1('rev')

    def test_matrix_free_explicit2_fwd(self):
        execute_model2('fwd')

    def test_matrix_free_explicit2_rev(self):
        execute_model2('rev')


class _ApplyLinearCounter(om.ExecComp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_lin_count = 0

    def _apply_linear(self, *args, **kwargs):
        super()._apply_linear(*args, **kwargs)
        self._apply_lin_count += 1


def build_nested_groups(nlevels, linsolver_class):
    # construct a nested group model with a component at each group level
    # and a component at the bottom that has one input for each component output.
    # The new method should only call _apply_linear on the bottom component once.
    p = om.Problem()
    current = model = p.model
    current.linear_solver = linsolver_class()
    p.model.add_subsystem('ivc', om.IndepVarComp('x', 2.0))

    parpath = ''
    outs = ['ivc.x']

    for i in range(nlevels):
        current.add_subsystem(f"C{i}", _ApplyLinearCounter(f"y={i+3}*x"))
        g = current.add_subsystem(f"G{i}", om.Group())
        g.linear_solver = linsolver_class()
        current = g

        if parpath:
            outs.append(parpath + '.' + f"C{i}.y")
            model.connect('ivc.x', parpath + '.' + f"C{i}.x")
            parpath = parpath + '.' + f"G{i}"
        else:
            outs.append(f"C{i}.y")
            model.connect('ivc.x', f"C{i}.x")
            parpath = f"G{i}"

    # bottom of the tree
    leafname = f"C{nlevels+1}"
    parts = []
    for i, out in enumerate(outs):
        parts.append(f"x{i+1}*{i+2}")
    expr = 'y = ' + '+'.join(parts)
    print(expr)
    current.add_subsystem(leafname, _ApplyLinearCounter(expr))

    leafpath = parpath + '.' + leafname

    # connect everything
    for i, out in enumerate(outs):
        model.connect(out, leafpath + '.' + f"x{i+1}")
        print("connect ", out, "to", leafpath + '.' + f"x{i+1}")

    return p, leafpath


class TestRecursiveApplyFix2(unittest.TestCase):
    def test_4levels(self):
        p, lpath = build_nested_groups(4, om.LinearBlockGS)
        p.setup()
        p.run_model()



if __name__ == "__main__":
    p, leafpath = build_nested_groups(4, om.LinearBlockGS)
    p.setup()
    p.run_model()

    from openmdao.devtools.debug import tree
    tree(p)

    p.compute_totals(of=[leafpath + '.y'], wrt=['ivc.x'])

    leaf = p.model._get_subsystem(leafpath)
    print('ncalls:', leaf._apply_lin_count)

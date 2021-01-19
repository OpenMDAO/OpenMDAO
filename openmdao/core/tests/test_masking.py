import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, NewtonSolver, \
    ScipyKrylov, LinearRunOnce, LinearBlockGS

from openmdao.test_suite.components.double_sellar import DoubleSellar, DoubleSellarImplicit


def _build_model(mode, implicit=False):
    p = Problem()

    dv = p.model.add_subsystem('dv', IndepVarComp(), promotes=['*'])
    dv.add_output('z', [1.,1.])

    if implicit:
        p.model.add_subsystem('double_sellar', DoubleSellarImplicit())
    else:
        p.model.add_subsystem('double_sellar', DoubleSellar())
    p.model.connect('z', ['double_sellar.g1.z', 'double_sellar.g2.z'])

    p.model.add_design_var('z', lower=-10, upper=10)
    p.model.add_objective('double_sellar.g1.y1')

    p.setup(mode=mode)

    p.model.nonlinear_solver = NewtonSolver(solve_subsystems=False)

    return p


def _add_solvers(p):
    p.model.double_sellar.g1.linear_solver = DirectSolver(assemble_jac=True)
    p.model.double_sellar.g1.nonlinear_solver = NewtonSolver(solve_subsystems=False)

    p.model.double_sellar.g2.linear_solver = DirectSolver(assemble_jac=True)
    p.model.double_sellar.g2.nonlinear_solver = NewtonSolver(solve_subsystems=False)

    newton = p.model.nonlinear_solver = NewtonSolver(solve_subsystems=False)
    newton.linear_solver = ScipyKrylov()
    newton.linear_solver.precon = LinearBlockGS()

    p.model.options['assembled_jac_type'] = 'dense'
    p.model.linear_solver = ScipyKrylov(assemble_jac=True)
    p.model.linear_solver.precon = DirectSolver()


class CSCMaskingTestCase(unittest.TestCase):
    def test_base_fwd(self):
        p = _build_model('fwd')

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_base_subsolve_fwd(self):
        p = _build_model('fwd')
        p.model.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_fwd(self):
        p = _build_model('fwd')
        _add_solvers(p)

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_subsolve_fwd(self):
        p = _build_model('fwd')
        _add_solvers(p)
        p.model.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_lower_subsolve_fwd(self):
        p = _build_model('fwd')
        _add_solvers(p)
        p.model.double_sellar.g1.nonlinear_solver.options['solve_subsystems'] = True
        p.model.double_sellar.g2.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_base_rev(self):
        p = _build_model('rev')

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_base_subsolve_rev(self):
        p = _build_model('rev')
        p.model.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_rev(self):
        p = _build_model('rev')
        _add_solvers(p)

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_subsolve_rev(self):
        p = _build_model('rev')
        _add_solvers(p)
        p.model.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_lower_subsolve_rev(self):
        p = _build_model('rev')
        _add_solvers(p)
        p.model.double_sellar.g1.nonlinear_solver.options['solve_subsystems'] = True
        p.model.double_sellar.g2.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))


class CSCMaskingImplicitTestCase(unittest.TestCase):
    def test_base_fwd(self):
        p = _build_model('fwd', implicit=True)

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_base_subsolve_fwd(self):
        p = _build_model('fwd', implicit=True)
        p.model.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_fwd(self):
        p = _build_model('fwd', implicit=True)
        _add_solvers(p)

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_subsolve_fwd(self):
        p = _build_model('fwd', implicit=True)
        _add_solvers(p)
        p.model.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_lower_subsolve_fwd(self):
        p = _build_model('fwd', implicit=True)
        _add_solvers(p)
        p.model.double_sellar.g1.nonlinear_solver.options['solve_subsystems'] = True
        p.model.double_sellar.g2.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_base_rev(self):
        p = _build_model('rev', implicit=True)

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_base_subsolve_rev(self):
        p = _build_model('rev', implicit=True)
        p.model.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_rev(self):
        p = _build_model('rev', implicit=True)
        _add_solvers(p)

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_subsolve_rev(self):
        p = _build_model('rev', implicit=True)
        _add_solvers(p)
        p.model.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

    def test_mixed_lower_subsolve_rev(self):
        p = _build_model('rev', implicit=True)
        _add_solvers(p)
        p.model.double_sellar.g1.nonlinear_solver.options['solve_subsystems'] = True
        p.model.double_sellar.g2.nonlinear_solver.options['solve_subsystems'] = True

        p.run_model()

        assert_almost_equal(p['double_sellar.g1.y1'], np.array([5.47125755]))
        assert_almost_equal(p.compute_totals(return_format='array'),
                            np.array([[3.3775959, 2.17131165]]))

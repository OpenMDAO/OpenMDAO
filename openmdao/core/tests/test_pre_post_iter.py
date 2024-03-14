import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals, assert_near_equal
from openmdao.test_suite.components.exec_comp_for_test import ExecComp4Test
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI
from openmdao.drivers.pyoptsparse_driver import pyoptsparse

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None


class MissingPartialsComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('a', 1.0)
        self.add_input('b', 1.0)
        self.add_output('y', 1.0)
        self.add_output('z', 1.0)

        self.declare_partials(of=['y'], wrt=['a'])
        self.declare_partials(of=['z'], wrt=['b'])

    def compute(self, inputs, outputs):
        outputs['y'] = 2.0 * inputs['a']
        outputs['z'] = 3.0 * inputs['b']

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials['y', 'a'] = 2.0
        partials['z', 'b'] = 3.0


def setup_problem(do_pre_post_opt, mode, use_ivc=False, coloring=False, size=3, group=False,
                  force=(), approx=False, force_complex=False, recording=False, parallel=False):
    prob = om.Problem()
    prob.options['group_by_pre_opt_post'] = do_pre_post_opt

    if parallel:
        prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')
    else:
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False)
        prob.set_solver_print(level=0)

    model = prob.model

    if approx:
        model.approx_totals()

    if use_ivc:
        model.add_subsystem('ivc', om.IndepVarComp('x', np.ones(size)))

    if parallel:
        par = model.add_subsystem('par', om.ParallelGroup(), promotes=['*'])
        par.nonlinear_solver = om.NonlinearBlockJac()
        par.linear_solver = om.LinearBlockJac()
        parent = par
    else:
        parent = model

    if group:
        G1 = parent.add_subsystem('G1', om.Group(), promotes=['*'])
        G2 = parent.add_subsystem('G2', om.Group(), promotes=['*'])
        if parallel:
            G2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
            G2.linear_solver = om.DirectSolver(assemble_jac=True)
            # this guy wouldn't be included in the iteration loop were it not under Newton
            G1.add_subsystem('sub_post_comp', ExecComp4Test('y=.5*x', x=np.ones(size), y=np.zeros(size)))
    else:
        G1 = parent
        G2 = parent

    comps = {
        'pre1': G1.add_subsystem('pre1', ExecComp4Test('y=2.*x', x=np.ones(size), y=np.zeros(size))),
        'pre2': G1.add_subsystem('pre2', ExecComp4Test('y=3.*x - 7.*xx', x=np.ones(size), xx=np.ones(size), y=np.zeros(size))),

        'iter1': G1.add_subsystem('iter1', ExecComp4Test('y=x1 + x2*4. + x3',
                                                x1=np.ones(size), x2=np.ones(size),
                                                x3=np.ones(size), y=np.zeros(size))),
        'iter2': G1.add_subsystem('iter2', ExecComp4Test('y=.5*x', x=np.ones(size), y=np.zeros(size))),
        'iter4': G2.add_subsystem('iter4', ExecComp4Test('y=7.*x', x=np.ones(size), y=np.zeros(size))),
        'iter3': G2.add_subsystem('iter3', ExecComp4Test('y=6.*x', x=np.ones(size), y=np.zeros(size))),

        'post1': G2.add_subsystem('post1', ExecComp4Test('y=8.*x', x=np.ones(size), y=np.zeros(size))),
        'post2': G2.add_subsystem('post2', ExecComp4Test('y=x1*9. + x2*5. + x3*3.', x1=np.ones(size),
                                                x2=np.ones(size), x3=np.zeros(size),
                                                y=np.zeros(size))),
    }

    for name in force:
        if name in comps:
            comps[name].options['always_opt'] = True
        else:
            raise RuntimeError(f'"{name}" not in comps')

    if use_ivc:
        model.connect('ivc.x', 'iter1.x3')

    model.connect('pre1.y', ['iter1.x1', 'post2.x1', 'pre2.xx'])
    model.connect('pre2.y', 'iter1.x2')
    model.connect('iter1.y', ['iter2.x', 'iter4.x'])
    model.connect('iter2.y', 'post2.x2')
    model.connect('iter3.y', 'post1.x')
    model.connect('iter4.y', 'iter3.x')
    model.connect('post1.y', 'post2.x3')

    prob.model.add_design_var('iter1.x3', lower=-10, upper=10)
    prob.model.add_constraint('iter2.y', upper=10.)
    prob.model.add_objective('iter3.y', index=0)

    if coloring:
        prob.driver.declare_coloring()

    if recording:
        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True

        recorder = om.SqliteRecorder("sqlite_test_pre_post", record_viewer_data=False)

        model.add_recorder(recorder)
        prob.driver.add_recorder(recorder)

        for comp in comps.values():
            comp.add_recorder(recorder)

    prob.setup(mode=mode, force_alloc_complex=force_complex)

    # we don't want ExecComps to be colored because it makes the iter counting more complicated.
    for comp in model.system_iter(recurse=True, typ=ExecComp4Test):
        comp.options['do_coloring'] = False
        comp.options['has_diag_partials'] = True

    return prob

@use_tempdirs
class TestPrePostIter(unittest.TestCase):

    def setup_problem(self, do_pre_post_opt, mode, use_ivc=False, coloring=False, size=3, group=False,
                      force=(), approx=False, force_complex=False, recording=False, set_vois=True):
        prob = om.Problem()
        prob.options['group_by_pre_opt_post'] = do_pre_post_opt

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False)
        prob.set_solver_print(level=0)

        model = prob.model

        if approx:
            model.approx_totals()

        if use_ivc:
            model.add_subsystem('ivc', om.IndepVarComp('x', np.ones(size)))

        if group:
            G1 = model.add_subsystem('G1', om.Group(), promotes=['*'])
            G2 = model.add_subsystem('G2', om.Group(), promotes=['*'])
        else:
            G1 = model
            G2 = model

        comps = {
            'pre1': G1.add_subsystem('pre1', ExecComp4Test('y=2.*x', x=np.ones(size), y=np.zeros(size))),
            'pre2': G1.add_subsystem('pre2', ExecComp4Test('y=3.*x - 7.*xx', x=np.ones(size), xx=np.ones(size), y=np.zeros(size))),

            'iter1': G1.add_subsystem('iter1', ExecComp4Test('y=x1 + x2*4. + x3',
                                                    x1=np.ones(size), x2=np.ones(size),
                                                    x3=np.ones(size), y=np.zeros(size))),
            'iter2': G1.add_subsystem('iter2', ExecComp4Test('y=.5*x', x=np.ones(size), y=np.zeros(size))),
            'iter4': G2.add_subsystem('iter4', ExecComp4Test('y=7.*x', x=np.ones(size), y=np.zeros(size))),
            'iter3': G2.add_subsystem('iter3', ExecComp4Test('y=6.*x', x=np.ones(size), y=np.zeros(size))),

            'post1': G2.add_subsystem('post1', ExecComp4Test('y=8.*x', x=np.ones(size), y=np.zeros(size))),
            'post2': G2.add_subsystem('post2', ExecComp4Test('y=x1*9. + x2*5. + x3*3.', x1=np.ones(size),
                                                    x2=np.ones(size), x3=np.zeros(size),
                                                    y=np.zeros(size))),
        }

        for name in force:
            if name in comps:
                comps[name].options['always_opt'] = True
            else:
                raise RuntimeError(f'"{name}" not in comps')

        if use_ivc:
            model.connect('ivc.x', 'iter1.x3')

        model.connect('pre1.y', ['iter1.x1', 'post2.x1', 'pre2.xx'])
        model.connect('pre2.y', 'iter1.x2')
        model.connect('iter1.y', ['iter2.x', 'iter4.x'])
        model.connect('iter2.y', 'post2.x2')
        model.connect('iter3.y', 'post1.x')
        model.connect('iter4.y', 'iter3.x')
        model.connect('post1.y', 'post2.x3')

        if set_vois:
            prob.model.add_design_var('iter1.x3', lower=-10, upper=10)
            prob.model.add_constraint('iter2.y', upper=10.)
            prob.model.add_objective('iter3.y', index=0)

        if coloring:
            prob.driver.declare_coloring()

        if recording:
            model.recording_options['record_inputs'] = True
            model.recording_options['record_outputs'] = True
            model.recording_options['record_residuals'] = True

            recorder = om.SqliteRecorder("sqlite_test_pre_post", record_viewer_data=False)

            model.add_recorder(recorder)
            prob.driver.add_recorder(recorder)

            for comp in comps.values():
                comp.add_recorder(recorder)

        prob.setup(mode=mode, force_alloc_complex=force_complex)

        # we don't want ExecComps to be colored because it makes the iter counting more complicated.
        for comp in model.system_iter(recurse=True, typ=ExecComp4Test):
            comp.options['do_coloring'] = False
            comp.options['has_diag_partials'] = True

        return prob

    def test_pre_post_iter_rev(self):
        prob = setup_problem(do_pre_post_opt=True, mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_rev_grouped(self):
        prob = setup_problem(do_pre_post_opt=True, group=True, mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.G1.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.G1.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.G1.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.G1.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.G2.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.G2.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_rev_coloring(self):
        prob = setup_problem(do_pre_post_opt=True, coloring=True, mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_rev_coloring_grouped(self):
        prob = setup_problem(do_pre_post_opt=True, coloring=True, group=True, mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.G1.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.G1.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.G1.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.G1.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.G2.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.G2.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_auto_coloring_grouped_no_vois(self):
        # this computes totals and does total coloring without declaring dvs/objs/cons in the driver
        prob = self.setup_problem(do_pre_post_opt=True, coloring=True, group=True, mode='auto', set_vois=False)
        prob.final_setup()
        prob.run_model()

        coloring_info = prob.driver._coloring_info.copy()
        coloring_info.coloring = None
        coloring_info.dynamic = True

        J = prob.compute_totals(of=['iter2.y', 'iter3.y'], wrt=['iter1.x3'], coloring_info=coloring_info)

        data = prob.check_totals(of=['iter2.y', 'iter3.y'], wrt=['iter1.x3'], out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_rev_ivc(self):
        prob = setup_problem(do_pre_post_opt=True, use_ivc=True, mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_rev_ivc_grouped(self):
        prob = setup_problem(do_pre_post_opt=True, use_ivc=True, group=True, mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.G1.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.G1.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.G1.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.G1.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.G2.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.G2.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_rev_ivc_coloring(self):
        prob = setup_problem(do_pre_post_opt=True, use_ivc=True, coloring=True, mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd(self):
        prob = setup_problem(do_pre_post_opt=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_grouped(self):
        prob = setup_problem(do_pre_post_opt=True, group=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.G1.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.G1.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.G1.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.G1.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.G2.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.G2.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_coloring(self):
        prob = setup_problem(do_pre_post_opt=True, coloring=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_coloring_grouped(self):
        prob = setup_problem(do_pre_post_opt=True, coloring=True, group=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.G1.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.G1.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.G1.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.G1.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.G2.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.G2.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_coloring_grouped_force_post(self):
        prob = setup_problem(do_pre_post_opt=True, coloring=True, group=True,
                                  force=['post2'], mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model._pre_components, {'G1.pre1', 'G1.pre2', '_auto_ivc'})
        self.assertEqual(prob.model._post_components, set())

        self.assertEqual(prob.model.G1.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.G1.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.G1.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.G1.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.G2.post1.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.post2.num_nl_solves, 3)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_coloring_grouped_force_pre(self):
        prob = setup_problem(do_pre_post_opt=True, coloring=True, group=True,
                                  force=['pre1'], mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model._pre_components, set())
        self.assertEqual(prob.model._post_components, {'G2.post1', 'G2.post2'})

        self.assertEqual(prob.model.G1.pre1.num_nl_solves, 3)
        self.assertEqual(prob.model.G1.pre2.num_nl_solves, 3)

        self.assertEqual(prob.model.G1.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.G1.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.G2.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.G2.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.G2.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_ivc(self):
        prob = setup_problem(do_pre_post_opt=True, use_ivc=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_ivc_coloring(self):
        prob = setup_problem(do_pre_post_opt=True, use_ivc=True, coloring=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_approx(self):
        prob = setup_problem(do_pre_post_opt=True, mode='fwd', approx=True, force_complex=True)

        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 9)
        self.assertEqual(prob.model.iter2.num_nl_solves, 9)
        self.assertEqual(prob.model.iter3.num_nl_solves, 9)
        self.assertEqual(prob.model.iter4.num_nl_solves, 9)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_approx_grouped(self):
        prob = setup_problem(do_pre_post_opt=True, group=True, approx=True, force_complex=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.G1.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.G1.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.G1.iter1.num_nl_solves, 9)
        self.assertEqual(prob.model.G1.iter2.num_nl_solves, 9)
        self.assertEqual(prob.model.G2.iter3.num_nl_solves, 9)
        self.assertEqual(prob.model.G2.iter4.num_nl_solves, 9)

        self.assertEqual(prob.model.G2.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.G2.post2.num_nl_solves, 1)

        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_approx_coloring(self):
        prob = setup_problem(do_pre_post_opt=True, coloring=True, approx=True, force_complex=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 16)
        self.assertEqual(prob.model.iter2.num_nl_solves, 16)
        self.assertEqual(prob.model.iter3.num_nl_solves, 16)
        self.assertEqual(prob.model.iter4.num_nl_solves, 16)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_approx_ivc(self):
        prob = setup_problem(do_pre_post_opt=True, use_ivc=True, approx=True, force_complex=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 9)
        self.assertEqual(prob.model.iter2.num_nl_solves, 9)
        self.assertEqual(prob.model.iter3.num_nl_solves, 9)
        self.assertEqual(prob.model.iter4.num_nl_solves, 9)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_approx_ivc_coloring(self):
        prob = setup_problem(do_pre_post_opt=True, use_ivc=True, coloring=True, approx=True, force_complex=True, mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 16)
        self.assertEqual(prob.model.iter2.num_nl_solves, 16)
        self.assertEqual(prob.model.iter3.num_nl_solves, 16)
        self.assertEqual(prob.model.iter4.num_nl_solves, 16)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)

    def test_newton_with_densejac_under_full_model_fd(self):
        for dosplit in (True, False):
            with self.subTest(dosplit):
                prob = om.Problem(group_by_pre_opt_post=dosplit)
                prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False)
                prob.set_solver_print(level=0)

                model = prob.model = om.Group(assembled_jac_type='dense')

                model.add_subsystem('px', om.IndepVarComp('x', 1.0))
                model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

                model.add_subsystem('pre_comp', om.ExecComp('out = x + 5.'))
                model.connect('px.x', 'pre_comp.x')
                model.connect('pre_comp.out', 'x')
                sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

                sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
                sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
                sub.add_subsystem('post_comp', om.ExecComp('out = y1 + y2'), promotes=['y1', 'y2'])

                model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                        z=np.array([0.0, 0.0]), x=0.0),
                                    promotes=['obj', 'x', 'z', 'y1', 'y2'])

                model.add_subsystem('post_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
                model.add_subsystem('post_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

                sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                sub.linear_solver = om.DirectSolver(assemble_jac=True)

                model.add_design_var('z')
                model.add_objective('obj')

                prob.setup()
                prob.set_solver_print(level=0)
                prob.run_model()

                J = prob.compute_totals(return_format='flat_dict')
                assert_near_equal(J[('obj', 'z')], np.array([[9.62568658, 1.78576699]]), .00001)

    def test_reading_system_cases_pre_opt_post(self):
        prob = setup_problem(do_pre_post_opt=True, mode='fwd', recording=True)
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader('sqlite_test_pre_post')

        # check that we only have the three system sources
        self.assertEqual(sorted(cr.list_sources(out_stream=None)), ['driver', 'root', 'root.iter1', 'root.iter2', 'root.iter3', 'root.iter4', 'root.post1', 'root.post2', 'root.pre1', 'root.pre2'])

        # check source vars
        source_vars = cr.list_source_vars('root', out_stream=None)
        self.assertEqual(sorted(source_vars['inputs']),  ['iter1.x1', 'iter1.x2', 'iter1.x3', 'iter2.x', 'iter3.x', 'iter4.x', 'post1.x', 'post2.x1', 'post2.x2', 'post2.x3', 'pre1.x', 'pre2.x', 'pre2.xx'])
        self.assertEqual(sorted(source_vars['outputs']), ['iter1.x3', 'iter1.y', 'iter2.y', 'iter3.y', 'iter4.y', 'post1.y', 'post2.y', 'pre1.x', 'pre1.y', 'pre2.x', 'pre2.y'])

        # Test to see if we got the correct number of cases
        self.assertEqual(len(cr.list_cases('root', recurse=False, out_stream=None)), 5)
        self.assertEqual(len(cr.list_cases('root.iter1', recurse=False, out_stream=None)), 3)
        self.assertEqual(len(cr.list_cases('root.iter2', recurse=False, out_stream=None)), 3)
        self.assertEqual(len(cr.list_cases('root.iter3', recurse=False, out_stream=None)), 3)
        self.assertEqual(len(cr.list_cases('root.pre1', recurse=False, out_stream=None)), 1)
        self.assertEqual(len(cr.list_cases('root.pre2', recurse=False, out_stream=None)), 1)
        self.assertEqual(len(cr.list_cases('root.post1', recurse=False, out_stream=None)), 1)
        self.assertEqual(len(cr.list_cases('root.post2', recurse=False, out_stream=None)), 1)

        # Test to see if the case keys (iteration coords) come back correctly
        for i, iter_coord in enumerate(cr.list_cases('root.pre1', recurse=False, out_stream=None)):
            self.assertEqual(iter_coord, f'rank0:root._solve_nonlinear|{i}|NLRunOnce|{i}|pre1._solve_nonlinear|{i}')

        for i, iter_coord in enumerate(cr.list_cases('root.iter1', recurse=False, out_stream=None)):
            self.assertEqual(iter_coord, f'rank0:ScipyOptimize_SLSQP|{i}|root._solve_nonlinear|{i + 1}|NLRunOnce|0|iter1._solve_nonlinear|{i}')

    def test_incomplete_partials(self):
        p = om.Problem()
        model = p.model
        size = 3

        p.options['group_by_pre_opt_post'] = True

        p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False)
        p.set_solver_print(level=0)

        model.add_subsystem('pre1', ExecComp4Test('y=2.*x', shape=size))
        model.add_subsystem('pre2', ExecComp4Test('y=3.*x1 - 7.*x2', shape=size))

        model.add_subsystem('ivc', om.IndepVarComp('x', np.ones(size)))

        model.add_subsystem('iter1', ExecComp4Test('y=x*3.2', shape=size))
        model.add_subsystem('incomplete', ExecComp4Test(['y1=3.*x1', 'y2=5*x2'], shape=size))
        model.add_subsystem('obj', ExecComp4Test('obj=.5*x', shape=size))

        model.add_subsystem('post1', ExecComp4Test('y=8.*x', shape=size))

        model.connect('ivc.x', 'iter1.x')
        model.connect('iter1.y', 'incomplete.x2')
        model.connect('incomplete.y2', 'obj.x')
        model.connect('incomplete.y1', 'post1.x')
        model.connect('pre1.y', 'pre2.x2')
        model.connect('pre2.y', 'incomplete.x1')

        p.model.add_design_var('ivc.x', lower=-10, upper=10)
        p.model.add_objective('obj.obj', index=0)

        p.setup(mode='fwd', force_alloc_complex=True)

        # we don't want ExecComps to be colored because it makes the iter counting more complicated.
        for comp in model.system_iter(recurse=True, typ=ExecComp4Test):
            comp.options['do_coloring'] = False
            comp.options['has_diag_partials'] = True

        p.run_driver()

        self.assertEqual(p.model.pre1.num_nl_solves, 1)
        self.assertEqual(p.model.pre2.num_nl_solves, 1)

        self.assertEqual(p.model.incomplete.num_nl_solves, 4)
        self.assertEqual(p.model.iter1.num_nl_solves, 4)
        self.assertEqual(p.model.obj.num_nl_solves, 4)

        self.assertEqual(p.model.post1.num_nl_solves, 1)

        data = p.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)

    def test_comp_multiple_iter_lists(self):
        # Groups can be in multiple iteration lists, but not components.
        p = om.Problem()
        model = p.model
        G1 = model.add_subsystem('G1', om.Group(), promotes=['*'])
        G2 = model.add_subsystem('G2', om.Group(), promotes=['*'])
        G3 = model.add_subsystem('G3', om.Group(), promotes=['*'])

        G1.add_subsystem('C1', MissingPartialsComp())
        G1.add_subsystem('C2', MissingPartialsComp())
        G1.add_subsystem('C3', MissingPartialsComp())
        G2.add_subsystem('C4', MissingPartialsComp())
        G3.add_subsystem('C5', MissingPartialsComp())

        model.connect('C1.y', 'C2.a')
        model.connect('C1.z', 'C4.a')
        model.connect('C2.y', 'C3.a')
        model.connect('C3.z', 'C4.b')
        model.connect('C4.y', 'C5.a')

        model.add_design_var('C3.b')
        model.add_objective('C4.z')

        p.driver = om.ScipyOptimizeDriver()
        p.options["group_by_pre_opt_post"] = True

        p.setup()
        p.final_setup()

        self.assertEqual(model._pre_components, {'_auto_ivc', 'G1.C1', 'G1.C2'})
        self.assertEqual(model._iterated_components, {'G1.C3', 'G2.C4', '_auto_ivc'})
        self.assertEqual(model._post_components, {'G3.C5'})


@use_tempdirs
@unittest.skipUnless(pyoptsparse, "pyoptsparse is required.")
@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class PrePostMPITestCase(unittest.TestCase):
    N_PROCS = 2

    def test_newton_on_one_rank(self):
        prob = setup_problem(do_pre_post_opt=True, mode='fwd', parallel=True, group=True)

        prob.run_driver()


if __name__ == "__main__":
    unittest.main()

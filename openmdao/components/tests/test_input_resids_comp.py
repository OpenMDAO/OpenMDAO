import unittest

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal


class TestInputResidsComp(unittest.TestCase):

    def test_input_resids_comp(self):
        p = om.Problem()

        p.model.add_subsystem('exec_comp',
                              om.ExecComp(['res_a = a - x[0]',
                                           'res_b = b - x[1:]'],
                                           a={'shape': (1,)},
                                           b={'shape': (2,)},
                                           res_a={'shape': (1,)},
                                           res_b={'shape': (2,)},
                                           x={'shape':3}),
                                           promotes_inputs=['*'],
                                           promotes_outputs=['*'])

        resid_comp = p.model.add_subsystem('resid_comp',
                                           om.InputResidsComp(),
                                           promotes_inputs=['*'],
                                           promotes_outputs=['*'])

        resid_comp.add_output('x', shape=(3,))
        resid_comp.add_input('res_a', shape=(1,), ref=1.0)
        resid_comp.add_input('res_b', shape=(2,), ref=1.0)

        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p.set_val('a', 3.0)
        p.set_val('b', [4.0, 5.0])

        p.run_model()

        a = p.get_val('a')
        b = p.get_val('b')
        x = p.get_val('x')

        assert_near_equal(a, x[0], tolerance=1.0E-9)
        assert_near_equal(b, x[1:], tolerance=1.0E-9)

    def test_input_resids_comp_copy_shape(self):
        p = om.Problem()

        p.model.add_subsystem('exec_comp',
                              om.ExecComp(['res_a = a - x[0]',
                                           'res_b = b - x[1:]'],
                                           a={'shape': (1,)},
                                           b={'shape': (2,)},
                                           res_a={'shape': (1,)},
                                           res_b={'shape': (2,)},
                                           x={'shape':3}),
                                           promotes_inputs=['*'],
                                           promotes_outputs=['*'])

        resid_comp = p.model.add_subsystem('resid_comp',
                                           om.InputResidsComp(),
                                           promotes_inputs=['*'],
                                           promotes_outputs=['*'])

        resid_comp.add_output('x', shape=(3,))
        resid_comp.add_input('res_a', shape_by_conn=True, ref=1.0)
        resid_comp.add_input('res_b', shape_by_conn=True, ref=1.0)

        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p.set_val('a', 3.0)
        p.set_val('b', [4.0, 5.0])

        p.run_model()

        a = p.get_val('a')
        b = p.get_val('b')
        x = p.get_val('x')

        assert_near_equal(a, x[0], tolerance=1.0E-9)
        assert_near_equal(b, x[1:], tolerance=1.0E-9)

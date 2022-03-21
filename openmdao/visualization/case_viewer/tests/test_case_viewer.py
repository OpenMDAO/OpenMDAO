import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.visualization.case_viewer.case_viewer import _apply_slice, _apply_transform, \
    _get_var_meta, _get_opt_vars, _get_vars, _get_resids_vars, _get_resids_val, _get_plot_style


@use_tempdirs
class TestCaseRetrieval(unittest.TestCase):

    def setUp(self):
        # build the model
        prob = om.Problem()
        prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])

        # define the component whose output will be constrained
        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        # Design variables 'x' and 'y' span components, so we need to provide a common initial
        # value for them.
        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_recorder(om.SqliteRecorder('parab_record.sql'))

        prob.driver.recording_options['includes'] = ['*']
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_residuals'] = True

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        # to add the constraint to the model
        prob.model.add_constraint('const.g', lower=0, upper=10.)

        prob.setup()

        prob.run_driver()

    def test_get_vars(self):
        cr = om.CaseReader('parab_record.sql')
        case_names = cr.list_cases(out_stream=None)
        inputs = _get_vars(cr, case_names=case_names, var_types='inputs')
        outputs = _get_vars(cr, case_names=case_names, var_types='outputs')
        self.assertSetEqual(set(inputs), {'x', 'y'})
        self.assertSetEqual(set(outputs), {'const.g', 'parab.f_xy'})

    def test_get_opt_vars(self):
        cr = om.CaseReader('parab_record.sql')
        case_names = cr.list_cases(out_stream=None)
        desvars = _get_opt_vars(cr, case_names, var_type='desvars')
        constraints = _get_opt_vars(cr, case_names, var_type='constraints')
        objectives = _get_opt_vars(cr, case_names, var_type='objectives')

        self.assertSetEqual(set(desvars), {'x', 'y'})
        self.assertSetEqual(set(constraints), {'const.g'})
        self.assertSetEqual(set(objectives), {'parab.f_xy'})

        all = _get_opt_vars(cr, case_names)
        self.assertSetEqual(set(all), {'x', 'y', 'const.g', 'parab.f_xy'})

    def test_get_var_meta(self):
        cr = om.CaseReader('parab_record.sql')
        case_name = cr.list_cases(out_stream=None)[0]
        g_meta = _get_var_meta(cr, case_name, 'const.g')
        self.assertEqual(g_meta['prom_name'], 'const.g')
        self.assertEqual(g_meta['units'], None)
        self.assertEqual(g_meta['shape'], (1,))

        f_xy_meta = _get_var_meta(cr, case_name, 'parab.f_xy')
        self.assertEqual(f_xy_meta['prom_name'], 'parab.f_xy')
        self.assertEqual(f_xy_meta['units'], None)
        self.assertEqual(f_xy_meta['shape'], (1,))

        x_meta = _get_var_meta(cr, case_name, 'x')
        self.assertEqual(x_meta['prom_name'], 'x')
        self.assertEqual(x_meta['units'], None)
        self.assertEqual(x_meta['shape'], (1,))

    def test_get_resids_vars(self):
        cr = om.CaseReader('parab_record.sql')
        case_names = cr.list_cases(out_stream=None)
        vars_with_resids = _get_resids_vars(cr, case_names)
        self.assertSetEqual(set(vars_with_resids), {'const.g', 'parab.f_xy'})

    def test_get_resids_val(self):
        cr = om.CaseReader('parab_record.sql')
        case_name = cr.list_cases(out_stream=None)[0]
        case = cr.get_case(case_name)

        g_resids = _get_resids_val(case, 'const.g')
        f_resids = _get_resids_val(case, 'parab.f_xy')

        assert_near_equal(g_resids, [0.])
        assert_near_equal(f_resids, [0.])


class TestOtherFuncs(unittest.TestCase):

    def test_apply_slice(self):
        x = np.random.random((20, 3)) - 0.5

        assert_near_equal(_apply_slice(x, '[...]'), x[...])
        assert_near_equal(_apply_slice(x, '[2, 1::2]'), x[2, 1::2])
        assert_near_equal(_apply_slice(x, '[:, -1]'), x[:, -1])
        assert_near_equal(_apply_slice(x, '[:, [0, 1]]'), x[:, [0, 1]])

    def test_apply_transform(self):
        x = np.random.random((20, 3)) - 0.5

        max = np.max(x)
        assert_near_equal(_apply_transform(x, 'max'), max)

        min = np.min(x)
        assert_near_equal(_apply_transform(x, 'min'), min)

        maxabs = np.max(np.abs(x))
        assert_near_equal(_apply_transform(x, 'maxabs'), maxabs)

        minabs = np.min(np.abs(x))
        assert_near_equal(_apply_transform(x, 'minabs'), minabs)

        norm = np.linalg.norm(x)
        assert_near_equal(_apply_transform(x, 'norm'), norm)

        ravel = np.ravel(x)
        assert_near_equal(_apply_transform(x, 'ravel'), ravel)

        assert_near_equal(_apply_transform(x, 'None'), x)

    def test_get_plot_style(self):
        # Selected case index is greater or equal to than the number of cases.
        # Style all plots equally.
        lw, ms, s, alpha = _get_plot_style(1, 20, 20)
        self.assertEqual(lw, 1.0)
        self.assertEqual(ms, 2.0)
        self.assertEqual(s, 10)
        self.assertEqual(alpha, 1.0)

        # Selected case index is less than the number of cases
        # but idx != selected_case idx
        # This plot should have low visibility.
        lw, ms, s, alpha = _get_plot_style(1, 19, 20)
        self.assertEqual(lw, 0.5)
        self.assertEqual(ms, 2.0)
        self.assertEqual(s, 10)
        self.assertEqual(alpha, 0.1)

        # Selected case index is less than the number of cases
        # and idx == selected_case idx
        # This plot should have high visibility.
        lw, ms, s, alpha = _get_plot_style(19, 19, 20)
        self.assertEqual(lw, 2.0)
        self.assertEqual(ms, 6.0)
        self.assertEqual(s, 40)
        self.assertEqual(alpha, 1)

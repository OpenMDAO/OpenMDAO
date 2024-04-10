import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials


class MyCompApprox(om.ImplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()

    def setup(self):
        self.add_input('mm', np.ones(1))
        self.add_output('Re', np.ones((1, 1)))
        self.add_output('temp', np.ones((1, 1)))
        self.add_residual('res', shape=(2,))

    def setup_partials(self):
        self.declare_partials('res', ['*'], method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        mm = inputs['mm'].item()
        Re = outputs['Re'].item()
        temp = outputs['temp'][0][0].item()

        T = 389.97
        cf = 0.01
        RE = 1.479301E9 * .0260239151 * (T / 1.8 + 110.4) / (T / 1.8) ** 2
        comb = 4.593153E-6 * 0.8 * (T + 198.72) / (RE * mm * T ** 1.5)
        temp_ratio = 1.0 + 0.035 * mm * mm + 0.45 * (temp / T - 1.0)
        CFL = cf / (1.0 + 3.59 * np.sqrt(cf) * temp_ratio)
        residuals['res'][0] = Re - RE * mm
        residuals['res'][1] = (1.0 / (1.0 +  comb * temp ** 3 / CFL) + temp) * 0.5 - temp


class MyCompAnalytic(MyCompApprox):
    def setup_partials(self):
        self.declare_partials('res', ['*'])

    def linearize(self, inputs, outputs, partials):
        mm = inputs['mm'][0]
        temp = outputs['temp'][0][0]
        T = 389.97
        cf = 0.01
        RE = 1.479301E9 * .0260239151 * ((T / 1.8) + 110.4) / (T / 1.8) ** 2
        comb = 4.593153E-6 * 0.8 * (T + 198.72) / (RE * mm * T ** 1.5)
        dcomb_dmm = -4.593153E-6 * 0.8 * (T + 198.72) / (RE * mm * mm * T ** 1.5)
        temp_ratio = 1.0 +  0.035 * mm * mm + 0.45 * temp / T - 1.0
        CFL = cf / (1.0 + 3.59 * np.sqrt(cf) * temp_ratio)
        dCFL_dwtr = - cf * 3.59 * np.sqrt(cf) / (1.0 + 3.59 * np.sqrt(cf) * temp_ratio) ** 2
        den = 1.0 + comb * temp ** 3 / CFL
        dreswt_dcomb = -0.5 * temp ** 3 / (CFL * den ** 2)
        dreswt_dCFL = 0.5 * comb * temp ** 3 / (CFL * den) ** 2
        dreswt_dwt = -0.5 - 1.5 * comb * temp ** 2 / (CFL * den ** 2)

        partials['res', 'mm'] = np.array([[-RE],
                                          [dreswt_dcomb * dcomb_dmm +  dreswt_dCFL * dCFL_dwtr * 0.07 * mm]])
        partials['res', 'temp'] = np.array([[0.],
                                            [dreswt_dCFL * dCFL_dwtr * 0.45 / T + dreswt_dwt]])
        partials['res', 'Re'] = np.array([[1.],[0.]])


class MyCompApprox2(MyCompApprox):
    def setup(self):
        self.add_input('mm', np.ones(1))
        self.add_output('Re', np.ones((1, 1)))
        self.add_output('temp', np.ones((1, 1)))
        self.add_residual('res1', shape=(1,))
        self.add_residual('res2', shape=(1,))

    def setup_partials(self):
        self.declare_partials('res1', ['Re', 'mm'], method='fd')
        self.declare_partials('res2', ['temp', 'mm'], method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        mm = inputs['mm'][0]
        T = 389.97
        cf = 0.01
        temp = outputs['temp'][0][0]
        RE = 1.479301E9 * .0260239151 * (T / 1.8 + 110.4) / (T / 1.8) ** 2
        comb = 4.593153E-6 * 0.8 * (T + 198.72) / (RE * mm * T ** 1.5)
        temp_ratio = 1.0 + 0.035 * mm * mm + 0.45 * (temp / T - 1.0)
        CFL = cf / (1.0 + 3.59 * np.sqrt(cf) * temp_ratio)
        residuals['res1'] = outputs['Re'] - RE * mm
        residuals['res2'] = (1.0 / (1.0 +  comb * temp ** 3 / CFL) + temp) * 0.5 - temp


class MyCompSizeMismatch(MyCompApprox):
    def setup(self):
        self.add_input('mm', np.ones(1))
        self.add_output('Re', np.ones((1, 1)))
        self.add_output('temp', np.ones((1, 1)))
        self.add_residual('res1', shape=(1,))
        self.add_residual('res2', shape=(2,))

    def setup_partials(self):
        self.declare_partials('res1', ['Re', 'mm'], method='fd')
        self.declare_partials('res2', ['temp', 'mm'], method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass


class MyCompShapeMismatch(MyCompApprox):
    def setup(self):
        self.add_input('mm', np.ones(1))
        self.add_output('Re', np.ones((1, 1)))
        self.add_output('temp', np.ones((1, 1)))
        self.add_residual('res1', shape=(1,), ref=np.ones((1,2)))
        self.add_residual('res2', shape=(1,))

    def setup_partials(self):
        self.declare_partials('res1', ['Re', 'mm'], method='fd')
        self.declare_partials('res2', ['temp', 'mm'], method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass


class MyCompBadUnits(MyCompApprox):
    def setup(self):
        self.add_input('mm', np.ones(1))
        self.add_output('Re', np.ones((1, 1)))
        self.add_output('temp', np.ones((1, 1)))
        self.add_residual('res1', shape=(1,), units="foobar/baz")
        self.add_residual('res2', shape=(1,))

    def setup_partials(self):
        self.declare_partials('res1', ['Re', 'mm'], method='fd')
        self.declare_partials('res2', ['temp', 'mm'], method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass

class MyCompUnitsMismatchNoDerivs(MyCompApprox):
    def setup(self):
        self.add_input('mm', np.ones(1))
        self.add_output('Re', np.ones((1, 1)), res_units='ft')
        self.add_output('temp', np.ones((1, 1)))
        self.add_residual('res1', shape=(1,), units="inch")
        self.add_residual('res2', shape=(1,))

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass


class MyCompUnitsMismatch(MyCompUnitsMismatchNoDerivs):
    def setup_partials(self):
        self.declare_partials('res1', ['Re', 'mm'], method='fd')
        self.declare_partials('res2', ['temp', 'mm'], method='fd')


class MyCompRefMismatch(MyCompApprox):
    def setup(self):
        self.add_input('mm', np.ones(1))
        self.add_output('Re', np.ones((1, 1)), res_ref=3.)
        self.add_output('temp', np.ones((1, 1)))
        self.add_residual('res1', shape=(1,), ref=4.)
        self.add_residual('res2', shape=(1,))

    def setup_partials(self):
        self.declare_partials('res1', ['Re', 'mm'], method='fd')
        self.declare_partials('res2', ['temp', 'mm'], method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass


class MyCompRefMismatchDefault(MyCompApprox):
    def setup(self):
        self.add_input('mm', np.ones(1))
        self.add_output('Re', np.ones((1, 1)))
        self.add_output('temp', np.ones((1, 1)))
        self.add_residual('res1', shape=(1,), ref=4.)
        self.add_residual('res2', shape=(1,))

    def setup_partials(self):
        self.declare_partials('res1', ['Re', 'mm'], method='fd')
        self.declare_partials('res2', ['temp', 'mm'], method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass


class MyCompRefMismatchDefault2(MyCompApprox):
    def setup(self):
        self.add_input('mm', np.ones(1))
        self.add_output('Re', np.ones((1, 1)), res_ref=4.)
        self.add_output('temp', np.ones((1, 1)))
        self.add_residual('res1', shape=(1,))
        self.add_residual('res2', shape=(1,))

    def setup_partials(self):
        self.declare_partials('res1', ['Re', 'mm'], method='fd')
        self.declare_partials('res2', ['temp', 'mm'], method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass


class MyCompAnalytic2(MyCompApprox2):
    def setup_partials(self):
        self.declare_partials('res1', ['Re', 'mm'])
        self.declare_partials('res2', ['temp', 'mm'])

    def linearize(self, inputs, outputs, partials):
        mm = inputs['mm'][0]
        temp = outputs['temp'][0][0]
        T = 389.97
        cf = 0.01
        RE = 1.479301E9 * .0260239151 * ((T / 1.8) + 110.4) / (T / 1.8) ** 2
        comb = 4.593153E-6 * 0.8 * (T + 198.72) / (RE * mm * T ** 1.5)
        dcomb_dmm = -4.593153E-6 * 0.8 * (T + 198.72) / (RE * mm * mm * T ** 1.5)
        temp_ratio = 1.0 +  0.035 * mm * mm + 0.45 * temp / T - 1.0
        CFL = cf / (1.0 + 3.59 * np.sqrt(cf) * temp_ratio)
        dCFL_dwtr = - cf * 3.59 * np.sqrt(cf) / (1.0 + 3.59 * np.sqrt(cf) * temp_ratio) ** 2
        den = 1.0 + comb * temp ** 3 / CFL
        dreswt_dcomb = -0.5 * temp ** 3 / (CFL * den ** 2)
        dreswt_dCFL = 0.5 * comb * temp ** 3 / (CFL * den) ** 2
        dreswt_dwt = -0.5 - 1.5 * comb * temp ** 2 / (CFL * den ** 2)
        partials['res1', 'Re'] = 1.0
        partials['res1', 'mm'] = -RE
        partials['res2', 'mm'] = dreswt_dcomb * dcomb_dmm +  dreswt_dCFL * dCFL_dwtr * 0.07 * mm
        partials['res2', 'temp'] = (dreswt_dCFL * dCFL_dwtr * 0.45 / T + dreswt_dwt)


class ResidNamingTestCase(unittest.TestCase):
    def _build_model(self, comp_class):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('MyComp', comp_class(), promotes=['*'])

        model.add_objective('Re')
        model.add_design_var('mm')

        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.set_val("mm", val=0.2)

        prob.run_model()

        return prob

    def test_approx(self):
        prob = self._build_model(MyCompApprox)
        assert_check_partials(prob.check_partials(method='cs', out_stream=None), atol=1e-5)

        totals = prob.check_totals(method='cs', out_stream=None)
        for val in totals.values():
            assert_near_equal(val['rel error'][0], 0.0, 1e-10)

    def test_approx2(self):
        prob = self._build_model(MyCompApprox2)
        assert_check_partials(prob.check_partials(method='cs', out_stream=None), atol=1e-5)

        totals = prob.check_totals(method='cs', out_stream=None)
        for val in totals.values():
            assert_near_equal(val['rel error'][0], 0.0, 1e-10)

    def test_size_mismatch(self):
        with self.assertRaises(Exception) as cm:
            prob = self._build_model(MyCompSizeMismatch)

        self.assertEqual(cm.exception.args[0], "'MyComp' <class MyCompSizeMismatch>: The number of residuals (3) doesn't match number of outputs (2).  If any residuals are added using 'add_residuals', their total size must match the total size of the outputs.")

    def test_ref_shape_mismatch(self):
        with self.assertRaises(Exception) as cm:
            prob = self._build_model(MyCompShapeMismatch)

        self.assertEqual(cm.exception.args[0], "'MyComp' <class MyCompShapeMismatch>: When adding residual 'res1', expected shape (1,) but got shape (1, 2) for argument 'ref'.")

    def test_bad_unit(self):
        with self.assertRaises(Exception) as cm:
            prob = self._build_model(MyCompBadUnits)

        self.assertEqual(cm.exception.args[0], "'MyComp' <class MyCompBadUnits>: The units 'foobar/baz' are invalid.")

    def test_unit_mismatch(self):
        with self.assertRaises(Exception) as cm:
            prob = self._build_model(MyCompUnitsMismatch)

        self.assertEqual(cm.exception.args[0], "'MyComp' <class MyCompUnitsMismatch>: residual units 'inch' for residual 'res1' != output res_units 'ft' for output 'Re'.")

    def test_ref_mismatch(self):
        with self.assertRaises(Exception) as cm:
            prob = self._build_model(MyCompRefMismatch)

        self.assertEqual(cm.exception.args[0], "'MyComp' <class MyCompRefMismatch>: (4.0 != 3.0), 'ref' for residual 'res1' != 'res_ref' for output 'Re'.")

    def test_ref_mismatch_default_no_exception(self):
        prob = self._build_model(MyCompRefMismatchDefault)

    def test_ref_mismatch_default2_no_exception(self):
        prob = self._build_model(MyCompRefMismatchDefault2)

    def test_analytic(self):
        prob = self._build_model(MyCompAnalytic)
        assert_check_partials(prob.check_partials(method='cs', out_stream=None))

        totals = prob.check_totals(method='cs', out_stream=None)
        for val in totals.values():
            assert_near_equal(val['rel error'][0], 0.0, 1e-12)

    def test_analytic2(self):
        prob = self._build_model(MyCompAnalytic2)
        assert_check_partials(prob.check_partials(method='cs', out_stream=None))

        totals = prob.check_totals(method='cs', out_stream=None)
        for val in totals.values():
            assert_near_equal(val['rel error'][0], 0.0, 1e-12)


class _InputResidComp(om.ImplicitComponent):

    def __init__(self, jac, add_io_in_setup=True):
        self._jac = jac
        self._add_io_in_setup = add_io_in_setup
        super().__init__()

    def setup(self):
        if self._add_io_in_setup:
            self.add_output('x', shape=(3,))
            self.add_residual_from_input('aa', shape=(1,))
            self.add_residual_from_input('bb', shape=(2,))

    def add_residual_from_input(self, name, **kwargs):
        resid_name = 'resid_' + name
        shape = kwargs['shape'] if 'shape' in kwargs else (1,)
        size = np.prod(shape)

        self.add_input(name, **kwargs)
        self.add_residual(resid_name, **kwargs)

        if self._jac in ('fd', 'cs'):
            self.declare_partials(of=resid_name, wrt=name, method=self._jac)
        elif self._jac == 'dense':
            self.declare_partials(of=resid_name, wrt=name, val=np.eye(size))
        elif self._jac == 'sparse':
            ar = np.arange(size, dtype=int)
            self.declare_partials(of=resid_name, wrt=name, rows=ar, cols=ar, val=1.0)
        else:
            raise ValueError('invalid value for jac use one of ', ['fd', 'cs', 'dense', 'sparse'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals.set_val(inputs.asarray())


class _TestGroupConfig(om.Group):

    def setup(self):
        self.add_subsystem('exec_com', om.ExecComp(['res_a = a - x[0]', 'res_b = b - x[1:]'],
                                                   a={'shape': (1,)},
                                                   b={'shape': (2,)},
                                                   res_a={'shape': (1,)},
                                                   res_b={'shape': (2,)},
                                                   x={'shape':3}),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('resid_comp', _InputResidComp(add_io_in_setup=False, jac='fd'),
                           promotes_inputs=['*'], promotes_outputs=['*'])

    def configure(self):
        resid_comp = self._get_subsystem('resid_comp')
        resid_comp.add_output('x', shape=(3,))
        resid_comp.add_residual_from_input('res_a', shape=(1,))
        resid_comp.add_residual_from_input('res_b', shape=(2,))
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()


class TestAddResidualConfigure(unittest.TestCase):

    def test_add_residual_configure(self):
        p = om.Problem()
        p.model.add_subsystem('test_group', _TestGroupConfig())
        p.setup()

        p.set_val('test_group.a', 3.0)
        p.set_val('test_group.b', [4.0, 5.0])

        p.run_model()

        a = p.get_val('test_group.a')
        b = p.get_val('test_group.b')
        x = p.get_val('test_group.x')

        assert_near_equal(a, x[0], tolerance=1.0E-9)
        assert_near_equal(b, x[1:], tolerance=1.0E-9)


class _TestGroup(om.Group):

    def __init__(self, jac):
        self._jac = jac
        super().__init__()

    def setup(self):
        self.add_subsystem('exec_com', om.ExecComp(['aa = a - x[0]', 'bb = b - x[1:]'],
                                                   a={'shape': (1,)},
                                                   b={'shape': (2,)},
                                                   aa={'shape': (1,)},
                                                   bb={'shape': (2,)},
                                                   x={'shape':3}),
                           promotes=['*'])

        self.add_subsystem('resid_comp', _InputResidComp(jac=self._jac),
                           promotes_inputs=['*'], promotes_outputs=['*'])
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()


def run_rename_test(jac):
    p = om.Problem()
    p.model.add_subsystem('test_group', _TestGroup(jac=jac))
    p.setup()

    p.set_val('test_group.a', 3.0)
    p.set_val('test_group.b', [4.0, 5.0])

    p.run_model()

    a = p.get_val('test_group.a')
    b = p.get_val('test_group.b')
    x = p.get_val('test_group.x')

    assert_near_equal(a, x[0], tolerance=1.0E-9)
    assert_near_equal(b, x[1:], tolerance=1.0E-9)


class TestRenamedResidsDifferentJacs(unittest.TestCase):
    def test_fd(self):
        run_rename_test('fd')

    def test_cs(self):
        run_rename_test('cs')

    def test_dense(self):
        run_rename_test('dense')

    def test_sparse(self):
        run_rename_test('sparse')


if __name__ == '__main__':
    unittest.main()

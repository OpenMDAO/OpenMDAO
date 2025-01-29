import unittest

import numpy as np

import openmdao.api as om
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarNoDerivatives
from openmdao.test_suite.components.sellar_feature import SellarNoDerivativesCS
from openmdao.test_suite.components.partial_check_feature import BrokenDerivComp
from openmdao.utils.assert_utils import assert_check_partials, assert_no_approx_partials, assert_no_dict_jacobians
from openmdao.utils.testing_utils import snum_equal


class TestAssertUtils(unittest.TestCase):

    def test_assert_check_partials_no_exception_expected(self):

        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0 * inputs['x1'] + 4.0 * inputs['x2']

            def compute_partials(self, inputs,     partials):
                """Correct derivative."""
                J = partials
                J['y', 'x1'] = np.array([3.0])
                J['y', 'x2'] = np.array([4.0])

        prob = om.Problem()
        prob.model.add_subsystem('comp', MyComp())

        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    def test_assert_check_partials_exception_expected(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', BrokenDerivComp())

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        expected = """
==============================================================
assert_check_partials failed for the following Components
with absolute tolerance = 1e-06 and relative tolerance = 1e-06
==============================================================

---------------
Component: comp
---------------
Forward derivatives of 'y' wrt 'x1' do not match finite difference.
Mismatched elements: 1 / 1 (100%)
Max absolute difference: 1.
Max relative difference: 0.33333333
J_fd - J_fwd:
[[-1.]]

Forward derivatives of 'y' wrt 'x2' do not match finite difference.
Mismatched elements: 1 / 1 (100%)
Max absolute difference: 36.
Max relative difference: 9.
J_fd - J_fwd:
[[-36.]]
""".strip()

        if np.__version__.split('.')[0] > '1':
            expected = expected.replace('difference:', 'difference among violations:')

        try:
            assert_check_partials(data, atol=1.e-6, rtol=1.e-6, verbose=True)
        except Exception as err:
            if not snum_equal(err.args[0].strip(), expected):
                # just show normal string diff
                self.assertEqual(err.args[0], expected)
        else:
            self.fail('Exception expected.')

    def test_assert_no_approx_partials_exception_expected(self):

        prob = om.Problem()
        prob.model = SellarNoDerivativesCS()

        prob.setup()
        prob.final_setup()

        try:
            assert_no_approx_partials(prob.model, include_self=True, recurse=True)

        except AssertionError as err:
            expected_err = \
'''The following components use approximated partials:
    cycle.d1
        of=*               wrt=*               method=cs
    cycle.d2
        of=*               wrt=*               method=cs
'''
            self.assertEqual(str(err), expected_err)
        else:
            self.fail('Exception expected.')

    def test_assert_no_approx_partials_cs_method(self):

        prob = om.Problem()
        prob.model = SellarNoDerivativesCS()

        prob.setup()
        prob.final_setup()

        with self.assertRaises(AssertionError):
            assert_no_approx_partials(prob.model, include_self=True, recurse=True, method='any')

        with self.assertRaises(AssertionError) :
            assert_no_approx_partials(prob.model, include_self=True, recurse=True, method='cs')

        assert_no_approx_partials(prob.model, include_self=True, recurse=True, method='fd')

    def test_assert_no_approx_partials_fd_method(self):

        prob = om.Problem()
        prob.model = SellarNoDerivatives()  # uses FD

        prob.setup()
        prob.final_setup()

        with self.assertRaises(AssertionError):
            assert_no_approx_partials(prob.model, include_self=True, recurse=True, method='any')

        with self.assertRaises(AssertionError):
            assert_no_approx_partials(prob.model, include_self=True, recurse=True, method='fd')

        assert_no_approx_partials(prob.model, include_self=True, recurse=True, method='cs')

    def test_assert_no_approx_partials_exception_not_expected(self):

        prob = om.Problem()
        prob.model = DoubleSellar()

        prob.setup()

        assert_no_approx_partials(prob.model, include_self=True, recurse=True)

    def test_assert_no_approx_partials_exception_exclude_single_pathname(self):

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExecComp('y=x+1.', x=2.0))
        C1.declare_partials('y', 'x', method='fd')

        prob.setup()
        prob.final_setup()

        with self.assertRaises(AssertionError) :
            assert_no_approx_partials(prob.model, include_self=True, recurse=True)

        assert_no_approx_partials(prob.model, include_self=True, recurse=True, excludes='C1')

    def test_assert_no_approx_partials_exception_exclude_pathnames(self):

        prob = om.Problem()
        prob.model = SellarNoDerivativesCS()

        prob.setup()
        prob.final_setup()

        assert_no_approx_partials(prob.model, include_self=True, recurse=True,
                                  excludes=['cycle.d1', 'cycle.d2'])

    def test_assert_no_dict_jacobians_exception_expected(self):

        prob = om.Problem()
        prob.model = SellarNoDerivativesCS()

        prob.setup()
        prob.model.cycle._jacobian = DictionaryJacobian(prob.model.cycle)

        try:
            assert_no_dict_jacobians(prob.model, include_self=True, recurse=True)

        except AssertionError as err:
            expected_err = "The following groups use dictionary jacobians:\n\n    cycle"
            self.assertEqual(str(err), expected_err)
        else:
            self.fail('Exception expected.')

    def test_assert_no_dict_jacobians_exception_not_expected(self):

        model = om.Group(assembled_jac_type='dense')
        ivc = om.IndepVarComp()
        ivc.add_output('x', 3.0)
        ivc.add_output('y', -4.0)
        model.add_subsystem('des_vars', ivc)
        model.add_subsystem('parab_comp', Paraboloid())

        model.connect('des_vars.x', 'parab_comp.x')
        model.connect('des_vars.y', 'parab_comp.y')

        prob = om.Problem(model)
        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.setup()

        assert_no_dict_jacobians(prob.model, include_self=True, recurse=True)

    def test_assert_check_partials_nan(self):
        # Due to a bug, this case passed the assert when it should not have.

        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', 3.0)

                self.add_output('y', 5.5)

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                # Introduce an error.
                outputs['y'] = 3.0 * inputs['x']

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = np.nan

        prob = om.Problem()
        prob.model.add_subsystem('comp', MyComp())

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        expected = """
==============================================================
assert_check_partials failed for the following Components
with absolute tolerance = 1e-06 and relative tolerance = 1e-06
==============================================================

---------------
Component: comp
---------------
Forward derivatives of 'y' wrt 'x' do not match finite difference.
Mismatched elements: 1 / 1 (100%)
Max absolute difference: nan
Max relative difference: nan
J_fd - J_fwd:
[[nan]]
""".strip()

        if np.__version__.split('.')[0] > '1':
            expected = expected.replace('difference:', 'difference among violations:')

        try:
            assert_check_partials(data, atol=1.e-6, rtol=1.e-6, verbose=True)
        except ValueError as err:
            if not snum_equal(err.args[0].strip(), expected):
                self.assertEqual(err.args[0].strip(), expected)
        else:
            self.fail('Exception expected.')


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class TestCrossProductCompNx3(unittest.TestCase):

    def setUp(self):
        self.n = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.n, 3))
        ivc.add_output(name='b', shape=(self.n, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='cross_prod_comp',
                                   subsys=om.CrossProductComp(vec_size=self.n))

        self.p.model.connect('a', 'cross_prod_comp.a')
        self.p.model.connect('b', 'cross_prod_comp.b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.n, 3)
        self.p['b'] = np.random.rand(self.n, 3)

        self.p['a'][:, 0] = 2.0
        self.p['a'][:, 1] = 3.0
        self.p['a'][:, 2] = 4.0

        self.p['b'][:, 0] = 5.0
        self.p['b'][:, 1] = 6.0
        self.p['b'][:, 2] = 7.0

        self.p.run_model()

    def test_results(self):

        for i in range(self.n):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p['cross_prod_comp.c'][i, :]
            expected_i = np.cross(a_i, b_i)

            np.testing.assert_almost_equal(c_i, expected_i)

    def test_partials(self):
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)

class TestCrossProductCompNx3x1(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3, 1))
        ivc.add_output(name='b', shape=(self.nn, 3, 1))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='cross_prod_comp',
                                   subsys=om.CrossProductComp(vec_size=self.nn))

        self.p.model.connect('a', 'cross_prod_comp.a')
        self.p.model.connect('b', 'cross_prod_comp.b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn, 3, 1)
        self.p['b'] = np.random.rand(self.nn, 3, 1)

        self.p['a'][:, 0, 0] = 2.0
        self.p['a'][:, 1, 0] = 3.0
        self.p['a'][:, 2, 0] = 4.0

        self.p['b'][:, 0, 0] = 5.0
        self.p['b'][:, 1, 0] = 6.0
        self.p['b'][:, 2, 0] = 7.0

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :, 0]
            b_i = self.p['b'][i, :, 0]
            c_i = self.p['cross_prod_comp.c'][i, :]
            expected_i = np.cross(a_i, b_i)

            np.testing.assert_almost_equal(c_i, expected_i)

    def test_partials(self):
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)

class TestCrossProductCompNonVectorized(unittest.TestCase):

    def setUp(self):
        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(3, 1))
        ivc.add_output(name='b', shape=(3, 1))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='cross_prod_comp',
                                   subsys=om.CrossProductComp())

        self.p.model.connect('a', 'cross_prod_comp.a')
        self.p.model.connect('b', 'cross_prod_comp.b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(3, 1)
        self.p['b'] = np.random.rand(3, 1)

        self.p['a'][0, 0] = 2.0
        self.p['a'][1, 0] = 3.0
        self.p['a'][2, 0] = 4.0

        self.p['b'][0, 0] = 5.0
        self.p['b'][1, 0] = 6.0
        self.p['b'][2, 0] = 7.0

        self.p.run_model()

    def test_results(self):

        a_i = self.p['a'][:, 0]
        b_i = self.p['b'][:, 0]
        c_i = self.p['cross_prod_comp.c'][:]

        expected = np.cross(a_i, b_i)

        np.testing.assert_almost_equal(c_i.ravel(), expected)

    def test_partials(self):
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='ft')
        ivc.add_output(name='b', shape=(self.nn, 3), units='lbf')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='cross_prod_comp',
                                   subsys=om.CrossProductComp(vec_size=self.nn,
                                                              a_units='m', b_units='N',
                                                              c_units='N*m'))

        self.p.model.connect('a', 'cross_prod_comp.a')
        self.p.model.connect('b', 'cross_prod_comp.b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn, 3)
        self.p['b'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p.get_val('cross_prod_comp.c', units='ft*lbf')[i, :]
            expected_i = np.cross(a_i, b_i)

            assert_near_equal(c_i, expected_i, tolerance=1.0E-12)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='ft')
        ivc.add_output(name='b', shape=(self.nn, 3), units='lbf')
        ivc.add_output(name='x', shape=(self.nn, 3), units='ft')
        ivc.add_output(name='y', shape=(self.nn, 3), units='lbf')

        cpc = om.CrossProductComp(vec_size=self.nn,
                                  a_units='m', b_units='N', c_units='N*m')

        cpc.add_product('z', a_name='x', b_name='y', vec_size=self.nn,
                        a_units='m', b_units='N', c_units='N*m')

        model = om.Group()
        model.add_subsystem('ivc', subsys=ivc, promotes_outputs=['a', 'b', 'x', 'y'])

        model.add_subsystem('cross_prod_comp', subsys=cpc)

        model.connect('a', 'cross_prod_comp.a')
        model.connect('b', 'cross_prod_comp.b')
        model.connect('x', 'cross_prod_comp.x')
        model.connect('y', 'cross_prod_comp.y')

        p = self.p = om.Problem(model)
        p.setup()

        p['a'] = np.random.rand(self.nn, 3)
        p['b'] = np.random.rand(self.nn, 3)

        p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p.get_val('cross_prod_comp.c', units='ft*lbf')[i, :]
            expected_i = np.cross(a_i, b_i)

            assert_near_equal(c_i, expected_i, tolerance=1.0E-12)

            x_i = self.p['x'][i, :]
            y_i = self.p['y'][i, :]
            z_i = self.p.get_val('cross_prod_comp.z', units='ft*lbf')[i, :]
            expected_i = np.cross(x_i, y_i)

            assert_near_equal(z_i, expected_i, tolerance=1.0E-12)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleConfigure(unittest.TestCase):

    def setUp(self):

        class MyModel(om.Group):

            def setup(self):
                ivc = om.IndepVarComp()
                ivc.add_output(name='a', shape=(5, 3), units='ft')
                ivc.add_output(name='b', shape=(5, 3), units='lbf')

                cpc = om.CrossProductComp(vec_size=5, a_units='m', b_units='N', c_units='N*m')

                self.add_subsystem('ivc', ivc, promotes_outputs=['*'])
                self.add_subsystem('cpc', cpc)

                self.connect('a', 'cpc.a')
                self.connect('b', 'cpc.b')

            def configure(self):
                self.ivc.add_output(name='x', shape=(5, 3), units='ft')
                self.ivc.add_output(name='y', shape=(5, 3), units='lbf')

                self.cpc.add_product('z', a_name='x', b_name='y', vec_size=5,
                                     a_units='m', b_units='N', c_units='N*m')

                self.connect('x', 'cpc.x')
                self.connect('y', 'cpc.y')

        p = self.p = om.Problem(MyModel())
        p.setup()

        p['a'] = np.random.rand(5, 3)
        p['b'] = np.random.rand(5, 3)

        p.run_model()

    def test_results(self):

        for i in range(5):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p.get_val('cpc.c', units='ft*lbf')[i, :]
            expected_i = np.cross(a_i, b_i)

            assert_near_equal(c_i, expected_i, tolerance=1.0E-12)

            x_i = self.p['x'][i, :]
            y_i = self.p['y'][i, :]
            z_i = self.p.get_val('cpc.z', units='ft*lbf')[i, :]
            expected_i = np.cross(x_i, y_i)

            assert_near_equal(z_i, expected_i, tolerance=1.0E-12)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleCommonA(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='lbf')
        ivc.add_output(name='b', shape=(self.nn, 3), units='ft/s')
        ivc.add_output(name='y', shape=(self.nn, 3), units='ft/s')

        cpc = om.CrossProductComp(vec_size=self.nn,
                                a_units='N', b_units='m/s', c_units='W')

        cpc.add_product('z', b_name='y', vec_size=self.nn,
                        a_units='N', b_units='m/s', c_units='W')

        model = om.Group()
        model.add_subsystem(name='ivc', subsys=ivc,
                            promotes_outputs=['a', 'b', 'y'])

        model.add_subsystem(name='cross_prod_comp', subsys=cpc)

        model.connect('a', 'cross_prod_comp.a')
        model.connect('b', 'cross_prod_comp.b')
        model.connect('y', 'cross_prod_comp.y')

        p = self.p = om.Problem(model)
        p.setup()

        p['a'] = np.random.rand(self.nn, 3)
        p['b'] = np.random.rand(self.nn, 3)
        p['y'] = np.random.rand(self.nn, 3)

        p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p.get_val('cross_prod_comp.c', units='hp')[i]
            expected_i = np.cross(a_i, b_i) / 550.

            np.testing.assert_almost_equal(c_i, expected_i)

            y_i = self.p['y'][i, :]
            z_i = self.p.get_val('cross_prod_comp.z', units='hp')[i]
            expected_i = np.cross(a_i, y_i) / 550.
            np.testing.assert_almost_equal(z_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleCommonB(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='lbf')
        ivc.add_output(name='b', shape=(self.nn, 3), units='ft/s')
        ivc.add_output(name='x', shape=(self.nn, 3), units='lbf')

        cpc = om.CrossProductComp(vec_size=self.nn,
                                a_units='N', b_units='m/s', c_units='W')

        cpc.add_product('z', a_name='x', vec_size=self.nn,
                        a_units='N', b_units='m/s', c_units='W')

        model = om.Group()
        model.add_subsystem(name='ivc', subsys=ivc,
                            promotes_outputs=['a', 'b', 'x'])

        model.add_subsystem(name='cross_prod_comp', subsys=cpc)

        model.connect('a', 'cross_prod_comp.a')
        model.connect('b', 'cross_prod_comp.b')
        model.connect('x', 'cross_prod_comp.x')

        p = self.p = om.Problem(model)
        p.setup()

        p['a'] = np.random.rand(self.nn, 3)
        p['b'] = np.random.rand(self.nn, 3)
        p['x'] = np.random.rand(self.nn, 3)

        p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p.get_val('cross_prod_comp.c', units='hp')[i]
            expected_i = np.cross(a_i, b_i) / 550.

            np.testing.assert_almost_equal(c_i, expected_i)

            x_i = self.p['x'][i, :]
            z_i = self.p.get_val('cross_prod_comp.z', units='hp')[i]
            expected_i = np.cross(x_i, b_i) / 550.
            np.testing.assert_almost_equal(z_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleErrors(unittest.TestCase):

    def test_duplicate_outputs(self):
        cpc = om.CrossProductComp()

        with self.assertRaises(NameError) as ctx:
            cpc.add_product('c')

        self.assertEqual(str(ctx.exception), "<class CrossProductComp>: "
                         "Multiple definition of output 'c'.")

    def test_input_as_output(self):
        cpc = om.CrossProductComp()

        with self.assertRaises(NameError) as ctx:
            cpc.add_product('a', 'b', 'c')

        self.assertEqual(str(ctx.exception), "<class CrossProductComp>: 'a' specified as "
                         "an output, but it has already been defined as an input.")

    def test_output_as_input_a(self):
        cpc = om.CrossProductComp()

        with self.assertRaises(NameError) as ctx:
            cpc.add_product('z', 'c', 'b')

        self.assertEqual(str(ctx.exception), "<class CrossProductComp>: 'c' specified as "
                         "an input, but it has already been defined as an output.")

    def test_output_as_input_b(self):
        cpc = om.CrossProductComp()

        with self.assertRaises(NameError) as ctx:
            cpc.add_product('z', 'a', 'c')

        self.assertEqual(str(ctx.exception), "<class CrossProductComp>: 'c' specified as "
                         "an input, but it has already been defined as an output.")

    def test_a_vec_size_mismatch(self):
        cpc = om.CrossProductComp(vec_size=7)

        with self.assertRaises(ValueError) as ctx:
            cpc.add_product('z', 'a', 'y', vec_size=42)

        self.assertEqual(str(ctx.exception), "<class CrossProductComp>: "
                         "Conflicting vec_size=42 specified for input 'a', "
                         "which has already been defined with vec_size=7.")

    def test_a_units_mismatch(self):
        cpc = om.CrossProductComp()

        with self.assertRaises(ValueError) as ctx:
            cpc.add_product('z', 'a', 'b',a_units='ft')

        self.assertEqual(str(ctx.exception), "<class CrossProductComp>: "
                         "Conflicting units 'ft' specified for input 'a', "
                         "which has already been defined with units 'None'.")

    def test_b_vec_size_mismatch(self):
        cpc = om.CrossProductComp()

        with self.assertRaises(ValueError) as ctx:
            cpc.add_product('z', 'x', 'b', vec_size=10)

        self.assertEqual(str(ctx.exception), "<class CrossProductComp>: "
                         "Conflicting vec_size=10 specified for input 'b', "
                         "which has already been defined with vec_size=1.")

    def test_b_units_mismatch(self):
        cpc = om.CrossProductComp()

        with self.assertRaises(ValueError) as ctx:
            cpc.add_product('z', 'a', 'b', b_units='ft')

        self.assertEqual(str(ctx.exception), "<class CrossProductComp>: "
                         "Conflicting units 'ft' specified for input 'b', "
                         "which has already been defined with units 'None'.")


class TestFeature(unittest.TestCase):

    def test(self):
        import numpy as np

        import openmdao.api as om

        n = 24

        p = om.Problem()

        p.model.add_subsystem(name='cross_prod_comp',
                              subsys=om.CrossProductComp(vec_size=n,
                                                         a_name='r', b_name='F', c_name='torque',
                                                         a_units='m', b_units='N', c_units='N*m'),
                              promotes_inputs=['r', 'F'])

        p.setup()

        p.set_val('r', np.random.rand(n, 3))
        p.set_val('F', np.random.rand(n, 3))

        p.run_model()

        # Check the output in units of ft*lbf to ensure that our units work as expected.
        expected = []
        for i in range(n):
            a_i = p.get_val('r')[i, :]
            b_i = p.get_val('F')[i, :]
            expected.append(np.cross(a_i, b_i) * 0.73756215)

            actual_i = p.get_val('cross_prod_comp.torque', units='ft*lbf')[i]
            rel_error = np.abs(expected[i] - actual_i)/actual_i
            assert np.all(rel_error < 1e-8), f"Relative error: {rel_error}"

        assert_near_equal(p.get_val('cross_prod_comp.torque', units='ft*lbf'), np.array(expected), tolerance=1e-8)

    def test_multiple(self):
        import numpy as np

        import openmdao.api as om

        n = 24

        p = om.Problem()

        cpc = om.CrossProductComp(vec_size=n,
                                  a_name='r', b_name='F', c_name='torque',
                                  a_units='m', b_units='N', c_units='N*m')

        cpc.add_product(vec_size=n,
                        a_name='r', b_name='p', c_name='L',
                        a_units='m', b_units='kg*m/s', c_units='kg*m**2/s')

        p.model.add_subsystem(name='cross_prod_comp', subsys=cpc,
                              promotes_inputs=['r', 'F', 'p'])

        p.setup()

        p.set_val('r', np.random.rand(n, 3))
        p.set_val('F', np.random.rand(n, 3))
        p.set_val('p', np.random.rand(n, 3))

        p.run_model()

        # Check the output.
        expected_T = []
        expected_L = []
        for i in range(n):
            a_i = p.get_val('r')[i, :]
            b_i = p.get_val('F')[i, :]
            expected_T.append(np.cross(a_i, b_i))

            actual_i = p.get_val('cross_prod_comp.torque')[i]
            rel_error = np.abs(expected_T[i] - actual_i)/actual_i
            assert np.all(rel_error < 1e-8), f"Relative error: {rel_error}"

            b_i = p.get_val('p')[i, :]
            expected_L.append(np.cross(a_i, b_i))

            actual_i = p.get_val('cross_prod_comp.L')[i]
            rel_error = np.abs(expected_L[i] - actual_i)/actual_i
            assert np.all(rel_error < 1e-8), f"Relative error: {rel_error}"

        assert_near_equal(p.get_val('cross_prod_comp.torque'), np.array(expected_T), tolerance=1e-8)
        assert_near_equal(p.get_val('cross_prod_comp.L'), np.array(expected_L), tolerance=1e-8)


if __name__ == "__main__":
    unittest.main()

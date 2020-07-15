import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.units import convert_units
from openmdao.utils.assert_utils import assert_near_equal


class TestMatrixVectorProductComp3x3(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='A', shape=(self.nn, 3, 3))
        ivc.add_output(name='x', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['A', 'x'])

        self.p.model.add_subsystem(name='mat_vec_product_comp',
                                   subsys=om.MatrixVectorProductComp(vec_size=self.nn))

        self.p.model.connect('A', 'mat_vec_product_comp.A')
        self.p.model.connect('x', 'mat_vec_product_comp.x')

        self.p.setup(force_alloc_complex=True)

        self.p['A'] = np.random.rand(self.nn, 3, 3)
        self.p['x'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            A_i = self.p['A'][i, :, :]
            x_i = self.p['x'][i, :]
            b_i = self.p['mat_vec_product_comp.b'][i, :]

            expected_i = np.dot(A_i, x_i)
            np.testing.assert_almost_equal(b_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMatrixVectorProductComp6x4(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='A', shape=(self.nn, 6, 4))
        ivc.add_output(name='x', shape=(self.nn, 4))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['A', 'x'])

        self.p.model.add_subsystem(name='mat_vec_product_comp',
                                   subsys=om.MatrixVectorProductComp(vec_size=self.nn,
                                                                     A_shape=(6, 4)))

        self.p.model.connect('A', 'mat_vec_product_comp.A')
        self.p.model.connect('x', 'mat_vec_product_comp.x')

        self.p.setup(force_alloc_complex=True)

        self.p['A'] = np.random.rand(self.nn, 6, 4)
        self.p['x'] = np.random.rand(self.nn, 4)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            A_i = self.p['A'][i, :, :]
            x_i = self.p['x'][i, :]
            b_i = self.p['mat_vec_product_comp.b'][i, :]

            expected_i = np.dot(A_i, x_i)
            np.testing.assert_almost_equal(b_i, expected_i)

    def test_partials(self):
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMatrixVectorProductCompNonVectorized(unittest.TestCase):

    def setUp(self):
        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='A', shape=(3, 3))
        ivc.add_output(name='x', shape=(3, 1))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['A', 'x'])

        self.p.model.add_subsystem(name='mat_vec_product_comp',
                                   subsys=om.MatrixVectorProductComp())

        self.p.model.connect('A', 'mat_vec_product_comp.A')
        self.p.model.connect('x', 'mat_vec_product_comp.x')

        self.p.setup(force_alloc_complex=True)

        self.p['A'] = np.random.rand(3, 3)
        self.p['x'] = np.random.rand(3, 1)

        self.p.run_model()

    def test_results(self):

        A_i = self.p['A']
        x_i = self.p['x']
        b_i = self.p['mat_vec_product_comp.b']

        expected = np.dot(np.reshape(A_i, (3, 3)), np.reshape(x_i, (3,)))
        assert_near_equal(b_i, expected)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                assert_near_equal(
                                 actual=cpd[comp][var, wrt]['J_fwd'],
                                 desired=cpd[comp][var, wrt]['J_fd'])


class TestUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='A', shape=(self.nn, 3, 3), units='ft')
        ivc.add_output(name='x', shape=(self.nn, 3), units='lbf')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['A', 'x'])

        self.p.model.add_subsystem(name='mat_vec_product_comp',
                                   subsys=om.MatrixVectorProductComp(vec_size=self.nn,
                                                                     A_units='m', x_units='N',
                                                                     b_units='N*m'))

        self.p.model.connect('A', 'mat_vec_product_comp.A')
        self.p.model.connect('x', 'mat_vec_product_comp.x')

        self.p.setup(force_alloc_complex=True)

        self.p['A'] = np.random.rand(self.nn, 3, 3)
        self.p['x'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            A_i = self.p['A'][i, :, :]
            x_i = self.p['x'][i, :]
            b_i = self.p.get_val('mat_vec_product_comp.b', units='ft*lbf')[i, :]

            expected_i = np.dot(A_i, x_i)
            np.testing.assert_almost_equal(b_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 2

        ivc = om.IndepVarComp()
        ivc.add_output(name='A', shape=(self.nn, 5, 3), units='ft')
        ivc.add_output(name='x', shape=(self.nn, 3), units='lbf')
        ivc.add_output(name='B', shape=(self.nn, 5, 3), units='m')
        ivc.add_output(name='y', shape=(self.nn, 3), units='N')

        mvp = om.MatrixVectorProductComp(vec_size=self.nn, A_shape=(5, 3),
                                         A_units='m', x_units='N', b_units='N*m')

        mvp.add_product('c', A_name='B', x_name='y', A_shape=(5, 3), vec_size=self.nn,
                        A_units='m', x_units='N', b_units='N*m')

        model = om.Group()
        model.add_subsystem(name='ivc',
                            subsys=ivc,
                            promotes_outputs=['*'])

        model.add_subsystem(name='mat_vec_product_comp',
                            subsys=mvp,
                            promotes=['*'])

        self.p = om.Problem(model)
        self.p.setup(force_alloc_complex=True)

        A = np.random.rand(self.nn, 5, 3)
        x = np.random.rand(self.nn, 3)

        self.p['A'] = A
        self.p['x'] = x

        self.p['B'] = convert_units(A, 'ft', 'm')
        self.p['y'] = convert_units(x, 'lbf', 'N')

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            # b = Ax
            A_i = self.p['A'][i, :, :]
            x_i = self.p['x'][i, :]
            b_i = self.p.get_val('mat_vec_product_comp.b', units='ft*lbf')[i, :]

            expected_i = np.dot(A_i, x_i)
            np.testing.assert_almost_equal(b_i, expected_i)

            # c = By
            B_i = self.p['B'][i, :, :]
            y_i = self.p['y'][i, :]
            c_i = self.p.get_val('mat_vec_product_comp.c', units='N*m')[i, :]

            expected_i = np.dot(B_i, y_i)
            np.testing.assert_almost_equal(c_i, expected_i)

            # b & c should match after unit conversion
            np.testing.assert_almost_equal(convert_units(b_i, 'ft*lbf', 'N*m'), c_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(out_stream=None, method='cs')

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
                ivc.add_output(name='A', shape=(2, 5, 3), units='ft')
                ivc.add_output(name='x', shape=(2, 3), units='lbf')

                mvp = om.MatrixVectorProductComp(vec_size=2, A_shape=(5, 3),
                                                 A_units='m', x_units='N', b_units='N*m')

                self.add_subsystem('ivc', ivc, promotes_outputs=['*'])
                self.add_subsystem('mvp', mvp, promotes=['*'])

            def configure(self):
                self.ivc.add_output(name='B', shape=(2, 5, 3), units='m')
                self.ivc.add_output(name='y', shape=(2, 3), units='N')

                self.mvp.add_product('c', A_name='B', x_name='y',
                                     A_shape=(5, 3), vec_size=2,
                                     A_units='m', x_units='N', b_units='N*m')

        self.p = om.Problem(MyModel())
        self.p.setup(force_alloc_complex=True)

        A = np.random.rand(2, 5, 3)
        x = np.random.rand(2, 3)

        self.p['A'] = A
        self.p['x'] = x

        self.p['B'] = convert_units(A, 'ft', 'm')
        self.p['y'] = convert_units(x, 'lbf', 'N')

        self.p.run_model()

    def test_results(self):

        for i in range(2):
            # b = Ax
            A_i = self.p['A'][i, :, :]
            x_i = self.p['x'][i, :]
            b_i = self.p.get_val('mvp.b', units='ft*lbf')[i, :]

            expected_i = np.dot(A_i, x_i)
            np.testing.assert_almost_equal(b_i, expected_i)

            # c = By
            B_i = self.p['B'][i, :, :]
            y_i = self.p['y'][i, :]
            c_i = self.p.get_val('mvp.c', units='N*m')[i, :]

            expected_i = np.dot(B_i, y_i)
            np.testing.assert_almost_equal(c_i, expected_i)

            # b & c should match after unit conversion
            np.testing.assert_almost_equal(convert_units(b_i, 'ft*lbf', 'N*m'), c_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleCommonMatrix(unittest.TestCase):

    def setUp(self):
        self.nn = 2

        ivc = om.IndepVarComp()
        ivc.add_output(name='A', shape=(self.nn, 5, 3), units='ft')
        ivc.add_output(name='x', shape=(self.nn, 3), units='lbf')
        ivc.add_output(name='y', shape=(self.nn, 3), units='N')

        mvp = om.MatrixVectorProductComp(vec_size=self.nn, A_shape=(5, 3),
                                         A_units='m', x_units='N', b_units='N*m')

        mvp.add_product(A_name='A', x_name='y', b_name='c', vec_size=self.nn, A_shape=(5, 3),
                        b_units='N*m', A_units='m', x_units='N')

        model = om.Group()
        model.add_subsystem(name='ivc',
                            subsys=ivc,
                            promotes_outputs=['*'])

        model.add_subsystem(name='mat_vec_product_comp',
                            subsys=mvp,
                            promotes=['*'])

        self.p = om.Problem(model)
        self.p.setup(force_alloc_complex=True)

        A = np.random.rand(self.nn, 5, 3)
        x = np.random.rand(self.nn, 3)

        self.p['A'] = A
        self.p['x'] = x
        self.p['y'] = convert_units(x, 'lbf', 'N')

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            # b = Ax
            A_i = self.p['A'][i, :, :]
            x_i = self.p['x'][i, :]
            b_i = self.p.get_val('mat_vec_product_comp.b', units='ft*lbf')[i, :]

            expected_i = np.dot(A_i, x_i)
            np.testing.assert_almost_equal(b_i, expected_i)

            # c = Ay
            y_i = self.p['y'][i, :]
            c_i = self.p.get_val('mat_vec_product_comp.c', units='N*m')[i, :]

            expected_i = np.dot(convert_units(A_i, 'ft', 'm'), y_i)
            np.testing.assert_almost_equal(c_i, expected_i)

            # b & c should match after unit conversion
            np.testing.assert_almost_equal(convert_units(b_i, 'ft*lbf', 'N*m'), c_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleCommonVector(unittest.TestCase):

    def setUp(self):
        self.nn = 2

        ivc = om.IndepVarComp()
        ivc.add_output(name='A', shape=(self.nn, 5, 3), units='ft')
        ivc.add_output(name='B', shape=(self.nn, 7, 3), units='m')
        ivc.add_output(name='x', shape=(self.nn, 3), units='lbf')

        mvp = om.MatrixVectorProductComp(vec_size=self.nn, A_shape=(5, 3),
                                         A_units='m', x_units='N', b_units='N*m')

        mvp.add_product(A_name='B', x_name='x', b_name='c', vec_size=self.nn, A_shape=(7, 3),
                        A_units='m', x_units='N', b_units='N*m')

        model = om.Group()
        model.add_subsystem(name='ivc',
                            subsys=ivc,
                            promotes_outputs=['*'])

        model.add_subsystem(name='mat_vec_product_comp',
                            subsys=mvp,
                            promotes=['*'])

        self.p = om.Problem(model)
        self.p.setup(force_alloc_complex=True)

        A = np.random.rand(self.nn, 5, 3)
        B = np.random.rand(self.nn, 7, 3)
        x = np.random.rand(self.nn, 3)

        self.p['A'] = A
        self.p['B'] = B
        self.p['x'] = x

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            # b = Ax
            A_i = self.p['A'][i, :, :]
            x_i = self.p['x'][i, :]
            b_i = self.p.get_val('mat_vec_product_comp.b', units='ft*lbf')[i, :]

            expected_i = np.dot(A_i, x_i)
            np.testing.assert_almost_equal(b_i, expected_i)

            # c = Bx
            B_i = self.p['B'][i, :, :]
            c_i = self.p.get_val('mat_vec_product_comp.c', units='N*m')[i, :]

            expected_i = np.dot(B_i, convert_units(x_i, 'lbf', 'N'))
            np.testing.assert_almost_equal(c_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleErrors(unittest.TestCase):

    def test_duplicate_outputs(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(NameError) as ctx:
            mvp.add_product('b', 'B', 'y')

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "Multiple definition of output 'b'.")

    def test_input_as_output(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(NameError) as ctx:
            mvp.add_product('x', 'A', 'b')

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "'x' specified as an output, but it has already been "
                         "defined as an input.")

    def test_output_as_input_A(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(NameError) as ctx:
            mvp.add_product('c', 'b', 'A')

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "'b' specified as an input, but it has already been "
                         "defined as an output.")

    def test_output_as_input_x(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(NameError) as ctx:
            mvp.add_product('c', 'A', 'b')

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "'b' specified as an input, but it has already been "
                         "defined as an output.")

    def test_A_vec_size_mismatch(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(ValueError) as ctx:
            mvp.add_product('c', 'A', 'y', vec_size=10)

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "Conflicting vec_size=10 specified for matrix 'A', "
                         "which has already been defined with vec_size=1.")

    def test_A_shape_mismatch(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(ValueError) as ctx:
            mvp.add_product('c', 'A', 'y', A_shape=(5, 5))

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "Conflicting shape (5, 5) specified for matrix 'A', "
                         "which has already been defined with shape (3, 3).")

    def test_A_units_mismatch(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(ValueError) as ctx:
            mvp.add_product('c', 'A', 'y', A_units='ft')

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "Conflicting units 'ft' specified for matrix 'A', "
                         "which has already been defined with units 'None'.")

    def test_x_vec_size_mismatch(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(ValueError) as ctx:
            mvp.add_product('c', 'B', 'x', vec_size=10)

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "Conflicting vec_size=10 specified for vector 'x', "
                         "which has already been defined with vec_size=1.")

    def test_x_vec_size_mismatch(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(ValueError) as ctx:
            mvp.add_product('c', 'B', 'x', A_shape=(5, 5))

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "Matrix shape (5, 5) is incompatible with vector 'x', "
                         "which has already been defined with 3 column(s).")

    def test_x_units_mismatch(self):
        mvp = om.MatrixVectorProductComp()

        with self.assertRaises(ValueError) as ctx:
            mvp.add_product('c', 'A', 'x', x_units='ft')

        self.assertEqual(str(ctx.exception), "MatrixVectorProductComp: "
                         "Conflicting units 'ft' specified for vector 'x', "
                         "which has already been defined with units 'None'.")


class TestFeature(unittest.TestCase):

    def test(self):
        import numpy as np
        import openmdao.api as om

        nn = 2

        p = om.Problem()

        p.model.add_subsystem(name='mat_vec_product_comp',
                              subsys=om.MatrixVectorProductComp(A_name='Mat', vec_size=nn,
                                                                b_name='y', b_units='m',
                                                                x_units='m'),
                              promotes_inputs=['Mat', 'x'])

        p.setup()

        p.set_val('Mat', np.random.rand(nn, 3, 3))
        p.set_val('x', np.random.rand(nn, 3))

        p.run_model()

        assert_near_equal(p.get_val('mat_vec_product_comp.y', units='ft')[0, :],
                          np.dot(p['Mat'][0, :, :], p['x'][0, :]) * 3.2808399,
                          tolerance=1.0E-8)

        assert_near_equal(p.get_val('mat_vec_product_comp.y', units='ft')[1, :],
                          np.dot(p['Mat'][1, :, :], p['x'][1, :]) * 3.2808399,
                          tolerance=1.0E-8)

    def test_multiple(self):
        import numpy as np
        import openmdao.api as om

        nn = 2

        p = om.Problem()

        mvp = om.MatrixVectorProductComp(A_name='Mat', vec_size=nn,
                                         b_name='y', b_units='m',
                                         x_units='m')

        mvp.add_product(A_name='Mat', vec_size=nn,
                        b_name='z', b_units='m',
                        x_name='w', x_units='m')

        p.model.add_subsystem(name='mat_vec_product_comp',
                              subsys=mvp,
                              promotes_inputs=['Mat', 'x', 'w'])

        p.setup()

        p.set_val('Mat', np.random.rand(nn, 3, 3))
        p.set_val('x', np.random.rand(nn, 3))
        p.set_val('w', np.random.rand(nn, 3))

        p.run_model()

        assert_near_equal(p.get_val('mat_vec_product_comp.y', units='ft')[0, :],
                          np.dot(p['Mat'][0, :, :], p['x'][0, :]) * 3.2808399,
                          tolerance=1.0E-8)

        assert_near_equal(p.get_val('mat_vec_product_comp.y', units='ft')[1, :],
                          np.dot(p['Mat'][1, :, :], p['x'][1, :]) * 3.2808399,
                          tolerance=1.0E-8)

        assert_near_equal(p.get_val('mat_vec_product_comp.z', units='ft')[0, :],
                          np.dot(p['Mat'][0, :, :], p['w'][0, :]) * 3.2808399,
                          tolerance=1.0E-8)

        assert_near_equal(p.get_val('mat_vec_product_comp.z', units='ft')[1, :],
                          np.dot(p['Mat'][1, :, :], p['w'][1, :]) * 3.2808399,
                          tolerance=1.0E-8)


if __name__ == "__main__":
    unittest.main()

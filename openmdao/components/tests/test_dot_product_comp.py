"""
Unit test for DotProductComp.
"""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class TestDotProductCompNx3(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='dot_prod_comp',
                                   subsys=om.DotProductComp(vec_size=self.nn))

        self.p.model.connect('a', 'dot_prod_comp.a')
        self.p.model.connect('b', 'dot_prod_comp.b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn, 3)
        self.p['b'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p['dot_prod_comp.c'][i]
            expected_i = np.dot(a_i, b_i)

            np.testing.assert_almost_equal(c_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestDotProductCompNx4(unittest.TestCase):
    def setUp(self):
        self.nn = 100

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 4))
        ivc.add_output(name='b', shape=(self.nn, 4))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='dot_prod_comp',
                                   subsys=om.DotProductComp(vec_size=self.nn, length=4))

        self.p.model.connect('a', 'dot_prod_comp.a')
        self.p.model.connect('b', 'dot_prod_comp.b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn, 4)
        self.p['b'] = np.random.rand(self.nn, 4)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p['dot_prod_comp.c'][i]
            expected_i = np.dot(a_i, b_i)
            np.testing.assert_almost_equal(c_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None)

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
        ivc.add_output(name='a', shape=(self.nn, 3), units='lbf')
        ivc.add_output(name='b', shape=(self.nn, 3), units='ft/s')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='dot_prod_comp',
                                   subsys=om.DotProductComp(vec_size=self.nn,
                                                            a_units='N', b_units='m/s',
                                                            c_units='W'))

        self.p.model.connect('a', 'dot_prod_comp.a')
        self.p.model.connect('b', 'dot_prod_comp.b')

        self.p.setup()

        self.p['a'] = np.random.rand(self.nn, 3)
        self.p['b'] = np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p.get_val('dot_prod_comp.c', units='hp')[i]
            expected_i = np.dot(a_i, b_i) / 550.0

            np.testing.assert_almost_equal(c_i, expected_i)

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
        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(5, 3), units='lbf')
        ivc.add_output(name='b', shape=(5, 3), units='ft/s')
        ivc.add_output(name='x', shape=(3, 7), units='N')
        ivc.add_output(name='y', shape=(3, 7), units='m/s')

        dpc = om.DotProductComp(vec_size=5,  # default length=3
                                a_units='N', b_units='m/s', c_units='W')

        dpc.add_product(a_name='x', b_name='y', c_name='z', vec_size=3, length=7,
                        a_units='N', b_units='m/s', c_units='hp')

        model = om.Group()
        model.add_subsystem(name='ivc', subsys=ivc,
                            promotes_outputs=['a', 'b', 'x', 'y'])

        model.add_subsystem(name='dot_prod_comp', subsys=dpc)

        model.connect('a', 'dot_prod_comp.a')
        model.connect('b', 'dot_prod_comp.b')
        model.connect('x', 'dot_prod_comp.x')
        model.connect('y', 'dot_prod_comp.y')

        p = self.p = om.Problem(model)
        p.setup()

        p['a'] = np.random.rand(5, 3)
        p['b'] = np.random.rand(5, 3)
        p['x'] = np.random.rand(3, 7)
        p['y'] = np.random.rand(3, 7)

        p.run_model()

    def test_results(self):

        for i in range(5):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p.get_val('dot_prod_comp.c', units='hp')[i]
            expected_i = np.dot(a_i, b_i) / 550.0

            np.testing.assert_almost_equal(c_i, expected_i)

        for i in range(3):
            x_i = self.p['x'][i, :]
            y_i = self.p['y'][i, :]
            z_i = self.p.get_val('dot_prod_comp.z', units='hp')[i]
            expected_i = np.dot(x_i, y_i)
            np.testing.assert_almost_equal(z_i, expected_i)

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
                ivc.add_output(name='a', shape=(5, 3), units='lbf')
                ivc.add_output(name='b', shape=(5, 3), units='ft/s')

                dpc = om.DotProductComp(vec_size=5,  # default length=3
                                        a_units='N', b_units='m/s', c_units='W')

                self.add_subsystem('ivc', ivc, promotes_outputs=['*'])
                self.add_subsystem('dpc', dpc)

                self.connect('a', 'dpc.a')
                self.connect('b', 'dpc.b')

            def configure(self):
                self.ivc.add_output(name='x', shape=(3, 7), units='N')
                self.ivc.add_output(name='y', shape=(3, 7), units='m/s')

                self.dpc.add_product(a_name='x', b_name='y', c_name='z', vec_size=3, length=7,
                                     a_units='N', b_units='m/s', c_units='hp')


                self.connect('x', 'dpc.x')
                self.connect('y', 'dpc.y')

        p = self.p = om.Problem(MyModel())
        p.setup()

        p['a'] = np.random.rand(5, 3)
        p['b'] = np.random.rand(5, 3)
        p['x'] = np.random.rand(3, 7)
        p['y'] = np.random.rand(3, 7)

        p.run_model()

    def test_results(self):

        for i in range(5):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p.get_val('dpc.c', units='hp')[i]
            expected_i = np.dot(a_i, b_i) / 550.0

            np.testing.assert_almost_equal(c_i, expected_i)

        for i in range(3):
            x_i = self.p['x'][i, :]
            y_i = self.p['y'][i, :]
            z_i = self.p.get_val('dpc.z', units='hp')[i]
            expected_i = np.dot(x_i, y_i)
            np.testing.assert_almost_equal(z_i, expected_i)

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

        dpc = om.DotProductComp(vec_size=self.nn,  # default length=3
                                a_units='N', b_units='m/s', c_units='W')

        dpc.add_product('z', b_name='y', vec_size=self.nn, length=3,
                        a_units='N', b_units='m/s', c_units='W')

        model = om.Group()
        model.add_subsystem(name='ivc', subsys=ivc,
                            promotes_outputs=['a', 'b', 'y'])

        model.add_subsystem(name='dot_prod_comp', subsys=dpc)

        model.connect('a', 'dot_prod_comp.a')
        model.connect('b', 'dot_prod_comp.b')
        model.connect('y', 'dot_prod_comp.y')

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
            c_i = self.p.get_val('dot_prod_comp.c', units='hp')[i]
            expected_i = np.dot(a_i, b_i) / 550.

            np.testing.assert_almost_equal(c_i, expected_i)

            y_i = self.p['y'][i, :]
            z_i = self.p.get_val('dot_prod_comp.z', units='hp')[i]
            expected_i = np.dot(a_i, y_i) / 550.
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

        dpc = om.DotProductComp(vec_size=self.nn,  # default length=3
                                a_units='N', b_units='m/s', c_units='W')

        dpc.add_product('z', a_name='x', vec_size=self.nn, length=3,
                        a_units='N', b_units='m/s', c_units='W')

        model = om.Group()
        model.add_subsystem(name='ivc', subsys=ivc,
                            promotes_outputs=['a', 'b', 'x'])

        model.add_subsystem(name='dot_prod_comp', subsys=dpc)

        model.connect('a', 'dot_prod_comp.a')
        model.connect('b', 'dot_prod_comp.b')
        model.connect('x', 'dot_prod_comp.x')

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
            c_i = self.p.get_val('dot_prod_comp.c', units='hp')[i]
            expected_i = np.dot(a_i, b_i) / 550.

            np.testing.assert_almost_equal(c_i, expected_i)

            x_i = self.p['x'][i, :]
            z_i = self.p.get_val('dot_prod_comp.z', units='hp')[i]
            expected_i = np.dot(x_i, b_i) / 550.
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
        dpc = om.DotProductComp()

        with self.assertRaises(NameError) as ctx:
            dpc.add_product('c')

        self.assertEqual(str(ctx.exception), "DotProductComp: "
                         "Multiple definition of output 'c'.")

    def test_input_as_output(self):
        dpc = om.DotProductComp()

        with self.assertRaises(NameError) as ctx:
            dpc.add_product('a', 'b', 'c')

        self.assertEqual(str(ctx.exception), "DotProductComp: 'a' specified as "
                         "an output, but it has already been defined as an input.")

    def test_output_as_input_a(self):
        dpc = om.DotProductComp()

        with self.assertRaises(NameError) as ctx:
            dpc.add_product('z', 'c', 'b')

        self.assertEqual(str(ctx.exception), "DotProductComp: 'c' specified as "
                         "an input, but it has already been defined as an output.")

    def test_output_as_input_b(self):
        dpc = om.DotProductComp()

        with self.assertRaises(NameError) as ctx:
            dpc.add_product('z', 'b', 'c')

        self.assertEqual(str(ctx.exception), "DotProductComp: 'c' specified as "
                         "an input, but it has already been defined as an output.")

    def test_a_vec_size_mismatch(self):
        dpc = om.DotProductComp()

        with self.assertRaises(ValueError) as ctx:
            dpc.add_product('z', 'a', 'y', vec_size=10)

        self.assertEqual(str(ctx.exception), "DotProductComp: "
                         "Conflicting vec_size=10 specified for input 'a', "
                         "which has already been defined with vec_size=1.")

    def test_a_length_mismatch(self):
        dpc = om.DotProductComp()

        with self.assertRaises(ValueError) as ctx:
            dpc.add_product('z', 'a', 'y', length=10)

        self.assertEqual(str(ctx.exception), "DotProductComp: "
                         "Conflicting length=10 specified for input 'a', "
                         "which has already been defined with length=3.")

    def test_a_units_mismatch(self):
        dpc = om.DotProductComp()

        with self.assertRaises(ValueError) as ctx:
            dpc.add_product('z', 'a', 'b',a_units='ft')

        self.assertEqual(str(ctx.exception), "DotProductComp: "
                         "Conflicting units 'ft' specified for input 'a', "
                         "which has already been defined with units 'None'.")

    def test_b_vec_size_mismatch(self):
        dpc = om.DotProductComp()

        with self.assertRaises(ValueError) as ctx:
            dpc.add_product('z', 'x', 'b', vec_size=10)

        self.assertEqual(str(ctx.exception), "DotProductComp: "
                         "Conflicting vec_size=10 specified for input 'b', "
                         "which has already been defined with vec_size=1.")

    def test_b_length_mismatch(self):
        dpc = om.DotProductComp()

        with self.assertRaises(ValueError) as ctx:
            dpc.add_product('z', 'x', 'b', length=10)

        self.assertEqual(str(ctx.exception), "DotProductComp: "
                         "Conflicting length=10 specified for input 'b', "
                         "which has already been defined with length=3.")

    def test_b_units_mismatch(self):
        dpc = om.DotProductComp()

        with self.assertRaises(ValueError) as ctx:
            dpc.add_product('z', 'a', 'b', b_units='ft')

        self.assertEqual(str(ctx.exception), "DotProductComp: "
                         "Conflicting units 'ft' specified for input 'b', "
                         "which has already been defined with units 'None'.")


class TestFeature(unittest.TestCase):

    def test(self):
        """
        A simple example to compute power as the dot product of force and velocity vectors
        at 24 points simultaneously.
        """
        import numpy as np

        import openmdao.api as om

        n = 24

        p = om.Problem()

        dp_comp = om.DotProductComp(vec_size=n, length=3, a_name='F', b_name='v', c_name='P',
                                    a_units='N', b_units='m/s', c_units='W')

        p.model.add_subsystem(name='dot_prod_comp', subsys=dp_comp,
                             promotes_inputs=[('F', 'force'), ('v', 'vel')])

        p.setup()

        p.set_val('force', np.random.rand(n, 3))
        p.set_val('vel', np.random.rand(n, 3))

        p.run_model()

        # Verify the results against numpy.dot in a for loop.
        expected = []
        for i in range(n):
            a_i = p.get_val('force')[i, :]
            b_i = p.get_val('vel')[i, :]
            expected.append(np.dot(a_i, b_i))

            actual_i = p.get_val('dot_prod_comp.P')[i]
            rel_error = np.abs(expected[i] - actual_i)/actual_i
            assert rel_error < 1e-9, f"Relative error: {rel_error}"

        assert_near_equal(p.get_val('dot_prod_comp.P', units='kW'), np.array(expected)/1000.)

    def test_multiple(self):
        """
        Simultaneously compute work as the dot product of force and displacement vectors
        and power as the dot product of force and velocity vectors at 24 points.
        """
        import numpy as np

        import openmdao.api as om

        n = 24

        p = om.Problem()

        dp_comp = om.DotProductComp(vec_size=n, length=3,
                                    a_name='F', b_name='d', c_name='W',
                                    a_units='N', b_units='m', c_units='J')

        dp_comp.add_product(vec_size=n, length=3,
                            a_name='F', b_name='v', c_name='P',
                            a_units='N', b_units='m/s', c_units='W')

        p.model.add_subsystem(name='dot_prod_comp', subsys=dp_comp,
                              promotes_inputs=[('F', 'force'), ('d', 'disp'), ('v', 'vel')])

        p.setup()

        p.set_val('force', np.random.rand(n, 3))
        p.set_val('disp', np.random.rand(n, 3))
        p.set_val('vel', np.random.rand(n, 3))

        p.run_model()

        # Verify the results against numpy.dot in a for loop.
        expected_P = []
        expected_W = []
        for i in range(n):
            a_i = p.get_val('force')[i, :]

            b_i = p.get_val('disp')[i, :]
            expected_W.append(np.dot(a_i, b_i))

            actual_i = p.get_val('dot_prod_comp.W')[i]
            rel_error = np.abs(actual_i - expected_W[i])/actual_i
            assert rel_error < 1e-9, f"Relative error: {rel_error}"

            b_i = p.get_val('vel')[i, :]
            expected_P.append(np.dot(a_i, b_i))

            actual_i = p.get_val('dot_prod_comp.P')[i]
            rel_error = np.abs(expected_P[i] - actual_i)/actual_i
            assert rel_error < 1e-9, f"Relative error: {rel_error}"

        assert_near_equal(p.get_val('dot_prod_comp.W', units='kJ'), np.array(expected_W)/1000.)
        assert_near_equal(p.get_val('dot_prod_comp.P', units='kW'), np.array(expected_P)/1000.)


if __name__ == '__main__':
    unittest.main()

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
        cpd = self.p.check_partials(compact_print=True)

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
        cpd = self.p.check_partials(compact_print=True)

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
        cpd = self.p.check_partials(compact_print=True)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='lbf')
        ivc.add_output(name='b', shape=(self.nn, 3), units='ft/s')
        ivc.add_output(name='x', shape=(self.nn, 7), units='N')
        ivc.add_output(name='y', shape=(self.nn, 7), units='m/s')

        dpc = om.DotProductComp(vec_size=self.nn,
                                a_units='N', b_units='m/s', c_units='W')

        dpc.add_product(a_name='x', b_name='y', c_name='z', vec_size=self.nn, length=7,
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

        p['a'] = np.random.rand(self.nn, 3)
        p['b'] = np.random.rand(self.nn, 3)
        p['x'] = np.random.rand(self.nn, 7)
        p['y'] = np.random.rand(self.nn, 7)

        p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            b_i = self.p['b'][i, :]
            c_i = self.p.get_val('dot_prod_comp.c', units='hp')[i]
            expected_i = np.dot(a_i, b_i) / 550.0

            np.testing.assert_almost_equal(c_i, expected_i)

            x_i = self.p['x'][i, :]
            y_i = self.p['y'][i, :]
            z_i = self.p.get_val('dot_prod_comp.z', units='hp')[i]
            expected_i = np.dot(x_i, y_i)
            np.testing.assert_almost_equal(z_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestMultipleErrors(unittest.TestCase):

    def test_duplicate_outputs(self):
        dpc = om.DotProductComp()
        dpc.add_product('a', 'b', 'c')

        model = om.Group()
        model.add_subsystem('dpc', dpc)

        p = om.Problem(model)

        with self.assertRaises(NameError) as ctx:
            p.setup()

        self.assertEqual(str(ctx.exception), "DotProductComp (dpc): "
                         "Multiple definition of output 'c'.")

    def test_b_vec_size_mismatch(self):
        dpc = om.DotProductComp()
        dpc.add_product('a', 'y', 'z', vec_size=10)

        model = om.Group()
        model.add_subsystem('dpc', dpc)

        p = om.Problem(model)

        with self.assertRaises(ValueError) as ctx:
            p.setup()

        self.assertEqual(str(ctx.exception), "DotProductComp (dpc): "
                         "Conflicting vec_size specified for input 'a', 1 versus 10.")

    def test_a_length_mismatch(self):
        dpc = om.DotProductComp()
        dpc.add_product('a', 'y', 'z', length=10)

        model = om.Group()
        model.add_subsystem('dpc', dpc)

        p = om.Problem(model)

        with self.assertRaises(ValueError) as ctx:
            p.setup()

        self.assertEqual(str(ctx.exception), "DotProductComp (dpc): "
                         "Conflicting length specified for input 'a', 3 versus 10.")

    def test_a_units_mismatch(self):
        dpc = om.DotProductComp()
        dpc.add_product('a', 'b', 'z', a_units='ft')

        model = om.Group()
        model.add_subsystem('dpc', dpc)

        p = om.Problem(model)

        with self.assertRaises(ValueError) as ctx:
            p.setup()

        self.assertEqual(str(ctx.exception), "DotProductComp (dpc): "
                         "Conflicting units specified for input 'a', 'None' and 'ft'.")

    def test_b_vec_size_mismatch(self):
        dpc = om.DotProductComp()
        dpc.add_product('x', 'b', 'z', vec_size=10)

        model = om.Group()
        model.add_subsystem('dpc', dpc)

        p = om.Problem(model)

        with self.assertRaises(ValueError) as ctx:
            p.setup()

        self.assertEqual(str(ctx.exception), "DotProductComp (dpc): "
                         "Conflicting vec_size specified for input 'b', 1 versus 10.")

    def test_b_length_mismatch(self):
        dpc = om.DotProductComp()
        dpc.add_product('x', 'b', 'z', length=10)

        model = om.Group()
        model.add_subsystem('dpc', dpc)

        p = om.Problem(model)

        with self.assertRaises(ValueError) as ctx:
            p.setup()

        self.assertEqual(str(ctx.exception), "DotProductComp (dpc): "
                         "Conflicting length specified for input 'b', 3 versus 10.")

    def test_b_units_mismatch(self):
        dpc = om.DotProductComp()
        dpc.add_product('a', 'b', 'z', b_units='ft')

        model = om.Group()
        model.add_subsystem('dpc', dpc)

        p = om.Problem(model)

        with self.assertRaises(ValueError) as ctx:
            p.setup()

        self.assertEqual(str(ctx.exception), "DotProductComp (dpc): "
                         "Conflicting units specified for input 'b', 'None' and 'ft'.")


class TestFeature(unittest.TestCase):

    def test(self):
        """
        A simple example to compute power as the dot product of force and velocity vectors
        at 100 points simultaneously.
        """
        import numpy as np

        import openmdao.api as om

        n = 100

        p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='force', shape=(n, 3), units='N')
        ivc.add_output(name='vel', shape=(n, 3), units='m/s')

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['force', 'vel'])

        dp_comp = om.DotProductComp(vec_size=n, length=3, a_name='F', b_name='v', c_name='P',
                                    a_units='N', b_units='m/s', c_units='W')

        p.model.add_subsystem(name='dot_prod_comp', subsys=dp_comp)

        p.model.connect('force', 'dot_prod_comp.F')
        p.model.connect('vel', 'dot_prod_comp.v')

        p.setup()

        p['force'] = np.random.rand(n, 3)
        p['vel'] = np.random.rand(n, 3)

        p.run_model()

        print(p.get_val('dot_prod_comp.P', units='W'))

        # Verify the results against numpy.dot in a for loop.
        for i in range(n):
            a_i = p['force'][i, :]
            b_i = p['vel'][i, :]
            expected_i = np.dot(a_i, b_i) / 1000.0
            assert_near_equal(p.get_val('dot_prod_comp.P', units='kW')[i], expected_i)

    def test_multiple(self):
        """
        A simple example to compute power as the dot product of force and velocity vectors
        at 100 points simultaneously.
        """
        import numpy as np

        import openmdao.api as om

        n = 100

        p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='force', shape=(n, 3), units='N')
        ivc.add_output(name='vel', shape=(n, 3), units='m/s')

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['force', 'vel'])

        dp_comp = om.DotProductComp(vec_size=n, length=3, a_name='F', b_name='v', c_name='P',
                                    a_units='N', b_units='m/s', c_units='W')

        p.model.add_subsystem(name='dot_prod_comp', subsys=dp_comp)

        p.model.connect('force', 'dot_prod_comp.F')
        p.model.connect('vel', 'dot_prod_comp.v')

        p.setup()

        p['force'] = np.random.rand(n, 3)
        p['vel'] = np.random.rand(n, 3)

        p.run_model()

        print(p.get_val('dot_prod_comp.P', units='W'))

        # Verify the results against numpy.dot in a for loop.
        for i in range(n):
            a_i = p['force'][i, :]
            b_i = p['vel'][i, :]
            expected_i = np.dot(a_i, b_i) / 1000.0
            assert_near_equal(p.get_val('dot_prod_comp.P', units='kW')[i], expected_i)


if __name__ == '__main__':
    unittest.main()

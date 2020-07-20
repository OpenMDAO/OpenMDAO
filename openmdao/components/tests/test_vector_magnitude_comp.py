import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.units import convert_units
from openmdao.utils.assert_utils import assert_near_equal


class TestVectorMagnitudeCompNx3(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        self.p.model.add_subsystem(name='vec_mag_comp',
                                   subsys=om.VectorMagnitudeComp(vec_size=self.nn))

        self.p.model.connect('a', 'vec_mag_comp.a')

        self.p.setup()

        self.p['a'] = 1.0 + np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            mag_i = self.p['vec_mag_comp.a_mag'][i]
            expected_i = np.sqrt(np.dot(a_i, a_i))

            np.testing.assert_almost_equal(mag_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='fd', step=1.0E-9, out_stream=None)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=5)


class TestVectorMagnitudeCompNx4(unittest.TestCase):
    def setUp(self):
        self.nn = 100

        self.p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 4))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        self.p.model.add_subsystem(name='vec_mag_comp',
                                   subsys=om.VectorMagnitudeComp(vec_size=self.nn, length=4))

        self.p.model.connect('a', 'vec_mag_comp.a')

        self.p.setup()

        self.p['a'] = 1.0 + np.random.rand(self.nn, 4)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            mag_i = self.p['vec_mag_comp.a_mag'][i]
            expected_i = np.sqrt(np.dot(a_i, a_i))

            np.testing.assert_almost_equal(mag_i, expected_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='fd', step=1.0E-9, out_stream=None)

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
        ivc.add_output(name='a', shape=(self.nn, 3), units='m')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        self.p.model.add_subsystem(name='vec_mag_comp',
                                   subsys=om.VectorMagnitudeComp(vec_size=self.nn, units='m'))

        self.p.model.connect('a', 'vec_mag_comp.a')

        self.p.setup()

        self.p['a'] = 1.0 + np.random.rand(self.nn, 3)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            c_i = self.p.get_val('vec_mag_comp.a_mag', units='ft')[i]
            expected_i = np.sqrt(np.dot(a_i, a_i)) / 0.3048

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
        self.nn = 5

        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='m')
        ivc.add_output(name='b', shape=(2*self.nn, 2), units='ft')

        vmc = om.VectorMagnitudeComp(vec_size=self.nn, units='m')
        vmc.add_magnitude('b_mag', 'b', vec_size=2*self.nn, length=2, units='ft')

        model = om.Group()

        model.add_subsystem('ivc', subsys=ivc, promotes_outputs=['a', 'b'])
        model.add_subsystem('vmc', subsys=vmc)

        model.connect('a', 'vmc.a')
        model.connect('b', 'vmc.b')

        p = self.p = om.Problem(model)
        p.setup()

        p['a'] = 1.0 + np.random.rand(self.nn, 3)
        p['b'] = 1.0 + np.random.rand(2*self.nn, 2)

        p.run_model()

    def test_results(self):

        for i in range(self.nn):
            a_i = self.p['a'][i, :]
            am_i = self.p.get_val('vmc.a_mag', units='ft')[i]
            expected_i = np.sqrt(np.dot(a_i, a_i)) / 0.3048

            np.testing.assert_almost_equal(am_i, expected_i)

            b_i = self.p['b'][i, :]
            bm_i = self.p.get_val('vmc.b_mag', units='m')[i]
            expected_i = np.sqrt(np.dot(b_i, b_i)) * 0.3048

            np.testing.assert_almost_equal(bm_i, expected_i)

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
                self.add_subsystem('ivc', om.IndepVarComp('a', shape=(5, 3), units='m'))
                self.add_subsystem('vmc', om.VectorMagnitudeComp(vec_size=5, units='m'))
                self.connect('ivc.a', 'vmc.a')

            def configure(self):
                self.ivc.add_output(name='b', shape=(10, 2), units='ft')
                self.vmc.add_magnitude('b_mag', 'b', vec_size=10, length=2, units='ft')
                self.connect('ivc.b', 'vmc.b')

        p = self.p = om.Problem(MyModel())
        p.setup()

        p.set_val('ivc.a', 1.0 + np.random.rand(5, 3))
        p.set_val('ivc.b', 1.0 + np.random.rand(10, 2))

        p.run_model()

    def test_results(self):

        for i in range(5):
            a_i = self.p.get_val('vmc.a')[i, :]
            am_i = self.p.get_val('vmc.a_mag', units='ft')[i]
            expected_i = np.sqrt(np.dot(a_i, a_i)) / 0.3048

            np.testing.assert_almost_equal(am_i, expected_i)

        for i in range(10):
            b_i = self.p.get_val('vmc.b')[i, :]
            bm_i = self.p.get_val('vmc.b_mag', units='m')[i]
            expected_i = np.sqrt(np.dot(b_i, b_i)) * 0.3048

            np.testing.assert_almost_equal(bm_i, expected_i)

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
        vmc = om.VectorMagnitudeComp()

        with self.assertRaises(NameError) as ctx:
            vmc.add_magnitude('a_mag', 'aa')

        self.assertEqual(str(ctx.exception), "VectorMagnitudeComp: "
                         "Multiple definition of output 'a_mag'.")

    def test_input_as_output(self):
        vmc = om.VectorMagnitudeComp()

        with self.assertRaises(NameError) as ctx:
            vmc.add_magnitude('a', 'aa')

        self.assertEqual(str(ctx.exception), "VectorMagnitudeComp: 'a' specified as"
                         " an output, but it has already been defined as an input.")

    def test_output_as_input(self):
        vmc = om.VectorMagnitudeComp()

        with self.assertRaises(NameError) as ctx:
            vmc.add_magnitude('aa', 'a_mag')

        self.assertEqual(str(ctx.exception), "VectorMagnitudeComp: 'a_mag' specified as"
                         " an input, but it has already been defined as an output.")

    def test_vec_size_mismatch(self):
        vmc = om.VectorMagnitudeComp()

        with self.assertRaises(ValueError) as ctx:
            vmc.add_magnitude('a_mag2', 'a', vec_size=10)

        self.assertEqual(str(ctx.exception), "VectorMagnitudeComp: "
                         "Conflicting vec_size=10 specified for input 'a', "
                         "which has already been defined with vec_size=1.")

    def test_length_mismatch(self):
        vmc = om.VectorMagnitudeComp()

        with self.assertRaises(ValueError) as ctx:
            vmc.add_magnitude('a_mag2', 'a', length=5)

        self.assertEqual(str(ctx.exception), "VectorMagnitudeComp: "
                         "Conflicting length=5 specified for input 'a', "
                         "which has already been defined with length=1.")

    def test_units_mismatch(self):
        vmc = om.VectorMagnitudeComp()

        with self.assertRaises(ValueError) as ctx:
            vmc.add_magnitude('a_mag2', 'a', units='ft')

        self.assertEqual(str(ctx.exception), "VectorMagnitudeComp: "
                         "Conflicting units 'ft' specified for input 'a', "
                         "which has already been defined with units 'None'.")


class TestFeature(unittest.TestCase):

    def test(self):
        """
        A simple example to compute the magnitude of 3-vectors at at 100 points simultaneously.
        """
        import numpy as np
        import openmdao.api as om

        n = 100

        p = om.Problem()

        comp = om.VectorMagnitudeComp(vec_size=n, length=3,
                                      in_name='r', mag_name='r_mag', units='km')

        p.model.add_subsystem(name='vec_mag_comp', subsys=comp,
                              promotes_inputs=[('r', 'pos')])

        p.setup()

        p.set_val('pos', 1.0 + np.random.rand(n, 3))

        p.run_model()

        # Verify the results against numpy.dot in a for loop.
        expected = []
        for i in range(n):
            a_i = p.get_val('pos')[i, :]
            expected.append(np.sqrt(np.dot(a_i, a_i)))

            actual_i = p.get_val('vec_mag_comp.r_mag')[i]
            rel_error = np.abs(expected[i] - actual_i)/actual_i
            assert rel_error < 1e-9, f"Relative error: {rel_error}"

        assert_near_equal(p.get_val('vec_mag_comp.r_mag'), np.array(expected))

    def test_multiple(self):
        """
        A simple example to compute the magnitude of 3-vectors at at 100 points simultaneously.
        """
        import numpy as np
        import openmdao.api as om

        n = 100

        p = om.Problem()

        comp = om.VectorMagnitudeComp(vec_size=n, length=3,
                                      in_name='r', mag_name='r_mag', units='km')

        comp.add_magnitude(vec_size=n, length=3,
                           in_name='b', mag_name='b_mag', units='ft')

        p.model.add_subsystem(name='vec_mag_comp', subsys=comp,
                              promotes_inputs=['r', 'b'])

        p.setup()

        p.set_val('r', 1.0 + np.random.rand(n, 3))
        p.set_val('b', 1.0 + np.random.rand(n, 3))

        p.run_model()

        # Verify the results against numpy.dot in a for loop.
        expected_r = []
        expected_b = []
        for i in range(n):
            a_i = p.get_val('r')[i, :]
            expected_r.append(np.sqrt(np.dot(a_i, a_i)))

            actual_i = p.get_val('vec_mag_comp.r_mag')[i]
            rel_error = np.abs(expected_r[i] - actual_i)/actual_i
            assert rel_error < 1e-9, f"Relative error: {rel_error}"

            b_i = p.get_val('b')[i, :]
            expected_b.append(np.sqrt(np.dot(b_i, b_i)))

            actual_i = p.get_val('vec_mag_comp.b_mag')[i]
            rel_error = np.abs(expected_b[i] - actual_i)/actual_i
            assert rel_error < 1e-9, f"Relative error: {rel_error}"

        assert_near_equal(p.get_val('vec_mag_comp.r_mag'), np.array(expected_r))
        assert_near_equal(p.get_val('vec_mag_comp.b_mag'), np.array(expected_b))


if __name__ == '__main__':
    unittest.main()
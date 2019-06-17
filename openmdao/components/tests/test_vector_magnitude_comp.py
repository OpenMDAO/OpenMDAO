from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, VectorMagnitudeComp


class TestVectorMagnitudeCompNx3(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem()

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        self.p.model.add_subsystem(name='vec_mag_comp',
                                   subsys=VectorMagnitudeComp(vec_size=self.nn))

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
        cpd = self.p.check_partials(compact_print=False, method='fd', step=1.0E-9)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=5)


class TestVectorMagnitudeCompNx4(unittest.TestCase):
    def setUp(self):
        self.nn = 100

        self.p = Problem()

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 4))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        self.p.model.add_subsystem(name='vec_mag_comp',
                                   subsys=VectorMagnitudeComp(vec_size=self.nn, length=4))

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
        cpd = self.p.check_partials(compact_print=False, method='fd', step=1.0E-9)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem()

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='m')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        self.p.model.add_subsystem(name='vec_mag_comp',
                                   subsys=VectorMagnitudeComp(vec_size=self.nn, units='m'))

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
        cpd = self.p.check_partials(compact_print=True)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=6)


class TestFeature(unittest.TestCase):

    def test(self):
        """
        A simple example to compute the magnitude of 3-vectors at at 100 points simultaneously.
        """
        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp, VectorMagnitudeComp
        from openmdao.utils.assert_utils import assert_rel_error

        n = 100

        p = Problem()

        ivc = IndepVarComp()
        ivc.add_output(name='pos', shape=(n, 3), units='m')

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['pos'])

        dp_comp = VectorMagnitudeComp(vec_size=n, length=3, in_name='r', mag_name='r_mag',
                                      units='km')

        p.model.add_subsystem(name='vec_mag_comp', subsys=dp_comp)

        p.model.connect('pos', 'vec_mag_comp.r')

        p.setup()

        p['pos'] = 1.0 + np.random.rand(n, 3)

        p.run_model()

        # Verify the results against numpy.dot in a for loop.
        for i in range(n):
            a_i = p['pos'][i, :]
            expected_i = np.sqrt(np.dot(a_i, a_i)) / 1000.0
            assert_rel_error(self, p.get_val('vec_mag_comp.r_mag')[i], expected_i)


if __name__ == '__main__':
    unittest.main()

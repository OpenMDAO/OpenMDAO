from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, DotProductComp


class TestDotProductCompNx3(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3))
        ivc.add_output(name='b', shape=(self.nn, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='dot_prod_comp',
                                   subsys=DotProductComp(vec_size=self.nn))

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

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 4))
        ivc.add_output(name='b', shape=(self.nn, 4))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='dot_prod_comp',
                                   subsys=DotProductComp(vec_size=self.nn, length=4))

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


class TestForDocs(unittest.TestCase):

    def test(self):
        """
        A simple example to compute power as the dot-product of force and velocity vectors
        at 100 points simultaneously.
        """
        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp, DotProductComp
        from openmdao.utils.assert_utils import assert_rel_error

        n = 100

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='force', shape=(n, 3))
        ivc.add_output(name='vel', shape=(n, 3))

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['force', 'vel'])

        dp_comp = DotProductComp(vec_size=n, length=3, a_name='F', b_name='v', c_name='P',
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
            assert_rel_error(self, p.get_val('dot_prod_comp.P', units='kW')[i], expected_i)


if __name__ == '__main__':
    unittest.main()

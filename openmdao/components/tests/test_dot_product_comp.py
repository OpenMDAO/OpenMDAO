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


class TestDotProductCompForDocs(unittest.TestCase):

    def test(self):
        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp
        from dymos.components import DotProductComp

        nn = 100

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(nn, 4))
        ivc.add_output(name='b', shape=(nn, 4))

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['a', 'b'])

        p.model.add_subsystem(name='dot_prod_comp',
                              subsys=DotProductComp(num_nodes=nn, vec_size=4))

        p.model.connect('a', 'dot_prod_comp.a')
        p.model.connect('b', 'dot_prod_comp.b')

        p.setup()

        p['a'] = np.random.rand(nn, 4)
        p['b'] = np.random.rand(nn, 4)

        p.run_model()

        expected = np.zeros((nn,))

        for i in range(nn):
            a_i = p['a'][i, :]
            b_i = p['b'][i, :]
            expected[i] = np.dot(a_i, b_i)

        self.assertTrue(np.all(np.abs(p['dot_prod_comp.c'] - expected) < 1.0E-12))


if __name__ == '__main__':
    unittest.main()

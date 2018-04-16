from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, CrossProductComp


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

        self.p.model.add_subsystem(name='cross_prod_comp',
                                   subsys=CrossProductComp(vec_size=self.nn))

        self.p.model.connect('a', 'cross_prod_comp.a')
        self.p.model.connect('b', 'cross_prod_comp.b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn, 3)
        self.p['b'] = np.random.rand(self.nn, 3)

        self.p['a'][:, 0] = 2.0
        self.p['a'][:, 1] = 3.0
        self.p['a'][:, 2] = 4.0

        self.p['b'][:, 0] = 5.0
        self.p['b'][:, 1] = 6.0
        self.p['b'][:, 2] = 7.0

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
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

class TestDotProductCompNx3x1(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3, 1))
        ivc.add_output(name='b', shape=(self.nn, 3, 1))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='cross_prod_comp',
                                   subsys=CrossProductComp(vec_size=self.nn))

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

class TestDotProductCompNonVectorized(unittest.TestCase):

    def setUp(self):
        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(3, 1))
        ivc.add_output(name='b', shape=(3, 1))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='cross_prod_comp',
                                   subsys=CrossProductComp())

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



class TestForDocs(unittest.TestCase):

    def test(self):
        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp, CrossProductComp

        nn = 100

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(nn, 3))
        ivc.add_output(name='b', shape=(nn, 3))

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['a', 'b'])

        p.model.add_subsystem(name='cross_prod_comp',
                              subsys=CrossProductComp(vec_size=nn))

        p.model.connect('a', 'cross_prod_comp.a')
        p.model.connect('b', 'cross_prod_comp.b')

        p.setup()

        p['a'] = np.random.rand(nn, 3)
        p['b'] = np.random.rand(nn, 3)

        p['a'][:, 0] = 2.0
        p['a'][:, 1] = 3.0
        p['a'][:, 2] = 4.0

        p['b'][:, 0] = 5.0
        p['b'][:, 1] = 6.0
        p['b'][:, 2] = 7.0

        p.run_model()

        expected = np.zeros_like(p['a'])

        for i in range(nn):
            a_i = p['a'][i, :]
            b_i = p['b'][i, :]
            # c_i = pp['cross_prod_comp.c'][i, :]
            expected[i, :] = np.cross(a_i, b_i)

        self.assertTrue(np.all(np.abs(p['cross_prod_comp.c'] - expected) < 1.0E-12))


if __name__ == "__main__":
    unittest.main()

from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, CrossProductComp
from openmdao.utils.assert_utils import assert_rel_error


class TestDotProductCompNx3(unittest.TestCase):

    def setUp(self):
        self.n = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.n, 3))
        ivc.add_output(name='b', shape=(self.n, 3))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='cross_prod_comp',
                                   subsys=CrossProductComp(vec_size=self.n))

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


class TestUnits(unittest.TestCase):

    def setUp(self):
        self.nn = 5

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 3), units='ft')
        ivc.add_output(name='b', shape=(self.nn, 3), units='lbf')

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        self.p.model.add_subsystem(name='cross_prod_comp',
                                   subsys=CrossProductComp(vec_size=self.nn,
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

            assert_rel_error(self, c_i, expected_i, tolerance=1.0E-12)

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
        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp, CrossProductComp
        from openmdao.utils.assert_utils import assert_rel_error

        n = 100

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='r', shape=(n, 3))
        ivc.add_output(name='F', shape=(n, 3))

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['r', 'F'])

        p.model.add_subsystem(name='cross_prod_comp',
                              subsys=CrossProductComp(vec_size=n,
                                                      a_name='r', b_name='F', c_name='torque',
                                                      a_units='m', b_units='N', c_units='N*m'))

        p.model.connect('r', 'cross_prod_comp.r')
        p.model.connect('F', 'cross_prod_comp.F')

        p.setup()

        p['r'] = np.random.rand(n, 3)
        p['F'] = np.random.rand(n, 3)

        p.run_model()

        # Check the output in units of ft*lbf to ensure that our units work as expected.
        for i in range(n):
            a_i = p['r'][i, :]
            b_i = p['F'][i, :]
            expected_i = np.cross(a_i, b_i) * 0.73756215
            assert_rel_error(self,
                             p.get_val('cross_prod_comp.torque', units='ft*lbf')[i, :],
                             expected_i, tolerance=1.0E-8)


if __name__ == "__main__":
    unittest.main()

from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from openmdao.api import MuxComp


class TestMuxCompOptions(unittest.TestCase):

    def test_invalid_axis_scalar(self):
        nn = 10

        p = Problem(model=Group())

        ivc = IndepVarComp()
        for i in range(nn):
            ivc.add_output(name='a_{0}'.format(i), val=1.0)

        p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['*'])

        demux_comp = p.model.add_subsystem(name='mux_comp',
                                                subsys=MuxComp(vec_size=nn))

        demux_comp.add_var('a', shape=(1,), axis=2)

        for i in range(nn):
            p.model.connect('a_{0}'.format(i), 'mux_comp.a_{0}'.format(i))

        with self.assertRaises(ValueError) as ctx:
            p.setup()
        self.assertEqual(str(ctx.exception),
                         'Cannot mux a 1D inputs for a along axis greater than 1 (2)')

    def test_invalid_axis_1D(self):
        nn = 10

        a_size = 7
        b_size = 3

        p = Problem(model=Group())

        ivc = IndepVarComp()
        for i in range(nn):
            ivc.add_output(name='a_{0}'.format(i), shape=(a_size,))
            ivc.add_output(name='b_{0}'.format(i), shape=(b_size,))

        p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['*'])

        demux_comp = p.model.add_subsystem(name='mux_comp',
                                                subsys=MuxComp(vec_size=nn))

        demux_comp.add_var('a', shape=(a_size,), axis=0)
        demux_comp.add_var('b', shape=(b_size,), axis=2)

        for i in range(nn):
            p.model.connect('a_{0}'.format(i), 'mux_comp.a_{0}'.format(i))
            p.model.connect('b_{0}'.format(i), 'mux_comp.b_{0}'.format(i))

        with self.assertRaises(ValueError) as ctx:
            p.setup()
        self.assertEqual(str(ctx.exception),
                         'Cannot mux a 1D inputs for b along axis greater than 1 (2)')


class TestMuxCompScalar(unittest.TestCase):

    def setUp(self):
        self.nn = 10

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        for i in range(self.nn):
            ivc.add_output(name='a_{0}'.format(i), val=1.0)
            ivc.add_output(name='b_{0}'.format(i), val=1.0)

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['*'])

        demux_comp = self.p.model.add_subsystem(name='mux_comp',
                                                subsys=MuxComp(vec_size=self.nn))

        demux_comp.add_var('a', shape=(1,), axis=0)
        demux_comp.add_var('b', shape=(1,), axis=1)

        for i in range(self.nn):
            self.p.model.connect('a_{0}'.format(i), 'mux_comp.a_{0}'.format(i))
            self.p.model.connect('b_{0}'.format(i), 'mux_comp.b_{0}'.format(i))

        self.p.setup(force_alloc_complex=True)

        for i in range(self.nn):
            self.p['a_{0}'.format(i)] = np.random.rand(1)
            self.p['b_{0}'.format(i)] = np.random.rand(1)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            out_i = self.p['mux_comp.a'][i]
            in_i = self.p['a_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

            out_i = self.p['mux_comp.b'][0, i]
            in_i = self.p['b_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='cs', out_stream=None)
        assert_check_partials(cpd, atol=1.0E-8, rtol=1.0E-8)


class TestDemuxComp1D(unittest.TestCase):

    def setUp(self):
        self.nn = 10

        a_size = 7
        b_size = 3

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        for i in range(self.nn):
            ivc.add_output(name='a_{0}'.format(i), shape=(a_size,))
            ivc.add_output(name='b_{0}'.format(i), shape=(b_size,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['*'])

        demux_comp = self.p.model.add_subsystem(name='mux_comp',
                                                subsys=MuxComp(vec_size=self.nn))

        demux_comp.add_var('a', shape=(a_size,), axis=0)
        demux_comp.add_var('b', shape=(b_size,), axis=1)

        for i in range(self.nn):
            self.p.model.connect('a_{0}'.format(i), 'mux_comp.a_{0}'.format(i))
            self.p.model.connect('b_{0}'.format(i), 'mux_comp.b_{0}'.format(i))

        self.p.setup(force_alloc_complex=True)

        for i in range(self.nn):
            self.p['a_{0}'.format(i)] = np.random.rand(a_size)
            self.p['b_{0}'.format(i)] = np.random.rand(b_size)

        self.p.run_model()

    def test_results(self):
        pass
        for i in range(self.nn):
            out_i = self.p['mux_comp.a'][i, ...]
            in_i = self.p['a_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

            out_i = self.p['mux_comp.b'][:, i]
            in_i = self.p['b_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='cs', out_stream=None)
        assert_check_partials(cpd, atol=1.0E-8, rtol=1.0E-8)


class TestDemuxComp2D(unittest.TestCase):

    def setUp(self):
        self.nn = 10

        a_shape = (3, 3)
        b_shape = (2, 4)

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        for i in range(self.nn):
            ivc.add_output(name='a_{0}'.format(i), shape=a_shape)
            ivc.add_output(name='b_{0}'.format(i), shape=b_shape)
            ivc.add_output(name='c_{0}'.format(i), shape=b_shape)

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['*'])

        demux_comp = self.p.model.add_subsystem(name='mux_comp',
                                                subsys=MuxComp(vec_size=self.nn))

        demux_comp.add_var('a', shape=a_shape, axis=0)
        demux_comp.add_var('b', shape=b_shape, axis=1)
        demux_comp.add_var('c', shape=b_shape, axis=2)

        for i in range(self.nn):
            self.p.model.connect('a_{0}'.format(i), 'mux_comp.a_{0}'.format(i))
            self.p.model.connect('b_{0}'.format(i), 'mux_comp.b_{0}'.format(i))
            self.p.model.connect('c_{0}'.format(i), 'mux_comp.c_{0}'.format(i))

        self.p.setup(force_alloc_complex=True)

        for i in range(self.nn):
            self.p['a_{0}'.format(i)] = np.random.rand(*a_shape)
            self.p['b_{0}'.format(i)] = np.random.rand(*b_shape)
            self.p['c_{0}'.format(i)] = np.random.rand(*b_shape)

        self.p.run_model()

    def test_results(self):
        for i in range(self.nn):
            out_i = self.p['mux_comp.a'][i, ...]
            in_i = self.p['a_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

            out_i = self.p['mux_comp.b'][:, i, :]
            in_i = self.p['b_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

            out_i = self.p['mux_comp.c'][:, :, i]
            in_i = self.p['c_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='cs', out_stream=None)
        assert_check_partials(cpd, atol=1.0E-8, rtol=1.0E-8)


class TestForDocs(unittest.TestCase):

    def test(self):
        """
        An example demonstrating a trivial use case of MuxComp
        """
        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp, MuxComp, VectorMagnitudeComp
        from openmdao.utils.assert_utils import assert_rel_error

        # The number of elements to be muxed
        n = 3

        # The size of each element to be muxed
        m = 100

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='x', shape=(m,), units='m')
        ivc.add_output(name='y', shape=(m,), units='m')
        ivc.add_output(name='z', shape=(m,), units='m')

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['x', 'y', 'z'])

        mux_comp = p.model.add_subsystem(name='mux',
                                         subsys=MuxComp(vec_size=n))

        mux_comp.add_var('r', shape=(m,), axis=1, units='m')

        p.model.add_subsystem(name='vec_mag_comp',
                              subsys=VectorMagnitudeComp(vec_size=m, length=n, in_name='r',
                                                         mag_name='r_mag', units='m'))

        p.model.connect('x', 'mux.r_0')
        p.model.connect('y', 'mux.r_1')
        p.model.connect('z', 'mux.r_2')
        p.model.connect('mux.r', 'vec_mag_comp.r')

        p.setup()

        p['x'] = 1 + np.random.rand(m)
        p['y'] = 1 + np.random.rand(m)
        p['z'] = 1 + np.random.rand(m)

        p.run_model()

        # Verify the results against numpy.dot in a for loop.
        for i in range(n):
            r_i = [p['x'][i], p['y'][i], p['z'][i]]
            expected_i = np.sqrt(np.dot(r_i, r_i))
            assert_rel_error(self, p.get_val('vec_mag_comp.r_mag')[i], expected_i)


if __name__ == '__main__':
    unittest.main()

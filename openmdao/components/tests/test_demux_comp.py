from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from openmdao.api import DemuxComp


class TestDemuxCompOptions(unittest.TestCase):

    def test_invalid_axis(self):
        nn = 10

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(nn,))

        p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a'])

        demux_comp = p.model.add_subsystem(name='demux_comp', subsys=DemuxComp(vec_size=nn))

        demux_comp.add_var('a', shape=(nn,), axis=1)

        p.model.connect('a', 'demux_comp.a')

        with self.assertRaises(RuntimeError) as ctx:
            p.setup()
        self.assertEqual(str(ctx.exception),
                         'Invalid axis (1) for variable of shape (10,)')
            
    def test_axis_with_wrong_size(self):
        nn = 10

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(nn, 7))
        ivc.add_output(name='b', shape=(3, nn))

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['a', 'b'])

        demux_comp = p.model.add_subsystem(name='demux_comp', subsys=DemuxComp(vec_size=nn))

        demux_comp.add_var('a', shape=(nn, 7), axis=1)
        demux_comp.add_var('b', shape=(3, nn), axis=1)

        p.model.connect('a', 'demux_comp.a')
        p.model.connect('b', 'demux_comp.b')

        with self.assertRaises(RuntimeError) as ctx:
            p.setup()
        self.assertEqual(str(ctx.exception),
                         'Variable a cannot be demuxed along axis 1. Axis size is '
                         '7 but vec_size is 10.')


class TestDemuxComp1D(unittest.TestCase):

    def setUp(self):
        self.nn = 10

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn,))
        ivc.add_output(name='b', shape=(self.nn,))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        demux_comp = self.p.model.add_subsystem(name='demux_comp',
                                                subsys=DemuxComp(vec_size=self.nn))

        demux_comp.add_var('a', shape=(self.nn,))
        demux_comp.add_var('b', shape=(self.nn,))

        self.p.model.connect('a', 'demux_comp.a')
        self.p.model.connect('b', 'demux_comp.b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn)
        self.p['b'] = np.random.rand(self.nn)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            in_i = self.p['a'][i]
            out_i = self.p['demux_comp.a_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)
            in_i = self.p['b'][i]
            out_i = self.p['demux_comp.b_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='cs', out_stream=None)
        assert_check_partials(cpd, atol=1.0E-8, rtol=1.0E-8)


class TestDemuxComp2D(unittest.TestCase):

    def setUp(self):
        self.nn = 10

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='a', shape=(self.nn, 7))
        ivc.add_output(name='b', shape=(3, self.nn))

        self.p.model.add_subsystem(name='ivc',
                                   subsys=ivc,
                                   promotes_outputs=['a', 'b'])

        demux_comp = self.p.model.add_subsystem(name='demux_comp',
                                                subsys=DemuxComp(vec_size=self.nn))

        demux_comp.add_var('a', shape=(self.nn, 7), axis=0)
        demux_comp.add_var('b', shape=(3, self.nn), axis=1)

        self.p.model.connect('a', 'demux_comp.a')
        self.p.model.connect('b', 'demux_comp.b')

        self.p.setup(force_alloc_complex=True)

        self.p['a'] = np.random.rand(self.nn, 7)
        self.p['b'] = np.random.rand(3, self.nn)

        self.p.run_model()

    def test_results(self):

        for i in range(self.nn):
            in_i = np.take(self.p['a'], indices=i, axis=0)
            out_i = self.p['demux_comp.a_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

            in_i = np.take(self.p['b'], indices=i, axis=1)
            out_i = self.p['demux_comp.b_{0}'.format(i)]
            assert_rel_error(self, in_i, out_i)

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='cs', out_stream=None)
        assert_check_partials(cpd, atol=1.0E-8, rtol=1.0E-8)


class TestForDocs(unittest.TestCase):

    def test(self):
        """
        An example demonstrating a trivial use case of DemuxComp
        """
        import numpy as np
        from openmdao.api import Problem, Group, IndepVarComp, DemuxComp, ExecComp
        from openmdao.utils.assert_utils import assert_rel_error

        # The number of elements to be demuxed
        n = 3

        # The size of each element to be demuxed
        m = 100

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='pos_ecef', shape=(m, 3), units='km')

        p.model.add_subsystem(name='ivc',
                              subsys=ivc,
                              promotes_outputs=['pos_ecef'])

        mux_comp = p.model.add_subsystem(name='demux',
                                         subsys=DemuxComp(vec_size=n))

        mux_comp.add_var('pos', shape=(m, n), axis=1, units='km')

        p.model.add_subsystem(name='longitude_comp',
                              subsys=ExecComp('long = atan(y/x)',
                                              x={'value': np.ones(m), 'units': 'km'},
                                              y={'value': np.ones(m), 'units': 'km'},
                                              long={'value': np.ones(m), 'units': 'rad'}))

        p.model.connect('demux.pos_0', 'longitude_comp.x')
        p.model.connect('demux.pos_1', 'longitude_comp.y')
        p.model.connect('pos_ecef', 'demux.pos')

        p.setup()

        p['pos_ecef'][:, 0] = 6378 * np.cos(np.linspace(0, 2*np.pi, m))
        p['pos_ecef'][:, 1] = 6378 * np.sin(np.linspace(0, 2*np.pi, m))
        p['pos_ecef'][:, 2] = 0.0

        p.run_model()

        expected = np.arctan( p['pos_ecef'][:, 1] / p['pos_ecef'][:, 0])
        assert_rel_error(self, p.get_val('longitude_comp.long'), expected)


if __name__ == '__main__':
    unittest.main()

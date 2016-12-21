""" Tests related to connecing inputs to outputs."""

import unittest
import numpy as np
from six import text_type, PY3
from six.moves import cStringIO
import warnings

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ExplicitComponent


class TestConnections(unittest.TestCase):

    def setUp(self, c1units=None, c3units=None):
        self.p = Problem(root=Group())
        root = self.p.root

        self.G1 = root.add_subsystem("G1", Group())
        self.G2 = self.G1.add_subsystem("G2", Group())
        self.C1 = self.G2.add_subsystem("C1", ExecComp('y=x*2.0', units=c1units))
        self.C2 = self.G2.add_subsystem("C2", IndepVarComp('x', 1.0))

        self.G3 = root.add_subsystem("G3", Group())
        self.G4 = self.G3.add_subsystem("G4", Group())
        self.C3 = self.G4.add_subsystem("C3", ExecComp('y=x*2.0', units=c3units))
        self.C4 = self.G4.add_subsystem("C4", ExecComp('y=x*2.0'))

    def test_no_conns(self):
        self.p.setup(check=False)

        self.p['G1.G2.C1.x'] = 111.
        self.p['G3.G4.C3.x'] = 222.
        self.p['G3.G4.C4.x'] = 333.

        self.assertEqual(self.C1._inputs['x'], 111.)
        self.assertEqual(self.C3._inputs['x'], 222.)
        self.assertEqual(self.C4._inputs['x'], 333.)

    def test_inp_inp_explicit_conn_w_src(self):
        raise unittest.SkipTest("explicit input-input connections not supported yet")
        self.p.root.connect('G3.G4.C3.x', 'G3.G4.C4.x') # connect inputs
        self.p.root.connect('G1.G2.C2.x', 'G3.G4.C3.x') # connect src to one of connected inputs
        self.p.setup(check=False)

        self.p['G1.G2.C2.x'] = 999.
        self.assertEqual(self.C3._inputs['x'], 0.)
        self.assertEqual(self.C4._inputs['x'], 0.)

        self.p.run()
        self.assertEqual(self.C3._inputs['x'], 999.)
        self.assertEqual(self.C4._inputs['x'], 999.)

    def test_pull_size_from_source(self):
        raise unittest.SkipTest("setting input size based on src size not supported yet")

        class Src(ExplicitComponent):

            def initialize_variables(self):

                self.add_input('x', 2.0)
                self.add_output('y1', np.zeros((3, )))
                self.add_output('y2', shape=((3, )))

            def solve_nonlinear(self, inputs, outputs, resids):
                x = inputs['x']

                outputs['y1'] = x * np.array( [1.0, 2.0, 3.0])
                outputs['y2'] = x * np.array( [1.0, 2.0, 3.0])

        class Tgt(ExplicitComponent):

            def initialize_variables(self):

                self.add_input('x1')
                self.add_input('x2')
                self.add_output('y1', 0.0)
                self.add_output('y2', 0.0)

            def solve_nonlinear(self, inputs, outputs, resids):
                x1 = inputs['x1']
                x2 = inputs['x2']

                outputs['y1'] = np.sum(x1)
                outputs['y2'] = np.sum(x2)

        p = Problem()
        p.root = Group()
        p.root.add_subsystem('src', Src())
        p.root.add_subsystem('tgt', Tgt())

        p.root.connect('src.y1', 'tgt.x1')
        p.root.connect('src.y2', 'tgt.x2')

        p.setup(check=False)
        p.run()

        self.assertEqual(p['tgt.y1'], 12.0)
        self.assertEqual(p['tgt.y2'], 12.0)

    def test_pull_size_from_source_with_indices(self):
        raise unittest.SkipTest("setting input size based on src size not supported yet")

        class Src(ExplicitComponent):

            def initialize_variables(self):

                self.add_input('x', 2.0)
                self.add_output('y1', np.zeros((3, )))
                self.add_output('y2', shape=((3, )))
                self.add_output('y3', 3.0)

            def solve_nonlinear(self, inputs, outputs, resids):
                """ counts up. """

                x = inputs['x']

                outputs['y1'] = x * np.array( [1.0, 2.0, 3.0])
                outputs['y2'] = x * np.array( [1.0, 2.0, 3.0])
                outputs['y3'] = x * 4.0

        class Tgt(ExplicitComponent):

            def initialize_variables(self):

                self.add_input('x1')
                self.add_input('x2')
                self.add_input('x3')
                self.add_output('y1', 0.0)
                self.add_output('y2', 0.0)
                self.add_output('y3', 0.0)

            def solve_nonlinear(self, inputs, outputs, resids):
                """ counts up. """

                x1 = inputs['x1']
                x2 = inputs['x2']
                x3 = inputs['x3']

                outputs['y1'] = np.sum(x1)
                outputs['y2'] = np.sum(x2)
                outputs['y3'] = np.sum(x3)

        top = Problem()
        top.root = Group()
        top.root.add_subsystem('src', Src())
        top.root.add_subsystem('tgt', Tgt())

        top.root.connect('src.y1', 'tgt.x1', src_indices=(0, 1))
        top.root.connect('src.y2', 'tgt.x2', src_indices=(0, 1))
        top.root.connect('src.y3', 'tgt.x3')

        top.setup(check=False)
        top.run()

        self.assertEqual(top['tgt.y1'], 6.0)
        self.assertEqual(top['tgt.y2'], 6.0)
        self.assertEqual(top['tgt.y3'], 8.0)

    def test_inp_inp_conn_no_src(self):
        raise unittest.SkipTest("no setup testing yet")
        self.p.root.connect('G3.G4.C3.x', 'G3.G4.C4.x')

        stream = cStringIO()
        self.p.setup(out_stream=stream)

        self.p['G3.G4.C3.x'] = 999.
        self.assertEqual(self.p.root.G3.G4.C3._inputs['x'], 999.)
        self.assertEqual(self.p.root.G3.G4.C4._inputs['x'], 999.)

        content = stream.getvalue()
        self.assertTrue("The following parameters have no associated unknowns:\nG1.G2.C1.x\nG3.G4.C3.x\nG3.G4.C4.x" in content)
        self.assertTrue("The following components have no connections:\nG1.G2.C1\nG1.G2.C2\nG3.G4.C3\nG3.G4.C4\n" in content)
        self.assertTrue("No recorders have been specified, so no data will be saved." in content)

    def test_diff_conn_input_vals(self):
        raise unittest.SkipTest("no checking yet of connected inputs without a src")

        # set different initial values
        self.C1._inputs['x'] = 7.
        self.C3._inputs['x'] = 5.

        # connect two inputs
        self.p.root.connect('G1.G2.C1.x', 'G3.G4.C3.x')

        try:
            self.p.setup(check=False)
        except Exception as err:
            self.assertTrue(
                "The following sourceless connected inputs have different initial values: "
                "[('G1.G2.C1.x', 7.0), ('G3.G4.C3.x', 5.0)].  Connect one of them to the output of "
                "an IndepVarComp to ensure that they have the same initial value." in str(err))
        else:
            self.fail("Exception expected")

    def test_diff_conn_input_units(self):
        raise unittest.SkipTest("no compatability checking of connected inputs yet")

        # set different but compatible units
        self.setUp(c1units={'x': 'ft'}, c3units={'x': 'inch'})

        # connect two inputs
        self.p.root.connect('G1.G2.C1.x', 'G3.G4.C3.x')

        try:
            self.p.setup(check=False)
        except Exception as err:
            msg = "The following connected inputs have no source and different units: [('G1.G2.C1.x', 'ft'), ('G3.G4.C3.x', 'inch')]. Connect 'G1.G2.C1.x' to a source (such as an IndepVarComp) with defined units."
            self.assertTrue(msg in str(err))
        else:
            self.fail("Exception expected")

    def test_diff_conn_input_units_swap(self):
        raise unittest.SkipTest("no compatability checking of connected inputs yet")

        # set different but compatible units
        self.setUp(c1units={'x': 'ft'}, c3units={'x': 'inch'})

        # connect two inputs
        self.p.root.connect('G3.G4.C3.x', 'G1.G2.C1.x')

        try:
            self.p.setup(check=False)
        except Exception as err:
            msg = "The following connected inputs have no source and different units: [('G1.G2.C1.x', 'ft'), ('G3.G4.C3.x', 'inch')]. Connect 'G3.G4.C3.x' to a source (such as an IndepVarComp) with defined units."
            self.assertTrue(msg in str(err))
        else:
            self.fail("Exception expected")

    def test_diff_conn_input_units_w_src(self):
        raise unittest.SkipTest("no compatability checking of connected inputs yet")

        p = Problem(root=Group())
        root = p.root

        num_comps = 50

        desvars = root.add_subsystem("desvars", IndepVarComp('dvar1', 1.0))

        # add a bunch of comps
        for i in range(num_comps):
            if i % 2 == 0:
                units = "ft"
            else:
                units = "m"

            root.add_subsystem("C%d"%i, ExecComp('y=x*2.0', units={'x':units}))

        # connect all of their inputs (which have different units)
        for i in range(1, num_comps):
            root.connect("C%d.x"%(i-1), "C%d.x"%i)

        try:
            p.setup(check=False)
        except Exception as err:
            self.assertTrue("The following connected inputs have no source and different units" in
                            str(err))
        else:
            self.fail("Exception expected")

        # now, connect a source and the error should go away

        p.cleanup()

        root.connect('desvars.dvar1', 'C10.x')

        p.setup(check=False)


class TestConnectionsPromoted(unittest.TestCase):

    def test_inp_inp_promoted_no_src(self):
        raise unittest.SkipTest("connected inputs w/o src not supported yet")

        p = Problem(root=Group())
        root = p.root

        G1 = root.add_subsystem("G1", Group())
        G2 = G1.add_subsystem("G2", Group())
        C1 = G2.add_subsystem("C1", ExecComp('y=x*2.0'))
        C2 = G2.add_subsystem("C2", ExecComp('y=x*2.0'))

        G3 = root.add_subsystem("G3", Group())
        G4 = G3.add_subsystem("G4", Group(), promotes=['x'])
        C3 = G4.add_subsystem("C3", ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add_subsystem("C4", ExecComp('y=x*2.0'), promotes=['x'])

        p.setup(check=False)

        # setting promoted name should set both inputs mapped to that name
        p['G3.x'] = 999.
        self.assertEqual(C3._inputs['x'], 999.)
        self.assertEqual(C4._inputs['x'], 999.)

    def test_inp_inp_promoted_w_prom_src(self):
        p = Problem(root=Group())
        root = p.root

        G1 = root.add_subsystem("G1", Group(), promotes=['x'])
        G2 = G1.add_subsystem("G2", Group(), promotes=['x'])
        C1 = G2.add_subsystem("C1", ExecComp('y=x*2.0'))
        C2 = G2.add_subsystem("C2", IndepVarComp('x', 1.0), promotes=['x'])

        G3 = root.add_subsystem("G3", Group(), promotes=['x'])
        G4 = G3.add_subsystem("G4", Group(), promotes=['x'])
        C3 = G4.add_subsystem("C3", ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add_subsystem("C4", ExecComp('y=x*2.0'), promotes=['x'])

        p.setup(check=False)
        p.root.suppress_solver_output = True

        # setting promoted name will set the value into the outputs, but will
        # not propagate it to the inputs. That will happen during run().
        p['x'] = 999.
        self.assertEqual(C3._inputs['x'], 0.)
        self.assertEqual(C4._inputs['x'], 0.)

        p.run()
        self.assertEqual(C3._inputs['x'], 999.)
        self.assertEqual(C4._inputs['x'], 999.)

    def test_inp_inp_promoted_w_explicit_src(self):
        p = Problem(root=Group())
        root = p.root

        G1 = root.add_subsystem("G1", Group())
        G2 = G1.add_subsystem("G2", Group(), promotes=['x'])
        C1 = G2.add_subsystem("C1", ExecComp('y=x*2.0'))
        C2 = G2.add_subsystem("C2", IndepVarComp('x', 1.0), promotes=['x'])

        G3 = root.add_subsystem("G3", Group())
        G4 = G3.add_subsystem("G4", Group(), promotes=['x'])
        C3 = G4.add_subsystem("C3", ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add_subsystem("C4", ExecComp('y=x*2.0'), promotes=['x'])

        p.root.connect('G1.x', 'G3.x')
        p.setup(check=False)
        p.root.suppress_solver_output = True

        # setting promoted name will set the value into the outputs, but will
        # not propagate it to the inputs. That will happen during run().
        p['G1.x'] = 999.
        self.assertEqual(C3._inputs['x'], 0.)
        self.assertEqual(C4._inputs['x'], 0.)

        p.run()
        self.assertEqual(C3._inputs['x'], 999.)
        self.assertEqual(C4._inputs['x'], 999.)

    def test_unit_conv_message(self):
        raise unittest.SkipTest("no units yet")
        prob = Problem(root=Group())
        root = prob.root

        root.add_subsystem("C1", ExecComp('y=x*2.0', units={'x':'ft'}), promotes=['x'])
        root.add_subsystem("C2", ExecComp('y=x*2.0', units={'x':'inch'}), promotes=['x'])
        root.add_subsystem("C3", ExecComp('y=x*2.0', units={'x':'m'}), promotes=['x'])

        try:
            prob.setup(check=False)
        except Exception as err:
            msg = "The following connected inputs are promoted to 'x', but have different units: [('C1.x', 'ft'), ('C2.x', 'inch'), ('C3.x', 'm')]. Connect 'x' to a source (such as an IndepVarComp) with defined units."
            self.assertTrue(msg in str(err))
        else:
            self.fail("Exception expected")

        # Remedy the problem with an Indepvarcomp

        prob = Problem(root=Group())
        root = prob.root

        root.add_subsystem("C1", ExecComp('y=x*2.0', units={'x':'ft'}), promotes=['x'])
        root.add_subsystem("C2", ExecComp('y=x*2.0', units={'x':'inch'}), promotes=['x'])
        root.add_subsystem("C3", ExecComp('y=x*2.0', units={'x':'m'}), promotes=['x'])
        root.add_subsystem('p', IndepVarComp('x', 1.0, units='cm'), promotes=['x'])

        prob.setup(check=False)


#class TestUBCS(unittest.TestCase):

    #def test_ubcs(self):
        #p = Problem(root=Group())
        #root = p.root
        #root._ln_solver = ScipyGMRES()

        #self.P1 = root.add_subsystem("P1", IndepVarComp('x', 1.0))
        #self.C1 = root.add_subsystem("C1", ExecComp('y=x1*2.0+x2*3.0', x2=1.0))
        #self.C2 = root.add_subsystem("C2", ExecComp('y=x1*2.0+x2'))
        #self.C3 = root.add_subsystem("C3", ExecComp('y=x*2.0'))
        #self.C4 = root.add_subsystem("C4", ExecComp('y=x1*2.0 + x2*5.0'))
        #self.C5 = root.add_subsystem("C5", ExecComp('y=x1*2.0 + x2*7.0'))

        #root.connect("P1.x", "C1.x1")
        #root.connect("C1.y", ("C2.x1", "C3.x"))
        #root.connect("C2.y", "C4.x1")
        #root.connect("C3.y", "C4.x2")

        ## input-input connection
        #root.connect("C1.x2", "C5.x2")

        ## create a cycle
        #root.connect("C4.y", "C1.x2")

        ## set a bogus value for C4.y
        #self.C4._init_unknowns_dict['y']['val'] = -999.

        #p.setup(check=False)

        #ubcs, tgts = p._get_ubc_vars(root.connections)

        #self.assertEqual(ubcs, ['C1.x2'])
        #self.assertEqual(tgts, set(['C1']))

        #p.run()

        ## TODO: for now, we've just decided to force component devs to give proper initial
        ## values for their outputs.  If later we decide to use push scatters or some other
        ## means to fix the issue, uncomment this.
        ##assert_rel_error(self, p['C1.y'], 5.0, 1e-6)


#class Sink1(ExplicitComponent):
    #def initialize_variables(self):
        #self.add_input('x', val=0.0, units='m')


#class Sink2(ExplicitComponent):
    #def initialize_variables(self):
        #self.add_input('x', val=0.0, units='mm')

#class TestConnSetup(unittest.TestCase):

    #def test_setup(self):
        #top = Problem()

        #root = top.root = Group()

        #root.add_subsystem('src1', IndepVarComp('x', 0.0, units='m'))
        #root.add_subsystem('sink1', Sink1())
        #root.add_subsystem('sink2', Sink2())

        #root.connect('src1.x', 'sink1.x')
        #root.connect('src1.x', 'sink2.x')

        #stream = cStringIO()
        #results = top.setup(check=True, out_stream=stream)

        ## Not in standard setup anymore
        #self.assertTrue('unit_diffs' not in results)


#class TestSourceIndicesAreInts(unittest.TestCase):

    #def test_src_indices_as_float_array_fails(self):
        #top = Problem()

        #root = top.root = Group()

        #root.add_subsystem('src1', IndepVarComp('x', np.zeros(5,), units='m'))
        #root.add_subsystem('sink1', Sink1())

        #with self.assertRaises(TypeError) as cm:
            #root.connect('src1.x', 'sink1.x', src_indices=np.zeros(1))

    #def test_src_indices_as_int_array_passes(self):
        #top = Problem()

        #root = top.root = Group()

        #root.add_subsystem('src1', IndepVarComp('x', np.zeros(5,), units='m'))
        #root.add_subsystem('sink1', Sink1())

        #try:
            #root.connect('src1.x', 'sink1.x', src_indices=np.zeros(1,dtype=int))
        #except TypeError:
            #self.fail('Issuing a connection with src_indices as int raised a TypeError')

    #def test_src_indices_as_float_list_fails(self):
        #top = Problem()

        #root = top.root = Group()

        #root.add_subsystem('src1', IndepVarComp('x', np.zeros(5,), units='m'))
        #root.add_subsystem('sink1', Sink1())

        #with self.assertRaises(TypeError) as cm:
            #root.connect('src1.x', 'sink1.x', src_indices=[1.0])

    #def test_src_indices_as_int_array_passes(self):
        #top = Problem()

        #root = top.root = Group()

        #root.add_subsystem('src1', IndepVarComp('x', np.zeros(5,), units='m'))
        #root.add_subsystem('sink1', Sink1())

        #try:
            #root.connect('src1.x', 'sink1.x', src_indices=np.ones(1,dtype=int))
        #except TypeError:
            #self.fail('Issuing a connection with src_indices as int raised a TypeError')


if __name__ == "__main__":
    unittest.main()

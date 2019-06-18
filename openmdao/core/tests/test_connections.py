""" Tests related to connecing inputs to outputs."""

import unittest
import numpy as np

from six.moves import cStringIO, range
from six import assertRaisesRegex

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ExplicitComponent
from openmdao.utils.assert_utils import assert_rel_error


class TestConnections(unittest.TestCase):

    def setUp(self):
        self.setup_model(None, None)

    def setup_model(self, c1meta=None, c3meta=None):
        self.p = Problem()
        root = self.p.model

        if c1meta is None:
            c1meta = {}

        if c3meta is None:
            c3meta = {}

        self.G1 = root.add_subsystem("G1", Group())
        self.G2 = self.G1.add_subsystem("G2", Group())
        self.C1 = self.G2.add_subsystem("C1", ExecComp('y=x*2.0', **c1meta))
        self.C2 = self.G2.add_subsystem("C2", IndepVarComp('x', 1.0))

        self.G3 = root.add_subsystem("G3", Group())
        self.G4 = self.G3.add_subsystem("G4", Group())
        self.C3 = self.G4.add_subsystem("C3", ExecComp('y=x*2.0', **c3meta))
        self.C4 = self.G4.add_subsystem("C4", ExecComp('y=x*2.0'))

    def test_no_conns(self):
        self.p.setup()

        self.p['G1.G2.C1.x'] = 111.
        self.p['G3.G4.C3.x'] = 222.
        self.p['G3.G4.C4.x'] = 333.

        self.p.final_setup()

        self.assertEqual(self.C1._inputs['x'], 111.)
        self.assertEqual(self.C3._inputs['x'], 222.)
        self.assertEqual(self.C4._inputs['x'], 333.)

    def test_inp_inp_explicit_conn_w_src(self):
        raise unittest.SkipTest("explicit input-input connections not supported yet")
        self.p.model.connect('G3.G4.C3.x', 'G3.G4.C4.x')  # connect inputs
        self.p.model.connect('G1.G2.C2.x', 'G3.G4.C3.x')  # connect src to one of connected inputs
        self.p.setup()

        self.p['G1.G2.C2.x'] = 999.
        self.assertEqual(self.C3._inputs['x'], 0.)
        self.assertEqual(self.C4._inputs['x'], 0.)

        self.p.run_model()
        self.assertEqual(self.C3._inputs['x'], 999.)
        self.assertEqual(self.C4._inputs['x'], 999.)

    def test_pull_size_from_source(self):
        raise unittest.SkipTest("setting input size based on src size not supported yet")

        class Src(ExplicitComponent):

            def setup(self):

                self.add_input('x', 2.0)
                self.add_output('y1', np.zeros((3, )))
                self.add_output('y2', shape=((3, )))

            def solve_nonlinear(self, inputs, outputs, resids):
                x = inputs['x']

                outputs['y1'] = x * np.array([1.0, 2.0, 3.0])
                outputs['y2'] = x * np.array([1.0, 2.0, 3.0])

        class Tgt(ExplicitComponent):

            def setup(self):

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
        p.model.add_subsystem('src', Src())
        p.model.add_subsystem('tgt', Tgt())

        p.model.connect('src.y1', 'tgt.x1')
        p.model.connect('src.y2', 'tgt.x2')

        p.setup()
        p.run_model()

        self.assertEqual(p['tgt.y1'], 12.0)
        self.assertEqual(p['tgt.y2'], 12.0)

    def test_pull_size_from_source_with_indices(self):
        raise unittest.SkipTest("setting input size based on src size not supported yet")

        class Src(ExplicitComponent):

            def setup(self):

                self.add_input('x', 2.0)
                self.add_output('y1', np.zeros((3, )))
                self.add_output('y2', shape=((3, )))
                self.add_output('y3', 3.0)

            def solve_nonlinear(self, inputs, outputs, resids):
                """ counts up. """

                x = inputs['x']

                outputs['y1'] = x * np.array([1.0, 2.0, 3.0])
                outputs['y2'] = x * np.array([1.0, 2.0, 3.0])
                outputs['y3'] = x * 4.0

        class Tgt(ExplicitComponent):

            def setup(self):

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
        top.model.add_subsystem('src', Src())
        top.model.add_subsystem('tgt', Tgt())

        top.model.connect('src.y1', 'tgt.x1', src_indices=(0, 1))
        top.model.connect('src.y2', 'tgt.x2', src_indices=(0, 1))
        top.model.connect('src.y3', 'tgt.x3')

        top.setup()
        top.run_model()

        self.assertEqual(top['tgt.y1'], 6.0)
        self.assertEqual(top['tgt.y2'], 6.0)
        self.assertEqual(top['tgt.y3'], 8.0)

    def test_inp_inp_conn_no_src(self):
        raise unittest.SkipTest("no setup testing yet")
        self.p.model.connect('G3.G4.C3.x', 'G3.G4.C4.x')

        stream = cStringIO()
        self.p.setup(out_stream=stream)

        self.p['G3.G4.C3.x'] = 999.
        self.assertEqual(self.p.model.G3.G4.C3._inputs['x'], 999.)
        self.assertEqual(self.p.model.G3.G4.C4._inputs['x'], 999.)

        content = stream.getvalue()
        self.assertTrue("The following parameters have no associated unknowns:\n"
                        "G1.G2.C1.x\nG3.G4.C3.x\nG3.G4.C4.x" in content)
        self.assertTrue("The following components have no connections:\n"
                        "G1.G2.C1\nG1.G2.C2\nG3.G4.C3\nG3.G4.C4\n" in content)
        self.assertTrue("No recorders have been specified, so no data will be saved." in content)

    def test_diff_conn_input_vals(self):
        raise unittest.SkipTest("no checking yet of connected inputs without a src")

        # set different initial values
        self.C1._inputs['x'] = 7.
        self.C3._inputs['x'] = 5.

        # connect two inputs
        self.p.model.connect('G1.G2.C1.x', 'G3.G4.C3.x')

        try:
            self.p.setup()
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
        self.setup_model(c1meta={'x': {'units': 'ft'}}, c3meta={'x': {'units': 'inch'}})

        # connect two inputs
        self.p.model.connect('G1.G2.C1.x', 'G3.G4.C3.x')

        try:
            self.p.setup()
        except Exception as err:
            msg = "The following connected inputs have no source and different units: " \
                  "[('G1.G2.C1.x', 'ft'), ('G3.G4.C3.x', 'inch')]. " \
                  "Connect 'G1.G2.C1.x' to a source (such as an IndepVarComp) with defined units."
            self.assertTrue(msg in str(err))
        else:
            self.fail("Exception expected")

    def test_diff_conn_input_units_swap(self):
        raise unittest.SkipTest("no compatability checking of connected inputs yet")

        # set different but compatible units
        self.setup_model(c1meta={'x': {'units': 'ft'}}, c3meta={'x': {'units': 'inch'}})

        # connect two inputs
        self.p.model.connect('G3.G4.C3.x', 'G1.G2.C1.x')

        try:
            self.p.setup()
        except Exception as err:
            msg = "The following connected inputs have no source and different units: " \
                  "[('G1.G2.C1.x', 'ft'), ('G3.G4.C3.x', 'inch')]. " \
                  "Connect 'G3.G4.C3.x' to a source (such as an IndepVarComp) with defined units."
            self.assertTrue(msg in str(err))
        else:
            self.fail("Exception expected")

    def test_diff_conn_input_units_w_src(self):
        raise unittest.SkipTest("no compatability checking of connected inputs yet")

        p = Problem()
        root = p.model

        num_comps = 50

        root.add_subsystem("desvars", IndepVarComp('dvar1', 1.0))

        # add a bunch of comps
        for i in range(num_comps):
            if i % 2 == 0:
                units = "ft"
            else:
                units = "m"

            root.add_subsystem("C%d" % i, ExecComp('y=x*2.0', units={'x': units}))

        # connect all of their inputs (which have different units)
        for i in range(1, num_comps):
            root.connect("C%d.x" % (i-1), "C%d.x" % i)

        try:
            p.setup()
        except Exception as err:
            self.assertTrue("The following connected inputs have no source and different units" in
                            str(err))
        else:
            self.fail("Exception expected")

        # now, connect a source and the error should go away

        p.cleanup()

        root.connect('desvars.dvar1', 'C10.x')

        p.setup()


class TestConnectionsPromoted(unittest.TestCase):

    def test_inp_inp_promoted_no_src(self):

        p = Problem()
        root = p.model

        G1 = root.add_subsystem("G1", Group())
        G2 = G1.add_subsystem("G2", Group())
        G2.add_subsystem("C1", ExecComp('y=x*2.0'))
        G2.add_subsystem("C2", ExecComp('y=x*2.0'))

        G3 = root.add_subsystem("G3", Group())
        G4 = G3.add_subsystem("G4", Group(), promotes=['x'])
        G4.add_subsystem("C3", ExecComp('y=x*2.0'), promotes=['x'])
        G4.add_subsystem("C4", ExecComp('y=x*2.0'), promotes=['x'])

        p.setup()
        p.final_setup()

        # setting promoted name should set both inputs mapped to that name
        with self.assertRaises(Exception) as context:
            p['G3.x'] = 999.
        self.assertEqual(str(context.exception),
                         "The promoted name G3.x is invalid because it refers to multiple inputs: "
                         "[G3.G4.C3.x, G3.G4.C4.x] that are not connected to an output variable.")

    def test_inp_inp_promoted_w_prom_src(self):
        p = Problem()
        root = p.model

        G1 = root.add_subsystem("G1", Group(), promotes=['x'])
        G2 = G1.add_subsystem("G2", Group(), promotes=['x'])
        G2.add_subsystem("C1", ExecComp('y=x*2.0'))
        G2.add_subsystem("C2", IndepVarComp('x', 1.0), promotes=['x'])

        G3 = root.add_subsystem("G3", Group(), promotes=['x'])
        G4 = G3.add_subsystem("G4", Group(), promotes=['x'])
        C3 = G4.add_subsystem("C3", ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add_subsystem("C4", ExecComp('y=x*2.0'), promotes=['x'])

        p.setup()
        p.set_solver_print(level=0)

        # setting promoted name will set the value into the outputs, but will
        # not propagate it to the inputs. That will happen during run_model().
        p['x'] = 999.

        p.run_model()
        self.assertEqual(C3._inputs['x'], 999.)
        self.assertEqual(C4._inputs['x'], 999.)

    def test_inp_inp_promoted_w_explicit_src(self):
        p = Problem()
        root = p.model

        G1 = root.add_subsystem("G1", Group())
        G2 = G1.add_subsystem("G2", Group(), promotes=['x'])
        G2.add_subsystem("C1", ExecComp('y=x*2.0'))
        G2.add_subsystem("C2", IndepVarComp('x', 1.0), promotes=['x'])

        G3 = root.add_subsystem("G3", Group())
        G4 = G3.add_subsystem("G4", Group(), promotes=['x'])
        C3 = G4.add_subsystem("C3", ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add_subsystem("C4", ExecComp('y=x*2.0'), promotes=['x'])

        p.model.connect('G1.x', 'G3.x')
        p.setup()
        p.set_solver_print(level=0)

        # setting promoted name will set the value into the outputs, but will
        # not propagate it to the inputs. That will happen during run_model().
        p['G1.x'] = 999.

        p.run_model()
        self.assertEqual(C3._inputs['x'], 999.)
        self.assertEqual(C4._inputs['x'], 999.)

    def test_unit_conv_message(self):
        raise unittest.SkipTest("no units yet")
        prob = Problem()
        root = prob.model

        root.add_subsystem("C1", ExecComp('y=x*2.0', units={'x': 'ft'}), promotes=['x'])
        root.add_subsystem("C2", ExecComp('y=x*2.0', units={'x': 'inch'}), promotes=['x'])
        root.add_subsystem("C3", ExecComp('y=x*2.0', units={'x': 'm'}), promotes=['x'])

        try:
            prob.setup()
        except Exception as err:
            msg = "The following connected inputs are promoted to 'x', but have different units: " \
                  "[('C1.x', 'ft'), ('C2.x', 'inch'), ('C3.x', 'm')]. " \
                  "Connect 'x' to a source (such as an IndepVarComp) with defined units."
            self.assertTrue(msg in str(err))
        else:
            self.fail("Exception expected")

        # Remedy the problem with an Indepvarcomp

        prob = Problem()
        root = prob.model

        root.add_subsystem("C1", ExecComp('y=x*2.0', units={'x': 'ft'}), promotes=['x'])
        root.add_subsystem("C2", ExecComp('y=x*2.0', units={'x': 'inch'}), promotes=['x'])
        root.add_subsystem("C3", ExecComp('y=x*2.0', units={'x': 'm'}), promotes=['x'])
        root.add_subsystem('p', IndepVarComp('x', 1.0, units='cm'), promotes=['x'])

        prob.setup()

    def test_overlapping_system_names(self):
        # This ensures that _setup_connections does not think g1 and g1a are the same system
        prob = Problem()
        model = prob.model

        g1 = model.add_subsystem('g1', Group())
        g1a = model.add_subsystem('g1a', Group())

        g1.add_subsystem('c', ExecComp('y=x'))
        g1a.add_subsystem('c', ExecComp('y=x'))

        model.connect('g1.c.y', 'g1a.c.x')
        model.connect('g1a.c.y', 'g1.c.x')

        prob.setup(check=True)


class TestConnectionsIndices(unittest.TestCase):

    def setUp(self):
        class ArrayComp(ExplicitComponent):
            def setup(self):
                self.add_input('inp', val=np.ones((2)))
                self.add_input('inp1', val=0)
                self.add_output('out', val=np.zeros((2)))

            def compute(self, inputs, outputs):
                outputs['out'] = inputs['inp'] * 2.

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('blammo', val=3.)
        indep_var_comp.add_output('arrout', val=np.ones(5))

        prob = Problem()
        prob.model.add_subsystem('idvp', indep_var_comp)
        prob.model.add_subsystem('arraycomp', ArrayComp())

        self.prob = prob

    def test_bad_shapes(self):
        # Should not be allowed because the source and target shapes do not match
        self.prob.model.connect('idvp.blammo', 'arraycomp.inp')

        expected = (r"The source and target shapes do not match or are ambiguous for the "
                    r"connection 'idvp.blammo' to 'arraycomp.inp'."
                    r" Expected \(2.*,\) but got \(1.*,\).")

        with assertRaisesRegex(self, ValueError, expected):
            self.prob.setup()

    def test_bad_length(self):
        # Should not be allowed because the length of src_indices is greater than
        # the shape of arraycomp.inp
        self.prob.model.connect('idvp.blammo', 'arraycomp.inp', src_indices=[0, 1, 0])

        expected = (r"The source indices \[0 1 0\] do not specify a valid shape "
                    r"for the connection 'idvp.blammo' to 'arraycomp.inp'. "
                    r"The target shape is \(2.*,\) but indices are \(3.*,\).")

        with assertRaisesRegex(self, ValueError, expected):
            self.prob.setup()

    def test_bad_value(self):
        # Should not be allowed because the index value within src_indices is outside
        # the valid range for the source
        self.prob.model.connect('idvp.arrout', 'arraycomp.inp1', src_indices=[100000])

        expected = ("The source indices do not specify a valid index for the "
                    "connection 'idvp.arrout' to 'arraycomp.inp1'. "
                    "Index '100000' is out of range for source dimension of "
                    "size 5.")

        try:
            self.prob.setup()
        except ValueError as err:
            self.assertEqual(str(err), expected)
        else:
            self.fail('Exception expected.')


class TestShapes(unittest.TestCase):
    def test_connect_flat_array_to_row_vector(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.arange(10)))
        p.model.add_subsystem('C1',
                              ExecComp('y=dot(x, A)',
                                       x={'value': np.zeros((1, 10))},
                                       A={'value': np.eye(10)},
                                       y={'value': np.zeros((1, 10))}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.y'], np.arange(10)[np.newaxis, :])

    def test_connect_flat_array_to_col_vector(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.arange(10)))
        p.model.add_subsystem('C1',
                              ExecComp('y=dot(A, x)',
                                       x={'value': np.zeros((10, 1))},
                                       A={'value': np.eye(10)},
                                       y={'value': np.zeros((10, 1))}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.y'], np.arange(10)[:, np.newaxis])

    def test_connect_row_vector_to_flat_array(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.arange(10)[np.newaxis, :]))
        p.model.add_subsystem('C1', ExecComp('y=5*x',
                                             x={'value': np.zeros(10)},
                                             y={'value': np.zeros(10)}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.y'], 5 * np.arange(10))

    def test_connect_col_vector_to_flat_array(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.arange(10)[:, np.newaxis]))
        p.model.add_subsystem('C1', ExecComp('y=5*x',
                                             x={'value': np.zeros(10)},
                                             y={'value': np.zeros(10)}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.y'], 5 * np.arange(10))

    def test_connect_flat_to_3d_array(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.arange(10)))
        p.model.add_subsystem('C1', ExecComp('y=5*x',
                                             x={'value': np.zeros((1, 10, 1))},
                                             y={'value': np.zeros((1, 10, 1))}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.y'], 5 * np.arange(10)[np.newaxis, :, np.newaxis])

    def test_connect_flat_nd_to_flat_nd(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x',
                                                    val=np.arange(10)[np.newaxis, :, np.newaxis,
                                                                      np.newaxis]))
        p.model.add_subsystem('C1', ExecComp('y=5*x',
                                             x={'value': np.zeros((1, 1, 1, 10))},
                                             y={'value': np.zeros((1, 1, 1, 10))}))
        p.model.connect('indep.x', 'C1.x')
        p.setup()
        p.run_model()
        assert_rel_error(self, p['C1.y'],
                         5 * np.arange(10)[np.newaxis, np.newaxis, np.newaxis, :])

    def test_connect_incompatible_shapes(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.arange(10)[np.newaxis, :,
                                                                           np.newaxis, np.newaxis]))
        p.model.add_subsystem('C1', ExecComp('y=5*x',
                                             x={'value': np.zeros((5, 2))},
                                             y={'value': np.zeros((5, 2))}))
        p.model.connect('indep.x', 'C1.x')

        with self.assertRaises(Exception) as context:
            p.setup()
        self.assertEqual(str(context.exception),
                         "The source and target shapes do not match or are ambiguous "
                         "for the connection 'indep.x' to 'C1.x'. Expected (5, 2) but "
                         "got (1, 10, 1, 1).")


class TestMultiConns(unittest.TestCase):
    def test_mult_conns(self):

        class SubGroup(Group):
            def setup(self):
                self.add_subsystem('c1', ExecComp('y = 2*x', x=np.ones(4), y=2*np.ones(4)),
                                   promotes=['y', 'x'])
                self.add_subsystem('c2', ExecComp('z = 2*y', y=np.ones(4), z=2*np.ones(4)),
                                   promotes=['z', 'y'])

        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('x', 10*np.ones(4))
        indeps.add_output('y', np.ones(4))

        prob.model.add_subsystem('sub', SubGroup())

        prob.model.connect('x', 'sub.x')
        prob.model.connect('y', 'sub.y')

        with self.assertRaises(Exception) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "The following inputs have multiple connections: "
                         "sub.c2.y from ['indeps.y', 'sub.c1.y']")

    def test_mixed_conns_same_level(self):

        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', 10*np.ones(4))

        # c2.y is implicitly connected to c1.y
        prob.model.add_subsystem('c1', ExecComp('y = 2*x', x=np.ones(4), y=2*np.ones(4)),
                                 promotes=['y'])
        prob.model.add_subsystem('c2', ExecComp('z = 2*y', y=np.ones(4), z=2*np.ones(4)),
                                 promotes=['y'])

        # make a second, explicit, connection to y (which is c2.y promoted)
        prob.model.connect('indeps.x', 'y')

        with self.assertRaises(Exception) as context:
            prob.setup()
            prob.final_setup()

        self.assertEqual(str(context.exception),
                         "Input 'c2.y' cannot be connected to 'indeps.x' "
                         "because it's already connected to 'c1.y'")


if __name__ == "__main__":
    unittest.main()

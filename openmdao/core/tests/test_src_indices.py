import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_totals
from openmdao.utils.testing_utils import use_tempdirs

class Inner(om.Group):
    def setup(self):
        comp = om.ExecComp('y=2*x', x=np.zeros((3, 2)), y=np.zeros((3, 2)))
        self.add_subsystem('comp', comp)


class Outer(om.Group):
    def setup(self):
        self.add_subsystem('inner', Inner())
    def configure(self):
        self.promotes('inner', inputs=[('comp.x', 'desvar_x')],
                      src_indices=np.array([0, 1, 0, 1, 0, 1]), flat_src_indices=True, src_shape=2)


class SrcIndicesTestCase(unittest.TestCase):
    def test_one_nesting(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('src', om.ExecComp('y=3*x', x=np.zeros((7)), y=np.zeros((7))))
        model.add_subsystem('outer', Outer())
        model.connect('src.y', 'outer.desvar_x', src_indices=[2, 4], flat_src_indices=True)
        prob.setup()
        srcval = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
        prob.set_val('src.x', srcval)
        prob.run_model()
        assert_near_equal(prob.get_val('outer.desvar_x'), [15., 27.], 1e-6)
        expected = np.array([[15., 27.],
                             [15., 27.],
                             [15., 27.]])
        assert_near_equal(prob.get_val('outer.inner.comp.x'), expected, 1e-6)

    def test_broadcast_scalar_connection(self):
        """
        OpenMDAO should allow promotion to an input that appears to have a certain shape.
        Users should be allowed to connect an arbitrary variable with compatible src_indices to that input.
        """
        p = om.Problem()

        # c1 contains scalar calculations
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('y', val=5.0*np.ones(4,))
        ivc.add_output('x', val=np.arange(4, dtype=float) + 1)

        g1 = p.model.add_subsystem('g1', om.Group())
        # c2 is vectorized calculations
        g1.add_subsystem('c2', om.ExecComp('z = a * y', shape=(4,)))

        # The ultimate source of a and y may be scalar, or have some other arbitrary shape
        g1.promotes('c2', inputs=['a'], src_indices=[0, 0, 0, 0], src_shape=(1,))
        g1.promotes('c2', inputs=['y'], src_indices=[0, 0, 0, 0], src_shape=(4,))

        p.model.connect('ivc.y', 'g1.y')

        # Now connect only a portion of some other output to a, which appears as a scalar input
        p.model.connect('ivc.x', 'g1.a', src_indices=[-1])

        p.setup()

        p.run_model()

        assert_near_equal(p['g1.a'], 4.)
        assert_near_equal(p.model.g1.get_val('a'), 4.)
        assert_near_equal(p['g1.y'], [5., 5., 5., 5.])
        assert_near_equal(p.model.g1.get_val('y'), [5., 5., 5., 5.])
        assert_near_equal(p['g1.c2.y'], [5., 5., 5., 5.])
        assert_near_equal(p['g1.c2.a'], [4., 4., 4., 4.])
        assert_near_equal(p['g1.c2.z'], [20., 20., 20., 20.])

    def test_multiple_inputs_different_src_indices(self):
        """
        When different variables get promoted to the same name, but have different src_indices, this
        should be supported as long as the src_shape of the different promotions is compatible or
        made so by the set_input_defaults call.
        """
        p = om.Problem()

        g1 = p.model.add_subsystem('g1', om.Group(), promotes_inputs=['b'])
        # c1 contains scalar calculations
        g1.add_subsystem('c1', om.ExecComp('y = a0 + b', shape=(1,)),
                         promotes_inputs=[('a0', 'a'), 'b'], promotes_outputs=['y'])
        g2 = g1.add_subsystem('g2', om.Group())
        # c2 is vectorized calculations
        g2.add_subsystem('c2', om.ExecComp('z = a * y', shape=(4,)), promotes_inputs=['a', 'y'])

        g1.promotes('g2', inputs=['y'], src_indices=[0, 0, 0, 0], src_shape=(1,))
        g1.promotes('g2', inputs=['a'], src_indices=[0, 0, 0, 0], src_shape=(1,))

        p.model.promotes('g1', inputs=['a'], src_indices=[0], src_shape=(1,))

        p.setup()

        p['a'] = 99
        p['b'] = 2

        p.run_model()

        assert_near_equal(p['a'], 99.)
        assert_near_equal(p['b'], 2.)
        assert_near_equal(p.model.get_val('a'), 99.)
        assert_near_equal(p.model.get_val('b'), 2.)
        assert_near_equal(p['g1.y'], 101.)
        assert_near_equal(p.model.g1.get_val('y'), 101.)
        assert_near_equal(p.model.g1.get_val('a'), 99.)
        assert_near_equal(p['g1.c1.a0'], 99.)
        assert_near_equal(p['g1.c1.b'], 2.)
        assert_near_equal(p.model.g1.g2.get_val('y'), [101.] * 4)
        assert_near_equal(p.model.g1.g2.get_val('a'), [99.] * 4)
        assert_near_equal(p['g1.g2.c2.a'], [99.] * 4)
        assert_near_equal(p['g1.g2.c2.y'], [101.] * 4)
        assert_near_equal(p['g1.g2.c2.z'], [9999.] * 4)

    def test_src_indices_nested_promotes(self):
        """
        Promoting a variable up multiple levels with different src_indices at each level.
        """
        p = om.Problem()

        g1 = p.model.add_subsystem('g1', om.Group(), promotes_inputs=['b'])
        # c1 contains scalar calculations
        g1.add_subsystem('c1', om.ExecComp('y = a0 + b', shape=(1,)),
                         promotes_inputs=[('a0', 'a'), 'b'], promotes_outputs=['y'])

        g2 = g1.add_subsystem('g2', om.Group())
        # c2 is vectorized calculations
        g2.add_subsystem('c2',  om.ExecComp('z = a * y', shape=(4,)), promotes_inputs=['a', 'y'])

        g1.promotes('g2', inputs=['a'], src_indices=[0, 0, 0, 0], src_shape=(1,))
        g1.promotes('g2', inputs=['y'], src_indices=[0, 0, 0, 0], src_shape=(1,))

        p.model.promotes('g1', inputs=['a'], src_indices=[0], src_shape=(1,))

        p.setup()

        p['a'] = 99
        p['b'] = 2

        p.run_model()

        assert_near_equal(p['g1.y'], 101.)
        assert_near_equal(p['g1.g2.c2.z'], [9999.] * 4)

    def test_mixed_src_indices_no_src_indices(self):
        class C1(om.ExplicitComponent):
            def setup(self):
                self.add_input('diameter', 0.0, units='m')
                self.add_output('z_start', 0.0, units='m')

            def compute(self, inputs, outputs):
                outputs['z_start'] = inputs['diameter'] * 2.


        class C2(om.ExplicitComponent):

            def setup(self):
                self.add_input('diameter', np.zeros(3), units='m')

            def compute(self, inputs, outputs):
                pass

        # C1 has src_indices and C2 doesn't.
        prob = om.Problem()
        prob.model.add_subsystem('C1', C1())
        prob.model.promotes('C1', inputs=['diameter'], src_indices=[0])
        prob.model.add_subsystem('C2', C2(), promotes=['diameter'])  # size 3

        prob.setup()

        prob['diameter'] = np.ones(3) * 1.5

        prob.run_model()

        assert_near_equal(prob['C1.z_start'], 3.)

        assert_near_equal(prob['diameter'], np.ones(3) * 1.5)
        assert_near_equal(prob['C1.diameter'], 1.5)
        assert_near_equal(prob['C2.diameter'], np.ones(3) * 1.5)

    def test_flat_src_inds_2_levels(self):
        class Burn1(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.ExecComp(['y1=x*2'], y1=np.ones(4), x=np.ones(4)),
                                   promotes_outputs=['*'])

                self.add_subsystem('comp2', om.ExecComp(['y2=x*3'], y2=np.ones(4), x=np.ones(4)),
                                   promotes_outputs=['*'])

            def configure(self):
                self.promotes('comp1', inputs=[('x', 'design:x')],
                              src_indices=[0, 0, 0, 0], flat_src_indices=True)

                self.set_input_defaults('design:x', 75.3)


        class Traj(om.Group):
            def setup(self):
                self.add_subsystem('burn1', Burn1(), promotes_outputs=['*'])

            def configure(self):
                self.promotes('burn1', inputs=['design:x'],
                              src_indices=[0, 0, 0, 0], flat_src_indices=True)

        prob = om.Problem(model=Traj())

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['design:x'], [75.3])
        assert_near_equal(prob['y1'], [75.3*2]*4)
        assert_near_equal(prob['y2'], [1*3]*4)

    def test_src_inds_2_subs(self):

        class RHS(om.Group):

            def initialize(self):
                self.options.declare('size', 1)

            def setup(self):
                size = self.options['size']
                self.add_subsystem('comp1', om.ExecComp(['y1=x*2'], y1=np.ones(size), x=np.ones(size)),
                                promotes_inputs=['*'], promotes_outputs=['*'])

                # test with second absolute path for 'x'
                self.add_subsystem('comp2', om.ExecComp(['y2=x*3'], y2=np.ones(size), x=np.ones(size)),
                                promotes_inputs=['*'], promotes_outputs=['*'])


        class Phase(om.Group):
            def setup(self):
                self.add_subsystem('rhs', RHS(size=4))

            def configure(self):
                self.promotes('rhs', inputs=[('x', 'design:x')],
                            src_indices=[0, 0, 0, 0], flat_src_indices=True)

                # this doesn't set the value because it's connected to a non-auto_ivc that
                # already has a value of 1.0 that overrides it.
                self.set_input_defaults('design:x', 75.3)


        class Traj(om.Group):
            def setup(self):
                self.add_subsystem('src', om.ExecComp(['q = b*3'], b=1.5))

                self.add_subsystem('phase', Phase())

                self.connect('src.q', 'phase.design:x')

        prob = om.Problem(model=Traj())

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['phase.design:x'], 4.5)
        assert_near_equal(prob['phase.rhs.y1'], [4.5*2]*4)
        assert_near_equal(prob['phase.rhs.y2'], [4.5*3]*4)

    def test_sub_sub_promotes(self):
        class Burn1(om.Group):
            def setup(self):
                self.add_subsystem('comp1', om.ExecComp(['y1=x*2'], y1=np.ones(4), x=np.ones(4)),
                                promotes_outputs=['*'])
                self.add_subsystem('comp2', om.ExecComp(['y2=x*2'], y2=np.ones(4), x=np.ones(4)),
                                promotes_outputs=['*'])

            def configure(self):
                self.promotes('comp1', inputs=[('x', 'design:x')], src_indices=[0, 0, 0, 0], flat_src_indices=True)
                self.set_input_defaults('design:x', 75.3)

        class Phases(om.ParallelGroup):
            def setup(self):
                self.add_subsystem('burn1', Burn1(), promotes_outputs=['*'])

        class Traj(om.Group):
            def setup(self):
                self.add_subsystem('phases', Phases(), promotes_outputs=['*'])

            def configure(self):
                # this promotes was leaving a leftover entry in _group_inputs that resulted
                # from an earlier _setup_var_data call where the input in question had a different
                # promoted name.
                self.phases.promotes('burn1', inputs=['design:x'])

        prob = om.Problem(model=Traj())
        prob.setup()
        prob.run_model()

        assert_near_equal(prob['phases.design:x'], 75.3)
        assert_near_equal(prob['y1'], [75.3*2]*4)

    def test_src_shape_mismatch(self):
        p = om.Problem(name='src_shape_mismatch')
        G = p.model.add_subsystem('G', om.Group(), promotes_inputs=['x'])

        G.set_input_defaults('x', src_shape=(3,2))

        g1 = G.add_subsystem('g1', om.Group(), promotes_inputs=['x'])
        g1.add_subsystem('C1', om.ExecComp('y = 3*x', shape=3))

        g1.promotes('C1', inputs=['x'], src_indices=om.slicer[:, 1], src_shape=(3,3))

        g2 = G.add_subsystem('g2', om.Group(), promotes_inputs=['x'])
        g2.add_subsystem('C2', om.ExecComp('y = 2*x', shape=2))

        g2.promotes('C2', inputs=['x'], src_indices=[1,5], src_shape=(3,2), flat_src_indices=True)

        with self.assertRaises(Exception) as cm:
            p.setup()
            p.final_setup()

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'src_shape_mismatch':"
           "\n   <model> <class Group>: When connecting '_auto_ivc.v0' to 'G.g1.C1.x': Promoted "
           "src_shape of (3, 3) for 'G.g1.C1.x' differs from src_shape (3, 2) for 'x'.")

    def test_src_indices_on_promotes(self):
        src_shape = (3, 3)
        tgt_shape = (2, 2)
        src_indices = [4, 5, 7, 9]
        flat_src_indices = True

        class MyComp(om.ExplicitComponent):
            def __init__(self, input_shape):
                super().__init__()
                self._input_shape = input_shape
            def setup(self):
                self.add_input('x', val=np.zeros(self._input_shape))
                self.add_output('y', val=np.zeros(self._input_shape))
            def compute(self, inputs, outputs):
                outputs['y'] = 2.0 * inputs['x']

        p = om.Problem(name='src_indices_on_promotes')
        p.model.add_subsystem('indeps', om.IndepVarComp('x', shape=src_shape))
        p.model.add_subsystem('C1', MyComp(tgt_shape))
        p.model.promotes('C1', any=['x'],
                            src_indices=src_indices,
                            flat_src_indices=flat_src_indices)
        p.model.set_input_defaults('x', src_shape=src_shape)
        with self.assertRaises(Exception) as cm:
            p.setup()
            p.final_setup()

        self.assertEqual(cm.exception.args[0],
            "\nCollected errors for problem 'src_indices_on_promotes':"
            "\n   <model> <class Group>: When promoting 'x' from system 'C1' with src_indices "
            "[4 5 7 9] and src_shape (3, 3): index 9 is out of bounds for source dimension of size 9."
            "\n   <model> <class Group>: The source indices [4 5 7 9] do not specify a valid shape "
            "for the connection '_auto_ivc.v0' to 'C1.x'. (target shape=(2, 2), indices_shape=(4,)):"
            " index 9 is out of bounds for axis 0 with size 9")

    def test_connect_slice_src_indices_not_full_size(self):
        p = om.Problem()

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp())
        M_np = np.arange(9)[::-1].reshape((3, 3))
        ivc.add_output('M', val=M_np)

        exec = p.model.add_subsystem('exec', om.ExecComp())

        exec.add_expr('A = B', A={'shape': (3,)}, B={'shape': (3,)})

        p.model.connect('ivc.M', 'exec.B', src_indices=om.slicer[:1])

        p.setup()

        p.run_model()

        assert_near_equal(p.get_val('exec.B').ravel(), p.get_val('ivc.M')[:1].ravel())
        assert_near_equal(p.get_val('exec.A').ravel(), p.get_val('ivc.M')[:1].ravel())

    def test_promote_slice_src_indices_not_full_size(self):
        p = om.Problem()

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp())
        M_np = np.arange(9)[::-1].reshape((3, 3))
        ivc.add_output('M', val=M_np)

        exec = p.model.add_subsystem('exec', om.ExecComp())

        exec.add_expr('A = B', A={'shape': (3,)}, B={'shape': (3,)})

        p.model.promotes('ivc', outputs=['M'])
        p.model.promotes('exec', inputs=[('B', 'M')], src_indices=om.slicer[:1], src_shape=(3, 3))

        p.setup()

        p.run_model()

        assert_near_equal(p.get_val('exec.B').ravel(), p.get_val('ivc.M')[:1].ravel())
        assert_near_equal(p.get_val('exec.A').ravel(), p.get_val('ivc.M')[:1].ravel())

class SrcIndicesFeatureTestCase(unittest.TestCase):
    def test_multi_promotes(self):

        p = om.Problem()
        G = p.model.add_subsystem('G', om.Group())

        # At the top level, we assume that the source, and our input 'x', has a shape of (3,3),
        # and after we slice it with [:,:-1], lower levels will see their source having a shape of (3,2)
        p.model.promotes('G', inputs=['x'], src_indices=om.slicer[:,:-1], src_shape=(3,3))

        # This specifies that G.x assumes a source shape of (3,2)
        G.set_input_defaults('x', src_shape=(3,2))

        g1 = G.add_subsystem('g1', om.Group(), promotes_inputs=['x'])
        g1.add_subsystem('C1', om.ExecComp('y = 3*x', x=np.random.random(3), y=np.random.random(3)))

        # C1.x has a shape of 3, so we apply a slice of [:, 1] to our source which has a shape
        # of (3,2) to give us our final shape of 3.
        g1.promotes('C1', inputs=['x'], src_indices=om.slicer[:, 1], src_shape=(3,2))

        g2 = G.add_subsystem('g2', om.Group(), promotes_inputs=['x'])
        g2.add_subsystem('C2', om.ExecComp('y = 2*x', shape=2))

        # C2.x has a shape of 2, so we apply flat source indices of [1,5] to our source which has
        # a shape of (3,2) to give us our final shape of 2.
        g2.promotes('C2', inputs=['x'], src_indices=[1,5], flat_src_indices=True, src_shape=(3,2))

        p.setup()

        inp = np.arange(9).reshape((3,3)) + 1.

        # 'x' at the top level has a shape of (3,3) because no src_indices have been applied yet
        p.set_val('x', inp)
        p.run_model()

        assert_near_equal(p['x'], inp)
        assert_near_equal(p.model.G.get_val('x'), inp[:, :-1])
        assert_near_equal(p['G.g1.C1.y'], inp[:, :-1][:, 1]*3.)
        assert_near_equal(p['G.g2.C2.y'], inp[:, :-1].flatten()[[1,5]]*2.)


@use_tempdirs
class SrcIndicesMPITestCase(unittest.TestCase):
    N_PROCS = 2

    def test_multi_promotes_mpi(self):
        p = om.Problem()
        par = p.model.add_subsystem('par', om.ParallelGroup())
        g1 = par.add_subsystem('g1', om.Group(), promotes_inputs=['x'])
        g2 = par.add_subsystem('g2', om.Group(), promotes_inputs=['x'])
        g1.add_subsystem('C1', om.ExecComp('y = 3*x', shape=3))
        g2.add_subsystem('C2', om.ExecComp('y = 2*x', shape=2))
        g1.promotes('C1', inputs=['x'], src_indices=om.slicer[:, 1], src_shape=(3,2))
        g2.promotes('C2', inputs=['x'], src_indices=[1,5], src_shape=(3,2), flat_src_indices=True)

        # we want the connection to x to have a shape of (3,2), which differs from the
        # shapes of either of the connected absolute inputs.
        par.set_input_defaults('x', src_shape=(3,2))

        # we want the auto_ivc output to have a shape of (3,3)
        p.model.promotes('par', inputs=['x'], src_indices=om.slicer[:,:-1], src_shape=(3,3))

        p.setup()

        commsize = p.comm.size
        inp = np.random.random((3,3))
        if commsize > 1:
            if p.comm.rank == 0:
                p.comm.bcast(inp, root=0)
            else:
                inp = p.comm.bcast(None, root=0)

        reduced_inp = inp[:, :-1]

        p.set_val('x', inp)
        p.run_model()

        if commsize == 1 or p.comm.rank == 0:
            assert_near_equal(p['par.g1.C1.y'], reduced_inp[:, 1]*3.)
        elif commsize == 1 or p.comm.rank == 1:
            assert_near_equal(p['par.g2.C2.y'], reduced_inp.flatten()[[1,5]]*2.)


class SrcIndicesSerialMultipoint2(unittest.TestCase):

    def test_multipoint2(self):
        p = om.Problem()
        par = p.model.add_subsystem('par', om.ParallelGroup())

        g1 = par.add_subsystem('g1', om.Group(), promotes_inputs=['x'])
        g1.add_subsystem('C1', om.ExecComp('y = 3*x', shape=1))
        g1.promotes('C1', inputs=['x'], src_indices=[0], src_shape=(2,))

        g2 = par.add_subsystem('g2', om.Group(), promotes_inputs=['x'])
        g2.add_subsystem('C2', om.ExecComp('y = 2*x', shape=1))
        g2.promotes('C2', inputs=['x'], src_indices=[1], src_shape=(2,))

        p.model.set_input_defaults('par.x', val=[7., -5.])

        p.setup()
        p.run_model()


@use_tempdirs
class Multipoint2TooManyProcs(SrcIndicesSerialMultipoint2):
    N_PROCS = 3


class SrcIndicesSerialMultipoint3(unittest.TestCase):

    def test_multipoint2(self):
        p = om.Problem()
        par = p.model.add_subsystem('par', om.ParallelGroup())

        g1 = par.add_subsystem('g1', om.Group(), promotes_inputs=['x'])
        g1.add_subsystem('C1', om.ExecComp('y = 3*x', shape=1))
        g1.promotes('C1', inputs=['x'], src_indices=[0], src_shape=(3,))

        g2 = par.add_subsystem('g2', om.Group(), promotes_inputs=['x'])
        g2.add_subsystem('C2', om.ExecComp('y = 2*x', shape=1))
        g2.promotes('C2', inputs=['x'], src_indices=[1], src_shape=(3,))

        g3 = par.add_subsystem('g3', om.Group(), promotes_inputs=['x'])
        g3.add_subsystem('C3', om.ExecComp('y = 5*x', shape=1))
        g3.promotes('C3', inputs=['x'], src_indices=[2], src_shape=(3,))

        p.model.set_input_defaults('par.x', val=[7., -5., 2.])

        p.setup()
        p.run_model()

        assert_check_totals(p.check_totals(of=['par.g1.C1.y', 'par.g2.C2.y', 'par.g3.C3.y'], wrt=['par.x']))


@use_tempdirs
class Multipoint3SameProcs(SrcIndicesSerialMultipoint3):
    N_PROCS = 3


@use_tempdirs
class Multipoint3TooManyProcs(SrcIndicesSerialMultipoint3):
    N_PROCS = 4

@use_tempdirs
class DoubleNestedParallelMultipointTestCase(unittest.TestCase):
    N_PROCS = 6

    def test_nested_pars(self):
        p = om.Problem()
        par = p.model.add_subsystem('par', om.ParallelGroup(), promotes_inputs=['x'])

        G1 = par.add_subsystem('G1', om.Group(), promotes_inputs=['x'])
        G1p = G1.add_subsystem('G1p', om.ParallelGroup(), promotes_inputs=['x'])
        G1p.add_subsystem('C1_1', om.ExecComp('y = 3.0*x'))
        G1p.add_subsystem('C1_2', om.ExecComp('y = -3.0*x'))
        G1p.promotes('C1_1', inputs=['x'], src_indices=[0])
        G1p.promotes('C1_2', inputs=['x'], src_indices=[0])

        G2 = par.add_subsystem('G2', om.Group(), promotes_inputs=['x'])
        G2p = G2.add_subsystem('G2p', om.ParallelGroup(), promotes_inputs=['x'])
        G2p.add_subsystem('C2_1', om.ExecComp('y = 5.0*x'))
        G2p.add_subsystem('C2_2', om.ExecComp('y = -5.0*x'))
        G2p.promotes('C2_1', inputs=['x'], src_indices=[1])
        G2p.promotes('C2_2', inputs=['x'], src_indices=[1])

        G3 = par.add_subsystem('G3', om.Group(), promotes_inputs=['x'], max_procs=1)  # no nested parallel group here
        G3s = G3.add_subsystem('G3s', om.Group(), promotes_inputs=['x'])
        G3s.add_subsystem('C3_1', om.ExecComp('y = 7.0*x'))
        G3s.add_subsystem('C3_2', om.ExecComp('y = -7.0*x'))
        G3s.promotes('C3_1', inputs=['x'], src_indices=[2])
        G3s.promotes('C3_2', inputs=['x'], src_indices=[2])

        par.set_input_defaults('x', val=[.5, 1.5, 2.5])

        p.setup()
        p.run_model()

        assert_check_totals(p.check_totals(of=['par.G1.G1p.C1_1.y', 'par.G1.G1p.C1_2.y',
                                               'par.G2.G2p.C2_1.y', 'par.G2.G2p.C2_2.y',
                                               'par.G3.G3s.C3_1.y', 'par.G3.G3s.C3_2.y'], wrt=['x'],
                                           show_only_incorrect=True))


class TestNestedInputDefaults(unittest.TestCase):
    N_PROCS = 3

    def test_nested(self):
        # this was failing as reported in github issue #2441
        # It passes if the two setup() calls don't raise any exceptions.

        class Grp(om.Group):
            def setup(self):
                self.add_subsystem("lin", om.ExecComp("y = x" ), promotes_inputs=["x"])
                self.add_subsystem("quad", om.ExecComp("z = x*x"), promotes_inputs=["x"])
                # self.set_input_defaults("x", 0.85)

        class Vec(om.Group):
            def initialize(self):
                self.options.declare("num", default=1)
            def setup(self):
                n = self.options["num"]

                self.set_input_defaults("x", np.full(n, 0.85))

                for node in range(n):
                    name = f"comp_{node}"
                    self.add_subsystem(name, Grp())
                    self.promotes(name, inputs=["x"], src_indices=[node])

        # This one seems to require that set_input_defaults (on line 8) is NOT called in Comp.setup()
        n = 3
        p = om.Problem(Vec(num=n))
        p.setup()

        # This one seems to require that set_input_defaults (on line 8) IS called in Comp.setup()
        p2 = om.Problem(Grp())
        p2.setup()


if __name__ == '__main__':
    unittest.main()

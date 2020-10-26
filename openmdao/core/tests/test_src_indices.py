import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class Inner(om.Group):
    def setup(self):
        comp = om.ExecComp('y=2*x', x=np.zeros((3, 2)), y=np.zeros((3, 2)))
        self.add_subsystem('comp', comp)


class Outer(om.Group):
    def setup(self):
        self.add_subsystem('inner', Inner())
    def configure(self):
        self.promotes('inner', inputs=[('comp.x', 'desvar_x')],
                      src_indices=np.array([[0, 1], [0, 1], [0, 1]]), flat_src_indices=True, src_shape=2)


class SrcIndicesTestCase(unittest.TestCase):
    def test_one_nesting(self):
        prob = om.Problem()
        model = prob.model
        comp = model.add_subsystem('src', om.ExecComp('y=3*x', x=np.zeros((7)), y=np.zeros((7))))
        model.add_subsystem('outer', Outer())
        model.connect('src.y', 'outer.desvar_x', src_indices=[2, 4], flat_src_indices=True)
        prob.setup()
        srcval = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
        prob.set_val('src.x', srcval)
        prob.run_model()
        assert_near_equal(prob.get_val('outer.desvar_x'), srcval * 3, 1e-6)
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
        c2 = g1.add_subsystem('c2', om.ExecComp('z = a * y', shape=(4,)))

        # The ultimate source of a and y may be scalar, or have some other arbitrary shape
        g1.promotes('c2', inputs=['a', 'y'], src_indices=[0, 0, 0, 0], src_shape=(1,))

        p.model.connect('ivc.y', 'g1.y')

        # Now connect only a portion of some other output to a, which appears as a scalar input
        # (This currently breaks because we're specifying the src_indices of an input twice.)
        p.model.connect('ivc.x', 'g1.a', src_indices=[-1])

        p.setup()

        p.run_model()

        assert_near_equal(p['g1.c2.z'], [20.] * 4)

    def test_multiple_inputs_different_src_indices(self):
        """
        When different variables get promoted to the same name, but have different src_indices, this
        should be supported as long as the src_shape of the different promotions is compatible or
        made so by the set_input_defaults call.
        """
        p = om.Problem()

        g1 = p.model.add_subsystem('g1', om.Group(), promotes_inputs=['b'])
        # c1 contains scalar calculations
        c1 = g1.add_subsystem('c1', om.ExecComp('y = a0 + b', shape=(1,)),
                              promotes_inputs=[('a0', 'a'), 'b'], promotes_outputs=['y'])
        g2 = g1.add_subsystem('g2', om.Group())
        # c2 is vectorized calculations
        c2 = g2.add_subsystem('c2', om.ExecComp('z = a * y', shape=(4,)), promotes_inputs=['a', 'y'])

        g1.promotes('g2', inputs=['y'], src_indices=[0, 0, 0, 0], src_shape=(1,))
        g1.promotes('g2', inputs=['a'], src_indices=[0, 0, 0, 0], src_shape=(1,))

        p.model.promotes('g1', inputs=['a'], src_indices=[0], src_shape=(1,))

        p.setup()

        p['a'] = 99
        p['b'] = 2

        p.run_model()

        assert_near_equal(p['g1.y'], 101.)
        assert_near_equal(p['g1.g2.c2.z'], [9999.] * 4)

    def test_src_indices_nested_promotes(self):
        """
        Promoting a variable up multiple levels with different src_indices at each level.
        """
        p = om.Problem()

        g1 = p.model.add_subsystem('g1', om.Group(), promotes_inputs=['b'])
        # c1 contains scalar calculations
        c1 = g1.add_subsystem('c1', om.ExecComp('y = a0 + b', shape=(1,)),
                              promotes_inputs=[('a0', 'a'), 'b'], promotes_outputs=['y'])

        g2 = g1.add_subsystem('g2', om.Group())
        # c2 is vectorized calculations
        c2 = g2.add_subsystem('c2',  om.ExecComp('z = a * y', shape=(4,)), promotes_inputs=['a', 'y'])

        g1.promotes('g2', inputs=['a'], src_indices=[0, 0, 0, 0], src_shape=(4,))
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
                self.add_input('diameter', 0.0, units='m', src_indices=[0])
                self.add_output('z_start', 0.0, units='m')

            def compute(self, inputs, outputs):
                outputs['z_start'] = inputs['diameter'] * 2.


        class C2(om.ExplicitComponent):

            def setup(self):
                self.add_input('diameter', np.zeros(3), units='m')

            def compute(self, inputs, outputs):
                pass

        # this test passes if setup doesn't raise an exception.
        # C1 has src_indices and C2 doesn't.
        prob = om.Problem()
        prob.model.add_subsystem('C1', C1(), promotes=['diameter'])
        prob.model.add_subsystem('C2', C2(), promotes=['diameter'])

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
                self.add_subsystem('burn1', Burn1(),
                                promotes_outputs=['*'])

            def configure(self):
                self.promotes('burn1', inputs=['design:x'],
                            src_indices=[0, 0, 0, 0], flat_src_indices=True)

        prob = om.Problem(model=Traj())

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['design:x'], 75.3)
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


class SrcIndicesMPITestCase(unittest.TestCase):
    N_PROCS = 2

    def test_multi_promotes_mpi(self):
        p = om.Problem()
        par = p.model.add_subsystem('par', om.ParallelGroup())
        g1 = par.add_subsystem('g1', om.Group())
        g2 = par.add_subsystem('g2', om.Group())
        g1.add_subsystem('C1', om.ExecComp('y = 3*x', shape=3))
        g2.add_subsystem('C2', om.ExecComp('y = 2*x', shape=2))
        g1.promotes('C1', inputs=['x'], src_indices=om.slicer[:, 1], src_shape=(3,2), flat_src_indices=True)
        g2.promotes('C2', inputs=['x'], src_indices=[1,5], src_shape=(3,2), flat_src_indices=True)
        par.promotes('g1', inputs=['x'])
        par.promotes('g2', inputs=['x'])

        # we want the connection to x to have a shape of (3,2), which differs from the
        # shapes of either of the connected absolute inputs.
        par.set_input_defaults('x', src_shape=(3,2))

        # we want the auto_ivc output to have a shape of (3,3)
        p.model.promotes('par', inputs=['x'], src_indices=om.slicer[:,:-1], src_shape=(3,3))

        p.setup()

        inp = np.random.random((3,3))
        if p.comm.size > 1:
            if p.comm.rank == 0:
                p.comm.bcast(inp, root=0)
            else:
                inp = p.comm.bcast(None, root=0)

        reduced_inp = inp[:, :-1]

        p.set_val('_auto_ivc.v0', inp)
        p.run_model()

        if p.comm.rank == 0:
            assert_near_equal(p['par.g1.C1.y'], reduced_inp[:, 1]*3.)
        else:
            assert_near_equal(p['par.g2.C2.y'], reduced_inp.flatten()[[1,5]]*2.)


if __name__ == '__main__':
    unittest.main()
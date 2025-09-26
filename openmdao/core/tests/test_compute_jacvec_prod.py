import unittest

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


def get_comp(size):
    return om.ExecComp('out = sum(3*x**2) + inp', x=np.ones(size - 1), inp=0.0, out=0.0)


class SubProbComp(om.ExplicitComponent):
    """
    This component contains a sub-Problem with a component that will be solved over num_nodes
    points instead of creating num_nodes instances of that same component and connecting them
    together.
    """
    def __init__(self, input_size, num_nodes, mode, **kwargs):
        super().__init__(**kwargs)
        self.prob = None
        self.size = input_size
        self.num_nodes = num_nodes
        self.mode = mode

    def _setup_subprob(self):
        self.prob = p = om.Problem(comm=self.comm)
        model = self.prob.model

        model.add_subsystem('comp', get_comp(self.size))

        p.setup()
        p.final_setup()

    def setup(self):
        self._setup_subprob()

        self.add_input('x', np.zeros(self.size - 1))
        self.add_input('inp', val=0.0)
        self.add_output('out', val=0.0)
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        p = self.prob
        p['comp.x'] = inputs['x']
        p['comp.inp'] = inputs['inp']
        inp = inputs['inp']
        for i in range(self.num_nodes):
            p['comp.inp'] = inp
            p.run_model()
            inp = p['comp.out']

        outputs['out'] = p['comp.out']

    def _compute_partials_fwd(self, inputs, partials):
        p = self.prob
        x = inputs['x']
        p['comp.x'] = x
        p['comp.inp'] = inputs['inp']

        seed = {'comp.x':np.zeros(x.size), 'comp.inp': np.zeros(1)}
        p.run_model()
        p.model._linearize(None)
        for rhsname in seed:
            for rhs_i in range(seed[rhsname].size):
                seed['comp.x'][:] = 0.0
                seed['comp.inp'][:] = 0.0
                seed[rhsname][rhs_i] = 1.0
                for i in range(self.num_nodes):
                    p.model._vectors['output']['linear'].set_val(0.0)
                    p.model._vectors['residual']['linear'].set_val(0.0)
                    jvp = p.compute_jacvec_product(of=['comp.out'], wrt=['comp.x','comp.inp'], mode='fwd', seed=seed)
                    seed['comp.inp'][:] = jvp['comp.out']

                if rhsname == 'comp.x':
                    partials[self.pathname + '.out', self.pathname +'.x'][0, rhs_i] = jvp[self.pathname + '.out'].item()
                else:
                    partials[self.pathname + '.out', self.pathname + '.inp'][0, 0] = jvp[self.pathname + '.out'].item()

    def _compute_partials_rev(self, inputs, partials):
        p = self.prob
        p['comp.x'] = inputs['x']
        p['comp.inp'] = inputs['inp']
        seed = {'comp.out': np.ones(1)}

        stack = []
        comp = p.model.comp
        comp._inputs['inp'] = inputs['inp']
        # store the inputs to each comp (the comp at each node point) by doing nonlinear solves
        # and storing what the inputs are for each node point.  We'll set these inputs back
        # later when we linearize about each node point.
        for i in range(self.num_nodes):
            stack.append(comp._inputs['inp'][0])
            comp._inputs['x'] = inputs['x']
            comp._solve_nonlinear()
            comp._inputs['inp'] = comp._outputs['out']

        for i in range(self.num_nodes):
            p.model._vectors['output']['linear'].set_val(0.0)
            p.model._vectors['residual']['linear'].set_val(0.0)
            comp._inputs['inp'] = stack.pop()
            comp._inputs['x'] = inputs['x']
            p.model._linearize(None)
            jvp = p.compute_jacvec_product(of=['comp.out'], wrt=['comp.x','comp.inp'], mode='rev', seed=seed)
            seed['comp.out'][:] = jvp['comp.inp']

            # all of the comp.x's are connected to the same indepvarcomp, so we have
            # to accumulate their contributions together
            partials[self.pathname + '.out', self.pathname + '.x'] += jvp['comp.x']

            # this one doesn't get accumulated because each comp.inp contributes to the
            # previous comp's .out (or to comp.inp in the case of the first comp) only.
            # Note that we have to handle this explicitly here because normally in OpenMDAO
            # we accumulate derivatives when we do reverse transfers.  We can't do that
            # here because we only have one instance of our component, so instead of
            # accumulating into separate 'comp.out' variables for each comp instance,
            # we would be accumulating into a single comp.out variable, which would make
            # our derivative too big.
            partials[self.pathname + '.out', self.pathname + '.inp'] = jvp['comp.inp']

    def compute_partials(self, inputs, partials):
        # note that typically you would only have to define partials for one direction,
        # either fwd OR rev, not both.
        if self.mode == 'fwd':
            self._compute_partials_fwd(inputs, partials)
        else:
            self._compute_partials_rev(inputs, partials)


class TestPComputeJacvecProd(unittest.TestCase):

    def _build_om_model(self, size):
        p = om.Problem()
        model = p.model
        comp = model.add_subsystem('comp', om.IndepVarComp('x', val=np.zeros(size - 1)))
        comp.add_output('inp', val=0.0)

        model.add_subsystem('C1', get_comp(size))
        model.add_subsystem('C2', get_comp(size))
        model.add_subsystem('C3', get_comp(size))

        model.connect('comp.x', ['C1.x', 'C2.x', 'C3.x'])
        model.connect('comp.inp', 'C1.inp')
        model.connect('C1.out', 'C2.inp')
        model.connect('C2.out', 'C3.inp')

        return p

    def _build_cjv_model(self, size, mode):
        p = om.Problem()

        p.model.add_subsystem('comp', SubProbComp(input_size=size, num_nodes=3, mode=mode))

        p.setup(mode=mode)

        return p

    def test_fwd(self):
        size = 5
        p = self._build_om_model(size)
        p.setup(mode='fwd')
        p['comp.x'] = np.arange(size-1, dtype=float) + 1. #np.random.random(size - 1)
        p['comp.inp'] = np.array([7.])  #np.random.random(1)[0]
        p.final_setup()

        p2 = self._build_cjv_model(size, 'fwd')

        p2['comp.x'] = p['comp.x']
        p2['comp.inp'] = p['comp.inp']

        p2.run_model()
        J2 = p2.compute_totals(of=['comp.out'], wrt=['comp.x', 'comp.inp'], return_format='array')

        p.run_model()
        J = p.compute_totals(of=['C3.out'], wrt=['comp.x', 'comp.inp'], return_format='array')

        self.assertEqual(p['C3.out'], p2['comp.out'])
        np.testing.assert_allclose(J2, J)


    def test_rev(self):
        size = 5
        p = self._build_om_model(size)
        p.setup(mode='rev')
        p['comp.x'] = np.arange(size-1, dtype=float) + 1. #np.random.random(size - 1)
        p['comp.inp'] = np.array([7.])  #np.random.random(1)[0]
        p.final_setup()

        p2 = self._build_cjv_model(size, 'rev')

        p2['comp.x'] = p['comp.x']
        p2['comp.inp'] = p['comp.inp']

        p.run_model()
        J = p.compute_totals(of=['C3.out'], wrt=['comp.x', 'comp.inp'], return_format='array')

        p2.run_model()
        self.assertEqual(p['C3.out'], p2['comp.out'])

        J2 = p2.compute_totals(of=['comp.out'], wrt=['comp.x', 'comp.inp'], return_format='array')

        np.testing.assert_allclose(J2, J)


@unittest.skipIf(jnp is None, "jax is not installed")
class TestProbComputeJacvecProdPromoted(unittest.TestCase):

    def test_promoted_names(self):
        class ObjComp(om.JaxExplicitComponent):

            def setup(self):
                self.add_input('Θ', shape_by_conn=True)
                self.add_input('p', shape_by_conn=True)
                self.add_output('f', shape=(1,))

            def compute_primal(self, Θ, p):
                f = (Θ[0] - p[0])**2 + Θ[0] * Θ[1] + (Θ[1] + p[1])**2 - p[2]
                return jnp.array([f])

        class ConComp(om.JaxExplicitComponent):

            def setup(self):
                self.add_input('Θ', shape_by_conn=True)
                self.add_input('p', shape_by_conn=True)
                self.add_output('g', shape=(1,))

            def compute_primal(self, Θ, p):
                g = 2*Θ[0] + Θ[1]
                return jnp.array([g])

        prob = om.Problem()
        prob.model.add_subsystem('f_comp', ObjComp(), promotes_inputs=['*'], promotes_outputs=['*'])
        prob.model.add_subsystem('g_comp', ConComp(), promotes_inputs=['*'], promotes_outputs=['*'])

        prob.model.add_design_var('Θ', upper=[6., None])
        prob.model.add_constraint('g', equals=0.)
        prob.model.add_objective('f')

        prob.driver = om.ScipyOptimizeDriver()

        prob.setup()

        # Set parameter values
        prob.set_val('p', np.array([3.0, 4.0, 3.0]))

        # Initial guess
        prob.set_val('Θ', np.array([1.0, 1.0]))

        prob.run_model()

        dg_dΘ = prob.compute_totals(of='g', wrt='Θ', return_format='array')

        # The constraint jac should be [2, 1]
        jvp0 = prob.compute_jacvec_product(of=['g'],
                                           wrt=['Θ'],
                                           mode='fwd',
                                           seed={'Θ': np.array([1.0, 0.0])},
                                           linearize=True)

        jvp1 = prob.compute_jacvec_product(of=['g'],
                                           wrt=['Θ'],
                                           mode='fwd',
                                           seed={'Θ': np.array([0.0, 1.0])},
                                           linearize=False)

        assert_near_equal(jvp0['g'], dg_dΘ @ np.array([1.0, 0.0]))
        assert_near_equal(jvp1['g'], dg_dΘ @ np.array([0.0, 1.0]))

        # Reverse mode
        jvp0 = prob.compute_jacvec_product(of=['g'],
                                           wrt=['Θ'],
                                           mode='rev',
                                           seed={'g': np.array([1.0])},
                                           linearize=True)
        # Should yield a row of dg_dΘ (the entire jacobian in this case)
        assert_near_equal(jvp0['Θ'], dg_dΘ.ravel())

import unittest
import sys
import itertools

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, \
    assert_check_totals, assert_sparsity_matches_fd
import openmdao.api as om

from openmdao.utils.jax_utils import jax, jnp
from openmdao.utils.testing_utils import parameterized_name

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized


class DotProd(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))

        self.declare_partials(of='z', wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])


class DotProdMult(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])
        outputs['zz'] = inputs['y'] * 2.5


class DotProdMultPrimalNoDeclPartials(om.JaxExplicitComponent):
    def initialize(self):
        self.options['default_to_dyn_shapes'] = True

    def setup(self):
        self.add_input('x')
        self.add_input('y')
        self.add_output('z')
        self.add_output('zz')

    def compute_primal(self, x, y):
        z = jnp.dot(x, y)
        zz = y * 2.5
        return z, zz


class DotProdMultPrimal(DotProdMultPrimalNoDeclPartials):
    def setup(self):
        super().setup()

    def setup_partials(self):
        self.declare_partials(of=['z'], wrt=['x', 'y'])
        self.declare_partials(of=['zz'], wrt=['y'])


class DotProdMultPrimalOption(om.JaxExplicitComponent):
    def __init__(self, stat=2., **kwargs):
        super().__init__(**kwargs)
        self.stat = stat

    def initialize(self):
        self.options.declare('mult', default=1.0, types=float, allow_none=False)

    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

    def get_self_statics(self):
        return (self.options['mult'], self.stat)

    def compute_primal(self, x, y):
        z = jnp.dot(x, y)
        zz = y * self.options['mult'] * self.stat
        return z, zz


class DotProductMultDiscretePrimal(om.JaxExplicitComponent):
    def __init__(self, xshape=None, yshape=None, **kwargs):
        super().__init__(**kwargs)
        self.xshape = xshape
        self.yshape = yshape

    def setup(self):
        if self.xshape is None:
            self.add_input('x', shape_by_conn=True)
        else:
            self.add_input('x', shape=self.xshape)

        if self.yshape is None:
            self.add_input('y', shape_by_conn=True)
        else:
            self.add_input('y', shape=self.yshape)

        self.add_discrete_input('disc_in', val=2)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')
        self.add_discrete_output('disc_out', val=3)

    def compute_primal(self, x, y, disc_in):
        def pos(x, y):
            return jnp.dot(x, y), y * 3.0

        def neg(x, y):
            return -jnp.dot(x, y), y * 2.5

        disc_out = -disc_in
        z, zz = jax.lax.cond(disc_in >= 0, pos, neg, x, y)

        return (z, zz, disc_out)


class BadShape(om.JaxExplicitComponent):
    def setup(self):
        self.add_input('x', shape=(2,3))
        self.add_input('y', shape=(3,4))
        self.add_output('z', shape=(2,3))
        self.add_output('zz', shape=())

        self.add_discrete_input('disc_in', val=2)
        self.add_discrete_output('disc_out', val=3)

    def compute_primal(self, x, y, disc_in):
        def pos(x, y):
            return jnp.dot(x, y), y * 3.0

        def neg(x, y):
            return -jnp.dot(x, y), y * 2.5

        disc_out = -disc_in
        z, zz = jax.lax.cond(disc_in >= 0, pos, neg, x, y)

        return (z, zz, disc_out)


x_shape = (2, 3)
y_shape = (3, 4)
method_dict = {'fd': 'cs', 'cs': 'fd', 'jax': 'fd'}


@unittest.skipIf(jax is None, 'jax is not available')
class TestJaxComp(unittest.TestCase):
    @parameterized.expand(itertools.product(['fwd', 'rev']), name_func=parameterized_name)
    def test_shape_check(self, mode):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones((2,3))))
        ivc.add_output('y', val=np.ones((3,4)))
        ivc.add_discrete_output('disc_out', val=3)
        comp = p.model.add_subsystem('comp', BadShape())

        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')
        p.model.connect('ivc.disc_out', 'comp.disc_in')

        p.setup(mode=mode)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()

        with self.assertRaises(Exception) as cm:
            p.run_model()

        msg = ("'comp' <class BadShape>:"
               "\n   Shape mismatch for output 'z': expected (2, 3) but got (2, 4)."
               "\n   Shape mismatch for output 'zz': expected () but got (3, 4).")
        self.assertEqual(cm.exception.args[0], msg)

    @parameterized.expand(itertools.product(['fwd', 'rev'], ['matfree', 'jac'], ['jax', 'fd'],
                                            ['coloring', 'nocoloring']),
                          name_func=parameterized_name)
    def test_jax_explicit_comp2primal(self, mode, matrix_free, derivs_method, slvtype):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', DotProdMultPrimal(derivs_method=derivs_method))
        p.model.add_subsystem('objcomp', om.ExecComp('y=x+1.'))
        if derivs_method == 'jax':
            comp.matrix_free = matrix_free == 'matfree'
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')
        p.model.connect('comp.zz', 'objcomp.x', src_indices=[0], flat_src_indices=True)
        if slvtype == 'coloring':
            comp.declare_coloring()
            p.model.add_constraint('comp.zz', lower=0., upper=1000.)
            p.model.add_constraint('comp.z', lower=0., upper=1000.)
            p.model.add_design_var('ivc.x', lower=0., upper=1000.)
            p.model.add_design_var('ivc.y', lower=0., upper=1000.)
            p.model.add_objective('objcomp.y')
            p.driver = om.ScipyOptimizeDriver(maxiter=2)
            p.driver.declare_coloring()

        p.setup(mode=mode, force_alloc_complex=True)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        method = method_dict[derivs_method]

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method=method, show_only_incorrect=True))
        assert_check_partials(p.check_partials(method=method, show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)

        if slvtype == 'coloring':
            p.run_driver()
            coloring = p.driver._coloring_info.coloring
            self.assertTrue(coloring is not None, 'coloring is None')
            self.assertEqual(coloring.total_solves(), 6 if mode=='fwd' else 4)

    @parameterized.expand(itertools.product(['fwd', 'rev'], ['jax','fd', 'cs'],
                                            ['coloring', 'nocoloring']), name_func=parameterized_name)
    def test_jax_explicit_comp2primal_nodecl(self, mode, derivs_method, slvtype):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', DotProdMultPrimalNoDeclPartials(derivs_method=derivs_method))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        if slvtype == 'coloring':
            comp.declare_coloring()
            p.model.add_constraint('comp.zz', lower=0.)
            p.model.add_constraint('comp.z', lower=0.)
            p.model.add_design_var('ivc.x', lower=0.)
            p.model.add_design_var('ivc.y', lower=0.)
            p.driver.declare_coloring()

        p.setup(mode=mode, force_alloc_complex=True)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        method = method_dict[derivs_method]

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['ivc.x', 'ivc.y'],
                                           method=method, show_only_incorrect=True))
        assert_check_partials(p.check_partials(method=method, show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)

    @parameterized.expand(itertools.product(['fwd', 'rev'], ['jax','fd', 'cs'],
                                            ['coloring', 'nocoloring']), name_func=parameterized_name)
    def test_jax_explicit_comp2primal_nodecl_shape_by_conn(self, mode, derivs_method, slvtype):
        # this component defines its own compute_primal method
        p = om.Problem()
        comp = p.model.add_subsystem('comp', DotProdMultPrimalNoDeclPartials(derivs_method=derivs_method))

        if slvtype == 'coloring':
            comp.declare_coloring()
            p.model.add_constraint('comp.zz', lower=0.)
            p.model.add_constraint('comp.z', lower=0.)
            p.model.add_design_var('comp.x', lower=0.)
            p.model.add_design_var('comp.y', lower=0.)
            p.driver.declare_coloring()

        p.setup(mode=mode, force_alloc_complex=True)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('comp.x', x)
        p.set_val('comp.y', y)
        p.final_setup()
        p.run_model()

        method = method_dict[derivs_method]

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method=method, show_only_incorrect=True))
        assert_check_partials(p.check_partials(method=method, show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)

    def test_auto_shape_by_conn(self):
        class Transpose(om.JaxExplicitComponent):
            def setup(self):
                self.add_input('x')
                self.add_output('y')

            def compute_primal(self, x):
                return x.T

        p = om.Problem()
        p.model.add_subsystem('C1', DotProdMultPrimalNoDeclPartials())
        p.model.add_subsystem('T', Transpose(default_to_dyn_shapes=True))
        p.model.add_subsystem('C2', DotProdMultPrimalNoDeclPartials())

        p.model.connect('C1.z', 'C2.x')
        p.model.connect('C1.zz', 'T.x')
        p.model.connect('T.y', 'C2.y')
        p.model.connect('C2.z', 'C1.x')

        p.setup()
        p.set_val('C1.x', np.ones((2, 3)))
        p.set_val('C1.y', np.ones((3, 4)) * 3.)

        p.run_model()

    @parameterized.expand(itertools.product(['fwd', 'rev'],[(), (3,), (2, 3)]), name_func=parameterized_name)
    def test_super_simple_decl(self, mode, shape):
        class SuperSimpleJaxComp(om.JaxExplicitComponent):
            def compute_primal(self, a, b, c):
                x = a * b
                y = x + c
                z = y * 2.0
                return x, y, z

        p = om.Problem()
        p.model.add_subsystem('comp', SuperSimpleJaxComp(default_to_dyn_shapes=True))
        p.setup(mode=mode)
        if shape == ():
            p.set_val('comp.a', 2.0)
            p.set_val('comp.b', 3.0)
            p.set_val('comp.c', 4.0)
        else:
            p.set_val('comp.a', np.ones(shape) * 2.0)
            p.set_val('comp.b', np.ones(shape) * 3.0)
            p.set_val('comp.c', np.ones(shape) * 4.0)

        p.run_model()

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False], ['jax','fd']), name_func=parameterized_name)
    def test_jax_explicit_comp2primal_w_option(self, mode, matrix_free, derivs_method):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', DotProdMultPrimalOption(mult=1.5, derivs_method=derivs_method))
        comp.matrix_free = matrix_free if derivs_method == 'jax' else False

        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode=mode, force_alloc_complex=True)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape) * 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        method = method_dict[derivs_method]

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 3.0)
        assert_check_partials(comp.check_partials(method=method, show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method=method, show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)

        comp.options['mult'] = 3.5
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 7.0)
        assert_check_partials(p.check_partials(method=method, show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method=method, show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)
        comp.stat = 1./3.5
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y)
        assert_check_partials(p.check_partials(method=method, show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method=method, show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False], ['jax','fd']), name_func=parameterized_name)
    def test_jax_explicit_comp_with_option(self, mode, matrix_free, derivs_method):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', DotProdMultPrimalOption(mult=1.7, derivs_method=derivs_method))
        comp.matrix_free = matrix_free if derivs_method == 'jax' else False

        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode=mode, force_alloc_complex=True)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 1.7 * 2.)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method=method_dict[derivs_method], show_only_incorrect=True))
        assert_check_partials(p.check_partials(method=method_dict[derivs_method], show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)

        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        comp.options['mult'] = 1.9
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 1.9 * 2.)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method=method_dict[derivs_method], show_only_incorrect=True))
        assert_check_partials(p.check_partials(method=method_dict[derivs_method], show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False], ['jax','fd']), name_func=parameterized_name)
    def test_jax_explicit_comp_with_discrete_primal(self, mode, matrix_free, derivs_method):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        ivc.add_discrete_output('disc_out', val=3)
        comp = p.model.add_subsystem('comp', DotProductMultDiscretePrimal(derivs_method=derivs_method))
        comp.matrix_free = matrix_free if derivs_method == 'jax' else False

        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')
        p.model.connect('ivc.disc_out', 'comp.disc_in')

        p.setup(mode=mode, force_alloc_complex=True)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        method = method_dict[derivs_method]

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 3.0)
        assert_check_partials(p.check_partials(method=method, show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                             method=method, show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)

        p.set_val('ivc.disc_out', -2)
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), -np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_partials(p.check_partials(method=method, show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method=method, show_only_incorrect=True))
        assert_sparsity_matches_fd(comp, outstream=None)

    def test_jax_subjacs_info_entries(self):
        p = om.Problem()
        G = p.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', DotProdMultPrimalNoDeclPartials())

        p.setup()

        p['G.comp.x'] = np.array([1, 2])
        p['G.comp.y'] = np.array([3, 4])

        p.run_model()

        self.assertEqual(len(G._subjacs_info), 5)
        self.assertEqual(set(G._subjacs_info.keys()),
                         {('G.comp.z', 'G.comp.x'), ('G.comp.zz', 'G.comp.y'),
                          ('G.comp.z', 'G.comp.y'), ('G.comp.zz', 'G.comp.zz'),
                          ('G.comp.z', 'G.comp.z')})


class CompRetValue(om.JaxExplicitComponent):
    def __init__(self, shape, nins=1, nouts=1, **kwargs):
        self.compute_primal = getattr(self, f'compute_primal_{nins}_{nouts}')
        super().__init__(**kwargs)
        self.shape = shape
        self.nins = nins
        self.nouts = nouts

    def setup(self):
        if self.shape == ():
            for i in range(self.nins):
                self.add_input(f'x{i}', val=1.)
            for i in range(self.nouts):
                self.add_output(f'y{i}', val=0.)
        else:
            for i in range(self.nins):
                self.add_input(f'x{i}', val=jnp.ones(self.shape))
            for i in range(self.nouts):
                self.add_output(f'y{i}', val=jnp.zeros(self.shape))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute_primal_1_1(self, x0):
        return x0**2

    def compute_primal_2_1(self, x0, x1):
        return x0**2 + x1**2

    def compute_primal_1_2(self, x0):
        return x0**2, x0*2

    def compute_primal_2_2(self, x0, x1):
        return x0**2, x1**2


class CompRetTuple(om.JaxExplicitComponent):
    def __init__(self, shape, nins=1, nouts=1, **kwargs):
        self.compute_primal = getattr(self, f'compute_primal_{nins}_{nouts}')
        super().__init__(**kwargs)
        self.shape = shape
        self.nins = nins
        self.nouts = nouts

    def setup(self):
        if self.shape == ():
            for i in range(self.nins):
                self.add_input(f'x{i}', val=1.)
            for i in range(self.nouts):
                self.add_output(f'y{i}', val=0.)
        else:
            for i in range(self.nins):
                self.add_input(f'x{i}', val=jnp.ones(self.shape))
            for i in range(self.nouts):
                self.add_output(f'y{i}', val=jnp.zeros(self.shape))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute_primal_1_1(self, x0):
        return (x0**2,)

    def compute_primal_2_1(self, x0, x1):
        return (x0**2 + x1**2,)

    def compute_primal_1_2(self, x0):
        return (x0**2, x0*2)

    def compute_primal_2_2(self, x0, x1):
        return (x0**2, x1**2)


class TopGrp(om.Group):
    def __init__(self, shape, ret_tuple=False, nins=1, nouts=1, matrix_free=False, derivs_method='jax',
                    use_coloring=False, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.ret_tuple = ret_tuple
        self.nins = nins
        self.nouts = nouts
        self.matrix_free = matrix_free if derivs_method == 'jax' else False
        self.derivs_method = derivs_method
        self.use_coloring = use_coloring

    def setup(self):
        self.add_subsystem('ivc', om.IndepVarComp())
        if self.shape == ():
            for i in range(self.nins):
                self.ivc.add_output(f'x{i}', 0.)

            if self.ret_tuple:
                comp = self.add_subsystem('comp', CompRetTuple(shape=self.shape, nins=self.nins,
                                                                nouts=self.nouts, derivs_method=self.derivs_method))
            else:
                comp = self.add_subsystem('comp', CompRetValue(shape=self.shape, nins=self.nins,
                                                                nouts=self.nouts, derivs_method=self.derivs_method))
        else:
            for i in range(self.nins):
                self.ivc.add_output(f'x{i}', np.zeros(self.shape))
            if self.ret_tuple:
                comp = self.add_subsystem('comp', CompRetTuple(shape=self.shape, nins=self.nins,
                                                                nouts=self.nouts, derivs_method=self.derivs_method))
            else:
                comp = self.add_subsystem('comp', CompRetValue(shape=self.shape, nins=self.nins,
                                                                nouts=self.nouts, derivs_method=self.derivs_method))

        comp.matrix_free = self.matrix_free
        if self.use_coloring:
            comp.declare_coloring()

        for i in range(self.nins):
            self.connect(f'ivc.x{i}', f'comp.x{i}')



@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
class TestJaxShapesAndReturns(unittest.TestCase):
    @parameterized.expand(itertools.product([(), (2,), (2,3)], [(1, 1), (2, 2), (1, 2), (2, 1)],
                                            [True, False], [True, False], ['fwd', 'rev'], ['jax','fd'],
                                            ['coloring', 'nocoloring']), name_func=parameterized_name)
    def test_compute_primal_return_shapes(self, shape, sizetup, ret_tuple, matrix_free, mode, derivs_method, slvtype):
        nins, nouts = sizetup
        prob = om.Problem()
        prob.model = TopGrp(shape=shape, ret_tuple=ret_tuple, nins=nins, nouts=nouts,
                            matrix_free=matrix_free, derivs_method=derivs_method, use_coloring=slvtype=='coloring')
        prob.set_solver_print(level=0)

        ofs = [f'comp.y{i}' for i in range(nouts)]
        wrts = [f'ivc.x{i}' for i in range(nins)]

        do_coloring = shape==(2,3) and slvtype == 'coloring'

        # only color for 2x3 for now, others too dense
        if do_coloring:
            prob.driver = om.ScipyOptimizeDriver(maxiter=2)
            for wrt in wrts:
                prob.model.add_design_var(wrt)
            for of in ofs:
                prob.model.add_constraint(of, lower=200.)

            prob.model.add_subsystem('objcomp', om.ExecComp('y=x+1.'))
            prob.model.connect(f'comp.y{nouts - 1}', 'objcomp.x', src_indices=[0], flat_src_indices=True)
            prob.model.add_objective('objcomp.y')
            ofs.append('objcomp.y')
            prob.driver.declare_coloring()

        prob.setup(force_alloc_complex=True, check=False, mode=mode)
        prob.run_model()
        prob.compute_totals(of=ofs, wrt=wrts)

        method = method_dict[derivs_method]

        assert_check_partials(prob.check_partials(method=method, out_stream=None), atol=1e-5)
        assert_check_totals(prob.check_totals(of=ofs, wrt=wrts, method=method, out_stream=None), atol=1e-5)

        for c in prob.model.system_iter(recurse=True, typ=om.JaxExplicitComponent):
            assert_sparsity_matches_fd(c, direction=mode, outstream=None)

        if do_coloring:
            prob['ivc.x0'] = np.ones(shape) * -10.
            prob.run_driver()
            coloring = prob.driver._coloring_info.coloring
            self.assertTrue(coloring is not None, 'coloring is None')
            self.assertTrue(coloring.total_solves() <= 2)


if __name__ == '__main__':
    unittest.main()

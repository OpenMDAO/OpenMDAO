import unittest
import sys
import itertools

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
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
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

    def compute_primal(self, x, y):
        z = jnp.dot(x, y)
        zz = y * 2.5
        return z, zz


class DotProdMultPrimal(DotProdMultPrimalNoDeclPartials):
    def setup(self):
        super().setup()
        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])


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

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def get_self_statics(self):
        return (self.options['mult'], self.stat)

    def compute_primal(self, x, y):
        z = jnp.dot(x, y)
        zz = y * self.options['mult'] * self.stat
        return z, zz



class DotProdMultOption(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('mult', default=2.5, desc='multiplier', types=(float,))

    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute(self, inputs, outputs):
        outputs['z'] = np.dot(inputs['x'], inputs['y'])
        outputs['zz'] = inputs['y'] * self.options['mult']



class DotProductMultDiscrete(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_discrete_input('disc_in', val=2)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')
        self.add_discrete_output('disc_out', val=3)

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['disc_out'] = -discrete_inputs['disc_in']
        if discrete_inputs['disc_in'] > 0:
            outputs['z'] = np.dot(inputs['x'], inputs['y'])
        else:
            outputs['z'] = -np.dot(inputs['x'], inputs['y'])

        if discrete_outputs['disc_out'] > 0:
            outputs['zz'] = inputs['y'] * 2.5
        else:
            outputs['zz'] = inputs['y'] * 3.0



class DotProductMultDiscretePrimal(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True)
        self.add_input('y', shape_by_conn=True)
        self.add_discrete_input('disc_in', val=2)
        self.add_output('z', compute_shape=lambda shapes: (shapes['x'][0], shapes['y'][1]))
        self.add_output('zz', copy_shape='y')
        self.add_discrete_output('disc_out', val=3)

        self.declare_partials(of=['z', 'zz'], wrt=['x', 'y'])

    def compute_primal(self, x, y, disc_in):
        def pos(x, y):
            return jnp.dot(x, y), y * 3.0

        def neg(x, y):
            return -jnp.dot(x, y), y * 2.5

        disc_out = -disc_in
        z, zz = jax.lax.cond(disc_in >= 0, pos, neg, x, y)

        self._discrete_outputs.set_vals((disc_out,))
        return (z, zz, disc_out)

x_shape = (2, 3)
y_shape = (3, 4)


@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
class TestJaxComp(unittest.TestCase):

    @parameterized.expand(itertools.product(['fwd', 'rev'], [(True, 'jax'), (False, 'jax'), (False, 'fd')]), name_func=parameterized_name)
    def test_jax_explicit_comp(self, mode, tup):
        matrix_free, derivs_method = tup
        xshape = x_shape
        yshape = y_shape
        p = om.Problem()
        # create an IVC manually so we can set the shapes.  Otherwise must set shape in the component
        # itself.
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(xshape)))
        ivc.add_output('y', val=np.ones(yshape))
        comp = p.model.add_subsystem('comp', DotProd(derivs_method=derivs_method))
        comp.matrix_free = matrix_free
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode=mode, force_alloc_complex=True)

        x = np.arange(1,np.prod(xshape)+1).reshape(xshape) * 2.0
        y = np.arange(1,np.prod(yshape)+1).reshape(yshape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        p.check_totals(of=['comp.z'], wrt=['comp.x', 'comp.y'], method='cs', show_only_incorrect=True)
        assert_check_partials(p.check_partials(method='cs', show_only_incorrect=True))

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]), name_func=parameterized_name)
    def test_jax_explicit_comp2(self, mode, matrix_free):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', DotProdMult(derivs_method='jax'))
        comp.matrix_free = matrix_free
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode=mode)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'],
                                           wrt=['comp.x', 'comp.y'], method='fd',
                                           show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]), name_func=parameterized_name)
    def test_jax_explicit_comp2primal(self, mode, matrix_free):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', DotProdMultPrimal(derivs_method='jax'))
        comp.matrix_free = matrix_free
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode=mode)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    @parameterized.expand(itertools.product(['fwd', 'rev'], ['fd', 'cs']), name_func=parameterized_name)
    def test_jax_explicit_comp2primal_nodecl(self, mode, derivs_method):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        p.model.add_subsystem('comp', DotProdMultPrimalNoDeclPartials(derivs_method=derivs_method))
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode=mode, force_alloc_complex=True)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        if derivs_method == 'fd':
            method = 'cs'
        else:
            method = 'fd'

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method=method, show_only_incorrect=True))
        assert_check_partials(p.check_partials(method=method, show_only_incorrect=True))

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]), name_func=parameterized_name)
    def test_jax_explicit_comp2primal_w_option(self, mode, matrix_free):
        # this component defines its own compute_primal method
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', DotProdMultPrimalOption(derivs_method='jax', mult=1.5))
        comp.matrix_free = matrix_free

        # deriv shape: [[(2, 4, 2, 3), (2, 4, 3, 4)], [(3, 4, 2, 3), (3, 4, 3, 4)]]
        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode=mode)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape) * 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 3.0)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        comp.options['mult'] = 3.5
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 7.0)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        comp.stat = 1./3.5
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]), name_func=parameterized_name)
    def test_jax_explicit_comp_with_option(self, mode, matrix_free):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        comp = p.model.add_subsystem('comp', DotProdMultOption(mult=1.7, derivs_method='jax'))
        comp.matrix_free = matrix_free

        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')

        p.setup(mode=mode)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 1.7)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        comp.options['mult'] = 1.9
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 1.9)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]), name_func=parameterized_name)
    def test_jax_explicit_comp_with_discrete(self, mode, matrix_free):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        ivc.add_discrete_output('disc_out', val=3)
        comp = p.model.add_subsystem('comp', DotProductMultDiscrete(derivs_method='jax'))
        comp.matrix_free = matrix_free

        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')
        p.model.connect('ivc.disc_out', 'comp.disc_in')

        p.setup(mode=mode)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 3.0)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        p.set_val('ivc.disc_out', -2)
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), -np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]), name_func=parameterized_name)
    def test_jax_explicit_comp_with_discrete_primal(self, mode, matrix_free):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        ivc.add_discrete_output('disc_out', val=3)
        comp = p.model.add_subsystem('comp', DotProductMultDiscretePrimal(derivs_method='jax'))
        comp.matrix_free = matrix_free

        p.model.connect('ivc.x', 'comp.x')
        p.model.connect('ivc.y', 'comp.y')
        p.model.connect('ivc.disc_out', 'comp.disc_in')

        p.setup(mode=mode)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 3.0)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                             method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

        p.set_val('ivc.disc_out', -2)
        p.run_model()
        assert_near_equal(p.get_val('comp.z'), -np.dot(x, y))
        assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        assert_check_totals(p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'],
                                           method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))


if sys.version_info >= (3, 9):


    class CompRetValue(om.JaxExplicitComponent):
        def __init__(self, shape, nins=1, nouts=1, **kwargs):
            super().__init__(**kwargs)
            self.shape = shape
            self.nins = nins
            self.nouts = nouts

            self.compute_primal = getattr(self, f'compute_primal_{self.nins}_{self.nouts}')

        def setup(self):
            if self.shape == ():
                for i in range(self.nins):
                    self.add_input(f'x{i}', val=0.)
                for i in range(self.nouts):
                    self.add_output(f'y{i}', val=0.)
            else:
                for i in range(self.nins):
                    self.add_input(f'x{i}', val=jnp.zeros(self.shape))
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
            super().__init__(**kwargs)
            self.shape = shape
            self.nins = nins
            self.nouts = nouts

            self.compute_primal = getattr(self, f'compute_primal_{self.nins}_{self.nouts}')

        def setup(self):
            if self.shape == ():
                for i in range(self.nins):
                    self.add_input(f'x{i}', val=0.)
                for i in range(self.nouts):
                    self.add_output(f'y{i}', val=0.)
            else:
                for i in range(self.nins):
                    self.add_input(f'x{i}', val=jnp.zeros(self.shape))
                for i in range(self.nouts):
                    self.add_output(f'y{i}', val=jnp.zeros(self.shape))

        def setup_partials(self):
            self.declare_partials('*', '*')

        def compute_primal_1_1(self, x0):
            return (x0**2,)

        def compute_primal_2_1(self, x0, x1):
            return (x0**2 + x1**2,)

        def compute_primal_1_2(self, x0):
            return x0**2, x0*2

        def compute_primal_2_2(self, x0, x1):
            return x0**2, x1**2


    class TopGrp(om.Group):
        def __init__(self, shape, ret_tuple=False, nins=1, nouts=1, matrix_free=False, **kwargs):
            super().__init__(**kwargs)
            self.shape = shape
            self.ret_tuple = ret_tuple
            self.nins = nins
            self.nouts = nouts
            self.matrix_free = matrix_free

        def setup(self):
            self.add_subsystem('ivc', om.IndepVarComp())
            if self.shape == ():
                for i in range(self.nins):
                    self.ivc.add_output(f'x{i}', 0.)

                if self.ret_tuple:
                    comp = self.add_subsystem('comp', CompRetTuple(shape=self.shape, nins=self.nins, nouts=self.nouts))
                else:
                    comp = self.add_subsystem('comp', CompRetValue(shape=self.shape, nins=self.nins, nouts=self.nouts))
            else:
                for i in range(self.nins):
                    self.ivc.add_output(f'x{i}', np.zeros(self.shape))
                if self.ret_tuple:
                    comp = self.add_subsystem('comp', CompRetTuple(shape=self.shape, nins=self.nins, nouts=self.nouts))
                else:
                    comp = self.add_subsystem('comp', CompRetValue(shape=self.shape, nins=self.nins, nouts=self.nouts))

            comp.matrix_free = self.matrix_free

            for io in range(self.nouts):
                for ii in range(self.nins):
                    if ii == io:
                        self.connect(f'ivc.x{io}', f'comp.x{ii}')



@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
class TestJaxShapesAndReturns(unittest.TestCase):
    @parameterized.expand(itertools.product([(), (2,), (2,3)], [(1, 1), (2, 2), (1, 2), (2, 1)],
                                            [True, False], [True, False], ['fwd', 'rev']), name_func=parameterized_name)
    def test_compute_primal_return_shapes(self, shape, sizetup, ret_tuple, matrix_free, mode):
        nins, nouts = sizetup
        prob = om.Problem()
        prob.model = TopGrp(shape=shape, ret_tuple=ret_tuple, nins=nins, nouts=nouts, matrix_free=matrix_free)
        prob.set_solver_print(level=0)

        ofs = [f'comp.y{i}' for i in range(nouts)]
        wrts = [f'ivc.x{i}' for i in range(nins)]

        prob.setup(force_alloc_complex=True, check=False, mode=mode)
        prob.final_setup()
        prob.compute_totals(of=ofs, wrt=wrts)

        assert_check_partials(prob.check_partials(method='cs', out_stream=None), atol=1e-5)
        assert_check_totals(prob.check_totals(of=ofs, wrt=wrts, method='cs', out_stream=None), atol=1e-5)

    # TODO: test with mixed np and jnp in compute

if __name__ == '__main__':
    unittest.main()

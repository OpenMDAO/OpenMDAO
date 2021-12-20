import unittest
import numpy as np

import openmdao.func_api as omf
from openmdao.utils.assert_utils import assert_warning, assert_no_warning

try:
    import jax
except ImportError:
    jax = None


class TestFuncAPI(unittest.TestCase):
    def test_inout_var(self):
        def func(a, b, c):
            x = a * b
            y = b * c
            return x, y

        f = (omf.wrap(func)
             .add_input('a', units='m')
             .add_input('b', units='inch')
             .add_input('c', units='ft')
             .add_output('x', units='cm')
             .add_output('y', units='km'))

        invar_meta = list(f.get_input_meta())
        self.assertEqual(list(f.get_input_names()), ['a', 'b', 'c'])
        self.assertEqual(invar_meta[0][1]['val'], 1.0)
        self.assertEqual(invar_meta[0][1]['shape'], ())
        self.assertEqual(invar_meta[0][1]['units'], 'm')

        self.assertEqual(invar_meta[1][1]['val'], 1.0)
        self.assertEqual(invar_meta[1][1]['shape'], ())
        self.assertEqual(invar_meta[1][1]['units'], 'inch')

        self.assertEqual(invar_meta[2][1]['val'], 1.0)
        self.assertEqual(invar_meta[2][1]['shape'], ())
        self.assertEqual(invar_meta[2][1]['units'], 'ft')

        outvar_meta = list(f.get_output_meta())
        self.assertEqual(list(f.get_output_names()), ['x', 'y'])
        self.assertEqual(outvar_meta[0][1]['val'], 1.0)
        self.assertEqual(outvar_meta[0][1]['shape'], ())
        self.assertEqual(outvar_meta[0][1]['units'], 'cm')

        self.assertEqual(outvar_meta[1][1]['val'], 1.0)
        self.assertEqual(outvar_meta[1][1]['shape'], ())
        self.assertEqual(outvar_meta[1][1]['units'], 'km')

    def test_inout_vars(self):
        def func(a, b, c):
            x = a * b
            y = b * c
            return x, y

        f = (omf.wrap(func)
             .add_inputs(a={'units': 'm'}, b={'units': 'inch', 'shape': 3, 'val': 7.}, c={'units': 'ft'})
             .add_outputs(x={'units': 'cm', 'shape': 3}, y={'units': 'km', 'shape': 3}))

        invar_meta = list(f.get_input_meta())
        names = [n for n, _ in invar_meta]
        self.assertEqual(names, ['a', 'b', 'c'])

        self.assertEqual(invar_meta[0][1]['val'], 1.0)
        self.assertEqual(invar_meta[0][1]['shape'], ())
        self.assertEqual(invar_meta[0][1]['units'], 'm')

        np.testing.assert_allclose(invar_meta[1][1]['val'], np.ones(3)*7.)
        self.assertEqual(invar_meta[1][1]['shape'], (3,))
        self.assertEqual(invar_meta[1][1]['units'], 'inch')

        self.assertEqual(invar_meta[2][1]['val'], 1.0)
        self.assertEqual(invar_meta[2][1]['shape'], ())
        self.assertEqual(invar_meta[2][1]['units'], 'ft')


        outvar_meta = list(f.get_output_meta())
        names = [n for n, _ in outvar_meta]
        self.assertEqual(names, ['x', 'y'])

        np.testing.assert_allclose(outvar_meta[0][1]['val'], np.ones(3))
        self.assertEqual(outvar_meta[0][1]['shape'], (3,))
        self.assertEqual(outvar_meta[0][1]['units'], 'cm')

        np.testing.assert_allclose(outvar_meta[1][1]['val'], np.ones(3))
        self.assertEqual(outvar_meta[1][1]['shape'], (3,))
        self.assertEqual(outvar_meta[1][1]['units'], 'km')

    def test_nometa(self):
        def func(a, b, c):
            x = a * b
            y = b * c
            return x, y

        f = omf.wrap(func)
        invar_meta = list(f.get_input_meta())
        self.assertEqual(list(f.get_input_names()), ['a', 'b', 'c'])
        self.assertEqual(invar_meta[0][1]['val'], 1.0)
        self.assertEqual(invar_meta[0][1]['shape'], ())

        self.assertEqual(invar_meta[1][1]['val'], 1.0)
        self.assertEqual(invar_meta[1][1]['shape'], ())

        self.assertEqual(invar_meta[2][1]['val'], 1.0)
        self.assertEqual(invar_meta[2][1]['shape'], ())

        outvar_meta = list(f.get_output_meta())
        self.assertEqual(list(f.get_output_names()), ['x', 'y'])
        self.assertEqual(outvar_meta[0][1]['val'], 1.0)
        self.assertEqual(outvar_meta[0][1]['shape'], ())

        self.assertEqual(outvar_meta[1][1]['val'], 1.0)
        self.assertEqual(outvar_meta[1][1]['shape'], ())

    def test_infer_outnames(self):
        def func(a, b, c):
            x = a * b
            return x, c

        self.assertEqual(omf.wrap(func).get_return_names(), ['x', None])

    def test_infer_outnames_replace_inpname(self):
        def func(a, b, c):
            x = a * b
            return x, c

        f = (omf.wrap(func)
                .add_output('q'))  # replace second return value name with 'q' since 'c' is an input name

        self.assertEqual([n for n,_ in f.get_output_meta()], ['x', 'q'])

    def test_infer_outnames_err(self):
        def func(a, b, c):
            x = a * b
            y = b * c
            return x, y

        f = (omf.wrap(func)
                .add_output('q'))

        with self.assertRaises(Exception) as context:
            f.get_output_meta()

        self.assertEqual(context.exception.args[0],
                         "There must be an unnamed return value for every unmatched output name ['q'] but only found 0.")

    def test_set_out_names(self):
        def func(a, b, c):
            return a * b, b * c

        f = (omf.wrap(func)
                .output_names('x', 'y'))

        outvar_meta = list(f.get_output_meta())
        self.assertEqual(list(f.get_output_names()), ['x', 'y'])

        self.assertEqual(outvar_meta[0][1]['val'], 1.0)
        self.assertEqual(outvar_meta[0][1]['shape'], ())

        self.assertEqual(outvar_meta[1][1]['val'], 1.0)
        self.assertEqual(outvar_meta[1][1]['shape'], ())

    def test_defaults(self):
        def func(a):
            x = a * 2.0
            return x

        f = (omf.wrap(func)
                .defaults(units='cm', val=7., method='cs')
                .declare_partials(of='x', wrt='a')
                .declare_coloring(wrt='*'))

        invar_meta = list(f.get_input_meta())
        self.assertEqual(list(f.get_input_names()), ['a'])
        self.assertEqual(invar_meta[0][1]['val'], 7.0)
        self.assertEqual(invar_meta[0][1]['units'], 'cm')
        self.assertEqual(invar_meta[0][1]['shape'], ())

        outvar_meta = list(f.get_output_meta())
        self.assertEqual(list(f.get_output_names()), ['x'])
        self.assertEqual(outvar_meta[0][1]['val'], 7.0)
        self.assertEqual(outvar_meta[0][1]['shape'], ())
        self.assertEqual(outvar_meta[0][1]['units'], 'cm')

        partials_meta = list(f.get_declare_partials())
        self.assertEqual(partials_meta[0]['method'], 'cs')

        coloring_meta = f.get_declare_coloring()
        self.assertEqual(coloring_meta['method'], 'cs')

    def test_defaults_override(self):
        def func(a=4.):
            x = a * 2.0
            return x

        f = (omf.wrap(func)
                .defaults(units='cm', val=7.)
                .add_output('x', units='inch'))

        invar_meta = list(f.get_input_meta())
        self.assertEqual(list(f.get_input_names()), ['a'])
        self.assertEqual(invar_meta[0][1]['val'], 4.0)
        self.assertEqual(invar_meta[0][1]['units'], 'cm')
        self.assertEqual(invar_meta[0][1]['shape'], ())

        outvar_meta = list(f.get_output_meta())
        self.assertEqual(list(f.get_output_names()), ['x'])
        self.assertEqual(outvar_meta[0][1]['val'], 7.0)
        self.assertEqual(outvar_meta[0][1]['shape'], ())
        self.assertEqual(outvar_meta[0][1]['units'], 'inch')

    def test_declare_option(self):
        def func(a, opt):
            if opt == 'foo':
                x = a * 2.0
            else:
                x = a / 2.0
            return x

        f = (omf.wrap(func)
                .declare_option('opt', types=str, values=('foo', 'bar'), desc='an opt'))

        opt_meta = list(f.get_input_meta())[1]
        self.assertEqual(opt_meta[0], 'opt')
        self.assertEqual(opt_meta[1]['types'], str)
        self.assertEqual(opt_meta[1]['values'], ('foo', 'bar'))
        self.assertEqual(opt_meta[1]['desc'], 'an opt')

    def test_declare_partials(self):
        def func(a, b):
            x = a * b
            y = a / b
            return x, y

        f = (omf.wrap(func)
             .declare_partials(of='x', wrt=['a', 'b'], method='cs')
             .declare_partials(of='y', wrt=['a', 'b'], method='fd'))

        meta = list(f.get_declare_partials())
        self.assertEqual(meta[0], {'of': 'x', 'wrt': ['a', 'b'], 'method': 'cs'})
        self.assertEqual(meta[1], {'of': 'y', 'wrt': ['a', 'b'], 'method': 'fd'})

    @unittest.skipIf(jax is None, "jax is not installed")
    def test_declare_partials_jax_mixed(self):
        def func(a, b):
            x = a * b
            y = a / b
            return x, y

        with self.assertRaises(Exception) as cm:
            f = (omf.wrap(func)
                 .declare_partials(of='x', wrt=['a', 'b'], method='jax')
                 .declare_partials(of='y', wrt=['a', 'b'], method='fd'))

        self.assertEqual(cm.exception.args[0],
                         "If multiple calls to declare_partials() are made on the same function object and any set method='jax', then all must set method='jax'.")

    @unittest.skipIf(jax is None, "jax is not installed")
    def test_declare_partials_jax_mixed2(self):
        def func(a, b):
            x = a * b
            y = a / b
            return x, y

        with self.assertRaises(Exception) as cm:
            f = (omf.wrap(func)
                 .declare_partials(of='y', wrt=['a', 'b'], method='fd')
                 .declare_partials(of='x', wrt=['a', 'b'], method='jax'))

        self.assertEqual(cm.exception.args[0],
                         "If multiple calls to declare_partials() are made on the same function object and any set method='jax', then all must set method='jax'.")

    def test_declare_coloring(self):
        def func(a, b):
            x = a * b
            y = a / b
            return x, y

        f = (omf.wrap(func)
             .declare_coloring(wrt='*', method='cs'))

        meta = f.get_declare_coloring()
        self.assertEqual(meta, {'wrt': '*', 'method': 'cs'})

        with self.assertRaises(Exception) as cm:
            f2 = (omf.wrap(func)
                    .declare_coloring(wrt='a', method='cs')
                    .declare_coloring(wrt='b', method='cs'))

        self.assertEqual(str(cm.exception),
                         "declare_coloring has already been called.")


    @unittest.skipIf(jax is None, "jax is not installed")
    def test_jax_out_shape_compute(self):
        def func(a=np.ones((3,3)), b=np.ones((3,3))):
            x = a * b
            y = (a / b)[:,[1,2]]
            return x, y

        f = omf.wrap(func).declare_partials(of='*', wrt='*', method='jax')
        outvar_meta = list(f.get_output_meta())
        self.assertEqual(list(f.get_output_names()), ['x', 'y'])
        self.assertEqual(outvar_meta[0][0], 'x')
        self.assertEqual(outvar_meta[0][1]['shape'], (3,3))
        self.assertEqual(outvar_meta[1][0], 'y')
        self.assertEqual(outvar_meta[1][1]['shape'], (3,2))

    @unittest.skipIf(jax is None, "jax is not installed")
    def test_jax_out_shape_check(self):
        def func(a=np.ones((3,3)), b=np.ones((3,3))):
            x = a * b
            y = (a / b)[:,[1,2]]
            return x, y

        f = (omf.wrap(func)
                .add_outputs(x={}, y={'shape': (3,3)})
                .declare_partials(of='*', wrt='*', method='jax'))
        with self.assertRaises(Exception) as cm:
            outvar_meta = list(f.get_output_meta())

        msg = "shape from metadata for return value 'y' of (3, 3) doesn't match computed shape of (3, 2)."
        self.assertEqual(cm.exception.args[0], msg)

    def test_errors_and_warnings(self):
        def func(a=7., opt='foo'):
            if opt == 'foo':
                x = a * 2.0
            else:
                x = a / 2.0
            return x

        f = omf.wrap(func)

        with self.assertRaises(Exception) as context:
            f.declare_option('opt', types=str, valuess=('foo', 'bar'), desc='an opt')

        self.assertEqual(context.exception.args[0],
                         "In declare_option, metadata names ['valuess'] are not allowed.")

        with self.assertRaises(Exception) as context:
            f.declare_partials(of='*', wrrt='*', method='cs')

        self.assertEqual(context.exception.args[0],
                         "In declare_partials, metadata names ['wrrt'] are not allowed.")

        with self.assertRaises(Exception) as context:
            f.declare_coloring(wrrt='*', method='cs')

        self.assertEqual(context.exception.args[0],
                         "In declare_coloring, metadata names ['wrrt'] are not allowed.")

        with self.assertRaises(Exception) as context:
            f.add_input('a', shepe=4, units='m')

        self.assertEqual(context.exception.args[0],
                         "In add_input, metadata names ['shepe'] are not allowed.")

        with self.assertRaises(Exception) as context:
            f.add_input('a', val=4, units='m')

        self.assertEqual(context.exception.args[0],
                         "In add_input, metadata 'val' has already been added to function for input 'a'.")

        with self.assertRaises(Exception) as context:
            f.add_output('x', shepe=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In add_output, metadata names ['shepe'] are not allowed.")

        with self.assertRaises(Exception) as context:
            f.defaults(shepe=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In defaults, metadata names ['shepe'] are not allowed.")

        with self.assertRaises(Exception) as context:
            f.add_input('aa', shape=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In add_input, 'aa' is not an input to this function.")

        with self.assertRaises(Exception) as context:
            f.add_output('a', shape=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In add_output, 'a' already registered as an input.")

        f.add_output('x', shape=4, val=3.)
        with self.assertRaises(Exception) as context:
            f.add_output('x', shape=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In add_output, 'x' already registered as an output.")

        # multiple returns, but neither gives the name, so raise exception
        def func2(a):
            if a < 1.:
                return a + 1
            return None

        f = omf.wrap(func2)

        with self.assertRaises(Exception) as context:
            f._setup()

        self.assertEqual(context.exception.args[0],
                         "0 output names are specified in the metadata but there are 1 unnamed return values in the function.")

        # this func defines the same name in both returns, which is ok
        def func3(a):
            if a < 1.:
                bb = a + 1
                return bb
            bb = 8.
            return bb

        f = omf.wrap(func3)

        with assert_no_warning(UserWarning):
            f._setup()

        # this func defines the name in one return but not the other, which is ok
        def func4(a):
            if a < 1.:
                bb = a + 1
                return bb
            return a - 1

        f = omf.wrap(func4)

        with assert_no_warning(UserWarning):
            f._setup()

        # different numbers of return values, which isn't allowed
        def func5(a):
            if a < 1.:
                b = a + 1
                c = a * 2.
                return b, c
            return None

        f = omf.wrap(func5)

        with self.assertRaises(Exception) as context:
            f._setup()

        self.assertEqual(context.exception.args[0],
                         "During AST processing to determine the number and name of return values, the following error occurred: Function has multiple return statements with differing numbers of return values.")

        # different return value names. not allowed
        def func6(a):
            b = a + 1
            c = a * 2.
            x = c
            if a < 1.:
                return b, c
            return b, x

        f = omf.wrap(func6)

        with self.assertRaises(Exception) as context:
            f._setup()

        self.assertEqual(context.exception.args[0],
                         "During AST processing to determine the number and name of return values, the following error occurred: Function has multiple return statements with different return value names of ['c', 'x'] for return value 1.")

        def func7(a):
            b = a + 1
            c = a * 2.
            return b, c

        f = omf.wrap(func7).add_input('a', val=np.ones(2), shape=(2,2))

        with self.assertRaises(Exception) as context:
            f._setup()

        self.assertEqual(context.exception.args[0],
                         "Input 'a' value has shape (2,), but shape was specified as (2, 2).")

    def test_return_names(self):
        def func(a):
            b = a + 1
            # no return statement

        f = omf.wrap(func)
        self.assertEqual(f.get_return_names(), [])

if __name__ == '__main__':
    unittest.main()



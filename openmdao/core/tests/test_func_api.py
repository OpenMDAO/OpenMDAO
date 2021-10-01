import unittest
import numpy as np

import openmdao.func_api as omf


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
        self.assertEqual(outvar_meta[0][1]['deps'], {'b', 'a'})

        self.assertEqual(outvar_meta[1][1]['val'], 1.0)
        self.assertEqual(outvar_meta[1][1]['shape'], ())
        self.assertEqual(outvar_meta[1][1]['units'], 'km')
        self.assertEqual(outvar_meta[1][1]['deps'], {'b', 'c'})


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
        self.assertEqual(outvar_meta[0][1]['deps'], {'b', 'a'})

        np.testing.assert_allclose(outvar_meta[1][1]['val'], np.ones(3))
        self.assertEqual(outvar_meta[1][1]['shape'], (3,))
        self.assertEqual(outvar_meta[1][1]['units'], 'km')
        self.assertEqual(outvar_meta[1][1]['deps'], {'b', 'c'})

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
        self.assertEqual(outvar_meta[0][1]['deps'], {'b', 'a'})

        self.assertEqual(outvar_meta[1][1]['val'], 1.0)
        self.assertEqual(outvar_meta[1][1]['shape'], ())
        self.assertEqual(outvar_meta[1][1]['deps'], {'b', 'c'})

    def test_infer_outnames(self):
        def func(a, b, c):
            x = a * b
            return x, c

        deps = omf.get_function_deps(func)
        self.assertEqual(deps[0], ('x', {'b', 'a'}))
        # name of second return value is None since output cannot have same name as input
        self.assertEqual(deps[1], (None, {'c'}))

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
        self.assertEqual(outvar_meta[0][1]['deps'], {'b', 'a'})

        self.assertEqual(outvar_meta[1][1]['val'], 1.0)
        self.assertEqual(outvar_meta[1][1]['shape'], ())
        self.assertEqual(outvar_meta[1][1]['deps'], {'b', 'c'})

    def test_function_deps1(self):
        def func(a, b, c):
            x = a * b
            foo = b + np.sin(c)
            bar = np.cos(foo) + 7.
            baz = np.sin(foo, bar)
            y = baz + 1.
            return x, y

        deps = omf.get_function_deps(func)
        self.assertEqual(deps[0][0], 'x')
        self.assertEqual(deps[0][1], {'a', 'b'})
        self.assertEqual(deps[1][0], 'y')
        self.assertEqual(deps[1][1], {'b', 'c'})

    def test_function_deps2(self):
        def func(a, b, c):
            x = a * b
            foo = 4.
            bar = np.cos(foo) + 7.
            baz = np.sin(foo, bar)
            y = baz + 1.
            return x, y

        deps = omf.get_function_deps(func)
        self.assertEqual(deps[0][0], 'x')
        self.assertEqual(deps[0][1], {'a', 'b'})
        self.assertEqual(deps[1][0], 'y')
        self.assertEqual(deps[1][1], set())

    def test_function_deps3(self):
        def func(a, b, c):
            foo = sin(a) - cos(b)
            bar = np.cos(foo) + 7.
            baz = np.sin(foo, bar)
            y = baz + 1.
            x = y
            return x, y

        deps = omf.get_function_deps(func)
        self.assertEqual(deps[0][0], 'x')
        self.assertEqual(deps[0][1], {'a', 'b'})
        self.assertEqual(deps[1][0], 'y')
        self.assertEqual(deps[1][1], {'a', 'b'})

    def test_defaults(self):
        def func(a):
            x = a * 2.0
            return x

        f = (omf.wrap(func)
                .defaults(units='cm', val=7.))

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
        self.assertEqual(outvar_meta[0][1]['deps'], {'a'})

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
        self.assertEqual(outvar_meta[0][1]['deps'], {'a'})

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

    def test_declare_coloring(self):
        def func(a, b):
            x = a * b
            y = a / b
            return x, y

        f = (omf.wrap(func)
             .declare_coloring(wrt='*', method='cs'))

        meta = f.get_declare_coloring()
        self.assertEqual(meta, {'wrt': '*', 'method': 'cs'})

    @unittest.skipIf(omf.jax is None, "jax is not installed")
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

    @unittest.skipIf(omf.jax is None, "jax is not installed")
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

        msg = "Annotated shape for return value 'y' of (3, 3) doesn't match computed shape of (3, 2)."
        self.assertEqual(cm.exception.args[0], msg)

    def test_bad_args(self):
        def func(a, opt):
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
            f.add_input('a', shepe=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In add_input, metadata names ['shepe'] are not allowed.")

        with self.assertRaises(Exception) as context:
            f.add_output('x', shepe=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In add_output, metadata names ['shepe'] are not allowed.")

        with self.assertRaises(Exception) as context:
            f.defaults(shepe=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In defaults, metadata names ['shepe'] are not allowed.")

        with self.assertRaises(Exception) as context:
            f.add_output('a', shape=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In add_output, 'a' already registered as an input.")

        f.add_output('x', shape=4, val=3.)
        with self.assertRaises(Exception) as context:
            f.add_output('x', shape=4, val=3.)

        self.assertEqual(context.exception.args[0],
                         "In add_output, 'x' already registered as an output.")

if __name__ == '__main__':
    unittest.main()



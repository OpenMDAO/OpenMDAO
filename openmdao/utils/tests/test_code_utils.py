
import unittest
import sys
import os
import pickle

import numpy as np

from openmdao.utils.code_utils import get_nested_calls, LambdaPickleWrapper, get_return_names, \
    get_func_graph, get_function_deps
from openmdao.core.group import Group


class TestCodeUtils(unittest.TestCase):

    def test_get_nested_calls(self):
        devnull = open(os.devnull, "w")

        graph = get_nested_calls(Group, '_setup', stream=devnull)
        self.assertIn(('Group._setup_var_data', 'System._setup_var_data'),
                      graph.edges(), "System._setup_var_data not called by Group._setup_var_data")


@unittest.skipUnless(sys.version_info[:2] >= (3, 9), "requires Python 3.9+")
class TestLambdaPickleWrapper(unittest.TestCase):

    def test_init(self):
        func = lambda x: x + 1  # noqa
        wrapper = LambdaPickleWrapper(func)
        self.assertEqual(wrapper._func, func)

    def test_call(self):
        func = lambda x: x + 1  # noqa
        wrapper = LambdaPickleWrapper(func)
        self.assertEqual(wrapper(1), 2)

    def test_getstate(self):
        func = lambda x: x + 1  # noqa
        wrapper = LambdaPickleWrapper(func)
        state = wrapper.__getstate__()
        self.assertEqual(state['_func'], wrapper._getsrc())

    def test_setstate(self):
        func = lambda x: x + 1  # noqa
        wrapper = LambdaPickleWrapper(func)
        state = {'_func': 'lambda x: x + 2', '_src': None}
        wrapper.__setstate__(state)
        self.assertEqual(wrapper(1), 3)

    def test_getsrc(self):
        func = lambda x: x + 1  # noqa
        wrapper = LambdaPickleWrapper(func)
        src = wrapper._getsrc()
        self.assertEqual(src, 'lambda x: x + 1')

    def test_pickle(self):
        func = lambda x: x + 1  # noqa
        wrapper = LambdaPickleWrapper(func)
        pkl = pickle.dumps(wrapper)
        wrapper2 = pickle.loads(pkl)
        self.assertEqual(wrapper2(1), 2)


class TestGetReturnNames(unittest.TestCase):

    def test_single_return(self):
        def func():
            a = 1
            return a

        result = get_return_names(func)
        self.assertEqual(result, ['a'])

    def test_multiple_same_return(self):
        def func():
            a = 1
            if a > 0:
                return a
            else:
                return a

        result = get_return_names(func)
        self.assertEqual(result, ['a'])

    def test_tuple_return(self):
        def func():
            a = 1
            b = 2
            return a, b

        result = get_return_names(func)
        self.assertEqual(result, ['a', 'b'])

    def test_different_return_values(self):
        def func():
            a = 1
            b = 2
            if a > 0:
                return a
            else:
                return b

        with self.assertRaises(RuntimeError) as cm:
            get_return_names(func)

        self.assertEqual(cm.exception.args[0],
                         "Function has multiple return statements with different return value names of ['a', 'b'] for return value 0.")

    def test_mixed_return_values(self):
        class _Foo(object):
            def __init__(self, x):
                self.x = x

        def func():
            a = 1
            f = _Foo(a)
            if a > 0:
                return a, 2
            else:
                return a, f.x

        result = get_return_names(func)
        self.assertEqual(result, ['a', None])


class TestGraphFunction(unittest.TestCase):

    def test_single_return(self):
        def func():
            a = 1
            return a

        graph = get_func_graph(func)
        self.assertEqual(0, len(graph.edges()))
        partials = sorted(get_function_deps(func))
        self.assertEqual(sorted(partials), [])

    def test_unconected_output(self):
        def func(a, b):
            c = a + b
            return c, 2

        graph = get_func_graph(func)
        self.assertIn(('a', 'c'), graph.edges())
        self.assertIn(('b', 'c'), graph.edges())
        self.assertEqual(0, len(graph.edges('out1')))
        partials = sorted(get_function_deps(func))
        self.assertEqual(sorted(partials), [('c', 'a'), ('c', 'b')])

    def test_complicated(self):
        class _Foo(object):
            def __init__(self, a):
                self.a = a
        foo = _Foo(1)
        def func(a, b):
            c = np.tan(a + np.sin(b))
            d = c[0] + 1
            x = np.array([a, d])
            e = 1./np.cos(d*foo.a) * 2
            f = d
            f[0] = x * x
            return e, f

        graph = get_func_graph(func)
        expected = [('a', 'c'), ('a', 'x'), ('b', 'c'), ('d', 'x'), ('c', 'd'), ('d', 'e'), ('d', 'f'), ('x', 'f')]
        self.assertEqual(sorted(graph.edges()), sorted(expected))
        self.assertEqual(sorted(graph.nodes()), sorted(['a', 'b', 'c', 'd', 'e', 'f', 'x']))

        partials = sorted(get_function_deps(func))
        self.assertEqual(sorted(partials), [('e', 'a'), ('e', 'b'), ('f', 'a'), ('f', 'b')])

    def test_multiple_returns(self):
        def func(a, b):
            return a, b

        graph = get_func_graph(func)
        self.assertIn(('a', 'out0'), graph.edges())
        self.assertIn(('b', 'out1'), graph.edges())
        partials = sorted(get_function_deps(func))
        self.assertEqual(sorted(partials), [('out0', 'a'), ('out1', 'b')])

    def test_conditional_return(self):
        def func(a):
            if a > 0:
                return a
            else:
                return -a

        graph = get_func_graph(func)
        self.assertIn(('a', 'out0'), graph.edges())
        partials = sorted(get_function_deps(func))
        self.assertEqual(sorted(partials), [('out0', 'a')])

    def test_no_return(self):
        def func(a):
            b = a + 1  #noqa

        graph = get_func_graph(func)
        self.assertIn(('a', 'b'), graph.edges())
        partials = sorted(get_function_deps(func))
        self.assertEqual(sorted(partials), [])

    def test_function_with_args(self):
        def func(a, b):
            c = a + b
            return c

        graph = get_func_graph(func)
        self.assertIn(('a', 'c'), graph.edges())
        self.assertIn(('b', 'c'), graph.edges())
        partials = sorted(get_function_deps(func))
        self.assertEqual(sorted(partials), [('c', 'a'), ('c', 'b')])

    def test_nested_function(self):
        def func(a):
            def nested(b):
                return b + 1
            return nested(a)

        msg = "Function contains nested functions, which are not supported yet."
        with self.assertRaises(Exception) as cm:
            get_func_graph(func)
        self.assertEqual(cm.exception.args[0], msg)

    def test_function_with_tuple_return(self):
        def func(a, b):
            return a, b

        graph = get_func_graph(func)
        self.assertIn(('a', 'out0'), graph.edges())
        self.assertIn(('b', 'out1'), graph.edges())
        partials = sorted(get_function_deps(func))
        self.assertEqual(sorted(partials), [('out0', 'a'), ('out1', 'b')])

    def test_function_with_mixed_return(self):
        def func(a):
            if a > 0:
                return a, 2
            else:
                return a, -2

        graph = get_func_graph(func)
        self.assertIn(('a', 'out0'), graph.edges())
        partials = sorted(get_function_deps(func))
        self.assertEqual(sorted(partials), [('out0', 'a')])


if __name__ == '__main__':
    unittest.main()

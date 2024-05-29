
import unittest
import os

from openmdao.utils.code_utils import get_nested_calls, LambdaPickleWrapper
from openmdao.core.group import Group


class TestCodeUtils(unittest.TestCase):

    def test_get_nested_calls(self):
        devnull = open(os.devnull, "w")

        graph = get_nested_calls(Group, '_setup', stream=devnull)
        self.assertIn(('Group._setup_var_data', 'System._setup_var_data'),
                      graph.edges(), "System._setup_var_data not called by Group._setup_var_data")


class TestLambdaPickleWrapper(unittest.TestCase):

    def test_init(self):
        func = lambda x: x + 1
        wrapper = LambdaPickleWrapper(func)
        self.assertEqual(wrapper._func, func)

    def test_call(self):
        func = lambda x: x + 1
        wrapper = LambdaPickleWrapper(func)
        self.assertEqual(wrapper(1), 2)

    def test_getstate(self):
        func = lambda x: x + 1
        wrapper = LambdaPickleWrapper(func)
        state = wrapper.__getstate__()
        self.assertEqual(state['_func'], wrapper._getsrc())

    def test_setstate(self):
        func = lambda x: x + 1
        wrapper = LambdaPickleWrapper(func)
        state = {'_func': 'lambda x: x + 2', '_src': None}
        wrapper.__setstate__(state)
        self.assertEqual(wrapper(1), 3)

    def test_getsrc(self):
        func = lambda x: x + 1
        wrapper = LambdaPickleWrapper(func)
        src = wrapper._getsrc()
        self.assertEqual(src, 'lambda x: x + 1')


if __name__ == '__main__':
    unittest.main()


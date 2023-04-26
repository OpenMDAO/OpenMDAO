
import unittest
import os

from openmdao.utils.code_utils import get_nested_calls
from openmdao.core.group import Group


class TestCodeUtils(unittest.TestCase):

    def test_get_nested_calls(self):
        devnull = open(os.devnull, "w")

        graph = get_nested_calls(Group, '_setup', stream=devnull)
        self.assertIn(('Group._setup_var_data', 'System._setup_var_data'),
                      graph.edges(), "System._setup_var_data not called by Group._setup_var_data")


if __name__ == '__main__':
    unittest.main()


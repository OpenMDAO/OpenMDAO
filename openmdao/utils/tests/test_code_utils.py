
import unittest
import os

from openmdao.utils.code_utils import get_nested_calls
from openmdao.core.group import Group


class TestCodeUtils(unittest.TestCase):

    def test_get_nested_calls(self):
        devnull = open(os.devnull, "w")

        graph = get_nested_calls(Group, '_final_setup', stream=devnull)
        self.assertIn(('Group._compute_root_scale_factors', 'System._compute_root_scale_factors'),
                      graph.edges(), "System._compute_root_scale_factors not called by Group._compute_root_scale_factors")


if __name__ == '__main__':
    unittest.main()


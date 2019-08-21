from __future__ import print_function, division, absolute_import

import unittest

from openmdao.utils.general_utils import remove_whitespace

import re
import sys

s = ''.join(chr(c) for c in range(sys.maxunicode+1))
ws = ''.join(re.findall(r'\s', s))


class TestGeneralUtils(unittest.TestCase):

    test_str = ws+'abc'+ws+'def'+ws

    def test_remove_whitespace(self):
        self.assertTrue(remove_whitespace(self.test_str), 'abcdef')

        self.assertTrue(remove_whitespace(self.test_str, right=True), ws+'abc'+ws+'def')

        self.assertTrue(remove_whitespace(self.test_str, left=True), 'abc'+ws+'def'+ws)

        self.assertTrue(remove_whitespace(self.test_str, right=True, left=True), 'abc'+ws+'def')


if __name__ == "__main__":
    unittest.main()

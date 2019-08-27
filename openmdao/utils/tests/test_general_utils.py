import unittest

from openmdao.utils.general_utils import remove_whitespace

import re
import sys

from six import PY2

if PY2:
    # ASCII whitespace
    from string import whitespace as ws
else:
    # Unicode whitespace
    s = ''.join(chr(c) for c in range(sys.maxunicode+1))
    ws = ''.join(re.findall(r'\s', s))


class TestGeneralUtils(unittest.TestCase):

    test_str = ws+'abc'+ws+'def'+ws

    def test_remove_whitespace(self):
        self.assertEqual(remove_whitespace(self.test_str), 'abcdef')

        self.assertEqual(remove_whitespace(self.test_str, right=True), ws+'abc'+ws+'def')

        self.assertEqual(remove_whitespace(self.test_str, left=True), 'abc'+ws+'def'+ws)

        self.assertEqual(remove_whitespace(self.test_str, right=True, left=True), 'abc'+ws+'def')


if __name__ == "__main__":
    unittest.main()

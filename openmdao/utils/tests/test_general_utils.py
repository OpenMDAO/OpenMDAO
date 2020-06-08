import unittest

from openmdao.utils.general_utils import remove_whitespace, common_subpath

import re
import sys

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

    def test_common_subpath(self):
        plists = [
            ([], ''),
            (['foo.bar.baz'], 'foo.bar.baz'),
            (['a.b.c', 'a.b'], 'a.b'),
            (['a.b', 'a.b.c'], 'a.b'),
            (['foo.foo', 'foo.bar'], 'foo'),
            (['foo.foo', 'bar'], ''),
            (['xx.yy.zz.foo.bar', 'xx.yy.zz.blah', 'xx.yy.zz.blah.ho'], 'xx.yy.zz'),
        ]

        for plist, ans in plists:
            self.assertEqual(common_subpath(plist), ans)

if __name__ == "__main__":
    unittest.main()

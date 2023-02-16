import unittest
import copy
from openmdao.core.constants import _UNDEFINED


class Foo(object):
    def __init__(self):
        self.bar = _UNDEFINED


class ConstantsTestCase(unittest.TestCase):
    def test_repr_copy(self):
        cp = copy.copy(_UNDEFINED)
        self.assertTrue(cp is _UNDEFINED, "Constants don't match!")

    def test_repr_deepcopy(self):
        f = Foo()
        cpf = copy.deepcopy(f)
        self.assertTrue(cpf.bar is _UNDEFINED, "Constants don't match!")

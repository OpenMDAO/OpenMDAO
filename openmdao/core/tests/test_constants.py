import unittest
import copy
from openmdao.core.constants import _UNDEFINED
from openmdao.api import is_undefined


class Foo(object):
    def __init__(self):
        self.bar = _UNDEFINED


class ConstantsTestCase(unittest.TestCase):
    def test_repr_copy(self):
        cp = copy.copy(_UNDEFINED)
        self.assertTrue(is_undefined(cp), "Constants don't match!")

    def test_repr_deepcopy(self):
        f = Foo()
        cpf = copy.deepcopy(f)
        self.assertTrue(is_undefined(cpf.bar), "Constants don't match!")

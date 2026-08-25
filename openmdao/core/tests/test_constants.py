import unittest
import copy
from openmdao.core.constants import _UNDEFINED
from openmdao.api import is_undefined
from openmdao.utils.assert_utils import assert_warning
from openmdao.utils.om_warnings import OMDeprecationWarning


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

    def test_inf_bound_deprecated(self):
        msg = 'The INF_BOUND sentinel in OpenMDAO is deprecated'
        with assert_warning(OMDeprecationWarning, msg, contains_msg=True):
            from openmdao.core.constants import INF_BOUND
            self.assertEqual(INF_BOUND, 1.0E30)

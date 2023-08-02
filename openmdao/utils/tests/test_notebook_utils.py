""" Unit tests for the notebook_utils."""

import unittest

try:
    import IPython
except ImportError:
    IPython = False

import openmdao.api as om
from openmdao.utils.assert_utils import assert_warning

class StateOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary specific to states.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(StateOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=str,
                     desc='name of ODE state variable')

@unittest.skipUnless(IPython, "IPython is required")
class TestNotebookUtils(unittest.TestCase):

    @unittest.skipIf(not IPython, reason='Test requires IPython')
    def test_show_obj_options(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = True
        try:
            om.show_options_table("openmdao.utils.tests.test_notebook_utils.StateOptionsDictionary")
        except Exception as e:
            self.fail('show_options_table raised the following exception:\n' + str(e))

    def test_show_options_w_attr(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = True

        options = om.show_options_table("openmdao.components.balance_comp.BalanceComp")

        self.assertEqual(options, None)

    def test_show_options_table_warning(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = False

        msg = ("IPython is not installed. Run `pip install openmdao[notebooks]` or `pip install "
               "openmdao[docs]` to upgrade.")

        with assert_warning(UserWarning, msg):
            om.show_options_table("openmdao.components.balance_comp.BalanceComp")
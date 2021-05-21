""" Unit tests for the notebook_utils."""

import unittest
import tabulate
try:
    import IPython
except ImportError:
    IPython = False

import openmdao.api as om
from openmdao.utils.assert_utils import assert_warning

class StandaloneOptionsDictionary(om.OptionsDictionary):
    def __init__(self, read_only=False):
        super(StandaloneOptionsDictionary, self).__init__(read_only)
        self.declare('units', types=str, allow_none=True,
                     default='s', desc='Units for the integration variable')

@unittest.skipUnless(tabulate and IPython, "Tabulate and IPython are required")
class TestNotebookUtils(unittest.TestCase):

    def test_show_options_wo_attr(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = True

        options = om.show_options_table("openmdao.utils.test_notebook_utils.StandaloneOptionsDictionary")

        self.assertEqual(options, None)

    def test_show_options_w_attr(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = True

        options = om.show_options_table("openmdao.components.balance_comp.BalanceComp")

        self.assertEqual(options, None)

    def test_show_options_table_warning(self):
        msg = ("IPython is not installed. Run `pip install openmdao[notebooks]` or `pip install "
               "openmdao[docs]` to upgrade.")

        with assert_warning(UserWarning, msg):
            om.show_options_table("openmdao.components.balance_comp.BalanceComp")
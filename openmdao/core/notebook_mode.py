"""Import packages and set attributes for OpenMDAO notebooks"""

from openmdao.core.system import System
import openmdao.api as om

class NotebookMode():

    def __init__(self):

        super(NotebookMode, self).__init__()
        try:
            import tabulate
        except ImportError:
            print("Tabulate is not installed, run `pip install openmdao[notebooks]` to install "
                  "required packages")

        self.notebook = True

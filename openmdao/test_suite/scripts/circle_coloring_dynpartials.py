
import os

COLOR_TYPE = os.environ.get('COLOR_TYPE', 'auto')

from openmdao.api import ScipyOptimizeDriver
from openmdao.core.tests.test_coloring import run_opt

if __name__ == '__main__':
    p_color = run_opt(ScipyOptimizeDriver, COLOR_TYPE, optimizer='SLSQP', disp=False,
                      dynamic_total_coloring=True, partial_coloring=True)


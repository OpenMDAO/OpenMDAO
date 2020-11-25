import unittest
import tracemalloc
import gc
import openmdao.api as om
from openmdao.test_suite.components.sellar_feature import SellarMDA

# Compare memory usage after running setup 10 times to
# running setup 100 times
ITERS = [ 10, 100 ]

# The code with the leak problem would report a difference of memory
# used between 10 and 100 iter runs of 36.9MiB. After fixing, it was
# 0.2 KiB.
MAX_MEM_DIFF_KB = 200

class TestSetupMemLeak(unittest.TestCase):
    """ Test for memory leaks when calling setup() multiple times """

    def test_setup_memleak(self):
        # Record the memory still in use after each run
        mem_used = []

        # Setup of the Sellar poblem
        prob = om.Problem()

        tracemalloc.start()

        for memtest in ITERS:
            prob.model = SellarMDA()
            snapshots = []

            for i in range(memtest):
                prob.setup(check=False) # called here causes memory leak   
                prob.run_driver()
                totals = prob.compute_totals("z", "x")
                del totals

                # Force garbage collection now instead of waiting for an
                # optimal/arbitrary point since we're tracking memory usage
                gc.collect()
                snapshots.append(tracemalloc.take_snapshot())

            top_stats = snapshots[memtest - 1].compare_to(snapshots[0], 'lineno')

            total_mem = 0
            for stat in top_stats:
                tb_str = str(stat.traceback)
                # Ignore memory allocations by tracemalloc itself, which throw things off:
                if "/tracemalloc.py:" not in tb_str:
                    total_mem += stat.size
            
            mem_used.append(total_mem)

        tracemalloc.stop()    
        mem_diff = (mem_used[1] - mem_used[0])/1024

        self.assertLess(mem_diff, MAX_MEM_DIFF_KB,
            "Memory leak in setup(): %.1f KiB difference between %d and %d iter runs" %
                (mem_diff, ITERS[0], ITERS[1]))

if __name__ == '__main__':
    unittest.main()
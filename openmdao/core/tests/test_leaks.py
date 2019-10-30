
from contextlib import contextmanager

import openmdao.api as om


@contextmanager
def report_leaks(flags=gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL):
    gc.set_debug(flags)

    yield

    # Force a sweep
    print('Collecting')
    gc.collect()
    print('Done')

    # Report on what was left
    for o in gc.garbage:
        if isinstance(o, Graph):
            print('Retained: {} 0x{:x}'.format(o, id(o)))

    # Reset the debug flags before exiting to avoid dumping a lot
    # of extra information and making the example output more
    # confusing.
    gc.set_debug(0)

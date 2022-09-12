import unittest
import numpy as np

test_seed = 42

# limit our random strings to upper, lower case letters, numbers, :, _, and space
_char_map = [chr(i) for i in range(48, 59)]  # numbers + ':'
_char_map.extend([chr(i) for i in range(65, 91)])  # CAPS
_char_map.extend([chr(i) for i in range(97, 123)])  # lower case
_char_map.append(chr(95))  # '_'
_char_map.append(chr(32))  # ' '


#  generators for random table cells

def _str_gen(nvals, maxsize=25, seed=test_seed):
    randgen = np.random.default_rng(seed)
    ctop = len(_char_map)
    for i in range(nvals):
        sz = randgen.integers(low=1, high=maxsize + 1)
        yield ''.join([_char_map[c] for c in randgen.integers(ctop, size=sz)])


def _bool_gen(nvals, seed=test_seed):
    randgen = np.random.default_rng(seed)
    for i in range(nvals):
        yield randgen.random() > .5


def _real_gen(nvals, low=-10000., high=10000., seed=test_seed):
    randgen = np.random.default_rng(seed)
    mult = high - low
    for i in range(nvals):
        yield randgen.random() * mult + low


def _int_gen(nvals, low=-10000, high=10000, seed=test_seed):
    randgen = np.random.default_rng(seed)
    for i in range(nvals):
        yield randgen.integers(low=low, high=high)


_cell_creators = {
    'real': _real_gen,
    'int': _int_gen,
    'bool': _bool_gen,
    'str': _str_gen,
}

def _create_random_table_data(coltypes, nrows, seed=test_seed):
    colgens = []
    for t, kwargs in coltypes.items():
        colgens.append(_cell_creators[t](nrows, **kwargs))

    headers = [s for s in _str_gen(len(coltypes))]
    rows = []
    for i in range(nrows):
        rows.append([next(cg) for cg in colgens])

    return headers, rows


class TestTables(unittest.TestCase):

    def test_text(self):
        coltypes = {
            'str': {'maxsize': 40},
            'real': {'low': -1e10, 'high': 1e10},
            'real': {},
            'bool': {},
            'str': {'maxsize': 10},
            'int': {'low': -99, 'high': 2500},
            'str': {'maxsize': 60},
        }

        headers, data = _create_random_table_data(coltypes, 10)

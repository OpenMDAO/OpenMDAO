import unittest
import numpy as np

from openmdao.utils.table_builder import generate_table


# limit our random strings to upper, lower case letters, numbers, :, _, and space
_char_map = [chr(i) for i in range(48, 59)]  # numbers + ':'
_char_map.extend([chr(i) for i in range(65, 91)])  # CAPS
_char_map.extend([chr(i) for i in range(97, 123)])  # lower case
_char_map.append(chr(95))  # '_'


#  generators for random table cells

def _str_gen(nvals, nwords=20, maxsize=8, seed=None):
    randgen = np.random.default_rng(seed)
    ctop = len(_char_map)
    for i in range(nvals):
        words = []
        for _ in range(nwords):
            sz = randgen.integers(low=min(4, maxsize), high=maxsize + 1)
            words.append(''.join([_char_map[c] for c in randgen.integers(ctop, size=sz)]))
        yield ' '.join(words)


def _bool_gen(nvals, seed=None):
    randgen = np.random.default_rng(seed)
    for i in range(nvals):
        yield randgen.random() > .5


def _real_gen(nvals, low=-10000., high=10000., seed=None):
    randgen = np.random.default_rng(seed)
    mult = high - low
    for i in range(nvals):
        yield randgen.random() * mult + low


def _int_gen(nvals, low=-10000, high=10000, seed=None):
    randgen = np.random.default_rng(seed)
    for i in range(nvals):
        yield randgen.integers(low=low, high=high)


_cell_creators = {
    'real': _real_gen,
    'int': _int_gen,
    'bool': _bool_gen,
    'str': _str_gen,
}

def _create_random_table_data(coltypes, nrows, seed=None):
    colgens = []
    for t, kwargs in coltypes:
        colgens.append(_cell_creators[t](nrows, seed=seed, **kwargs))
        if seed is not None:
            seed += 1

    headers = [s for s in _str_gen(len(coltypes), nwords=2, maxsize=5, seed=seed)]
    rows = []
    for i in range(nrows):
        rows.append([next(cg) for cg in colgens])

    return headers, rows


# class TestTables(unittest.TestCase):

def random_table(tablefmt='text', nrows=10, coltypes=None, seed=None, **options):
    if coltypes is None:
        coltypes = [
            ('str', {'maxsize': 40}),
            ('real', {'low': -1e10, 'high': 1e10}),
            ('str', {'maxsize': 50}),
            ('bool', {}),
            ('str', {'maxsize': 10}),
            ('int', {'low': -99, 'high': 2500}),
        ]

    headers, data = _create_random_table_data(coltypes, nrows, seed=seed)

    return generate_table(data, tablefmt=tablefmt, headers=headers, **options)


class TableTestCase(unittest.TestCase):
    def test_basic(self):
        self.fail("no test")

    def test_missing_vals(self):
        self.fail("no test")

    def test_word_wrap(self):
        self.fail("no test")



if __name__ == '__main__':
    import sys

    try:
        formats = [sys.argv[1]]
    except Exception:
        formats = ['rst', 'github', 'text', 'html', 'tabulator']

    for fmt in formats:
        tab = random_table(tablefmt=fmt)
        tab.max_width = 120
        tab.display()

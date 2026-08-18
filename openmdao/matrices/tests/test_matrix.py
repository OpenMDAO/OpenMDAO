"""Tests for COOMatrix, CSCMatrix, and CSRMatrix."""
import contextlib
import io
import unittest

import numpy as np

from openmdao.core.constants import INT_DTYPE
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csc_matrix import CSCMatrix
from openmdao.matrices.csr_matrix import CSRMatrix


class MockSubjac:
    """
    Minimal mock of a Subjac for matrix unit testing.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key identifying this subjac.
    rows : array-like
        Row indices of nonzero entries in the assembled jacobian.
    cols : array-like
        Column indices of nonzero entries in the assembled jacobian.
    data : array-like
        Values of the nonzero entries.
    factor : float or None
        Optional scaling factor applied during update.
    """

    def __init__(self, key, rows, cols, data, factor=None):
        """
        Initialize the mock subjac.
        """
        self.key = key
        self._rows = np.asarray(rows, dtype=INT_DTYPE)
        self._cols = np.asarray(cols, dtype=INT_DTYPE)
        self._data = np.asarray(data, dtype=float)
        self.factor = factor

    def get_coo_data_size(self):
        """
        Return the number of nonzero entries.

        Returns
        -------
        int
            Number of nonzero entries.
        """
        return self._rows.size

    def as_coo_info(self, full=False, randgen=None):
        """
        Return COO data, rows, and cols.

        Parameters
        ----------
        full : bool
            Unused in mock — offsets already applied at construction.
        randgen : object or None
            Unused in mock.

        Returns
        -------
        tuple
            (data, rows, cols)
        """
        return self._data.copy(), self._rows.copy(), self._cols.copy()

    def get_as_coo_data(self, randgen=None):
        """
        Return the data values.

        Parameters
        ----------
        randgen : object or None
            Unused in mock.

        Returns
        -------
        ndarray
            Data values.
        """
        return self._data.copy()


def _run_update_cycle(matrix, subjacs, dtype=np.dtype('float64')):
    """
    Run one complete pre_update / update_from_submat / post_update cycle.

    Parameters
    ----------
    matrix : COOMatrix
        The matrix to update.
    subjacs : list of MockSubjac
        The subjacs to update from.
    dtype : dtype
        The dtype to use.
    """
    matrix._pre_update(dtype)
    for subjac in subjacs:
        matrix._update_from_submat(subjac, None)
    matrix._post_update()


class _MatrixUpdateTests:
    """Shared update cycle tests for all sparse matrix formats."""

    def _build(self, submats, shape):
        raise NotImplementedError

    def test_basic_update(self):
        """Subjac values are placed correctly into the compressed matrix."""
        s = MockSubjac(('a', 'b'), [0, 1], [0, 1], [3., 7.])
        mat = self._build({s.key: s}, (2, 2))
        _run_update_cycle(mat, [s])

        np.testing.assert_array_equal(mat._matrix.toarray(), np.array([[3., 0.], [0., 7.]]))

    def test_factor_scaling(self):
        """subjac.factor scales the data before inserting into the matrix."""
        s = MockSubjac(('a', 'b'), [0, 1], [0, 1], [2., 4.], factor=3.)
        mat = self._build({s.key: s}, (2, 2))
        _run_update_cycle(mat, [s])

        np.testing.assert_array_almost_equal(mat._matrix.toarray(), np.array([[6., 0.], [0., 12.]]))

    def test_multiple_update_cycles(self):
        """Repeated update cycles produce the same correct values each time."""
        s = MockSubjac(('a', 'b'), [0], [0], [5.])
        mat = self._build({s.key: s}, (2, 2))

        for _ in range(3):
            _run_update_cycle(mat, [s])
            self.assertAlmostEqual(mat._matrix.toarray()[0, 0], 5.)

    def test_todense(self):
        """todense returns a dense array with the correct values."""
        s = MockSubjac(('a', 'b'), [0, 1], [1, 0], [3., 7.])
        mat = self._build({s.key: s}, (2, 2))
        _run_update_cycle(mat, [s])

        self.assertIsInstance(mat.todense(), np.ndarray)
        np.testing.assert_array_equal(mat.todense(), np.array([[0., 3.], [7., 0.]]))

    def test_dtype_complex(self):
        """Dtype transitions to complex correctly for complex step."""
        s = MockSubjac(('a', 'b'), [0], [0], [1.])
        mat = self._build({s.key: s}, (2, 2))

        mat._pre_update(np.dtype('complex128'))
        self.assertEqual(mat._matrix.data.dtype.kind, 'c')

    def test_dtype_reverts_to_real(self):
        """Dtype transitions back to real after complex step."""
        s = MockSubjac(('a', 'b'), [0], [0], [1.])
        mat = self._build({s.key: s}, (2, 2))

        mat._pre_update(np.dtype('complex128'))
        self.assertEqual(mat._matrix.data.dtype.kind, 'c')

        mat._pre_update(np.dtype('float64'))
        self.assertEqual(mat._matrix.data.dtype.kind, 'f')

    def test_transpose(self):
        """transpose returns the correct transposed matrix."""
        s = MockSubjac(('a', 'b'), [0, 1], [1, 0], [3., 7.])
        mat = self._build({s.key: s}, (2, 2))
        _run_update_cycle(mat, [s])

        np.testing.assert_array_equal(mat.transpose().toarray(),
                                      np.array([[0., 7.], [3., 0.]]))

    def test_dump(self):
        """dump prints msginfo and sorted (row, col, value) entries to stdout."""
        s = MockSubjac(('a', 'b'), [1, 0], [0, 1], [3., 7.])
        mat = self._build({s.key: s}, (2, 2))
        _run_update_cycle(mat, [s])

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mat.dump('test')
        output = buf.getvalue()

        self.assertIn('test', output)
        self.assertIn('0, 1, 7.0', output)
        self.assertIn('1, 0, 3.0', output)


class _CompressedMatrixUpdateTests(_MatrixUpdateTests):
    """Shared update cycle tests for CSC and CSR formats."""

    def test_cross_subjac_duplicates_accumulate(self):
        """Two subjacs writing to the same position accumulate their values."""
        s1 = MockSubjac(('a', 'b'), [0], [0], [10.])
        s2 = MockSubjac(('c', 'd'), [0], [0], [5.])
        mat = self._build({s1.key: s1, s2.key: s2}, (2, 2))
        _run_update_cycle(mat, [s1, s2])

        self.assertAlmostEqual(mat._matrix.toarray()[0, 0], 15.)

    def test_within_subjac_duplicates_accumulate(self):
        """A subjac with repeated (row, col) entries accumulates via np.add.at."""
        s = MockSubjac(('a', 'b'), [0, 0], [0, 0], [10., 5.])
        mat = self._build({s.key: s}, (2, 2))
        _run_update_cycle(mat, [s])

        self.assertAlmostEqual(mat._matrix.toarray()[0, 0], 15.)

    def test_transpose_cached(self):
        """Transpose result is cached and reused."""
        s = MockSubjac(('a', 'b'), [0], [1], [3.])
        mat = self._build({s.key: s}, (2, 2))
        _run_update_cycle(mat, [s])

        t1 = mat.transpose()
        t2 = mat.transpose()
        self.assertIs(t1, t2)

    def test_transpose_cache_reset_on_update(self):
        """Transpose cache is invalidated after each update cycle."""
        s = MockSubjac(('a', 'b'), [0], [1], [3.])
        mat = self._build({s.key: s}, (2, 2))
        _run_update_cycle(mat, [s])
        mat.transpose()

        _run_update_cycle(mat, [s])
        self.assertIsNone(mat._matrix_T)


class _CompressedMatrixBuildTests:
    """Shared _build tests for CSC and CSR formats."""

    _map_attr = None  # set in subclasses: '_coo_to_csc_map' or '_coo_to_csr_map'

    def _build_matrix(self, submats, shape):
        raise NotImplementedError

    def test_map_cross_subjac_duplicates_same_index(self):
        """
        Two COO entries at the same (row, col) from different subjacs map to
        the same index in the compressed matrix.
        """
        s1 = MockSubjac(('a', 'b'), [0], [0], [1.])
        s2 = MockSubjac(('c', 'd'), [0], [0], [2.])
        mat = self._build_matrix({s1.key: s1, s2.key: s2}, (2, 2))

        np.testing.assert_array_equal(getattr(mat, self._map_attr), [0, 0])

    def test_no_within_subjac_duplicates_flagged(self):
        """Subjac with unique (row, col) pairs is not flagged as having duplicates."""
        s = MockSubjac(('a', 'b'), [0, 1, 2], [0, 1, 2], [1., 2., 3.])
        mat = self._build_matrix({s.key: s}, (3, 3))

        self.assertFalse(mat._has_within_subjac_duplicates[s.key])

    def test_within_subjac_duplicates_flagged(self):
        """Subjac with repeated (row, col) pairs is correctly flagged."""
        s = MockSubjac(('a', 'b'), [0, 0], [0, 0], [1., 2.])
        mat = self._build_matrix({s.key: s}, (2, 2))

        self.assertTrue(mat._has_within_subjac_duplicates[s.key])


class TestCOOMatrixBuild(unittest.TestCase):
    """Tests for COOMatrix._build."""

    def test_coo_slices_correct(self):
        """_build correctly records contiguous slices for each subjac."""
        s1 = MockSubjac(('a', 'b'), [0, 1], [0, 1], [1., 2.])
        s2 = MockSubjac(('c', 'd'), [0, 1, 2], [2, 2, 2], [3., 4., 5.])
        mat = COOMatrix({s1.key: s1, s2.key: s2})
        mat._build(3, 3)

        self.assertEqual(mat._coo_slices[s1.key], slice(0, 2))
        self.assertEqual(mat._coo_slices[s2.key], slice(2, 5))

    def test_total_nnz(self):
        """_build allocates arrays of the correct total size."""
        s1 = MockSubjac(('a', 'b'), [0, 1], [0, 1], [1., 2.])
        s2 = MockSubjac(('c', 'd'), [0], [2], [3.])
        mat = COOMatrix({s1.key: s1, s2.key: s2})
        mat._build(3, 3)

        self.assertEqual(mat._coo.data.size, 3)
        self.assertEqual(mat._coo.row.size, 3)
        self.assertEqual(mat._coo.col.size, 3)


class TestCOOMatrixUpdate(_MatrixUpdateTests, unittest.TestCase):
    """Tests for COOMatrix update cycle."""

    def _build(self, submats, shape):
        mat = COOMatrix(submats)
        mat._build(*shape)
        return mat


class TestCSCMatrixBuild(_CompressedMatrixBuildTests, unittest.TestCase):
    """Tests for CSCMatrix._build."""

    _map_attr = '_coo_to_csc_map'

    def _build_matrix(self, submats, shape):
        mat = CSCMatrix(submats)
        mat._build(*shape)
        return mat

    def test_map_values_out_of_csc_order(self):
        """_coo_to_csc_map contains the correct CSC data index for each COO entry.

        COO entries are given in a non-CSC order so the map is non-trivial.
        CSC sorts column-major (col ascending, then row ascending within each col).
        COO entry 0: (row=0, col=2) -> CSC index 2
        COO entry 1: (row=2, col=0) -> CSC index 0
        COO entry 2: (row=1, col=1) -> CSC index 1
        """
        s = MockSubjac(('a', 'b'), [0, 2, 1], [2, 0, 1], [1., 2., 3.])
        mat = CSCMatrix({s.key: s})
        mat._build(3, 3)

        np.testing.assert_array_equal(mat._coo_to_csc_map, [2, 0, 1])


class TestCSCMatrixUpdate(_CompressedMatrixUpdateTests, unittest.TestCase):
    """Tests for CSCMatrix update cycle."""

    def _build(self, submats, shape):
        mat = CSCMatrix(submats)
        mat._build(*shape)
        return mat


class TestCSRMatrixBuild(_CompressedMatrixBuildTests, unittest.TestCase):
    """Tests for CSRMatrix._build."""

    _map_attr = '_coo_to_csr_map'

    def _build_matrix(self, submats, shape):
        mat = CSRMatrix(submats)
        mat._build(*shape)
        return mat

    def test_map_values_out_of_csr_order(self):
        """_coo_to_csr_map contains the correct CSR data index for each COO entry.

        COO entries are given in a non-CSR order so the map is non-trivial.
        CSR sorts row-major (row ascending, then col ascending within each row).
        COO entry 0: (row=0, col=2) -> CSR index 0
        COO entry 1: (row=2, col=0) -> CSR index 2
        COO entry 2: (row=1, col=1) -> CSR index 1
        """
        s = MockSubjac(('a', 'b'), [0, 2, 1], [2, 0, 1], [1., 2., 3.])
        mat = CSRMatrix({s.key: s})
        mat._build(3, 3)

        np.testing.assert_array_equal(mat._coo_to_csr_map, [0, 2, 1])


class TestCSRMatrixUpdate(_CompressedMatrixUpdateTests, unittest.TestCase):
    """Tests for CSRMatrix update cycle."""

    def _build(self, submats, shape):
        mat = CSRMatrix(submats)
        mat._build(*shape)
        return mat


if __name__ == '__main__':
    unittest.main()

"""Define the default Vector class."""
from __future__ import division

from copy import deepcopy
import numbers

from six import iteritems, itervalues
from six.moves import zip

import numpy as np

from openmdao.vectors.vector import Vector, INT_DTYPE
from openmdao.vectors.default_transfer import DefaultTransfer


class DefaultVector(Vector):
    """
    Default NumPy vector.
    """

    TRANSFER = DefaultTransfer

    def _create_data(self):
        """
        Allocate data array.

        This happens only in the top level system.  Child systems use views of the array
        we allocate here.

        Returns
        -------
        ndarray
            zeros array of correct size to hold all of this vector's variables.
        """
        ncol = self._ncol
        size = np.sum(self._system()._var_sizes[self._name][self._typ][self._iproc, :])
        return np.zeros(size) if ncol == 1 else np.zeros((size, ncol))

    def _update_root_data(self):
        """
        Resize the root data if necesary (i.e., due to reconfiguration).
        """
        system = self._system()
        type_ = self._typ
        vec_name = self._name
        iproc = self._iproc
        root_vec = self._root_vector

        self._create_data()

        ext_sizes_t = system._ext_sizes[vec_name][type_]
        int_sizes_t = np.sum(system._var_sizes[vec_name][type_][iproc, :])
        old_sizes_total = len(root_vec._data)

        old_sizes = (
            ext_sizes_t[0],
            old_sizes_total - ext_sizes_t[0] - ext_sizes_t[1],
            ext_sizes_t[1],
        )
        new_sizes = (
            ext_sizes_t[0],
            int_sizes_t,
            ext_sizes_t[1],
        )

        root_vec._data = np.concatenate([
            root_vec._data[:old_sizes[0]],
            np.zeros(new_sizes[1]),
            root_vec._data[old_sizes[0] + old_sizes[1]:],
        ])

        if self._alloc_complex and root_vec._cplx_data.size != root_vec._data.size:
            root_vec._cplx_data = np.zeros(root_vec._data.size, dtype=complex)

        root_vec._initialize_views()

    def _extract_data(self):
        """
        Extract views of arrays from root_vector.

        Returns
        -------
        ndarray
            zeros array of correct size.
        """
        system = self._system()
        type_ = self._typ
        iproc = self._iproc
        root_vec = self._root_vector

        cplx_data = None
        scaling = {}
        if self._do_scaling:
            scaling['phys'] = {}
            scaling['norm'] = {}

        sizes = system._var_sizes[self._name][type_]
        ind1 = system._ext_sizes[self._name][type_][0]
        ind2 = ind1 + np.sum(sizes[iproc, :])

        data = root_vec._data[ind1:ind2]

        # Extract view for complex storage too.
        if self._alloc_complex:
            cplx_data = root_vec._cplx_data[ind1:ind2]

        if self._do_scaling:
            for typ in ('phys', 'norm'):
                root_scale = root_vec._scaling[typ]
                rs0 = root_scale[0]
                if rs0 is None:
                    scaling[typ] = (rs0, root_scale[1][ind1:ind2])
                else:
                    scaling[typ] = (rs0[ind1:ind2], root_scale[1][ind1:ind2])

        return data, cplx_data, scaling

    def _initialize_data(self, root_vector):
        """
        Internally allocate data array.

        Parameters
        ----------
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        if root_vector is None:  # we're the root
            self._data = self._create_data()

            if self._do_scaling:
                self._scaling = {}
                data = self._data
                if self._name == 'nonlinear':
                    self._scaling['phys'] = (np.zeros(data.size), np.ones(data.size))
                    self._scaling['norm'] = (np.zeros(data.size), np.ones(data.size))
                elif self._name == 'linear':
                    # reuse the nonlinear scaling vecs since they're the same as ours
                    nlvec = self._system()._root_vecs[self._kind]['nonlinear']
                    self._scaling['phys'] = (None, nlvec._scaling['phys'][1])
                    self._scaling['norm'] = (None, nlvec._scaling['norm'][1])
                else:
                    self._scaling['phys'] = (None, np.ones(data.size))
                    self._scaling['norm'] = (None, np.ones(data.size))

            # Allocate imaginary for complex step
            if self._alloc_complex:
                self._cplx_data = np.zeros(self._data.shape, dtype=np.complex)

        else:
            self._data, self._cplx_data, self._scaling = self._extract_data()

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Sets the following attributes:
        _views
        _views_flat
        """
        system = self._system()
        type_ = self._typ
        kind = self._kind
        iproc = self._iproc
        ncol = self._ncol

        do_scaling = self._do_scaling
        if do_scaling:
            factors = system._scale_factors
            scaling = self._scaling

        self._views = views = {}
        self._views_flat = views_flat = {}

        alloc_complex = self._alloc_complex
        self._cplx_views = cplx_views = {}
        self._cplx_views_flat = cplx_views_flat = {}

        allprocs_abs2idx_t = system._var_allprocs_abs2idx[self._name]
        sizes_t = system._var_sizes[self._name][type_]
        abs2meta = system._var_abs2meta
        for abs_name in system._var_relevant_names[self._name][type_]:
            idx = allprocs_abs2idx_t[abs_name]

            ind1 = np.sum(sizes_t[iproc, :idx])
            ind2 = ind1 + sizes_t[iproc, idx]
            shape = abs2meta[abs_name]['shape']
            if ncol > 1:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                shape = tuple(list(shape) + [ncol])

            views_flat[abs_name] = v = self._data[ind1:ind2]
            if shape != v.shape:
                v = v.view()
                v.shape = shape
            views[abs_name] = v

            if alloc_complex:
                cplx_views_flat[abs_name] = v = self._cplx_data[ind1:ind2]
                if shape != v.shape:
                    v = v.view()
                    v.shape = shape
                cplx_views[abs_name] = v

            if do_scaling:
                for scaleto in ('phys', 'norm'):
                    scale0, scale1 = factors[abs_name][kind, scaleto]
                    vec = scaling[scaleto]
                    if vec[0] is not None:
                        vec[0][ind1:ind2] = scale0
                    vec[1][ind1:ind2] = scale1

        self._names = frozenset(views)

    def _clone_data(self):
        """
        For each item in _data, replace it with a copy of the data.
        """
        self._data = self._data.copy()

        if self._under_complex_step and self._cplx_data is not None:
            self._cplx_data = self._cplx_data.copy()

    def __iadd__(self, vec):
        """
        Perform in-place vector addition.

        Parameters
        ----------
        vec : <Vector>
            vector to add to self.

        Returns
        -------
        <Vector>
            self + vec
        """
        self._data += vec._data
        return self

    def __isub__(self, vec):
        """
        Perform in-place vector substraction.

        Parameters
        ----------
        vec : <Vector>
            vector to subtract from self.

        Returns
        -------
        <Vector>
            self - vec
        """
        self._data -= vec._data
        return self

    def __imul__(self, val):
        """
        Perform in-place scalar multiplication.

        Parameters
        ----------
        val : int or float
            scalar to multiply self.

        Returns
        -------
        <Vector>
            self * val
        """
        self._data *= val
        return self

    def add_scal_vec(self, val, vec):
        """
        Perform in-place addition of a vector times a scalar.

        Parameters
        ----------
        val : int or float
            scalar.
        vec : <Vector>
            this vector times val is added to self.
        """
        self._data += val * vec._data

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        self._data[:] = vec._data

    def set_const(self, val):
        """
        Set the value of this vector to a constant scalar value.

        Parameters
        ----------
        val : int or float
            scalar to set self to.
        """
        self._data[:] = val

    def dot(self, vec):
        """
        Compute the dot product of the real parts of the current vec and the incoming vec.

        Parameters
        ----------
        vec : <Vector>
            The incoming vector being dotted with self.

        Returns
        -------
        float
            The computed dot product value.
        """
        return np.dot(self._data, vec._data)

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        return np.sum(self._data.real**2) ** 0.5

    def _enforce_bounds_vector(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds, backtracking the entire vector together.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        u = self

        # The assumption is that alpha * du has been added to self (i.e., u)
        # just prior to this method being called. We are currently in the
        # initialization of a line search, and we're trying to ensure that
        # the u does not violate bounds in the first iteration. If it does,
        # we modify the du vector directly.

        # This is the required change in step size, relative to the du vector.
        d_alpha = 0

        # Find the largest amount a bound is violated
        # where positive means a bound is violated - i.e. the required d_alpha.
        mask = du._data != 0
        if mask.any():
            abs_du_mask = np.abs(du._data[mask])
            u_mask = u._data[mask]

            # Check lower bound
            max_d_alpha = np.amax((lower_bounds._data[mask] - u_mask) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

            # Check upper bound
            max_d_alpha = np.amax((u_mask - upper_bounds._data[mask]) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

        if d_alpha > 0:
            # d_alpha will not be negative because it was initialized to be 0
            # and we've only done max operations.
            # d_alpha will not be greater than alpha because the assumption is that
            # the original point was valid - i.e., no bounds were violated.
            # Therefore 0 <= d_alpha <= alpha.

            # We first update u to reflect the required change to du.
            u.add_scal_vec(-d_alpha, du)

            # At this point, we normalize d_alpha by alpha to figure out the relative
            # amount that the du vector has to be reduced, then apply the reduction.
            du *= 1 - d_alpha / alpha

    def _enforce_bounds_scalar(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds on each scalar separately, then backtrack as a vector.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        u = self

        # The assumption is that alpha * step has been added to this vector
        # just prior to this method being called. We are currently in the
        # initialization of a line search, and we're trying to ensure that
        # the initial step does not violate bounds. If it does, we modify
        # the step vector directly.

        # enforce bounds on step in-place.
        u_data = u._data

        # If u > lower, we're just adding zero. Otherwise, we're adding
        # the step required to get up to the lower bound.
        # For du, we normalize by alpha since du eventually gets
        # multiplied by alpha.
        change_lower = np.maximum(u_data, lower_bounds._data) - u_data

        # If u < upper, we're just adding zero. Otherwise, we're adding
        # the step required to get down to the upper bound, but normalized
        # by alpha since du eventually gets multiplied by alpha.
        change_upper = np.minimum(u_data, upper_bounds._data) - u_data

        change = change_lower + change_upper

        u_data += change
        du._data += change / alpha

    def _enforce_bounds_wall(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds on each scalar separately, then backtrack along the wall.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        u = self

        # The assumption is that alpha * step has been added to this vector
        # just prior to this method being called. We are currently in the
        # initialization of a line search, and we're trying to ensure that
        # the initial step does not violate bounds. If it does, we modify
        # the step vector directly.

        # enforce bounds on step in-place.
        u_data = u._data
        du_data = du._data

        # If u > lower, we're just adding zero. Otherwise, we're adding
        # the step required to get up to the lower bound.
        # For du, we normalize by alpha since du eventually gets
        # multiplied by alpha.
        change_lower = np.maximum(u_data, lower_bounds._data) - u_data

        # If u < upper, we're just adding zero. Otherwise, we're adding
        # the step required to get down to the upper bound, but normalized
        # by alpha since du eventually gets multiplied by alpha.
        change_upper = np.minimum(u_data, upper_bounds._data) - u_data

        change = change_lower + change_upper

        u_data += change
        du_data += change / alpha

        # Now we ensure that we will backtrack along the wall during the
        # line search by setting the entries of du at the bounds to zero.
        changed_either = change.astype(bool)
        du_data[changed_either] = 0.

    def __getstate__(self):
        """
        Return state as a dict.

        For pickling vectors in case recording, we want to get rid of
        the system contained within Vectors, because MPI Comm objects cannot
        be pickled using Python3's pickle module.

        Returns
        -------
        dict
            state minus system member.
        """
        state = self.__dict__.copy()
        del state['_system']
        return state

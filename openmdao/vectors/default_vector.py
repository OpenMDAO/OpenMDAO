"""Define the default Vector and Transfer classes."""
from __future__ import division
from copy import deepcopy
import numbers

from six import iteritems, itervalues
from six.moves import zip

import numpy as np

from openmdao.vectors.vector import Vector, Transfer, INT_DTYPE

real_types = tuple([numbers.Real, np.float32, np.float64])


class DefaultTransfer(Transfer):
    """
    Default NumPy transfer.
    """

    def _initialize_transfer(self, in_vec, out_vec):
        """
        Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.

        Parameters
        ----------
        in_vec : <Vector>
            reference to the input vector.
        out_vec : <Vector>
            reference to the output vector.
        """
        in_inds = self._in_inds
        out_inds = self._out_inds

        # filter out any empty transfers
        outs = {}
        ins = {}
        for key in in_inds:
            if len(in_inds[key]) > 0:
                ins[key] = in_inds[key]
                outs[key] = out_inds[key]

        self._in_inds = ins
        self._out_inds = outs

    def transfer(self, in_vec, out_vec, mode='fwd'):
        """
        Perform transfer.

        Parameters
        ----------
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.

        """
        in_inds = self._in_inds
        out_inds = self._out_inds

        if mode == 'fwd':
            do_complex = in_vec._vector_info._under_complex_step and out_vec._alloc_complex

            for key in in_inds:
                in_set_name, out_set_name = key
                # this works whether the vecs have multi columns or not due to broadcasting
                in_vec._data[in_set_name][in_inds[key]] = \
                    out_vec._data[out_set_name][out_inds[key]]

                # Imaginary transfer
                # (for CS, so only need in fwd)
                if do_complex:
                    in_vec._imag_data[in_set_name][in_inds[key]] = \
                        out_vec._imag_data[out_set_name][out_inds[key]]

        else:  # rev
            for key in in_inds:
                in_set_name, out_set_name = key
                np.add.at(
                    out_vec._data[out_set_name], out_inds[key],
                    in_vec._data[in_set_name][in_inds[key]])


class DefaultVector(Vector):
    """
    Default NumPy vector.
    """

    TRANSFER = DefaultTransfer

    def _create_data(self):
        """
        Allocate list of arrays, one for each var_set.

        This happens only in the top level system.  Child systems use views of the arrays
        we allocate here.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        system = self._system
        type_ = self._typ
        iproc = self._iproc
        ncol = self._ncol

        sizes_byset_t = system._var_sizes_byset[self._name][type_]
        sizes_t = system._var_sizes[self._name][type_]

        data = {}
        indices = {}
        nsets = len(sizes_byset_t)  # if we only have 1 varset, we can do some speedups
        for set_name in system._num_var_byset[self._name][type_]:
            size = np.sum(sizes_byset_t[set_name][iproc, :])
            data[set_name] = np.zeros(size) if ncol == 1 else np.zeros((size, ncol))
            if nsets == 1:
                indices[set_name] = slice(None)
            else:
                indices[set_name] = np.empty(size, int)

        if nsets > 1:
            abs2meta = system._var_abs2meta
            allprocs_abs2idx_byset_t = system._var_allprocs_abs2idx_byset[self._name]
            allprocs_abs2idx_t = system._var_allprocs_abs2idx[self._name]
            for abs_name in system._var_relevant_names[self._name][type_]:
                set_name = abs2meta[abs_name]['var_set']

                idx_byset = allprocs_abs2idx_byset_t[abs_name]
                ind_byset1 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset])
                ind_byset2 = ind_byset1 + sizes_byset_t[set_name][iproc, idx_byset]

                idx = allprocs_abs2idx_t[abs_name]
                ind1 = np.sum(sizes_t[iproc, :idx])
                ind2 = ind1 + sizes_t[iproc, idx]

                indices[set_name][ind_byset1:ind_byset2] = np.arange(ind1, ind2)

        return data, indices

    def _update_root_data(self):
        """
        Resize the root data if necesary (i.e., due to reconfiguration).
        """
        system = self._system
        type_ = self._typ
        vec_name = self._name
        iproc = self._iproc
        root_vec = self._root_vector

        _, tmp_indices = self._create_data()

        ext_sizes_t = system._ext_sizes[vec_name][type_]
        int_sizes_t = np.sum(system._var_sizes[vec_name][type_][iproc, :])
        old_sizes_total = np.sum([len(data) for data in itervalues(root_vec._data)])

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

        sizes_byset = system._var_sizes_byset[vec_name]
        nsets = len(sizes_byset)

        if nsets > 1:
            for set_name in system._num_var_byset[self._name][type_]:
                if isinstance(tmp_indices[set_name], slice):
                    start, stop, _ = tmp_indices[set_name].indices(old_sizes_total)
                    tmp_indices[set_name] = np.arange(start, stop, dtype=int)

                if isinstance(root_vec._indices[set_name], slice):
                    start, stop, _ = \
                        root_vec._indices[set_name].indices(root_vec._data[set_name].size)
                    root_vec._indices[set_name] = np.arange(start, stop, dtype=int)

        for set_name in system._num_var_byset[self._name][type_]:
            ext_sizes_byset_t = system._ext_sizes_byset[vec_name][type_][set_name]
            int_sizes_byset_t = np.sum(sizes_byset[type_][set_name][iproc, :])
            old_sizes_total_byset = len(root_vec._data[set_name])

            old_sizes_byset = (
                ext_sizes_byset_t[0],
                old_sizes_total_byset - ext_sizes_byset_t[0] - ext_sizes_byset_t[1],
                ext_sizes_byset_t[1],
            )
            new_sizes_byset = (
                ext_sizes_byset_t[0],
                int_sizes_byset_t,
                ext_sizes_byset_t[1],
            )

            root_vec._data[set_name] = np.concatenate([
                root_vec._data[set_name][:old_sizes_byset[0]],
                np.zeros(new_sizes_byset[1]),
                root_vec._data[set_name][old_sizes_byset[0] + old_sizes_byset[1]:],
            ])

            if nsets == 1:
                root_vec._indices[set_name] = slice(None)
            else:
                root_vec._indices[set_name] = np.concatenate([
                    root_vec._indices[set_name][:old_sizes_byset[0]],
                    tmp_indices[set_name] + new_sizes[0],
                    root_vec._indices[set_name][old_sizes_byset[0] + old_sizes_byset[1]:]
                        + new_sizes[1] - old_sizes[1],
                ])

        root_vec._initialize_views()

    def _extract_data(self):
        """
        Extract views of arrays from root_vector.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        system = self._system
        type_ = self._typ
        iproc = self._iproc
        root_vec = self._root_vector

        offset = system._ext_sizes[self._name][type_][0]
        sizes_byset = system._var_sizes_byset[self._name][type_]

        data = {}
        imag_data = {}
        indices = {}
        scaling = {}
        if self._do_scaling:
            scaling['phys'] = {}
            scaling['norm'] = {}

        nsets = len(sizes_byset)  # if we only have 1 varset, we can do some speedups

        for set_name, sizes in iteritems(sizes_byset):
            ind_byset1 = system._ext_sizes_byset[self._name][type_][set_name][0]
            ind_byset2 = ind_byset1 + np.sum(sizes[iproc, :])

            data[set_name] = root_vec._data[set_name][ind_byset1:ind_byset2]
            if nsets == 1:
                indices[set_name] = slice(None)
            else:
                indices[set_name] = root_vec._indices[set_name][ind_byset1:ind_byset2] - offset

            # Extract view for imaginary part too
            if self._alloc_complex:
                if root_vec._alloc_complex:
                    imag_data[set_name] = root_vec._imag_data[set_name][ind_byset1:ind_byset2]
                else:
                    shape = root_vec._data[set_name][ind_byset1:ind_byset2].shape
                    imag_data[set_name] = np.zeros(shape)

            if self._do_scaling:
                for typ in ('phys', 'norm'):
                    root_scale = root_vec._scaling[typ][set_name]
                    rs0 = root_scale[0]
                    if rs0 is None:
                        scaling[typ][set_name] = (rs0, root_scale[1][ind_byset1:ind_byset2])
                    else:
                        scaling[typ][set_name] = (rs0[ind_byset1:ind_byset2],
                                                  root_scale[1][ind_byset1:ind_byset2])

        return data, imag_data, scaling, indices

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Sets the following attributes:
        _data
        _indices

        Parameters
        ----------
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        if root_vector is None:  # we're the root
            self._data, self._indices = self._create_data()

            if self._do_scaling:
                self._scaling = {'phys': {}, 'norm': {}}
                for set_name, data in iteritems(self._data):
                    if self._name == 'linear':
                        # reuse the nonlinear scaling vecs since they're the same as ours
                        nlvec = self._system._root_vecs[self._kind]['nonlinear']
                        self._scaling['phys'][set_name] = (None,
                                                           nlvec._scaling['phys'][set_name][1])
                        self._scaling['norm'][set_name] = (None,
                                                           nlvec._scaling['norm'][set_name][1])
                    else:
                        dphys1 = np.ones(data.size)
                        dnorm1 = np.ones(data.size)
                        if self._name == 'nonlinear':
                            dphys0 = np.zeros(data.size)
                            dnorm0 = np.zeros(data.size)
                        else:
                            dphys0 = dnorm0 = None
                        self._scaling['phys'][set_name] = (dphys0, dphys1)
                        self._scaling['norm'][set_name] = (dnorm0, dnorm1)

            # Allocate imaginary for complex step
            if self._alloc_complex:
                self._imag_data = deepcopy(self._data)

        else:
            self._data, self._imag_data, self._scaling, self._indices = self._extract_data()

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Sets the following attributes:
        _views
        _views_flat
        """
        system = self._system
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
        self._imag_views = imag_views = {}
        self._imag_views_flat = imag_views_flat = {}

        allprocs_abs2idx_byset_t = system._var_allprocs_abs2idx_byset[self._name]
        sizes_byset_t = system._var_sizes_byset[self._name][type_]
        abs2meta = system._var_abs2meta
        for abs_name in system._var_relevant_names[self._name][type_]:
            idx_byset = allprocs_abs2idx_byset_t[abs_name]
            set_name = abs2meta[abs_name]['var_set']

            ind_byset1 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset])
            ind_byset2 = ind_byset1 + sizes_byset_t[set_name][iproc, idx_byset]
            shape = abs2meta[abs_name]['shape']
            if ncol > 1:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                shape = tuple(list(shape) + [ncol])

            views_flat[abs_name] = v = self._data[set_name][ind_byset1:ind_byset2]
            if shape != v.shape:
                v = v.view()
                v.shape = shape
            views[abs_name] = v

            if alloc_complex:
                imag_views_flat[abs_name] = v = self._imag_data[set_name][ind_byset1:ind_byset2]
                if shape != v.shape:
                    v = v.view()
                    v.shape = shape
                imag_views[abs_name] = v

            if do_scaling:
                for scaleto in ('phys', 'norm'):
                    scale0, scale1 = factors[abs_name][kind, scaleto]
                    vec = scaling[scaleto][set_name]
                    if vec[0] is not None:
                        vec[0][ind_byset1:ind_byset2] = scale0
                    vec[1][ind_byset1:ind_byset2] = scale1

        self._names = frozenset(views)

    def _clone_data(self):
        """
        For each item in _data, replace it with a copy of the data.
        """
        for set_name, data in iteritems(self._data):
            self._data[set_name] = np.array(data)

        if self._vector_info._under_complex_step:
            for set_name, data in iteritems(self._imag_data):
                self._imag_data[set_name] = np.array(data)

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
        for set_name, data in iteritems(self._data):
            data += vec._data[set_name]

        if vec._alloc_complex and self._vector_info._under_complex_step:
            for set_name, data in iteritems(self._imag_data):
                data += vec._imag_data[set_name]
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
        for set_name, data in iteritems(self._data):
            data -= vec._data[set_name]
        if vec._alloc_complex and self._vector_info._under_complex_step:
            for set_name, data in iteritems(self._imag_data):
                data -= vec._imag_data[set_name]
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
        if self._vector_info._under_complex_step:
            r_val = np.real(val)
            i_val = np.imag(val)
            for key in self._data:
                r_data = self._data[key]
                i_data = self._imag_data[key]
                self._data[key] = r_val * r_data + i_val * i_data
                self._imag_data[key] = r_val * i_data + i_val * r_data
        else:
            for data in itervalues(self._data):
                data *= val
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
        if self._vector_info._under_complex_step:
            r_val = np.real(val)
            i_val = np.imag(val)
            for set_name, data in iteritems(self._data):
                data += r_val * vec._data[set_name] + i_val * vec._imag_data[set_name]
            for set_name, data in iteritems(self._imag_data):
                data += i_val * vec._data[set_name] + r_val * vec._imag_data[set_name]
        else:
            for set_name, data in iteritems(self._data):
                data += val * vec._data[set_name]

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        for set_name, data in iteritems(self._data):
            data[:] = vec._data[set_name]
        if self._vector_info._under_complex_step:
            for set_name, data in iteritems(self._imag_data):
                data[:] = vec._imag_data[set_name][:]

    def set_const(self, val):
        """
        Set the value of this vector to a constant scalar value.

        Parameters
        ----------
        val : int or float
            scalar to set self to.
        """
        for data in itervalues(self._data):
            data[:] = val

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
        global_sum = 0
        for set_name, data in iteritems(self._data):
            global_sum += np.dot(data, vec._data[set_name])

        return global_sum

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        global_sum = 0
        for data in itervalues(self._data):
            global_sum += np.sum(data**2)
        return global_sum ** 0.5

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

        # Loop over varsets and find the largest amount a bound is violated
        # where positive means a bound is violated - i.e. the required d_alpha.
        for u_data, du_data, lower_data, upper_data in zip(
                itervalues(u._data), itervalues(du._data),
                itervalues(lower_bounds._data), itervalues(upper_bounds._data)):

            mask = du_data != 0
            if mask.any():
                abs_du_mask = np.abs(du_data[mask])
                u_mask = u_data[mask]

                # Check lower bound
                max_d_alpha = np.amax((lower_data[mask] - u_mask) / abs_du_mask)
                if max_d_alpha > d_alpha:
                    d_alpha = max_d_alpha

                # Check upper bound
                max_d_alpha = np.amax((u_mask - upper_data[mask]) / abs_du_mask)
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

        # Loop over varsets and enforce bounds on step in-place.
        for u_data, du_data, lower_data, upper_data in zip(
                itervalues(u._data), itervalues(du._data),
                itervalues(lower_bounds._data), itervalues(upper_bounds._data)):

            # If u > lower, we're just adding zero. Otherwise, we're adding
            # the step required to get up to the lower bound.
            # For du, we normalize by alpha since du eventually gets
            # multiplied by alpha.
            change_lower = np.maximum(u_data, lower_data) - u_data

            # If u < upper, we're just adding zero. Otherwise, we're adding
            # the step required to get down to the upper bound, but normalized
            # by alpha since du eventually gets multiplied by alpha.
            change_upper = np.minimum(u_data, upper_data) - u_data

            change = change_lower + change_upper

            u_data += change
            du_data += change / alpha

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

        # Loop over varsets and enforce bounds on step in-place.
        for u_data, du_data, lower_data, upper_data in zip(
                itervalues(u._data), itervalues(du._data),
                itervalues(lower_bounds._data), itervalues(upper_bounds._data)):

            # If u > lower, we're just adding zero. Otherwise, we're adding
            # the step required to get up to the lower bound.
            # For du, we normalize by alpha since du eventually gets
            # multiplied by alpha.
            change_lower = np.maximum(u_data, lower_data) - u_data

            # If u < upper, we're just adding zero. Otherwise, we're adding
            # the step required to get down to the upper bound, but normalized
            # by alpha since du eventually gets multiplied by alpha.
            change_upper = np.minimum(u_data, upper_data) - u_data

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

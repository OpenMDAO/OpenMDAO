"""Define the default Vector and Transfer classes."""
from __future__ import division
import numpy as np

import numbers
from six import iteritems, itervalues
from six.moves import range, zip

from openmdao.vectors.vector import Vector, Transfer
from openmdao.vectors.vector import VectorX, TransferX

real_types = tuple([numbers.Real, np.float32, np.float64])


class DefaultTransferX(TransferX):
    """
    Default NumPy transfer.
    """

    def _initialize_transfer(self):
        """
        Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.
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

    def __call__(self, in_vec, out_vec, mode='fwd'):
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
            for key in in_inds:
                in_set_name, out_set_name = key
                in_vec._data[in_set_name][in_inds[key]] = \
                    out_vec._data[out_set_name][out_inds[key]]
        elif mode == 'rev':
            for key in in_inds:
                in_set_name, out_set_name = key
                np.add.at(
                    out_vec._data[out_set_name], out_inds[key],
                    in_vec._data[in_set_name][in_inds[key]])


class DefaultVectorX(VectorX):
    """
    Default NumPy vector.
    """

    TRANSFER = DefaultTransferX

    def _create_data(self):
        """
        Allocate list of arrays, one for each var_set.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        system = self._system
        type_ = self._typ
        iproc = self._iproc

        sizes_byset_t = system._varx_sizes_byset[type_]
        sizes_t = system._varx_sizes[type_]
        allprocs_abs2idx_t = system._varx_allprocs_abs2idx[type_]
        allprocs_abs2idx_byset_t = system._varx_allprocs_abs2idx_byset[type_]
        abs2meta_t = system._varx_abs2meta[type_]

        data = {}
        indices = {}
        for set_name in system._varx_set2iset[type_]:
            size = np.sum(sizes_byset_t[set_name][iproc, :])
            data[set_name] = np.zeros(size)
            indices[set_name] = np.zeros(size, int)

        for abs_name in system._varx_abs_names[type_]:
            idx = allprocs_abs2idx_t[abs_name]
            idx_byset = allprocs_abs2idx_byset_t[abs_name]
            set_name = abs2meta_t[abs_name]['var_set']

            ind1 = np.sum(sizes_t[iproc, :idx])
            ind2 = np.sum(sizes_t[iproc, :idx + 1])
            ind_byset1 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset])
            ind_byset2 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset + 1])

            set_name = abs2meta_t[abs_name]['var_set']
            indices[set_name][ind_byset1:ind_byset2] = np.arange(ind1, ind2)

        return data, indices

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

        offset = system._ext_sizes[type_][0]

        data = {}
        indices = {}
        for set_name in system._varx_set2iset[type_]:
            offset_byset = system._ext_sizes_byset[type_][set_name][0]
            ind_byset1 = offset_byset
            ind_byset2 = offset_byset + np.sum(system._varx_sizes_byset[type_][set_name][iproc, :])

            data[set_name] = root_vec._data[set_name][ind_byset1:ind_byset2]
            indices[set_name] = root_vec._indices[set_name][ind_byset1:ind_byset2] - offset

        return data, indices

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
        if root_vector is None:
            self._data, self._indices = self._create_data()
        else:
            self._data, self._indices = self._extract_data()

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Sets the following attributes:
        _views
        _views_flat
        _idxs
        """
        system = self._system
        type_ = self._typ
        iproc = self._iproc

        allprocs_abs2idx_t = system._varx_allprocs_abs2idx[type_]
        allprocs_abs2idx_byset_t = system._varx_allprocs_abs2idx_byset[type_]
        sizes_byset_t = system._varx_sizes_byset[type_]
        abs2meta_t = system._varx_abs2meta[type_]
        var_range_byset_t = system._varx_range_byset[type_]

        # idxs contains a 0 index for floats or a slice(None) for arrays so getitem
        # will return either a float or a properly shaped array respectively.
        idxs = {}
        views = {}
        views_flat = {}

        for abs_name in system._varx_abs_names[type_]:
            idx_byset = allprocs_abs2idx_byset_t[abs_name]
            set_name = abs2meta_t[abs_name]['var_set']

            ind_byset1 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset])
            ind_byset2 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset + 1])
            shape = abs2meta_t[abs_name]['shape']

            views_flat[abs_name] = self._data[set_name][ind_byset1:ind_byset2]
            views[abs_name] = self._data[set_name][ind_byset1:ind_byset2]
            views[abs_name].shape = shape

            # The shape entry overrides value's shape, which is why we don't
            # use the shape of val as the reference
            if np.prod(shape) == 1:
                idxs[abs_name] = 0
            else:
                idxs[abs_name] = slice(None)

        self._views = self._names = views
        self._views_flat = views_flat
        self._idxs = idxs

    def _clone_data(self):
        """
        For each item in _data, replace it with a copy of the data.
        """
        for set_name, data in iteritems(self._data):
            self._data[set_name] = np.array(data)

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
        for set_name, data in iteritems(self._data):
            data += val * vec._data[set_name]

    def elem_mult(self, vec):
        """
        Perform element-wise multiplication and store the result in this vector.

        Parameters
        ----------
        vec : <Vector>
            The vector to perform element-wise multiplication with.
        """
        for set_name, data in iteritems(self._data):
            data[:] *= vec._data[set_name]

    def elem_div(self, vec):
        """
        Perform element-wise division and store the result in this vector.

        Parameters
        ----------
        vec : <Vector>
            The vector to perform element-wise division with.
        """
        for set_name, data in iteritems(self._data):
            data[:] /= vec._data[set_name]

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

    def _scale(self, coeffs):
        """
        Change the scaling state.

        Parameters
        ----------
        coeffs : int ndarray[nvar_myproc, 2]
            0th and 1st order coefficients for scaling/unscaling.
        """
        pass

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
                u._data, du._data, lower_bounds._data, upper_bounds._data):

            mask = du_data != 0
            if mask.any():
                abs_du_mask = np.abs(du_data[mask])

                # Check lower bound
                max_d_alpha = np.amax((lower_data[mask] - u_data[mask]) / abs_du_mask)
                d_alpha = max(d_alpha, max_d_alpha)

                # Check upper bound
                max_d_alpha = np.amax((u_data[mask] - upper_data[mask]) / abs_du_mask)
                d_alpha = max(d_alpha, max_d_alpha)

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
                u._data, du._data, lower_bounds._data, upper_bounds._data):

            # If u > lower, we're just adding zero. Otherwise, we're adding
            # the step required to get up to the lower bound.
            # For du, we normalize by alpha since du eventually gets
            # multiplied by alpha.
            change = np.maximum(u_data, lower_data) - u_data
            u_data += change
            du_data += change / alpha

            # If u < upper, we're just adding zero. Otherwise, we're adding
            # the step required to get down to the upper bound, but normalized
            # by alpha since du eventually gets multiplied by alpha.
            change = np.minimum(u_data, upper_data) - u_data
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
                u._data, du._data, lower_bounds._data, upper_bounds._data):

            # If u > lower, we're just adding zero. Otherwise, we're adding
            # the step required to get up to the lower bound.
            # For du, we normalize by alpha since du eventually gets
            # multiplied by alpha.
            change_lower = np.maximum(u_data, lower_data) - u_data
            u_data += change_lower
            du_data += change_lower / alpha

            # If u < upper, we're just adding zero. Otherwise, we're adding
            # the step required to get down to the upper bound, but normalized
            # by alpha since du eventually gets multiplied by alpha.
            change_upper = np.minimum(u_data, upper_data) - u_data
            u_data += change_upper
            du_data += change_upper / alpha

            # Now we ensure that we will backtrack along the wall during the
            # line search by setting the entries of du at the bounds to zero.
            changed_either = change_lower.astype(bool) + change_upper.astype(bool)
            du_data[changed_either] = 0.


class DefaultTransfer(Transfer):
    """
    Default NumPy transfer.
    """

    def _initialize_transfer(self):
        """
        Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.
        """
        outs = {}
        ins = {}
        in_inds = self._in_inds
        out_inds = self._out_inds

        # filter out any empty transfers
        for key in in_inds:
            if len(in_inds[key]) > 0:
                ins[key] = in_inds[key]
                outs[key] = out_inds[key]

        self._in_inds = ins
        self._out_inds = outs

    def __call__(self, in_vec, out_vec, mode='fwd'):
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
            for key in in_inds:
                in_iset, out_iset = key
                in_vec._root_vector._data[in_iset][in_inds[key]] = \
                    out_vec._root_vector._data[out_iset][out_inds[key]]
        elif mode == 'rev':
            for key in in_inds:
                in_iset, out_iset = key
                np.add.at(out_vec._root_vector._data[out_iset], out_inds[key],
                          in_vec._root_vector._data[in_iset][in_inds[key]])


class DefaultVector(Vector):
    """
    Default NumPy vector.
    """

    TRANSFER = DefaultTransfer

    def _create_data(self):
        """
        Allocate list of arrays, one for each var_set.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        data = [np.zeros(np.sum(sizes[self._iproc, :]))
                for sizes in self._assembler._var_sizes_by_set[self._typ]]
        indices = [np.zeros(np.sum(sizes[self._iproc, :]), int)
                   for sizes in self._assembler._var_sizes_by_set[self._typ]]

        system = self._system
        assembler = self._assembler

        sizes_all = assembler._var_sizes_all[self._typ]
        sizes = assembler._var_sizes_by_set[self._typ]

        for abs_name in system._var_abs_names[self._typ]:
            idx = assembler._var_allprocs_abs2idx_io[abs_name]
            ivar_set, ivar = assembler._var_set_indices[self._typ][idx, :]

            ivar_all = idx
            ind1 = np.sum(sizes[ivar_set][self._iproc, :ivar])
            ind2 = np.sum(sizes[ivar_set][self._iproc, :ivar + 1])
            ind1_all = np.sum(sizes_all[self._iproc, :ivar_all])
            ind2_all = np.sum(sizes_all[self._iproc, :ivar_all + 1])
            indices[ivar_set][ind1:ind2] = np.arange(ind1_all, ind2_all)

        return data, indices

    def _extract_data(self):
        """
        Extract views of arrays from root_vector.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        system = self._system
        assembler = system._assembler

        sys_start, sys_end = self._system._var_allprocs_idx_range[self._typ]

        var_set_indices = assembler._var_set_indices[self._typ]
        sub_var_set_indices = var_set_indices[sys_start:sys_end, :]

        ind_offset = np.sum(assembler._var_sizes_all[self._typ][self._iproc, :sys_start])

        data = []
        indices = []
        for iset in range(len(assembler._var_sizes_by_set[self._typ])):
            bool_vector = sub_var_set_indices[:, 0] == iset
            data_inds = sub_var_set_indices[bool_vector, 1]
            if len(data_inds) > 0:
                sizes_array = assembler._var_sizes_by_set[self._typ][iset]
                ind1 = np.sum(sizes_array[self._iproc, :data_inds[0]])
                ind2 = np.sum(sizes_array[self._iproc, :data_inds[-1] + 1])
                data.append(self._root_vector._data[iset][ind1:ind2])
                indices.append(self._root_vector._indices[iset][ind1:ind2] - ind_offset)
            else:
                data.append(np.zeros(0))
                indices.append(np.zeros(0, int))

        return data, indices

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Sets the following attributes:

        - _data

        Parameters
        ----------
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        if root_vector is None:
            self._data, self._indices = self._create_data()
        else:
            self._data, self._indices = self._extract_data()

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Sets the following attributes:

        - _views
        - _views_flat
        - _idxs

        """
        system = self._system
        assembler = self._assembler

        views = {}
        views_flat = {}

        # contains a 0 index for floats or a slice(None) for arrays so getitem
        # will return either a float or a properly shaped array respectively.
        idxs = {}

        ind_offsets = {}

        for abs_name in system._var_abs_names[self._typ]:
            idx = assembler._var_allprocs_abs2idx_io[abs_name]
            iset, ivar = assembler._var_set_indices[self._typ][idx, :]
            sizes_array = assembler._var_sizes_by_set[self._typ][iset]
            ind1 = np.sum(sizes_array[self._iproc, :ivar])
            ind2 = np.sum(sizes_array[self._iproc, :ivar + 1])

            # TODO: Optimize by precomputing offsets
            if iset not in ind_offsets:
                ind_offsets[iset] = ind1
            ind1 -= ind_offsets[iset]
            ind2 -= ind_offsets[iset]

            metadata = system._var_abs2data_io[abs_name]['metadata']

            views[abs_name] = self._data[iset][ind1:ind2]
            views_flat[abs_name] = self._data[iset][ind1:ind2]
            views[abs_name].shape = metadata['shape']

            # The shape entry overrides value's shape, which is why we don't
            # use the shape of val as the reference
            if np.prod(metadata['shape']) == 1:
                idxs[abs_name] = 0
            else:
                idxs[abs_name] = slice(None)

        self._views = self._names = views
        self._views_flat = views_flat
        self._idxs = idxs

    def _clone_data(self):
        """
        For each item in _data, replace it with a copy of the data.
        """
        for iset in range(len(self._data)):
            data = self._data[iset]
            self._data[iset] = np.array(data)

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
        for data, vec_data in zip(self._data, vec._data):
            data += vec_data
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
        for data, vec_data in zip(self._data, vec._data):
            data -= vec_data
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
        for data in self._data:
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
        for data, vec_data in zip(self._data, vec._data):
            data += val * vec_data

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        for data, vec_data in zip(self._data, vec._data):
            data[:] = vec_data

    def set_const(self, val):
        """
        Set the value of this vector to a constant scalar value.

        Parameters
        ----------
        val : int or float
            scalar to set self to.
        """
        for data in self._data:
            data[:] = val

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        global_sum = 0
        for data in self._data:
            global_sum += np.sum(data**2)
        return global_sum ** 0.5

    def _scale(self, coeffs):
        """
        Change the scaling state.

        Parameters
        ----------
        coeffs : int ndarray[nvar_myproc, 2]
            0th and 1st order coefficients for scaling/unscaling.
        """
        for iset, data in enumerate(self._data):
            idx = self._ivar_map[iset]
            data *= coeffs[idx, 1]
            data += coeffs[idx, 0]

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
                u._data, du._data, lower_bounds._data, upper_bounds._data):

            mask = du_data != 0
            if mask.any():
                abs_du_mask = np.abs(du_data[mask])

                # Check lower bound
                max_d_alpha = np.amax((lower_data[mask] - u_data[mask]) / abs_du_mask)
                d_alpha = max(d_alpha, max_d_alpha)

                # Check upper bound
                max_d_alpha = np.amax((u_data[mask] - upper_data[mask]) / abs_du_mask)
                d_alpha = max(d_alpha, max_d_alpha)

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
                u._data, du._data, lower_bounds._data, upper_bounds._data):

            # If u > lower, we're just adding zero. Otherwise, we're adding
            # the step required to get up to the lower bound.
            # For du, we normalize by alpha since du eventually gets
            # multiplied by alpha.
            change = np.maximum(u_data, lower_data) - u_data
            u_data += change
            du_data += change / alpha

            # If u < upper, we're just adding zero. Otherwise, we're adding
            # the step required to get down to the upper bound, but normalized
            # by alpha since du eventually gets multiplied by alpha.
            change = np.minimum(u_data, upper_data) - u_data
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
                u._data, du._data, lower_bounds._data, upper_bounds._data):

            # If u > lower, we're just adding zero. Otherwise, we're adding
            # the step required to get up to the lower bound.
            # For du, we normalize by alpha since du eventually gets
            # multiplied by alpha.
            change_lower = np.maximum(u_data, lower_data) - u_data
            u_data += change_lower
            du_data += change_lower / alpha

            # If u < upper, we're just adding zero. Otherwise, we're adding
            # the step required to get down to the upper bound, but normalized
            # by alpha since du eventually gets multiplied by alpha.
            change_upper = np.minimum(u_data, upper_data) - u_data
            u_data += change_upper
            du_data += change_upper / alpha

            # Now we ensure that we will backtrack along the wall during the
            # line search by setting the entries of du at the bounds to zero.
            changed_either = change_lower.astype(bool) + change_upper.astype(bool)
            du_data[changed_either] = 0.

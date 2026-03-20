"""Finite difference derivative approximations."""
from collections import namedtuple

import numpy as np

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme, _is_group


# For rel_element step sizing, we need to track local indices for each idx_info tuple.
# Local indices are computed once (either per iteration in the generator or upfront for colored)
# and stored here to avoid repeatedly computing them in _run_point and _run_sub_point.
VecIdxInfo = namedtuple('IdxInfo', ['vec', 'idxs', 'local_idxs'])


DEFAULT_ORDER = {
    'forward': 1,
    'backward': 1,
    'central': 2,
}


def _generate_fd_coeff(form, order, system):
    """
    Create an FDForm namedtuple containing the deltas, coefficients, and current coefficient.

    Parameters
    ----------
    form : str
        Requested form of FD (e.g. 'forward', 'central', 'backward').
    order : int
        The order of accuracy of the requested FD scheme.
    system : System
        Containing system.

    Returns
    -------
    FDForm
        namedtuple containing the 'deltas', 'coeffs', and 'current_coeff'. These deltas and
        coefficients need to be scaled by the step size.
    """
    FDForm = namedtuple('FDForm', ['deltas', 'coeffs', 'current_coeff'])

    FD_COEFFS = {
        ('forward', 1): FDForm(deltas=np.array([1.0]),
                               coeffs=np.array([1.0]),
                               current_coeff=-1.0),
        ('backward', 1): FDForm(deltas=np.array([-1.0]),
                                coeffs=np.array([-1.0]),
                                current_coeff=1.0),
        ('central', 2): FDForm(deltas=np.array([1.0, -1.0]),
                               coeffs=np.array([0.5, -0.5]),
                               current_coeff=0.),
    }

    try:
        fd_form = FD_COEFFS[form, order]
    except KeyError:
        # TODO: Automatically generate requested form and store in dict.
        raise ValueError('{}: Finite Difference form="{}" and order={} are not '
                         'supported'.format(system.msginfo, form, order))
    return fd_form


class FiniteDifference(ApproximationScheme):
    r"""
    Approximation scheme using finite differences to estimate derivatives.

    For example, using the 'forward' form with a step size of 'h' will approximate the derivative in
    the following way:

    .. math::

        f'(x) = \frac{f(x+h) - f(x)}{h} + O(h).

    Attributes
    ----------
    _starting_outs : ndarray
        A copy of the starting outputs array used to restore the outputs to original values.
    _starting_ins : ndarray
        A copy of the starting inputs array used to restore the inputs to original values.
    _results_tmp : ndarray
        An array the same size as the system outputs. Used to store the results temporarily.
    """

    DEFAULT_OPTIONS = {
        'step': 1e-6,
        'form': 'forward',
        'order': None,
        'step_calc': 'abs',
        'directional': False,
        'minimum_step': 1e-12,
    }

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        super().__init__()
        self._starting_ins = self._starting_outs = self._results_tmp = None

    def _colored_column_iter(self, system, colored_approx_groups):
        """
        Perform colored approximations with local indices computed upfront in vec_ind_list.

        Augments vec_ind_list with VecIdxInfo namedtuples containing computed local indices
        before calling the parent method. This ensures local indices are computed once,
        not repeatedly in _run_point and _run_sub_point.

        Parameters
        ----------
        system : System
            System where this approximation is occurring.
        colored_approx_groups : list of tuples
            Tuples of (data, jaccols, vec_ind_list, nzrows, seed_vars).

        Yields
        ------
        int
            column index
        ndarray
            solution array corresponding to the jacobian column at the given column index
        """
        # Augment vec_ind_list in colored_approx_groups with VecIdxInfo namedtuples
        augmented_groups = []
        for data, jcols, vec_ind_list, nzrows, seed_vars in colored_approx_groups:
            augmented_vec_ind_list = self._augment_vec_ind_list(vec_ind_list)
            augmented_groups.append((data, jcols, augmented_vec_ind_list, nzrows, seed_vars))

        # Call parent's _colored_column_iter with augmented groups
        yield from super()._colored_column_iter(system, augmented_groups)

    def _vec_ind_iter(self, vec_ind_list):
        """
        Yield vector index list with local indices computed once per iteration for rel_element.

        For rel_element step sizing, compute local indices once per iteration (in this generator)
        and store them in VecIdxInfo namedtuples. This avoids recalculating them repeatedly
        in _run_point and _run_sub_point.

        Parameters
        ----------
        vec_ind_list : list
            List of (Vector, indices) tuples.

        Yields
        ------
        list
            List of VecIdxInfo namedtuples with computed local indices.
        int or ndarray or None
            The indices for the current iteration.
        """
        if self._totals_directions:
            # For directional case, augment the entire list
            augmented_list = []
            for vec, vec_idxs in vec_ind_list:
                if vec_idxs is None:
                    augmented_list.append(VecIdxInfo(vec=vec, idxs=None, local_idxs=None))
                else:
                    local_idxs = self._compute_local_indices(vec_idxs)
                    augmented_list.append(VecIdxInfo(vec=vec, idxs=vec_idxs, local_idxs=local_idxs))
            yield augmented_list, vec_ind_list[0][1]
        else:
            # For non-directional case, augment each entry
            entry = [None]
            for vec, vec_idxs in vec_ind_list:
                if vec_idxs is None:
                    continue
                for vinds in vec_idxs:
                    local_idxs = self._compute_local_indices(vinds)
                    entry[0] = VecIdxInfo(vec=vec, idxs=vinds, local_idxs=local_idxs)
                    yield entry, vinds

    def _make_vec_idx_info(self, vec, idxs):
        """
        Create a VecIdxInfo namedtuple with computed local indices.

        Parameters
        ----------
        vec : Vector or None
            The vector to perturb.
        idxs : int or ndarray or None
            Indices into the vector.

        Returns
        -------
        VecIdxInfo
            Namedtuple with vec, idxs, and precomputed local_idxs.
        """
        if idxs is None:
            local_idxs = None
        else:
            local_idxs = self._compute_local_indices(idxs)
        return VecIdxInfo(vec=vec, idxs=idxs, local_idxs=local_idxs)

    def _augment_vec_ind_list(self, vec_ind_list):
        """
        Convert vec_ind_list tuples to VecIdxInfo namedtuples with computed local indices.

        Parameters
        ----------
        vec_ind_list : list of tuples
            Each tuple is (vec, idxs).

        Returns
        -------
        list of VecIdxInfo
            Augmented namedtuples with computed local indices.
        """
        return [self._make_vec_idx_info(vec, idxs) for vec, idxs in vec_ind_list]

    def _compute_local_indices(self, idxs):
        """
        Compute local indices relative to the first index in the current group.

        For rel_element step sizing, when multiple wrt variables are being processed,
        indices are relative to the start of the current wrt variable. Local indices
        should be 0-based within that variable.

        Parameters
        ----------
        idxs : int or ndarray or None
            Indices to convert.

        Returns
        -------
        int or ndarray or None
            Local indices relative to minimum index, or None if input is None.
        """
        if idxs is None:
            return None

        idxs_arr = np.atleast_1d(idxs)
        # For a group of indices from the same variable, find the offset
        offset = int(idxs_arr.min())
        local = idxs_arr - offset

        # Return in original form (scalar or array)
        if np.ndim(idxs) == 0:
            return int(local[0]) if local.size > 0 else 0
        return local

    def add_approximation(self, wrt, system, kwargs, vector=None):
        """
        Use this approximation scheme to approximate the derivative d(of)/d(wrt).

        Parameters
        ----------
        wrt : str
            Absolute name of wrt variable.
        system : System
            Containing System.
        kwargs : dict
            Additional keyword arguments, to be interpreted by sub-classes.
        vector : ndarray or None
            Direction for difference when using directional derivatives.
        """
        options = self.DEFAULT_OPTIONS.copy()
        options.update(kwargs)

        if options['order'] is None:
            # User-submitted options for method=='fd' are all checked here.
            form = options['form']
            if form in DEFAULT_ORDER:
                options['order'] = DEFAULT_ORDER[options['form']]
            else:
                raise ValueError("{}: '{}' is not a valid form of finite difference; must be "
                                 "one of {}".format(system.msginfo, form,
                                                    list(DEFAULT_ORDER.keys())))

            step_calc = options['step_calc']
            step_calcs = ['abs', 'rel', 'rel_legacy', 'rel_avg', 'rel_element']
            if step_calc not in step_calcs:
                raise ValueError(f"{system.msginfo}: '{step_calc}' is not a valid setting for "
                                 f"step_calc; must be one of {step_calcs}.")

            elif options['directional'] and step_calc == 'rel_element':
                raise ValueError(f"{system.msginfo}: Option 'directional' is not supported when "
                                 "'step_calc' is set to 'rel_element.'")

        options['vector'] = vector
        if wrt in self._wrt_meta:
            self._wrt_meta[wrt].update(options)
        else:
            self._wrt_meta[wrt] = options
        self._reset()  # force later regen of approx_groups

    def _get_approx_data(self, system, wrt, meta):
        """
        Given approximation metadata, compute necessary deltas and coefficients.

        Parameters
        ----------
        system : System
            System whose derivatives are being approximated.
        wrt : str
            Name of wrt variable.
        meta : dict
            Metadata dict.

        Returns
        -------
        tuple
            Tuple of the form (deltas, coeffs, current_coeff)
        """
        form = meta['form']
        order = meta['order']
        step = meta['step']
        step_calc = meta['step_calc']
        minimum_step = meta['minimum_step']

        # FD forms are written as a collection of changes to inputs (deltas) and the associated
        # coefficients (coeffs). Since we do not need to (re)evaluate the current step, its
        # coefficient is stored seperately (current_coeff). For example,
        # f'(x) = (f(x+h) - f(x))/h + O(h) = 1/h * f(x+h) + (-1/h) * f(x) + O(h)
        # would be stored as deltas = [h], coeffs = [1/h], and current_coeff = -1/h.
        # A central second order accurate approximation for the first derivative would be stored
        # as deltas = [-2, -1, 1, 2] * h, coeffs = [1/12, -2/3, 2/3 , -1/12] * 1/h,
        # current_coeff = 0.
        fd_form = _generate_fd_coeff(form, order, system)

        if step_calc != 'abs':
            var_local = True
            if system._outputs._contains_abs(wrt):
                wrt_val = system._outputs._abs_get_val(wrt)
            elif system._inputs._contains_abs(wrt):
                wrt_val = system._inputs._abs_get_val(wrt)
            else:
                var_local = False

            if var_local:
                if step_calc == 'rel_legacy':
                    step *= np.linalg.norm(wrt_val)

                    if step < minimum_step:
                        step = minimum_step

                elif step_calc == 'rel_avg' or step_calc == 'rel':
                    step *= np.sum(np.abs(wrt_val)) / len(wrt_val)

                    if step < minimum_step:
                        step = minimum_step

                else:  # 'rel_element'
                    step = np.abs(wrt_val) * step

                    idx_zero = np.where(step < minimum_step)
                    if idx_zero:
                        step[idx_zero] = minimum_step

        if step_calc == 'rel_element':
            step_divide = 1.0 / step
            deltas = np.outer(fd_form.deltas, step)
            coeffs = np.outer(fd_form.coeffs, step_divide)
            current_coeff = fd_form.current_coeff * step_divide
        else:
            deltas = fd_form.deltas * step
            coeffs = fd_form.coeffs / step
            current_coeff = fd_form.current_coeff / step

        return deltas, coeffs, current_coeff

    def compute_approx_col_iter(self, system, under_cs=False):
        """
        Execute the system to compute the approximate sub-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.
        under_cs : bool
            True if we're currently under complex step at a higher level.

        Yields
        ------
        int
            column index
        ndarray
            solution array corresponding to the jacobian column at the given column index
        """
        if not self._wrt_meta:
            return

        self._starting_outs = system._outputs.asarray(copy=True)
        self._starting_resids = system._residuals.asarray(copy=True)
        self._starting_ins = system._inputs.asarray(copy=True)
        if _is_group(system):  # totals/semitotals
            self._results_tmp = self._starting_outs.copy()
        else:
            self._results_tmp = self._starting_resids.copy()

        # Turn on finite difference.
        system._set_finite_difference_mode(True)

        try:
            yield from self._compute_approx_col_iter(system, under_cs=under_cs)
        finally:
            # Turn off finite difference.
            system._set_finite_difference_mode(False)

        # reclaim some memory
        self._starting_ins = None
        self._starting_outs = None
        self._starting_resids = None
        self._results_tmp = None

    def _get_multiplier(self, data):
        """
        Return a multiplier to be applied to the jacobian.

        Always returns 1.0 for finite difference.

        Parameters
        ----------
        data : tuple
            Not used.

        Returns
        -------
        float
            1.0
        """
        return 1.0

    def _transform_result(self, array):
        """
        Return the given array.

        Parameters
        ----------
        array : ndarray
            Result array after doing a finite difference.

        Returns
        -------
        array
            The givan array, unchanged.
        """
        return array.real

    def _run_point(self, system, idx_info, data, results_array, total, idx_range=range(1)):
        """
        Alter the specified inputs by the given deltas, run the system, and return the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        idx_info : list of VecIdxInfo namedtuples
            List of VecIdxInfo(vec, idxs, local_idxs) with computed local indices.
        data : tuple of float
            Tuple of the form (deltas, coeffs, current_coeff)
        results_array : ndarray
            Where the results will be stored.
        total : bool
            If True total derivatives are being approximated, else partials.
        idx_range : range
            Range of vector indices for this wrt variable.

        Returns
        -------
        ndarray
            Copy of the outputs or residuals array after running the perturbed system.
        """
        deltas, coeffs, current_coeff = data
        rel_element = False

        if isinstance(current_coeff, np.ndarray) and current_coeff.size > 0:
            # rel_element - each element has its own relative step.
            rel_element = True

            if current_coeff[0]:
                current_vec = system._outputs if total else system._residuals
                # copy data from outputs (if doing total derivs) or residuals (if doing partials)
                results_array[:] = current_vec.asarray()

                for info in idx_info:
                    if info.vec is not None and info.local_idxs is not None:
                        results_array *= current_coeff[info.local_idxs]

            else:
                results_array[:] = 0.

        elif not isinstance(current_coeff, np.ndarray) and current_coeff:
            current_vec = system._outputs if total else system._residuals
            # copy data from outputs (if doing total derivs) or residuals (if doing partials)
            results_array[:] = current_vec.asarray()
            results_array *= current_coeff
        else:
            results_array[:] = 0.

        # Run the Finite Difference
        for delta, coeff in zip(deltas, coeffs):
            results = self._run_sub_point(system, idx_info, delta, total, idx_range=idx_range,
                                          rel_element=rel_element)

            if rel_element:
                for info in idx_info:
                    if info.vec is not None and info.local_idxs is not None:
                        results *= coeff[..., info.local_idxs]
            else:
                results *= coeff

            results_array += results

        return results_array

    def _run_sub_point(self, system, idx_info, delta, total, idx_range, rel_element=False):
        """
        Alter the specified inputs by the given delta, run the system, and return the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        idx_info : list of VecIdxInfo namedtuples
            List of VecIdxInfo(vec, idxs, local_idxs) with computed local indices.
        delta : float or ndarray
            Perturbation amount. If rel_element, array of per-element perturbations.
        total : bool
            If True total derivatives are being approximated, else partials.
        idx_range : range
            Range of vector indices for this wrt variable (unused, kept for compatibility).
        rel_element : bool
            If True, then each element has a different delta.

        Returns
        -------
        ndarray
            Copy of the outputs or residuals array after running the perturbed system.
        """
        for info in idx_info:
            vec = info.vec
            idxs = info.idxs
            local_idxs = info.local_idxs

            if vec is not None and idxs is not None:
                # Support rel_element stepsizing
                if rel_element and local_idxs is not None:
                    # Use the local indices (computed once in the generator or upfront)
                    # to select the correct delta values instead of recomputing indices
                    local_delta = delta[local_idxs]
                else:
                    local_delta = delta

                vec.iadd(local_delta, idxs)

        if total:
            system.run_solve_nonlinear()
            self._results_tmp[:] = system._outputs.asarray()
        else:
            system.run_apply_nonlinear()
            self._results_tmp[:] = system._residuals.asarray()

        system._residuals.set_val(self._starting_resids)

        # save results and restore starting inputs/outputs
        system._inputs.set_val(self._starting_ins)
        system._outputs.set_val(self._starting_outs)

        return self._results_tmp

    def apply_directional(self, data, direction):
        """
        Apply stepsize to direction and embed into approximation data.

        Parameters
        ----------
        data : tuple
            Tuple contains step size, and other info.
        direction : ndarray
            Vector containing derivative direction.

        Returns
        -------
        ndarray
            New tuple with new step direction.
        """
        deltas, coeffs, current_coeff = data
        return (np.outer(np.atleast_1d(deltas), direction), coeffs, current_coeff)

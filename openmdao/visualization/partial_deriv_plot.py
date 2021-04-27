"""
Visualization of data functions.
"""
import numpy as np


def partial_deriv_plot(of, wrt, check_partials_data, title=None, jac_method='J_fwd', tol=1e-10,
                       binary=True):
    """
    Visually examine the computed and finite differenced Jacobians.

    Parameters
    ----------
    of : string
        Variable whose derivatives will be computed.
    wrt : string
        Variable with respect to which the derivatives will be computed.
    check_partials_data : dict of dicts of dicts
        First key:
            is the component name;
        Second key:
            is the (output, input) tuple of strings;
        Third key:
            is one of ['rel error', 'abs error', 'magnitude', 'J_fd', 'J_fwd', 'J_rev'];
        For 'rel error', 'abs error', 'magnitude' the value is: A tuple containing norms for
            forward - fd, adjoint - fd, forward - adjoint.
        For 'J_fd', 'J_fwd', 'J_rev' the value is: A numpy array representing the computed
            Jacobian for the three different methods of computation.
    title : string (Optional)
        Title for the plot
        If None, use the values of the arguments "of" and "wrt"
    jac_method : string (Optional)
        Method of computating Jacobian
        Is one of ['J_fwd', 'J_rev']. Optional, default is 'J_fwd'.
    tol : float (Optional)
        The tolerance, below which the two numbers are considered the same for
        plotting purposes.
    binary : bool (Optional)
        If true, the plot will only show the presence of a non-zero derivative, not the value.
        Otherwise, plot the value. Default is true.

    Raises
    ------
    KeyError
        If one of the Jacobians is not available.

    Returns
    -------
    matplotlib.figure.Figure
        The top level container for all the plot elements in the plot.
    array of matplotlib.axes.Axes objects
        An array of Axes objects, one for each of the three subplots created.
    """
    # Get the first item in the dict, which will be the model
    model_name = list(check_partials_data)[0]
    model_jacs = check_partials_data[model_name]
    key = (of, wrt)
    model_jac = model_jacs[key]

    # get finite difference arrays
    if 'J_fd' not in model_jac:
        msg = 'Jacobian "{}" not found.'
        raise KeyError(msg.format('J_fd'))
    fd_full = model_jac['J_fd']
    fd_full_flat = fd_full.flatten()  # for getting max min later
    if binary:
        fd_binary = fd_full.copy()
        fd_binary[np.nonzero(fd_binary)] = 1.0

    # get computed arrays
    if jac_method not in model_jac:
        msg = 'Jacobian "{}" not found.'
        raise KeyError(msg.format(jac_method))
    computed_full = model_jac[jac_method]
    computed_full_flat = computed_full.flatten()  # for getting max min later
    if binary:
        computed_binary = computed_full.copy()
        computed_binary[np.nonzero(computed_binary)] = 1.0

    # get plotting scales
    stacked = np.hstack((fd_full_flat, computed_full_flat))
    vmin = np.amin(stacked)
    vmax = np.amax(stacked)

    # basics of plot
    # Cannot do this sooner because of import and matplotlib
    #  backend issues when testing
    import matplotlib.pyplot as plt
    BINARY_CMP = plt.cm.gray
    NON_BINARY_CMP = plt.cm.RdBu
    fig, ax = plt.subplots(ncols=3, figsize=(12, 6))
    if title is None:
        title = str(key)
    plt.suptitle(title)

    # plot Jacobians
    if binary:
        ax[0].imshow(fd_binary.real, interpolation='none',
                     cmap=BINARY_CMP, aspect='auto')
        im_computed = ax[1].imshow(computed_binary.real, interpolation='none', cmap=BINARY_CMP,
                                   aspect='auto')
    else:
        ax[0].imshow(fd_full.real, interpolation='none', vmin=vmin, vmax=vmax, cmap=NON_BINARY_CMP,
                     aspect='auto')
        im_computed = ax[1].imshow(computed_full.real, interpolation='none', vmin=vmin, vmax=vmax,
                                   cmap=NON_BINARY_CMP, aspect='auto')
    ax[0].set_title('Approximated Jacobian')
    ax[1].set_title('User-Defined Jacobian')
    # Legend
    fig.colorbar(im_computed, orientation='horizontal',
                 ax=ax[0:2].ravel().tolist())

    # plot difference between computed and finite differenced
    diff = computed_full.real - fd_full.real
    diff_flat = diff.flatten()
    vmin = np.amin(diff_flat)
    vmax = np.amax(diff_flat)
    if vmax - vmin < tol:  # Do not want range to be too small
        vmin = -1 * tol
        vmax = tol
    im_diff = ax[2].imshow(diff, interpolation='none', vmin=vmin, vmax=vmax, cmap=NON_BINARY_CMP,
                           aspect='auto')
    fig.colorbar(im_diff, orientation='horizontal', ax=ax[2], aspect=10)
    ax[2].set_title('Difference')

    plt.show()
    return fig, ax

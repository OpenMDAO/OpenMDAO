"""
Routines common to both autograd and tangent.
"""

from __future__ import print_function, division

import sys
import os
import traceback
import importlib

import numpy as np

from openmdao.utils.ad_autograd import _get_autograd_ad_func, _get_autograd_ad_jac
from openmdao.utils.ad_tangent import _get_tangent_ad_func, _get_tangent_ad_jac, _dot_prod_test
from openmdao.utils.general_utils import get_module_attr
from numpy.testing import assert_almost_equal
from openmdao.core.problem import Problem
from openmdao.core.explicitcomponent import Component, ExplicitComponent
import openmdao.utils.mod_wrapper as mod_wrapper
from openmdao.devtools.debug import compare_jacs


def _ad_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao ad' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs='?', help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Output file name. By default, output goes to stdout.')
    parser.add_argument('--noopt', action='store_true', dest='noopt',
                        help="Turn off optimization. (tangent only)")
    parser.add_argument('-m', '--method', default='tangent', action='store', dest='ad_method',
                        help='AD method (autograd, tangent).')
    parser.add_argument('-c', '--class', action='append', dest='classes', default=[],
                        help='Specify component class(es) to run AD on.')
    parser.add_argument('-v', '--verbose', default=0, type=int, action='store', dest='verbose',
                        help='Verbosity level.')
    parser.add_argument('--fwd', action='store_true', dest='fwd',
                        help="Show forward mode results.")
    parser.add_argument('--rev', action='store_true', dest='rev',
                        help="Show reverse mode results.")
    parser.add_argument('-t', '--tol', default=1.e-6, type=float, action='store', dest='tol',
                        help='Maximum difference in derivatives allowed.')


def _ad_exec(options):
    """
    Process command line args and perform AD.
    """
    if options.file:
        from openmdao.utils.om import _post_setup_exec
        options.file = [options.file]
        _post_setup_exec(options)
    else:
        _ad(None, options)


def _create_instances(prob, classes):
    """
    Create instances of the given classes and checks their partials.

    Note: this only works for classes that can be instantiated with no args.

    Parameters
    ----------
    prob : Problem
        The problem object.
    classes : list of str
        List of class names (full module path).

    Yields
    ------
    object
        Instance of specified class.
    dict
        Dictionary returned from check_partials.
    """
    insts = [get_module_attr(cpath)() for cpath in classes]
    for obj in insts:
        prob.model.add_subsystem(obj.__class__.__name__.lower() + '_', obj)

    prob.setup()
    prob.final_setup()
    invec = prob.model._inputs._data
    invec[:] = np.random.random(invec.size) + 1.0

    for obj in insts:
        print("\nClass:", obj.__class__.__name__)
        yield obj


def _find_instances(prob, classes, exclude=()):
    """
    Find instances of the given classes in the problem.

    Yields only the first instance of each class that it finds.

    Parameters
    ----------
    prob : Problem
        The problem object.
    classes : list of str
        List of class names.
    exclude : set of str or None
        Set of class names to exclude.

    Yields
    ------
    object
        Instance of specified class.

    """
    seen = set(exclude)
    for s in prob.model.system_iter(recurse=True, include_self=True, typ=Component):
        cname = s.__class__.__name__
        if cname not in seen and (cname in classes or not classes):
            seen.add(cname)
            yield s

    not_found = classes - seen
    if not_found:
        raise RuntimeError("Couldn't find an instance of the following classes: %s." % not_found)


def _ad(prob, options):
    """
    Fwd and/or rev AD for the compute or apply_nonlinear method of the given class.
    """
    if options.outfile is None:
        out = sys.stdout
    else:
        out = open(options.outfile, 'w')

    classes = set(options.classes)

    Problem._post_setup_func = None  # prevent infinite recursion

    if prob is None:
        prob = Problem()

    if classes and all(['.' in cpath for cpath in classes]):
        it = _create_instances(prob, classes)
        prob.run_model()
    else:
        prob.run_model()
        it = _find_instances(prob, classes, exclude=set(['IndepVarComp', 'ExecComp']))

    modes = []
    if options.fwd:
        modes.append('fwd')
    if options.rev:
        modes.append('rev')
    if not modes:
        modes = ['fwd', 'rev']

    tol = options.tol
    summary = {}

    for comp in it:
        print("\nClass:", type(comp).__name__)
        print("Instance:", comp.pathname)

        if options.ad_method == 'autograd':
            import autograd.numpy as agnp
            mod_wrapper.np = mod_wrapper.numpy = agnp

        summary[comp.__class__.__name__] = summ = {}

        summ['fwd'] = {'ran': False, 'diff': float('nan')}
        summ['rev'] = {'ran': False, 'diff': float('nan')}

        rel_offset = len(comp.pathname) + 1 if comp.pathname else 0

        type_ = 'Explicit' if isinstance(comp, ExplicitComponent) else 'Implicit'
        summ['type'] = type_
        summ['osize'] = comp._outputs._data.size
        summ['isize'] = comp._inputs._data.size
        summ['pref'] = 'fwd' if summ['osize'] >= summ['isize'] else 'rev'
        print("Type:", type_)

        save_inputs = comp._inputs._data.copy()
        Japprox = comp.compute_approx_partials(method='cs')

        for mode in modes:
            summ[mode] = {}
            try:
                comp._inputs._data[:] = save_inputs
                deriv_func, dmod = _get_tangent_ad_func(comp, mode, verbose=options.verbose,
                                                        optimize=not options.noopt)
                summ[mode]['func'] = deriv_func
                Jad = {}
                _get_tangent_ad_jac(comp, mode, deriv_func, Jad)

                del sys.modules[dmod.__name__]
                os.remove(dmod.__file__)
                try:
                    os.remove(dmod.__file__ + 'c')
                except FileNotFoundError:
                    pass

                results = list(compare_jacs(Japprox, Jad))
                maxmax = 0.
                max_keywid = 0
                for key, mxdiff, typ in results:
                    relkey = str((key[0][rel_offset:], key[1][rel_offset:]))
                    keywid = len(relkey)
                    if keywid > max_keywid:
                        max_keywid = keywid
                    if mxdiff > maxmax:
                        maxmax = mxdiff
                summ[mode]['diff'] = maxmax
                summ[mode]['ran'] = True

                print("\n%s J:" % mode.upper())
                print("{key:<{max_keywid}} {mxdiff:<12} ErrTyp {ref:<8}".format(
                    key='(of, wrt)', mxdiff='diff', ref='norm(Jref)', max_keywid=max_keywid))

                for key, mxdiff, typ in sorted(results, key=lambda x: x[1]):
                    if mxdiff < 1.e-12 and not np.any(Japprox[key]):
                        continue
                    relkey = str((key[0][rel_offset:], key[1][rel_offset:]))
                    print("{key:<{max_keywid}} {mxdiff:<12.7} ({typ})  {ref:<12.7}".format(
                        key=relkey, max_keywid=max_keywid, mxdiff=mxdiff, typ=typ,
                        ref=np.linalg.norm(Japprox[key])))

                print()

            except Exception:
                traceback.print_exc(file=sys.stdout)
                summ[mode]['ran'] = False
                summ[mode]['diff'] = float('nan')
                print("\n")

        if options.ad_method == 'autograd':
            mod_wrapper.np = mod_wrapper.numpy = np

        if (options.ad_method == 'tangent' and summ['fwd']['ran'] and summ['rev']['ran']):
            summ['dotprod'] = _dot_prod_test(comp, summ['fwd']['func'], summ['rev']['func'])
        else:
            summ['dotprod'] = float('nan')

    max_cname = max(len(s) for s in summary) + 2
    max_diff = 16
    bothgood = []
    fwdgood = []
    revgood = []
    bad = []

    toptemplate = \
        "{n:<{cwid}}{typ:<10}{fdiff:<{dwid}}{rdiff:<{dwid}}{dot:<{dwid}}{iosz:<12}{pref:<14}"
    template = \
        "{n:<{cwid}}{typ:<10}{fdiff:<{dwid}.4}{rdiff:<{dwid}.4}{dot:<{dwid}.4}{iosz:<12}{pref:<14}"
    print(toptemplate.format(n='Class', typ='Type', fdiff='Max Diff (fwd)',
                             rdiff='Max Diff (rev)', dot='Dotprod Test', pref='Preferred Mode',
                             cwid=max_cname, dwid=max_diff, iosz='(I/O) Size'))
    print('--------- both derivs ok ------------')
    for cname, s in sorted(summary.items(),
                           key=lambda x: max(x[1]['fwd']['diff'], x[1]['rev']['diff'])):
        typ = s['type']
        fwdran = s['fwd']['ran']
        fwdmax = s['fwd']['diff']

        revran = s['rev']['ran']
        revmax = s['rev']['diff']

        dptest = s['dotprod']

        line = template.format(n=cname, typ=typ, fdiff=fwdmax, rdiff=revmax, dot=dptest,
                               cwid=max_cname, dwid=max_diff,
                               iosz='(%d/%d)' % (s['isize'], s['osize']), pref=s['pref'])

        add = "  different: %s %s" % (fwdmax, revmax) if fwdmax != revmax else ''

        if fwdran and revran and fwdmax <= tol and revmax <= tol:
            bothgood.append(line)
            print(line + add)
        elif fwdran and fwdmax <= tol:
            fwdgood.append(line + add)
        elif revran and revmax <= tol:
            revgood.append(line + add)
        else:
            bad.append(line)

    if fwdgood:
        print('--------- fwd derivs ok ------------')
        for b in fwdgood:
            print(b)

    if revgood:
        print('--------- rev derivs ok ------------')
        for b in revgood:
            print(b)

    if bad:
        print('--------- both derivs bad ------------')
        for b in bad:
            print(b)

    tot = len(bothgood) + len(fwdgood) + len(revgood) + len(bad)
    print('\nSummary:  %d total, %d both good,  %d fwd good,  %d rev good' % (tot, len(bothgood),
                                                                              len(fwdgood),
                                                                              len(revgood)))
    exit()


def _ad_cmd(options):
    """
    Return the post_setup hook function for 'openmdao ad'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    return lambda prob: _ad(prob, options)

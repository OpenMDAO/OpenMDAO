"""
Helper function to find all the `cite` attributes throughout a model.
"""
from __future__ import print_function
from collections import OrderedDict
import sys

if sys.version_info[0] == 2:  # Not named on 2.6
    from StringIO import StringIO
else:
    from io import StringIO


def _check_cite(instance, citations):
    """
    Grab the cite attribute, if it exists.

    Parameters
    ----------
    instance : object
        the instance to check for citations on
    citations : dict
        the dictionary to add a citation to, if found
    """
    if instance.cite:
        klass = instance.__class__
        # return klass, cite
        citations[klass] = instance.cite


def find_citations(prob, out_stream=sys.stdout):
    """
    Compiles a list of citations from all classes in the problem.

    Parameters
    ----------
    prob : <Problem>
        The Problem instance to be searched
    out_stream : File like
        defaults to sys.stdout. False will prevent printed output

    Returns
    -------
    dict
        dict of citations keyed by class
    """
    # dict keyed by the class so we don't report multiple citations
    # for the same class showing up in multiple instances
    citations = OrderedDict()
    _check_cite(prob, citations)
    _check_cite(prob.driver, citations)

    # recurse down the model
    for subsys in prob.model.system_iter(include_self=True, recurse=True):
        _check_cite(subsys, citations)
        if subsys.nonlinear_solver is not None:
            _check_cite(subsys.nonlinear_solver, citations)
        if subsys.linear_solver is not None:
            _check_cite(subsys.linear_solver, citations)

    if out_stream:

        for klass, cite in citations.items():
            # print("Class: {}".format(klass), file=out_stream)
            out_stream.write("Class: {}\n".format(klass))
            lines = cite.split('\n')
            for line in lines:
                # print("    {}".format(line), file=out_stream)
                out_stream.write("    {}\n".format(line))

    return citations

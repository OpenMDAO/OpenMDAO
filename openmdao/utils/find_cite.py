"""
Helper function to find all the `cite` attributes throughout a model.
"""
from __future__ import print_function
from collections import OrderedDict
import inspect
import sys

from six import iteritems

from openmdao.utils.logger_utils import get_logger


def _check_cite(obj, citations):
    """
    Grab the cite attribute, if it exists.

    Parameters
    ----------
    obj : object
        the instance to check for citations on
    citations : dict
        the dictionary to add a citation to, if found
    """
    if inspect.isclass(obj):
        if obj.cite:
            citations[obj] = obj.cite
    if obj.cite:
        klass = obj.__class__
        # return klass, cite
        citations[klass] = obj.cite


def find_citations(prob):
    """
    Compile a list of citations from all classes in the problem.

    Parameters
    ----------
    prob : <Problem>
        The Problem instance to be searched

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
    _check_cite(prob._vector_class, citations)

    # recurse down the model
    for subsys in prob.model.system_iter(include_self=True, recurse=True):
        _check_cite(subsys, citations)
        if subsys.nonlinear_solver is not None:
            _check_cite(subsys.nonlinear_solver, citations)
        if subsys.linear_solver is not None:
            _check_cite(subsys.linear_solver, citations)

    return citations


def _filter_citations(citations, classes):
    """
    Filter a dict of citations to include only those matching the give class names.

    Parameters
    ----------
    citations : dict
        Dict of citations keyed by class.
    classes : list of str
        List of class names for classes to include in the displayed citations.

    Returns
    -------
    dict
        The filtered dict of citations.
    """
    if classes is None:
        return citations

    cits = OrderedDict()
    for klass, cit in iteritems(citations):
        if klass.__name__ in classes or '.'.join((klass.__module__, klass.__name__)) in classes:
            cits[klass] = cit
    return cits


def print_citations(prob, classes=None, out_stream='stdout'):
    """
    Write a list of citations from classes in the problem to the given stream.

    Parameters
    ----------
    prob : <Problem>
        The Problem instance to be searched
    classes : list of str
        List of class names for classes to include in the displayed citations.
    out_stream : 'stdout', 'stderr' or file-like
            Where to send human readable output. Default is 'stdout'.
            Set to None to suppress.
    """
    citations = _filter_citations(find_citations(prob), classes)

    if out_stream:
        logger = get_logger('citations', out_stream=out_stream)
        for klass, cite in citations.items():
            # print("Class: {}".format(klass), file=out_stream)
            logger.info("Class: {}".format(klass))
            lines = cite.split('\n')
            for line in lines:
                # print("    {}".format(line), file=out_stream)
                logger.info("    {}".format(line))

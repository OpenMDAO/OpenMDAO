.. _citing:

**************************
How to Cite OpenMDAO
**************************

Depending on which parts of OpenMDAO you are using, there are diferent papers that are appropriate to cite.
We've provided a helper function that will look at all the parts of your model and give you a list of the relevant citations.
The output of this helper function will tell you which citations you should include and which classes those citations relate to.

There are two ways use the citation helper

Directly from setup()
----------------------------------
You can get the list of citations by adding passing :code:`print_citations=True` as an
argument to the :ref:`Problem setup()<setup>`

.. embed-test::
    openmdao.test_suite.test_examples.tldr_paraboloid.TestParaboloidTLDR.test_feature_tldr_citation


With the `openmdao` command
----------------------------------

If you copy the above script into a file called `paraboloid.py` and remove the call to `find_citations`,
then you can get the citations from the command line using the :ref:`openmdao run script<om-command>`.


::

    openmdao cite paraboloid.py

which will give you the following output

::

    Class: <class 'openmdao.core.problem.Problem'>
        @inproceedings{2014_openmdao_derivs,
            Author = {Justin S. Gray and Tristan A. Hearn and Kenneth T. Moore and John Hwang and Joaquim Martins and Andrew Ning},
            Booktitle = {15th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference},
            Doi = {doi:10.2514/6.2014-2042},
            Month = {2014/07/08},
            Publisher = {American Institute of Aeronautics and Astronautics},
            Title = {Automatic Evaluation of Multidisciplinary Derivatives Using a Graph-Based Problem Formulation in OpenMDAO},
            Year = {2014}
        }
    Class: <class 'openmdao.drivers.scipy_optimizer.ScipyOptimizer'>

        @phdthesis{hwang_thesis_2015,
          author       = {John T. Hwang},
          title        = {A Modular Approach to Large-Scale Design Optimization of Aerospace Systems},
          school       = {University of Michigan},
          year         = 2015
        }
.. _citing:

**************************
How to Cite OpenMDAO
**************************

Depending on which parts of OpenMDAO you are using, there are diferent papers that are appropriate to cite.
OpenMDAO can tell you which citations are appropriate, accouting for what classes you're actually using in your model.

Here is a simple example

.. embed-test::
    openmdao.test_suite.test_examples.tldr_paraboloid.TestParaboloidTLDR.test_tldr
    :no-split:

With the `openmdao` command
----------------------------------

If you copy the above script into a file called `paraboloid.py`,
then you can get the citations from the command line using the :ref:`openmdao command-line script<om-command>`.

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
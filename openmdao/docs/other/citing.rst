.. _citing:

********************
How to Cite OpenMDAO
********************


There is a general OpenMDAO paper that includes a high-level overview of the framework,
including how implicit and explicit components leverage the underlying core APIs to support multidisciplinary modeling.
There is a walk-through example of how some of the key underlying mathematics of the framework are used and how analytic derivatives are computed.
Lastly, there are  examples of how and when to use some of the specialized algorithms for computing derivatives efficiently for different kinds of problems.

We hope the paper helps you understand the framework better, and most importantly,
helps you to solve some really nice MDO problems! If you do make use of OpenMDAO, please cite this paper.

.. code-block:: none

    @article{openmdao_2019,
    Author={Justin S. Gray and John T. Hwang and Joaquim R. R. A. Martins and Kenneth T. Moore and Bret A. Naylor},
    Title="{OpenMDAO: An Open-Source Framework for Multidisciplinary Design, Analysis, and Optimization}",
    Journal="{Structural and Multidisciplinary Optimization}",
    Year={2019},
    Volume={59},
    pages={1075-1104},
    issue={4},
    Publisher={Springer},
    pdf={http://mdolab.engin.umich.edu/sites/default/files/OpenMDAO_preprint_0.pdf},
    Doi={10.1007/s00158-019-02211-z},
    }


With the `openmdao` command
---------------------------

Depending on which parts of OpenMDAO you are using, there are also a few other papers that are appropriate to cite.
OpenMDAO can tell you which citations are appropriate, accounting for what classes you're actually using in your model.

If you copy the following script into a file called `paraboloid.py`,
then you can get the citations from the command line using the :ref:`openmdao command-line script<om-command>`.

.. embed-code::
    openmdao.test_suite.test_examples.tldr_paraboloid.TestParaboloidTLDR.test_tldr
    :layout: code, output

.. embed-shell-cmd::
    :cmd: openmdao cite paraboloid.py
    :dir: ../test_suite/components

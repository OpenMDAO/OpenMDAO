[![Build Status](https://travis-ci.org/OpenMDAO/blue.svg?branch=master)](https://travis-ci.org/OpenMDAO/blue)   [![Build status](https://ci.appveyor.com/api/projects/status/33kct0irhbgcg8m1?svg=true
)](https://ci.appveyor.com/project/OpenMDAO/blue/branch/master)  [![Documentation Status](https://readthedocs.org/projects/blue/badge/?version=latest   )](http://blue.readthedocs.org/en/latest/)  [![Coverage Status](https://coveralls.io/repos/OpenMDAO/blue/badge.svg?branch=master&service=github)](https://coveralls.io/github/OpenMDAO/blue?branch=master)
# blue
This is the PRE-ALPHA version of OpenMDAO 2.0
(we have codenamed it 'blue').

Important Note:
---------------

While the API is MOSTLY stable, we reserve the right to change things as needed.
Production runs should still be done in 1.7.x for now.

We will be making very frequent updates to this code. If you’re going to try it,
make sure you pull these updates often

Features of OpenMDAO 1.7.x Not Yet in 2.x
-----------------------------------------

Be aware that this is a PRE-ALPHA. 
Not all the features of 1.7.x exist in blue yet. 
Here is a list of things that have not yet been fully developed in 2.x:

* Case Recording (CSV, HDF5, and dump recording)
* Pass-by-object variables
* DOE (Design of Experiment) driver and all other case drivers
* Parallel Finite Difference
* File-wrapping utilities
* File variables
* Total Derivatives checking
* Group Finite Difference
* Complex Step approximation for Component/Group derivatives
* Parallel Adjoint and Parallel Forward derivative calculation performance speedup
* Constraint Sparsity specification for Pyoptsparse
* Active-set constraint calculation disabling
* Brent Solver
* Analysis Error handling

Installation Instructions:
--------------------------

Use git to clone the repository:

`git clone http://github.com/OpenMDAO/blue`

Use pip to install openmdao locally:

`cd blue/openmdao`

`pip install .`


Documentation Building Instructions:
------------------------------------

`cd openmdao/docs`

`make all`

This will build the docs into `openmdao/docs/_build/html`.

Then, just open  `openmdao/docs/_build/html/index.html` in a browser to begin.

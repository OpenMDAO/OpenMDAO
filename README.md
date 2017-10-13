
[![Build Status](https://travis-ci.org/OpenMDAO/blue.svg?branch=master)](https://travis-ci.org/OpenMDAO/blue)   [![Build status](https://ci.appveyor.com/api/projects/status/33kct0irhbgcg8m1?svg=true
)](https://ci.appveyor.com/project/OpenMDAO/blue/branch/master)  [![Coverage Status](https://coveralls.io/repos/github/OpenMDAO/blue/badge.svg?branch=master)](https://coveralls.io/github/OpenMDAO/blue?branch=master)

#OpenMDAO 2.0
This is the PRE-ALPHA version of OpenMDAO 2.0

Our docs are available at [http://openmdao.org/twodocs](http://openmdao.org/twodocs)

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

* Pass-by-object variables
* DOE (Design of Experiment) driver and all other case drivers
* Parallel Finite Difference
* File-wrapping utilities
* File variables
* Active-set constraint calculation disabling
* Brent Solver
* CaseRecording using CSV, HDF5, and dump recorders (SqliteRecorder and WebRecorder are currently supported)

Installation Instructions:
--------------------------

Use git to clone the repository:

`git clone http://github.com/OpenMDAO/blue`

Use pip to install openmdao locally:

`cd blue`

`pip install .`


Documentation Building Instructions:
------------------------------------

`cd openmdao/docs`

`make all`

This will build the docs into `openmdao/docs/_build/html`.

Then, just open  `openmdao/docs/_build/html/index.html` in a browser to begin.


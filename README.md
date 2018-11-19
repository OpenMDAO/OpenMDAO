
[![Build Status](https://travis-ci.org/OpenMDAO/OpenMDAO.svg?branch=master)](https://travis-ci.org/OpenMDAO/OpenMDAO)   [![Build status](https://ci.appveyor.com/api/projects/status/33kct0irhbgcg8m1?svg=true
)](https://ci.appveyor.com/project/OpenMDAO/blue/branch/master)  [![Coverage Status](https://coveralls.io/repos/github/OpenMDAO/OpenMDAO/badge.svg?branch=master)](https://coveralls.io/github/OpenMDAO/OpenMDAO?branch=master)




OpenMDAO Naming/Legacy OpenMDAO
-------------------------------

PLEASE NOTE: Until recently, this repository was named OpenMDAO/blue. If you had cloned that repository, please update
your repository name and remotes to reflect these changes.

The OpenMDAO 1.7.4 codebase repo has been renamed to OpenMDAO1, and it resides
at https://github.com/OpenMDAO/OpenMDAO1

The OpenMDAO 2.x.y code has taken the name OpenMDAO,
and it resides at https://github.com/OpenMDAO/OpenMDAO.

Installation of 2.x.y code will now work with `pip install openmdao`.
Installation of 1.7.4 code will now only work with a version specifier: `pip install openmdao==1.7.4`

To use the OpenMDAO v0.x legacy version
 (versions 0.13.0 and older) of the OpenMDAO-Framework, go here:
https://github.com/OpenMDAO/OpenMDAO-Framework


OpenMDAO 2
--------------

This is an ALPHA version of OpenMDAO 2

Our latest docs are available at [http://openmdao.org/twodocs/versions/latest](http://openmdao.org/twodocs/versions/latest)
Our archived 2 docs are available at [http://openmdao.org/twodocs](http://openmdao.org/twodocs)



Important Note:
---------------

While the API is MOSTLY stable, we reserve the right to change things as needed.

We will be making frequent updates to this code. If you’re going to try it,
make sure you pull these updates often.


Features of OpenMDAO 1.7.4 Not Yet in 2.x.y
-------------------------------------------

Be aware that this is an Alpha.
Not all the features of 1.7.4 exist in 2.x.y yet.

Here is a list of things that have not yet been developed in 2.x:

* Automatic ordering of groups/components based on data connections
* File variables
* Active-set constraint calculation disabling
* Brent Solver
* CaseRecording using CSV, HDF5, and dump recorders (SqliteRecorder and WebRecorder are currently supported)

Installation Instructions:
--------------------------
Option 1: Install from pypi:

`pip install openmdao`


Option 2: Use git to clone the repository:

`git clone http://github.com/OpenMDAO/OpenMDAO`

 Then use pip to install openmdao locally:

`cd OpenMDAO`

`pip install .`


Documentation Building Instructions:
------------------------------------

If you've cloned the repository, change to the docs directory:

`cd openmdao/docs`

`make clean; make all`

This will build the docs into `openmdao/docs/_build/html`.

Then, just open  `openmdao/docs/_build/html/index.html` in a browser to begin.

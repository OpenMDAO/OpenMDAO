# Welcome to OpenMDAO

OpenMDAO is an open-source high-performance computing platform for
systems analysis and multidisciplinary optimization, written in Python.
It enables you to decompose your models, making them easier to build and
maintain, while still solving them in a tightly coupled manner with
efficient parallel numerical methods.

The OpenMDAO project is primarily focused on supporting gradient-based
optimization with analytic derivatives to allow you to explore large
design spaces with hundreds or thousands of design variables, but the
framework also has a number of parallel computing features that can
work with gradient-free optimization, mixed-integer nonlinear
programming, and traditional design space exploration.

If you are using OpenMDAO, please [cite](other/citing.ipynb) us!

## User Guide

These are a collection of tutorial problems that teach you important concepts and techniques for using OpenMDAO.
For new users, you should work through all material in **Getting Started** and **Basic User Guide**.
That represents the minimum set of information you need to understand to be able to work with OpenMDAO models.

You will also find tutorials in the **Advanced User Guide** to be very helpful as you grow more familiar with OpenMDAO,
but you don't need to read these right away.
They explain important secondary concepts that you will run into when working with more complex OpenMDAO models.

- [Getting Started](getting_started/getting_started.md)
- [Basic User Guide](basic_user_guide/basic_user_guide.md)
- [Advanced User Guide](advanced_user_guide/advanced_user_guide.md)


## Reference Guide

These docs are intended to be used by as a reference by users looking for explanation of a particular feature in detail or
documentation of the arguments/options/settings for a specific method, Component, Driver, or Solver.

- [Features](features/features.md)
- [Examples](examples/examples.md)
- [Theory Manual](theory_manual/theory_manual.md)


## Other Useful Docs

- [Command Line Tools](other_useful_docs/om_command.ipynb)
- [How to Cite OpenMDAO](other/citing.ipynb)
- [Building a Tool on Top of OpenMDAO](other_useful_docs/building_a_tool/building_a_tool.md)
- [Conversion Guide for the Auto-IVC (IndepVarComp) Feature](other_useful_docs/auto_ivc_api_translation.ipynb)
- [Upgrading from OpenMDAO 2.10 to OpenMDAO 3](other_useful_docs/api_translation.ipynb)
- [File Wrapping](other_useful_docs/file_wrap.ipynb)
- [Source Docs](_srcdocs/index.md)
- [Developer Docs (if you’re going to contribute code)](other_useful_docs/developer_docs/developer_docs.md)

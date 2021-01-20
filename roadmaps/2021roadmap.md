OpenMDAO 2021 Development Roadmap
=================================

Author: Justin S. Gray  
Date: 2020-08-24

## What is this roadmap for?
This document represents the perspective of the OpenMDAO development team on what we feel will be the most important areas to focus on for framework development in 2021. 
It is intended to communicate this perspective to the OpenMDAO user community, and to provide a basis for feedback. 
It is not intended to be all-encompassing or a binding  development plan, but should be regarded as an indication of the current direction of the core development team. 
Also, it should be clearly understood that there is very probably more listed here than can actually be accomplished in a single year!

## Summary of major progress from 2020

### V3 release
OpenMDAO V3 release was completed. Python 2 no longer supported.
As of the date of this document OpenMDAO V3.2 has been released.

### Model Data and Visualization
We completed a major re-factor of the N2 diagram, including a new UI concept and modernized look and feel. 
Performance improvements for larger models were added to make the N2 more usable for deeply nested hierarchies.
More metadata is now included in the N2, allowing users to see details about groups, components, and variables via the information tool.
We did not make much progress on the inclusion of model data and plotting into the visualization. 
It took longer than expected to refactor the N2 as is.
This goal will be carried into 2021.

### HPC Support
While a standardized HPC test case was not built, there is a larger generalized effort Mphys which focuses on modular integration of PDE solvers.
This has resulted in use cases with aerostructural, aerothermal, and aeropropulsive applications.
The original planned development goal was to focus on reducing memory usage and duplicate data transfers. 
However, with the use cases mentioned above it became clear that some more foundational changes to the setup stack were needed. 
The memory efficiency goals were postponed in favor these API level improvements.
The improvements include the ability to create I/O during the configure method, 
the ability to query variable metadata from any level of the model hierarchy during configure,
and the addition of the auto-ivc capabilities.

### Improved Differentiation Tools
We refined and tested the partial derivative coloring features, which work well for FD and CS partial derivatives.
This feature proved useful and fairly efficient for highly sparse component calculations. 
It is particularly effective when you have vectorized array calculations that have purely diagonal partial derivative sparsity structure.
We did not have time to focus on true algorithmic differentiation capability.
This is still an area of interest, but the development team didn't have a strong
enough internal use case to focus on it.

It is still not clear that AD in python is a viable long term strategy.
We did some investigation into coupling with Julia, which does have a strong AD development effort.
See https://github.com/byuflowlab/OpenMDAO.jl for our collaboration with the BYU Flow Lab.
We found the julia wrapper to be effective and a good choice for more complex and costly calculations that would be difficult to implement efficiently in pure python. 
The added benefit of AD makes it a compelling future development path. 
There are some clear downsides though. 
The need to have expertise in both Julia and Python is a major one, 
though the syntax is close enough in most cases that this is not insurmountable.
The larger challenge is the difficulty of getting packages installed because you have to manage both a Python and a Julia package and install both.
It adds significant user hassle when codes are changing often.

### OpenMDAO POEM and Plugins
When we first proposed the POEM process, Justin Gray guessed that there would be less than 10 POEMS per year. 
He was wrong!
As of the date of this document, there were 31 POEMS proposed in 2020.
Most of these POEMS came from the devs, but we've found them to be a useful way to advertise the new functionality and to work out API details before implementation. 

# 2021 main focus areas
- Model visualization and data post processing
- Building an OpenMDAO training course
- Benchmarking OpenMDAO Performance for large scale HPC applications
----------------------------------------------
----------------------------------------------
----------------------------------------------

# Model Visualization and Data Post Processing

Overall, we feel that this is an area of OpenMDAO that can still use significant improvement. 
The 2020 N2 refactor improved things somewhat, but we think that tighter integration with case recorder data will greatly enhance the user experience.

## 1) Integrated case data into N2

### Goal:
Allow users to view the numerical values for all variables from a single case, or a set of cases within the N2 viewer.

### Potential Challenges:
The value of the stand-alone HTML based N2 has been clearly demonstrated.
It is portable and easily shared. We don't want to give up that functionality.
We can definitely integrate data from a single case within that stand-alone HTML structure. 
However, enabling multiple cases will be more challenging because file size will quickly get out of control and performance may be an issue.
We'll be forced to consider an external application for the N2.
We have the prototype based on Electron already, but are also considering a Jupyter based solution.  
Regardless, some kind of additional application will be required for multi-case capability.

## 2) Plotting tool for case data

### Goal:
Improve the usability of the case recorder databases, and allow rapid navigation and plotting of results.
We've noticed that users can sometimes struggle with our case databases, and we hope a graphical navigation tool for post processing and simple plotting will help with that.

### Potential Challenges
Similar to the multi-case data N2 concept, this functionality will likely be built into a separate stand alone application.
We have also considered a reduced functionality version of this feature built into the OpenMDAO command line tool, based on matplotlib or bokeh.
This lighter version may be done instead of a stand alone tool or in addition to it.

# Building an OpenMDAO Training Course
We've seen that as users get more comfortable with OpenMDAO the size of the models they can build grows quickly, 
both in terms of the number of components and the complexity of their couplings.
One problem can then arise when the problem complexity grows to the point that converging the solver or the optimizers becomes a real bottle neck.
We can and will provide better debugging tools, but there is a certain amount of practical knowledge and experience about solvers and optimizers that is still required.
Since OpenMDAO has enabled users to build models of this level of complexity,
we think some practical training is needed to make those models more useful for them.

## 1) Develop a class for building implicit models and improving their convergence

### Goal:
Provide users with a functional introduction to implicit systems, with a focus on when and how to apply specific kinds of solvers.
Specifically focus on the use of BalanceComps, their role in model building, and how to enhance convergence. 
Also focus on techniques for debugging a model when it won't converge.

## 2) Develop a class for practical optimization based around OpenMDAO
Having a flexible optimization framework is a bit of a double-edged sword,
because it allows you to create any problem formulation you want... even a bad one!
This course will teach the rules for proper optimization formulation,
techniques for improving convergence and performance, and debugging processes for when
the optimization inevitably doesn't do what you want it to!

# Benchmarking OpenMDAO Performance for large scale HPC applications
Initial benchmarking results comparing OpenMDAO based shape optimization
to a stand-alone implementation have shown that for smallish aerodynamic optimization problems 
OpenMDAO is about 10% slower than a more tailored implementation. 

This comparison was not a formal benchmark, more of a general sanity check. 
It gives us a sense that we are in the right ballpark, 
but more work is needed to properly characterize our performance in these applications. 
The ongoing Mphys effort, led by Kevin Jacobson provides a good opportunity to develop more rigorous benchmarks and look for bottlenecks. 

## 1) General performance improvements 

Though they are hard to predict, we expect to find several places where we can reduce the overhead and improve the efficiency of OpenMDAO. 
To find these, we'll be running larger test cases using Mphys and looking at performance profiling. 

## 2) No-copy transfers

## Goal: 

One source of potential wasted memory comes from the current implementation for data transfers. 
The OpenMDAO framework takes a highly general approach to data transfers that requires separate copies for outputs and input vectors. 
This is needed for situations where the output and input are not located on the same processor, because the data must be sent via MPI across from one location to the other. 
Another situation where copies are needed arises when using Jacobi style convergences. 

However, in many normal use cases you do not need to have separate copy of the output and inputs. 
In these cases, no-copy transfers could be implemented that would both save memory and compute overhead (no actual copying is done)
We have developed a theoretical way to make this work, 
but practical implementation will take some more effort. 

We think that this will offer some meaningful savings, especially for high fidelity use cases where we can avoid passing large state vectors between two components that are running sequentially and on the same processor. 

### Potential Challenges: 

It's possible that the implementation complexity of this solution will be too high to justify its gains. 
We'll need detailed profiling to test this. 

It's also possible that this transfer scheme will work well for high-fidelity tools but poorly for lower-fidelity tools. 
This has yet to be seen, but its a risk we recognize. 
Comparative benchmarking for a range of use cases will need to be done to test this. 



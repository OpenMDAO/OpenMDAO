OpenMDAO 2020 Development Roadmap
=================================

Author: Justin S. Gray   
Date: 2019-10-24

This document represents the perspective of the OpenMDAO development team on what we feel are 
the most important areas to focus on for framework development in 2020. It is intended to 
communicate this perspective to the OpenMDAO user community, and to provide a basis for
feedback. It is not intended to be all-encompassing or a binding development plan,
but should be regarded as an indication of the current direction of the core development team. 

There are four main areas of focus: 
- Releasing OpenMDAO V3.0
- Model & Data Visualization 
- New Differentiation Tools that Reduce User Effort
- Establishing A Community Contribution Process

----------------------------------------------
----------------------------------------------
----------------------------------------------
# Releasing OpenMDAO V3.0

In January 2020, we plan to release V3.0 of the framework. 
This update will **NOT** represent significant change to the codebase or functionality, 
but will include some important practical changes: 

## 1) Dropping support for Python 2.X and supporting only Python 3.6 and greater
- Scipy and Numpy have already dropped Python 2 support for future releases. 
- This will allow for a modest simplification of the codebase.

## 2) Removing all current deprecations 
- We have been making an earnest attempt not to break backwards compatibility through
  deprecations, but V3.0 will solidify the current non-deprecated APIs as the only APIs. 


----------------------------------------------
----------------------------------------------
----------------------------------------------

# Improvements for integration with distributed HPC codes

**Motivation** At the 2019 OpenMDAO workshop, a number of separate parties all independently discussed a weakness in the OpenMDAO setup process when it comes to integration with parallel, distributed memory codes like FEA and CFD solvers. 
Graeme Kennedy (Georgia Tech), Tim Brooks (Aurora Flight Sciences), Sandy Mader (University of Michigan), and Kevin Jacobson (NASA Langley) all separately broached the subject. 
Although there are current workable solutions, they are less efficient both in terms of memory and general code organization than we would all like. 

**Overall Goal:** To modify OpenMDAO to avoid any unnecessary data copying in cases where two components share an output/input on the same process and there is no unit conversion between them (OpenMDAO currently copies all outputs to a separate input vector), and to make any necessary extensions to the Component and Group APIs necessary to support the instantiation of these types of components when MPI COMMs must be shared between them. 
These changes, if at all possible, should not break any current APIs that do not deal directly with distributed components nor should the necessitate changes to existing non distributed components. 

## 1) Develop a stable of simple, but representative test cases that can serve as the basis for a broader code design discussion 
### Goal: 
- Capture community examples to summarize the key situations that are showing up in their usage, 
in order to provide the Dev Team with a clear understanding of all the various scenarios 
- Enable effective testing and benchmarking of new functionality at various scales without requiring execution of actual engineering codes 

### Potential Challenges: 
- Will likely require iterative interaction between the Dev Team and users who are providing test cases, in order to fully capture the nature of what they can and can not currently accomplish with existing APIs. 
It is very possible that we will need a separate small workshop specifically on this topic. 

## 2) Make low-level modifications to how OpenMDAO handles data-passing in key situations when copying can be avoided 
### Goal: 
- Make all OpenMDAO models (not just ones that run distributed across HPC resources) faster and more memory efficient by lowering required memory allocations for data vectors and reducing the need for wasteful copies 

### Potential Challenges: 
- The envisioned changes could add significant code complexity to the underlying data passing mechanisms, 
and will need to be handled very carefully to ensure a stable and maintainable result. 
It will probably take a while to implement and test this new idea. 
- It seems possible that, if too much for-looping is added to the data transfer scheme to support this, that in an attempt to speed things up for large HPC applications we may slow them down for low-fidelity applications.
So some benchmarking will be needed to try to ensure that does not happen, 
but if it does then the way forward is less clear and will require some careful reconsideration of how to retain performance for both types of applications. 


----------------------------------------------
----------------------------------------------
----------------------------------------------
# Model & Data Visualization 

**Motivation:** Provide tools that make it easier to quickly build and debug complex 
models with 100's of components organized into 10's of groups in 10's of hierarchy layers.

**Overall Goal:** Rely heavily on data stored by the CaseRecorder for all visualization. 
This makes the visualization more portable, as a case database can be easily shared with others
who could not necessarily run your model. It also makes visualization available after a
model has been run. 

## 1) Metamodel viewer   

### Goal:
- Allow users to inspect their MetaModel to check its accuracy, smoothness, etc.
- Encourage greater use of OpenMDAO MetaModel components.

### Potential Challenges:
- The current implementation is based on Bokeh, and performance isn't as fast as we'd like yet.
- Some functionality requires live prediction from a MetaModel instance, so we require a server
  process running in background. This means the functionality can't be built into the CaseRecorder.

### Notes: 
- An initial capability was released in OpenMDAO V2.9.0.


## 2) Improved N<sup>2</sup> model visualization tool  
### Goal:
- Make the tool more intuitive for new users.
- Make the tool more useful for navigating models with deep hierarchies and large number of 
  components/variables.

### Potential Challenges:
- The current user interface is stretched to its limit, and can't integrate any expanded 
  navigational features. We'll need a new concept for the UI.

### Notes:
- A POEM detailing proposed new API will be posted by end of January 2020 detailing proposed new API concept

## 3) Integrate N<sup>2</sup> with CaseRecorder data so the state of specific parts of the model can be inspected via case DBs. 
### Goal: 
- Leverage the N<sup>2</sup> interface to allow users to understand their models more efficiently.
- Potentially useful as a debugging tools.

### Potential Challenges: 
- It is unclear how will this functionality should be implemented. 
  Current possibilities are Jupyter Lab, or stand alone Electron based javascript app. 


----------------------------------------------
----------------------------------------------
----------------------------------------------
# New Differentiation Tools that Reduce User Effort

**Motivation:** While analytic derivatives provide massive performance gains, they also require 
significant user effort to implement. This creates a high activation energy that prevents many 
users from taking advantage of the most powerful feature of OpenMDAO. 

**Overall Goal:** Lower the effort required to implement analytic partial derivatives for components, 
offering a spectrum of options that can potentially trade required user effort with computational efficiency. 

## 1) Coloring applied to approximated partial derivatives 
### Goal: 
- Make coloring usable for FD and CS approximations.
- Offer much higher performance for components with very sparse partial derivative Jacobians 
  (e.g. vectorized components with no interaction between vector elements).
- When used with CS, this effectively offers a fast-forward mode AD.

### Potential Challenges: 
- The computational cost of coloring may be excessive for models with a lot of instances.
- Coloring algorithms are not perfect and may result in an invalid Jacobian, 
  so we need an effective way for users to check their colored partials. 
  The existing `check_partials` functionality can be leveraged, but may need some updates.
- In some cases, sparsity may not be high enough to offer meaningful performance gains. 
  We need to provide a means for the user to check this.

### Notes: 
- A prototype implementation is already available as an experimental feature.
- The partial coloring algorithm will also be useful for AD partials.


## 2) Algorithmic differentiation for component partial derivatives
    
### Goal: 
- Provide forward and reverse mode AD that works for a wide variety of general use cases including
  with many numpy functions.
- AD should have relatively good performance compared to hand differentiation.

### Potential Challenges: 
- AD tools will struggle with current OpenMDAO syntax for `compute` and `apply_nonlinear` methods, 
  so some kind of translation layer will be needed.
- Python AD libraries are not as well developed as those for other languages (e.g. C, Fortran, Julia).
- Certain coding practices are not compatible with AD, and will need to be avoided
  (e.g. modification of instance attributes, functions with side-effects).


----------------------------------------------
----------------------------------------------
----------------------------------------------
# Establishing A Community Contribution Process

## 1) Planning for the 2020 OpenMDAO workshop 

## 2) POEMs
- A POEM is a **p**roposal for **O**penMDAO **e**nhance**m**ent.
- This process is loosely based on the model used by cPython project (the PEP process).
- A new repository has been created for tracking POEMs (http://github.com/openmdao/POEMs).
- We are planning to work the POEM process on PR 1086 (https://github.com/OpenMDAO/OpenMDAO/pull/1086).

## 3) Establishing a formal OpenMDAO plugin system
- To go along with the POEM process, we need a method for users to add functionality without
  merging their code to the core codebase.

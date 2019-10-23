OpenMDAO 2020 Development Roadmap - 10/28/2019
=============================================

This roadmap represents the current plans of the OpenMDAO development team at the NASA Glenn Research Center. 
It is intended as a reference for the external community as to what the major development focus will be in the near and medium term.
It is not intended to be all encompassing or a binding development plan, 
but should be regarded as an very strong indication of the focus of the core development team. 

There are four main topic areas that will be focused on: 
- Releasing OpenMDAO V3.0! 
- Model & Data Visualization 
- New Differentiation Tools that Reduce User Effort
- Establishing A Community Contribution Process

----------------------------------------------
----------------------------------------------
----------------------------------------------
# Releasing OpenMDAO 3.0

In January 2020, we plan to release V3.0 of the framework. 
This update will **NOT** represent significant change to the codebase or functionality, 
but will include several key changes: 

## 1) Droping support for Python 2.0, and supporting only python 3.6 and greater
- Scipy and Numpy have already dropped Python 2 support for future releases. 
- Will allow us to modestly simplify our codebase 

## 2) Removing all current deprecations 
- We've made a strong attempt not to break backwards compatibility through deprecations where possible, 
but V3.0 will solidify the current primary (re: non deprecated) APIs as the only ones. 


----------------------------------------------
----------------------------------------------
----------------------------------------------
# Model & Data Visualization 

Motivation: Provide tools that make it easier to quickly build and debug complex 
models with 100's of components organized into 10's of groups in 10's of hierarchy layers

Overall Goal: Rely heavily on data stored in the CaseRecorder for all visualization. 
This makes the visualization more portable (i.e. can share case db with other) and 
also makes it available after a model was run. 

## 1) Metamodel viewer   

### Goal:
    - Allow users to inspect their MetaModel to check its accuracy/smoothness/etc 
    - Encourage greater use of OM MetaModel components 

### Potential Challenges:
    - Current implementation based on Bokeh, and performance isn't as fast as we'd like yet 
    - Some functionality requires live prediction from MetaModel instance, 
      so we require a server process running in background. 
      This means the functionality can't be built into the CaseRecorder

### Notes: 
    - initial capability was released in OpenMDAO V2.9.0


## 2) Improved n2 model viz tool  
### Goal:
    - Make the tool more intuitive for new users 
    - Make the tool more useful for navigating models with deep hierarchies 
      and large number of components/variables 

### Potential Challenges:
    - Current user interface is stretched to its limit, and can't 
      integrate any expanded navigational features. We'll need a new concept 
      for the UI 

### Notes:
    - two different proposals for new UI concept are outlines in POE-001 and POE-002

## 3) OVIS application for quickly plotting data from CaseRecorder data bases
### Goal: 
    - Make it simpler for users to inspect, navigate, and plot data from the case recorder 

### Potential Challenges: 
    - Going to be a separate stand-alone application. Will users be willing to download separate app? 
    - Separate application brings large development overhead for dev-team, 
      which may not be sustainable long term

----------------------------------------------
----------------------------------------------
----------------------------------------------
# New Differentiation Tools that Reduce User Effort

Motivation: While analytic derivatives provide massive performance gains, 
they also require significant user effort to implement. 
This creates a high activation energy that prevents many users from taking advantage of the most powerful feature of OpenMDAO. 

Overall Goal: Lower the effort required to implement analytic partial derivatives for components, 
offering a spectrum of options that can potentially trade required user effort with computational efficiency. 

## 1) Coloring applied to approximated partial derivatives 
### Goal: 
    - usable for FD and CS approximations 
    - offers much higher performance for components with very sparse partial derivative Jacobians 
      (e.g. vectorized components with no interaction between vector elements)
    - When used with CS, effectively offers a fast-forward mode AD

### Potential Challenges: 
    - computational cost of coloring may be excessive for models with lots of instances 
    - coloring algorithms are not perfect, and may result in invalid Jacobian, 
      so we need an effective way for users to check their colored partials. 
      (the existing check_partials functionality can be leveraged, but may need some updates)
    - in some cases, sparsity may not be high enough to offer meaningful performance gain. 
      how can user check this? 

### Notes: 
    - prototype implementation already available in experimental features 
    - partial coloring algorithm will also be useful for AD partials


## 2) Algorithmic differentiation for component partials derivatives
    
### Goal: 
- forward and reverse mode AD that works for a wide variety of general use cases including many numpy functions 
- Relatively good performance compared to hand differentiation

### Potential Challenges: 
- AD tools will struggle with current OpenMDAO syntax for `compute` and `apply_nonlinear` methods, 
  so some kind of translation layer will be needed
- Python AD libraries are not as well developed as other languages (e.g. C, Fortran, Julia)
- Certain coding practices are not compatible with AD, and will need to be avoided (e.g. modification of instance attributes, functions with side-effects)


----------------------------------------------
----------------------------------------------
----------------------------------------------
# Establishing A Community Contribution Process

## 1) Planning for the 2020 OpenMDAO workshop 

## 2) POEMs: **p**roposal for **O**penMDAO **e**nhance**m**en
- loosely based on the model used by cPython project (PEP process)
- A new repo has been created for tracking POEMs: 
  http://github.com/openmdao/POEMs
- Notes: We are planning to work the POEM process on PR 1086 (https://github.com/OpenMDAO/OpenMDAO/pull/1086)

## 3) Establishing a formal OpenMDAO plugin system
- To go alone with the POEM process, 
we need a manner for users to add functionality without merging their code to core codebase. 

 









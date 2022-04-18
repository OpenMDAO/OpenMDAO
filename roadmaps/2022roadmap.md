OpenMDAO 2022 Development Roadmap
=================================

Author: Rob Falck
Date: 2022-04-08

## What is this roadmap for?
This document represents the perspective of the OpenMDAO development team on what we feel will be the most important areas to focus on for framework development in 2021.
It is intended to communicate this perspective to the OpenMDAO user community, and to provide a basis for feedback.
It is not intended to be all-encompassing or a binding  development plan, but should be regarded as an indication of the current direction of the core development team.
Also, it should be clearly understood that there is very probably more listed here than can actually be accomplished in a single year!

## Summary of major progress from 2021

### Model Visualization and Data Postprocessing

Significant progress has been achieved here.
While we already had the ability to view some helpful outputs like the N2 diagram and the scaling report, new users didn't necessarily know about them or how to access them.
The new reports system means that a lot of the most common tools for visualization and feedback will "just be there" for users.
Moving reports to separate files reduces the standard output noise from OpenMDAO that tends to obscure warnings.
It also allows us to use richer HTML output more rather than relying on strictly ascii output.
Our focus on Jupyter notebooks has led to several notebook tools, including widgets for setting options, setting inputs, and plotting results from case record files.
This allows users to build tools based on OpenMDAO with a more obvious user interface.
The current case plotting tool runs on Jupyter notebooks.  More work is needed to make it available as a standalone utility.

### Building an OpenMDAO training course

While no substitution for a training course, the documentation update achieved by the team in 2021 using JupyterBook
is a signficant advance in interactivity.
The ability for users to test out code in the documentation online via Binder or Google Colab without installing anything locally significantly lowered the barrier to entry for OpenMDAO.
However, this means that building a signficant training course is still on the todo list for 2022.
The focus on various Jupyter notebook based tools should help this effort signifcantly, however.

### General performance improvements

We've made strides on improving performance.  In particular, total derivative coloring is substantially cheaper.

Metamodel interpolation has been a consistent bottleneck, largely due to allowing a variable number of dimensions for each interpolant.
By developing a series of fixed-dimension interpolants for structured data, we've been able to achieve considerably faster interpolation and differentiation.

While not on the 2021 roadmap, the new function wrapping capability is a big deal that lowers the barrier to entry for OpenMDAO.
Users no longer need to rewrite their existing functions as OpenMDAO components, and can put the `omf` function wrappers around them instead.
This is an avenue to the use of automatic differentiation via Google `jax`.

### No Copy Transfers

We worked on No Copy transfers for several weeks and weren't able to work all of the kinks out of the system.
To date this remains a goal.

# 2022 Focus Areas

## Training

With our advancements in Jupyter notebook-based tools and documentation, the next step is to develop an interactive training course with these as the foundation.

### Goal

Provide a series of Jupyter notebooks, separate from the documentation, which entail a training course on optimization.
Starting from basics of single disicipline optimization, work up in complexity to systems involving implicit behavior, multi-point optimization, and cover how to identify degenerate cases.

The training course should be paralleled by videos providing further depth than the example notebooks.
Teach users to get a feel for how to debug models when things aren't working.

## Aiming for Larger Components

### Goal

There are two ways in which we can focus on achieving larger monolithic components.
We've experimented with the use of subproblems, and we've noticed significant performance increases gained from keeping the subproblem's internal data out of the vectors of the outer problem.
We will document the use of subproblems.  If it becomes a common pattern, then some sort of official API for them might be warranted, but it's too early to make that call.

We will develop training and documentation of automatic differentiation in the form of Google's `jax` as well as `sympy` and its lambdify feature.
The ability to effectively use automatic differentiation is a prerequesite to computing second derivative information for those optimizers which can utilize it.

## Large multi-point parallelism

We've continued to push the parallel capability of OpenMDAO and are actively developing larger and larger applications.
In 2021 we accomplished a major overhaul of some of our distributed memory API.
As we start to push costlier applications, we're stressing the parallization capabilities that have long-existed in OpenMDAO: distributed components and parallel groups.

### Goal:
Leverage the reports system to provide better user feedback on processor utilization when running in parallel.
Continue to improve parallel performance, potentially developing an API for easier control over how processors are distributed on large-scale models.

## Hessian Information

### Goal

Expand on the MAUD implementation to provide second derivatives.
This information can be used by modern optimizers to significantly improve performance, but would all but require AD on reasonably complex models.
As a goal for this year we should try to get a optimization model working where analytic partial second derivatives are developed for a simle system and have OpenMDAO compute the total second derivatives.

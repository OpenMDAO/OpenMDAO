# OpenMDAO Development Map 2023
  
Author: Rob Falck  
Date: 2023-02-16 
  
## What is this roadmap for?  
As in previous years, this roadmap exists to provide general guidance for our development decisions in the coming year, and to provide a retrospective for our performance in the previous year.
  
## Performance Assessment for 2022 

As a general platform for the development of multidisciplinary analysis and optimization tools,OpenMDAO achieved significant uptake in its usage at NASA.
The agency is building its next generation of aircraft conceptual design tools on an OpenMDAO foundation.
High fidelity multi-physics analysis tools are being developed to work together in an OpenMDAO ecosystem.
OpenMDAO will have a significant role to play in the next generation of the agency's digital engineering strategy.

We were pleased to host the first in-person OpenMDAO workshop since 2019.
This provided the community with a look at how various organizations are using OpenMDAO, and provided valuable feedback for us as developers.
We get to see ways in which the tool has enabled some really impressive work by the community.
We also get to learn about challenges the community faces and develop ways to overcome these challenges.

It's fair to say that OpenMDAO has reached a level of maturity.
The fundamental capability is sound, and through some iteration we've arrived at a stable API.
With a strong, technically sound foundation much of our focus in the next year will be on continuing to increase the productivity of users by removing friction where we can.

- Our goal of developing an interactive training course has produced a great series of [Practical MDO](https://www.youtube.com/@OpenMDAO/featured) videos and corresponding notebooks.
- Work on the [Mphys](https://github.com/OpenMDAO/mphys) package has pushed our capability in terms of large models involving parallel computing. This work resulted in a [significant API change](https://github.com/OpenMDAO/POEMs/blob/master/POEM_075.md) to make the use of distributed components more clear, with some more improvements on the way.
- We continue to move away from reliance on feedback through the terminal and instead provide richer feedback through reports. We've received positive feedback on the standard set of reports we generate and will continue to improve upon these and develop new ones when warranted.
- The push to bring 2nd-derivative information into the Modular Approach to Unified Derivatives (MAUD) theory upon which OpenMDAO is based remains an open goal but there is still interest in that area.
 
## 2023 Focus Areas 

A broad goal for OpenMDAO is to make things simple for ~80% of our users who are doing typical analysis, while making things possible for the ~20% of our users who are advanced.

In the past we've always said that there are ways for users to incorporate OpenMDAO into their workflows.
In practice this means that users developing different tools somtimes come together and find that they need to rework their implementations so that their codes are compatible.
OpenMDAO _should_ be providing ways of doing things that cover the most common use cases, while not prohibiting users from doing things in their own way.
  
## Training  
  
The Practical MDO course is a great foundation on the general ideas behind multidisciplinary optimization.
In 2023, we intend to expand this into more OpenMDAO-focused content.

### Goal  
  
We'll continue to develop training materials and provide videos and course work more specific to OpenMDAO and related tools.
  
## OpenMDAO for Design Space Exploration

Much of the development of OpenMDAO has focused on optimization, and it's an amazing tool in that regard.
In this coming year, we intend to focus a bit more on the penultimate letter in the name: Analysis.

A common theme in talking with users this year has been:

	"OpenMDAO is amazing at optimization, but we want our team to be able to use their engineering experience to visualize and explore the tradespace themselves. Sometimes it's more important to get a good solution quickly rather than a perfect solution in a considerably longer time."

### Goal  

Enable the use of DOEDriver around optimization through the use of subproblems.
While getting derivative information _through_ a sub-optimization is non-trivial, the use of non-derivative-requiring drivers such as DOEDriver should be a relatively easy task for the user.
We will develop visualization tools similar to those used for metamodels to enable the user to quickly glean useful information from their parameter sweeps.


## A standard subproblem implementation 
  
Subproblems have seen considerable use this year.

They're useful for "functionalizing" OpenMDAO systems so that other tools can be wrapped around them and brought into the OpenMDAO ecosystem - this sort of application is used in tools like Dymos, which utilize `scipy.integrate.solve_ivp` and need to have an OpenMDAO system wrapped as a function.

They provide a clean mechanism through which drivers can be nested (such as in the previous topic).

They're also full of pitfalls when it comes to things like getting complex-step differentiation to work through them or with MPI.
  
### Goal:  

Develop a standard subproblem component that handles the tricky bits of the implementation for most use cases.
Quantify potential improvements that come from "hiding" data from deeper models through the subproblem interface.

## Local Model Configuration

When assembling complicated models, users often find themselves changing the way models behave.
In many cases, this involves making modifications in the setup methods of the source code of these models.
This is generally a poor practice, as it leads to modified source files that can cause unintended behavior when used elsewhere.
Users may forget about modifications they make and accidentally commit changed code to version control repositories.
We want to discourage this as much as possible, and provide users with a means by which they can tweak the behavior of a model from a run script.
To some extent, we've done this through [POEM 072](https://github.com/OpenMDAO/POEMs/blob/master/POEM_072.md)

### Goal

We want to continue to push this capability by providing a dictionary at the problem level that contains options to be consumed by subsystems within the problem model.

We may provide users with a standard file interface that allows them to provide model configuration in a flat file. This way, the code implementation would not need to be touched by the user in order to exercise the model.

## Hessian Information 

This one will remain a stretch goal for this year.
  
### Goal  
  
Expand on the MAUD implementation to provide second derivatives.  
This information can be used by modern optimizers to significantly improve performance, but would all but require AD on reasonably complex models.  
As a goal for this year we should try to get an optimization model working where analytic partial second derivatives are developed for a simple system and have OpenMDAO compute the total second derivatives.

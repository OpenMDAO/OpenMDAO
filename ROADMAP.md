##############################################
OpenMDAO Development RoadMap V0 - 10/28/2019
##############################################

This roadmap represents the current plans of the OpenMDAO development team at the NASA Glenn Research Center. 
It is intended as a reference for the external community as to what the major development focus is expected to be. 


Model & Data Visualization 
---------------------------
Overall Goal: Rely heavily on data stored in the CaseRecorder for all visualization. 
This makes the visualization more portable (i.e. can share case db with other) and 
also makes it available after a model was run. 

1) metamodel viewer 
    Goal: 
        - Allow users to inspect their MetaModel to check its accuracy/smoothness/etc 
        - Encourage greater use of OM MetaModel components 

    Potential Challenges: 
        - Current implementation based on Bokeh, and performance isn't as fast as we'd like yet 
        - Some functionality requires live prediction from MetaModel instance, 
          so we require a server process running in background. 
          This means the functionality can't be built into the CaseRecorder

    Notes: 
        - initial capability was released in OpenMDAO V2.9.0


2) Improved n2 model viz tool 
    Goal: 
        - Make the tool more intuitive for new users 
        - Make the tool more useful for navigating models with deep hierarchies 
          and large number of components/variables 

    Potential Challenges: 
        - Current user interface is stretched to its limit, and can't 
          integrate any expanded navigational features. We'll need a new concept 
          for the UI 

    Notes: 
        - two different proposals for new UI concept are outlines in POE-001 and POE-002

4) OVIS application for quickly plotting data from CaseRecorder data bases
    Goal: 
        - Make it simpler for users to inspect, navigate, and plot data from the case recorder 

    Potential Challenges: 
        - Likely going to be a separate stand-alone application. Will users be willing to download separate app? 
        - Separate application brings large development overhead for dev-team, 
          which may not be sustainable long term


New Differentiation Tools that Reduce User Effort
--------------------------------------------------

Motivation: Lower the effort required to implement analytic partial derivatives for components

Feature Proposals: 

1) Coloring applied to approximated partial derivatives 
    Goal: 
        - usable for FD and CS approximations 
        - offers much higher performance for components with very sparse partial derivative Jacobians 
          (e.g. vectorized components with no interaction between vector elements)
        - When used with CS, effectively offers a fast-forward mode AD

    Potential Challenges: 
        - computational cost of coloring may be excessive for models with lots of instances 
        - coloring algorithms are not perfect, and may result in invalid Jacobian, 
          so we need an effective way for users to check their colored partials. 
          (the existing check_partials functionality can be leveraged, but may need some updates)
        - in some cases, sparsity may not be high enough to offer meaningful performance gain. 
          how can user check this? 

    Notes: 
        - prototype implementation already available in experimental features 
        - partial coloring algorithm will also be useful for AD partials


2) Algorithmic differentiation for component partials derivatives
    
    Goal: 
        - forward and reverse mode AD that works for a wide variety of general use cases including many numpy functions 
        - Relatively good performance compared to hand differentiation

    Potential Challenges: 
        - AD tools will struggle with current OpenMDAO syntax for `compute` and `apply_nonlinear` methods, 
          so some kind of translation layer will be needed
        - Python AD libraries are not as well developed as other languages (e.g. C, Fortran, Julia)
        - Certain coding practices are not compatible with AD, and will need to be avoided (e.g. modification of instance attributes, functions with side-effects)



Establishing A Community Contribution Process
----------------------------------------------

1) Dev team is committing to host an annual OpenMDAO workshop! 

2) POE: PrOposal for Enhancement of openMdao
    - Based on the model used by cPython project (PEP process)
    - A new repo has been created for tracking POEMs: 
      http://github.com/openmdao/poe

    - Rules: 
        - the author of the POE is responsible for its curration 
        - New POEs are submitted via PR to the repo, with feedback taken via comments and PR's to proposers own branch
        - OpenMDAO Dev team will get final say as to weather to accept POE or not 
        - Regardless of acceptance or not, all POE PRs will be merged (with notation of exceptance or not) to keep a record of discussion
        - A POE may be submitted, reviewed, and accepted before any code has been written. 
          Contributors are encouraged to submit POEs before writing code if they are concerned 
          about wasting time coding something that would not get accepted. 
        - A POE may be submitted coincidentally with an accompanying PR containing an implementation of the proposed feature. 

    - Authors of POE are very strongly encouraged to take ownership of POE implementation. 
      Acceptance of a POE by the Devs is an indication that a well implemented feature will be accepted via PR once completed, 
      but is not a commitment by the Dev team to do that implementation (unless otherwise noted by the Dev team in the POE itself).
    - A primary consideration for the acceptence or rejection of a POE will be the amount of effort required to 
      maintain the new feature. Rejection of a POE on the grounds of future maintaince explicitly DOES NOT mean that the Dev team thinks it is a bad idea! 

    - Notes: 
        - We are planning to work the POE process on PR 1086 (https://github.com/OpenMDAO/OpenMDAO/pull/1086)









.. _`repository_structure`:

Repository Structure
====================

Project Documentation Structure
-------------------------------

First, to make things run smoothly, set up your project structure so that your documentation lies in the top-level project directory
in a directory named `/docs`. For instance, "openmdao/docs" or "openaerostruct/docs".  The reasons for this location:

    #. This is where openmdao's sourcedoc-generating script, `generate_docs` will be looking for docs.
    #. This is where the github-pages publishing package `travis-sphinx` will be looking for docs.

If you must put docs elsewhere for some reason, just be aware that it will require modifications to things in the above list.


I
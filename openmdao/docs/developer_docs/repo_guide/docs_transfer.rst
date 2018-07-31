Getting Docs For Your Plugin Transferred to github-pages
********************************************************

Taking your Github repository and building its docs on Travis CI, then moving them to github-pages can be a slightly complicated process.
Use this guide to help make it easier for your project.


Project Structure
-----------------

First set up your project structure so that your documentation lies in the root directory in a directory named `/docs`.
For instance, "openmdao/docs" or "openaerostruct/docs".  The reasons for this location:

    *. This is where openmdao's sourcedoc-generating script, will be looking for docs.
    *. This is where the github-pages publishing package `travis-sphinx` will be looking for docs.

If you must put docs elsewhere for some reason, just be aware that it will require modifications to things in the above list.



Importing Tools from OpenMDAO
-----------------------------

During this process, your will need access to a couple of things in OpenMDAO that
You need to be able to use things from OpenMDAO's repo:
openmdao/docs/_utils will get you things like our sourcedoc building script, `generate_docs`, which will be called from conf.py

openmdao/docs/_exts will get you access to our powerful embedding library, with things like `embed_code`,
and `embed_options`.  this will help you to include things into your documentation that will dynamically stay updated
with the code in your project or in the OpenMDAO project.  To get access to these items both in your local install
and on CI, you will need to do a git clone and `pip install -e .`, because the private modules docs/_exts and docs/_utils are not in the OpenMDAO
API when installed by `pip install openmdao`, so the imports will fail.

Your sphinx documentation will need its own conf.py and theme and style.css to customize them into something that will make them their own.

General Docs Settings

Conf.py

Autodocumentation Generator

Tagging Tool

Transfer to github-pages
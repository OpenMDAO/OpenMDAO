#-------Begin tag pre-processing-----------------------------
#A script that finds occurrences of the .. tags:: directive
#and sets up the structure of the tags directory.  One file
#is created for each subject tag, that file contains links to
#each instance of the tag throughout the docs.

import sys, os, shutil, re
from six import PY3

def make_tagdir():
    #clean up tagdir, create tagdir, return tagdir
    dir = os.path.dirname(__file__)
    tagdir = os.path.join(dir, "../tags")

    if os.path.isdir(tagdir):
      shutil.rmtree(tagdir)

    os.mkdir(tagdir)

    return tagdir

def make_tagfiles(docdirs, tagdir):
    #pull tags from each file, then make a file
    #for each tag, containing all links to tagged files.
    for docdir in docdirs:
        for dirpath, dirnames, filenames in os.walk(docdir):
            for filename in filenames:
                #the path to the file being read for tags
                sourcefile = os.path.join(dirpath, filename)
                #a file object for the file being read for tags
                if PY3:
                    with open(sourcefile, 'r', encoding="latin-1") as textfile:
                        #the text of the entire sourcefile
                        filetext = textfile.read()
                else:
                    with open(sourcefile, 'r') as textfile:
                        #the text of the entire sourcefile
                        filetext = textfile.read()
                #pull all tag directives out of the filetext
                matches = re.findall(".. tags::.*$", filetext)

                #for every instance of tag directive, get a list of tags
                for match in matches:
                    match=match.lstrip(".. tags::")
                    taglist=match.split(", ")

                    for tag in taglist:
                        filepath = os.path.join(tagdir, (tag+".rst"))

                        #if the tagfile doesn't exist, let's put in a header
                        if not os.path.exists(filepath):
                            tagfileheader="""
=========================
%s
=========================
  .. toctree::
     :titlesonly:
""" % tag

                            #write the header for this tag's file.
                            with open(filepath, 'a') as tagfile:
                                tagfile.write(tagfileheader)
                        #write a link into an existing tagfile.
                        with open(filepath, 'a') as tagfile:
                            tagfile.write("     ../%s\n" % (sourcefile))

def make_tagindex(tagdir):
    #once all the files exist, create a simple index.rst file
    indexfile = tagdir + "/index.rst"

    for filepath, dirnames, filenames in os.walk(tagdir):
        with open(indexfile, 'a') as index:
            index.write("""
================
Tags in OpenMDAO
================
.. toctree::
   :maxdepth: 1
   :glob:
   ./*
 """)

def tag(args=None):
    if args is None:
        args = sys.argv[1:]
    #set the directories in which to find tags
    docdirs=['features', 'style_guide', 'tutorials']
    tagdir = make_tagdir()
    make_tagfiles(docdirs, tagdir)
    make_tagindex(tagdir)

tag()
#-------End tag pre-processing-----------------------------

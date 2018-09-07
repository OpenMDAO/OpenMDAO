.. _`advanced_operations_automation`:

Automating Doc Build and Custom Doc Deployment from Travis CI
=============================================================

.. note::
    The current recommended procedure for new repos is to use github-pages as outlined :ref:`here <github_pages>`.
    These instructions exist to explain how OpenMDAO's docs work for now. If you are
    starting a new project, go follow the :ref:`github-pages instructions <github_pages>` instructions, as they are much simpler.

The following process is a one-time setup for the owner of a project to complete.  Once it is finished, it will become
transparent to other developers and users. The process outlined here is to get a project's docs to build on Travis CI, and then
transfer the built docs off to a web server. (This example illustrates OpenMDAO's process.)
The reason you'd use this method instead of just setting up readthedocs.org, is because on Travis, you can very specifically
customize the installation of your project.  Certain advanced items in OpenMDAO 2.0 just can't be installed on readthedocs, but can be on Travis.
Having a full install means that the docs will be more complete, because embedded tests that rely on, for instance, MPI, will actually work on Travis CI,
whereas they currently do not on readthedocs.org.


Getting Started and General Concept
-----------------------------------

The very first thing to do is to comb through your Travis CI build logs to ensure that your installation is
working the way that you expect, and that the docs are actually building successfully and
without errors on Travis on any/all platforms and Python distributions. If you’re using Sphinx, the docs will build into
your `<project>/docs/_build/html` directory. The concept is simply to copy the contents of that html directory over to a
server directory that is serving up html.  For the purposes of OpenMDAO/blue, this happens on `webXXX.webfaction.com:webapps/<doc_serving_app>`

You need to make several changes to your `.travis.yml` file in order to make this happen.
One of these steps, you will find out, is going to re-write your `.travis.yml` file and take out all the spacing,
comments, and formatting. So if you like the way your `.travis.yml` file looks, I suggest backing up that file beforehand.
You can then take the auto-generated lines that we’re about to generate, and stick them back in the pretty-looking `.travis.yml` file.

To get this doc transfer automated, we need to be able to move built docs from Travis to our private server without any required input,
because there is no human in the loop to type passwords or answer prompts.  This will require some ssh key wizardry.
The overall concept is to have an encrypted key already on the Travis server, then to decrypt it, and use it to do a
password-less transfer of the docs from Travis to our private server (in our case, Webfaction).  Here's the outline:

    ::

	#. Generate a dedicated SSH key for this job (it is easier to later isolate and to revoke it, if we ever must).
	#. Encrypt the private key to make it readable only by Travis CI .
	#. Copy the public key file into the remote SSH host’s allowed_hosts file.
	#. Modify the yaml further.
	#. Commit the modified .travis.yaml and encrypted key file into Git.




Specific Commands
-----------------

The commands to do the above look something like this:

**1. Generate the Key**
    Run this command from the top level of your project (in the example case, from the `openmdao/` level):

    ::

        ssh-keygen -t rsa -b 4096 -f ./travis_deploy_rsa

        -t chooses the type of key to generate, in this case, rsa
        -b  Specifies the number of bits in the key to create.  For RSA keys, the minimum
            size is 768 bits and the default is 2048 bits. I used 4096 because it was recommended by Travis.
        -f output into this keyfile, in this case, I want the key created right where I am in the dir structure.

    It will spit out both a private key file and a public key file: `travis_deploy_rsa` and `travis_deploy_rsa.pub`.

**2. Encrypt the Key**
    Again from the same directory, execute the next command to encrypt the PRIVATE key file that we just generated.
    (NOTE: this is the aforementioned command that will change your .travis.yml file! ):

    ::

        travis encrypt-file travis_deploy_rsa --add --debug --org

        --add adds the decryption command to the .travis.yml file automatically.
        --debug shows all the available output of the command.
        --org creates a key for the org side, not the paid side

    So, what does this command do?

    A. Well, it creates a file `travis_deploy_rsa.enc`, that is an encrypted file. We want to add ONLY this encrypted file to the current repository.  `git add travis_deploy_rsa.enc`  The reason we want to add travis_deploy_rsa.enc to the repo is because the repo gets cloned from github onto the Travis machines, and we want to ensure this file is up there to decrypt.

    B. Next, the above command edits the  `before install:` section of your .travis.yml file with the instruction to decrypt the key at the appropriate time. This is the part that you might want to just copy back into your original .travis.yml file.

        ::

            before_install:
                openssl aes-256-cbc -K $encrypted_67e08862e114_key -iv $encrypted_67e08862e114_iv
                    -in travis deploy.rsa.enc -out /tmp/travis_deploy.rsa -d;


        That openssl command above command has several arguments:

        ::

            aes-256-cbc is a subcommand that just calls out an decryption strategy

            -d decrypt
            -K/-iv key/iv in hex is the next argument
            -in is the file that needs encrypting
            -out is the decrypted key file being written out


        Note that you will need to manually edit the file to add checks against the event $TRAVIS_PULL_REQUEST, as seen in the block below.
        We do not want this whole process to happen during every pull request, but only after a merge has occurred.
        it's also an added security benefit, that these decryption processes are only run by your own account, and not by any person who can fork your code.
        There are a couple of other spots in the file where this check needs to be added, as you'll see below.

        ::

            before_install:
                - if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
                    openssl aes-256-cbc -K $encrypted_67e08862e114_key -iv $encrypted_67e08862e114_iv
                        -in travis deploy.rsa.enc -out /tmp/travis_deploy.rsa -d;
                  fi

    C. Finally, the command is supposed to create and assign two environment variables in the the Travis CI settings for the repository in question.  This was a big stumbling block for me, and it is why I added the `—debug` arg to the `travis encrypt-file` command.
       I was executing the correct command, but the identity I was signed in as (me) and the identity of the repo (OpenMDAO) didn’t match and so those env vars were never created.  Going to the travis-ci.org webpage for OpenMDAO and going into Settings and using
       the web interface to add two new env vars is the way around this problem.  But what are the env vars called, and what will their values be?  That’s where —debug comes in (actual values redacted):

        ::

            (openmdao2)$ travis encrypt-file deploy.rsa --add --debug
            ** Loading "/Users/xxxxxxxx/.travis/config.yml"
            ** GET "repos/OpenMDAO/openmdao"
            **   took 0.2 seconds
            encrypting deploy.rsa for OpenMDAO/openmdao
            storing result as deploy.rsa.enc
            storing secure env variables for decryption
            ** GET "settings/env_vars/?repository_id=XXXXXXX"
            **   took 0.051 seconds
            ** POST "settings/env_vars/?repository_id=XXXXXXX" "{\"env_var\":{\"public\":false,\"name\":\"encrypted_67eXXXXXXXXX_key\",\"value\":\"?????????????????????????????\"}}"
            **   took 0.064 seconds
            ** GET "settings/env_vars/?repository_id=XXXXXXX"
            **   took 0.049 seconds
            ** POST "settings/env_vars/?repository_id=XXXXXXX" "{\"env_var\":{\"public\":false,\"name\":\"encrypted_67eXXXXXXXXX_iv\",\"value\":\"??????????????????????????????\"}}"
            **   took 0.057 seconds

            Make sure to add deploy.rsa.enc to the git repository.
            Make sure not to add deploy.rsa to the git repository.
            Commit all changes to your .travis.yml.
            ** Deleting "/Users/xxxxxxxx/.travis/error.log"
            ** Storing "/Users/xxxxxxxx/.travis/config.yml"


       The command is attempting to POST those env vars, but they don’t seem to make it to the OpenMDAO account.
       However, the name and value are right there in the debug output, so they can easily be copied and pasted into the Travis CI web
       interface (https://travis-ci.org/<user>/<project>/settings ). Creating these env variables must be done, because the
       openssl decrypt command is going to refer to those env vars in the `-K` and `-iv` arguments.


**3. Copy Key to Web Server**
    To copy the key over to your web server.  In the specific case of OpenMDAO, let’s take a moment to explore what needs to be done on Webfaction.

    A. Need to create a web server application on Webfaction (for local NASA users).

     #. Go to panel.webfaction.com,
     #. Click Domains/Websites,
     #. Choose the Applications tab.
     #. Click the Add New Application button.
     #. Give your new app an appropriate name, for our example, I chose “openmdaodocs.”
     #. Make the app as type “Static Only (no .htaccess)."
     #. Click on Websites, choose openmdao_org,
     #. Choose, “reuse an existing application” and then pick your newapp and give it a url.
     #. After a moment, a folder will appear on webXXX, under ~/webapps/<name>, that is accessible at openmdao.org/<url>. Keep in mind that webXXX.webfaction.com:webapps/<name> will be your path to copy your docs to.

    B. Need to copy the public key generated above to our Webfaction server to allow passwordless entrance.

     1. On webXXX, in the ~/.ssh folder, there is a file called authorized_keys.
     2. Copy the contents of the travis_deploy_rsa.pub as an entry into the authorized_keys file.

**4. Modify the YAML Further**

   A. Late in the `before_install` section add this line:
        :code:`- echo -e "Host <server address>\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config`
        (This will turn off a human-prompt by Travis machine “are you willing to accept <server address> as a host (yes/no)”)

   B. Create a new subhead in your `addons`->`apt` called `ssh_known_hosts`, like this:

     ::

        addons:
            apt:
                sources:
                - ubuntu-toolchain-r-test
                packages:
                - gfortran
                - libblas-dev
                - liblapack-dev
                - libopenmpi-dev
                - openmpi-bin
                ssh_known_hosts:
                - <server address>


   C. Finally, add these sections to the end of your .travis.yml file, after your after_success section:

    `before_deploy` : The `before_deploy` makes sure the newly-decrypted key is the right permissions and that the Travis system is aware of it.

    ::

        before_deploy:
        - if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
            eval "$(ssh-agent -s)";
            chmod 600 /tmp/deploy.rsa;
            ssh-add /tmp/deploy.rsa;
          fi

    `deploy` is focused on actually transferring the docs.  Note there is a logic check that makes sure the doc copy only happens on ONE machine (don’t want 4 machines racing to rsync docs!), and only on a certain branch, and only after success.

    ::

        deploy:
          provider: script
          skip_cleanup: true
          script:
            - if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
                - if [ "$MPI" ] && [ "$PY" = "3.4" ]; then
                    cd openmdao/docs;
                    rsync -r --delete-after -v _build/html/* <username>@<server address>:webapps/openmdaodocs;
                  fi
              fi
          on:
            branch: master

**5. Commit the .travis.yml file and encrypted key file into Git**

    You have now heavily edited your .travis.yml, and created an encrypted key locally.  `git add` the encrypted key (.enc) file to your repository.
    Once you're done with everything and it all works, you should move or discard the private and public key files to make sure they do not end up in your repository.
    A quick `git commit` should finish things up.  Your pull request to enable automation should be only those two files.
    The remainder of the work was done in the travis-ci.org settings and on your web server.  Once this pull request is accepted to your
    repository, you should be able to check the logs to make sure it's working.  And of course, seeing new documentation transferred up on your web server is all the
    proof you need that things are working.
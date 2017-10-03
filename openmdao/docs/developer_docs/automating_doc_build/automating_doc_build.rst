**************************************************
Automating Doc Build and Deployment from Travis CI
**************************************************

The following is a process to get OpenMDAO's (or any project's) docs to build on Travis CI, and then transfer the built docs off to another server.
The reason you'd use this method instead of just setting up readthedocs.org, is because on Travis, you can very specifically
customize the installation of your project.  Certain advanced items in OpenMDAO 2.0 just can't be installed on readthedocs, but can be on Travis.

Getting Started and General Concept
-----------------------------------

The very first thing to do is to comb through your Travis CI build logs to ensure that your installation is
working the way that you expect, and that the docs are actually building successfully and
without errors on Travis on any/all platforms and Python distributions. If you’re using Sphinx, the docs will build into
your `<project>/docs/_build/html` directory. The concept is simply to copy the contents of that html directory over to a
server directory that is serving up html.  For the purposes of OpenMDAO/blue, this happens on `web543.webfaction.com:webapps/<doc_serving_app>`


You need to make several changes to your `.travis.yaml` file in order to make this happen.
One of these steps, you will find out, is going to re-write your `.travis.yaml` file and take out all the spacing,
comments, and formatting. So if you like the way your `.travis.yaml` file looks, I suggest backing up that file beforehand.
You can then take the auto-generated lines that we’re about to generate, and stick them back in the pretty-looking .travis.yml file.

To get this doc transfer automated, we need to be able to move things from Travis to our private server without any required input,
because there is no human in the loop to type passwords or answer prompts.  This will require some ssh key wizardry.
The overall concept is to have an encrypted key already on the Travis server, then to decrypt it, and use it to do a
password-less transfer of the docs from Travis to our private server (in our case, Webfaction).  Here's the outline:

    ::

	1. Generate a dedicated SSH key for this job (it is easier to isolate and to revoke it if we ever must).
	2. Encrypt the private key to make it readable only by Travis CI .
	3. Copy the public key file into the remote SSH host’s allowed_hosts file.
	4. Modify the yaml further.
	5. Commit the modified .travis.yaml and encrypted key file into Git.




Specific Commands
-----------------

The commands to do all the above look something like this:

**1. Generate the Key**
    Run this command from the top level of your project (in my case the `blue/` level):

    ::

        ssh-keygen -t rsa -b 4096 -f ./travis_deploy_rsa

        -t chooses the type of key to generate, in this case, rsa
        -b  Specifies the number of bits in the key to create.  For RSA keys, the minimum
            size is 768 bits and the default is 2048 bits. I used 4096 because it was recommended by Travis.
        -f output into this keyfile, in this case I want the key created right where I am in the dir structure.

    It will spit out both a private key file and a public key file: `travis_deploy_rsa` and `travis_deploy_rsa.pub`.

**2. Encrypt the Key**
    Again from the same directory, execute the next command to encrypt the PRIVATE key file that we just generated.
    (NOTE: this is the aforementioned command that will change your .travis.yml file! ):

    ::

        travis encrypt-file travis_deploy_rsa --add —debug

        —add adds the decryption command to the .travis.yml file automatically.
        —debug shows all the available output of the command.

    So, what does this command do?

    A. Well, it creates a file `travis_deploy_rsa.enc`, that is an encrypted file. We want to add ONLY this encrypted file to the current repository.  `git add travis_deploy_rsa.enc`  The reason we want to add travis_deploy_rsa.enc to the repo is because the repo gets cloned from github onto the Travis machines, and we want to ensure this file is up there to decrypt.

    B. Next, the above command edits the  `before install:` section of your .travis.yml file with the instruction to decrypt the key at the appropriate time. This is the part that you might want to just copy back into your original .travis.yml file.

        ::

        before_install:
            - openssl aes-256-cbc -K $encrypted_67e08862e114_key -iv $encrypted_67e08862e114_iv
                -in travis deploy.rsa.enc -out /tmp/travis_deploy.rsa -d;

    That openssl command above command has several arguments:

        ::

            aes-256-cbc is a subcommand that just calls out an decryption strategy

            -d decrypt
            -K/-iv key/iv in hex is the next argument
            -in is the file that needs encrypting
            -out is the decrypted key file being written out

    C. Finally, the command is SUPPOSED to create and assign two environment variables in the the travis settings for the repository in question.  This was a big stumbling block for me…and it is why I added the `—debug` arg to the `travis encrypt-file` command.
        I was executing the correct command, but the identity I was signed in as (me) and the identity of the repo (OpenMDAO) didn’t match and so those env vars were never created.  Going to the travis-ci.org webpage for OpenMDAO and going into Settings and using
        the web interface to add two new env vars is the way around this problem.  But what are the env vars called, and what will their values be?  That’s where —debug comes in:

        ::

            (blue2)$ travis encrypt-file deploy.rsa --add --debug
            ** Loading "/Users/xxxxxxxx/.travis/config.yml"
            ** GET "repos/OpenMDAO/blue"
            **   took 0.2 seconds
            encrypting deploy.rsa for OpenMDAO/blue
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

     1. Go to panel.webfaction.com,
     2. Click Domains/Websites,
     3. Choose the Applications tab.
     4. Click the Add New Application button.
     5. Give your new app an appropriate name, for our example, I chose “bluedocs.”
     6. Make the app as type “Static Only (no .htaccess)."
     7. Click on Websites, choose openmdao_org,
     8. Choose, “reuse an existing application” and then pick your newapp and give it a url.
     9. After a moment, a folder will appear on web543, under ~/webapps/<name>, that is accessible at openmdao.org/<url>. Keep in mind that web543.webfaction.com:webapps/<name> will be your path to copy your docs to.

    B. Need to copy the public key generated above to our Webfaction server to allow passwordless entrance.

     1. On web543, in the ~/.ssh folder, there is a file called authorized_keys.
     2. Copy the contents of the travis_deploy_rsa.pub as an entry into the authorized_keys file.

**4. Modify the YAML Further…**

   A. Late in the before_install section add this line:
        `- echo -e "Host web543.webfaction.com\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config`
        (This will turn off a human-prompt by Travis machine “are you willing to accept web543 as a host (yes/no)”)

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
                - web543.webfaction.com


   C. Finally, add these sections to the end of your .travis.yml file, after your after_success section:

    `before_deploy` : The `before_deploy` makes sure the newly-decrypted key is the right permissions and that the Travis system is aware of it.

      ::

        before_deploy:
        - eval "$(ssh-agent -s)";
        - chmod 600 /tmp/deploy.rsa;
        - ssh-add /tmp/deploy.rsa;


    `deploy` is focused on actually transferring the docs.  Note there is a section that makes sure the doc copy only happens on ONE machine (don’t want 4 machines racing to rsync docs!), and only on a certain branch, and only after success.

      ::

        deploy:
          provider: script
          skip_cleanup: true
          script:
          - if [ "$MPI" ] && [ "$PY" = "3.4" ]; then
              cd openmdao/docs;
              rsync -r --delete-after -v _build/html/* openmdao@web543.webfaction.com:webapps/bluedocs;
            fi
          on:
            branch: master

To summarize, you need to heavily edit your .travis.yml, create and `git add` an encrypted key (.enc) file.
Your pull req should be only those files.  The rest of the work is done in the travis-ci.org settings and
on your web server.



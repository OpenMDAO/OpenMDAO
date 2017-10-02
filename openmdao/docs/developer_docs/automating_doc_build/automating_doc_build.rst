Here is the process I used to get blue docs to build on Travis, and then transfer them off to another server.

The very first thing you need to do is to ensure that your installation is working and that the docs are actually building
without errors on Travis. If you’re using Sphinx, the docs will build into your <project>/docs/_build/html directory.
The concept is simply to copy the contents of that html directory over to a server directory that is serving up html.
For the purposes of OpenMDAO/blue, this happens on `web543.webfaction.com:webapps/<doc_serving_app>`


You need to make several changes to your .travis.yaml file in order to make this happen.
One of these steps, I found out the hard way, is going to re-write your yaml file and take out all the spacing, comments and formatting.
So if you like the way your yaml looks, I suggest backing up that file beforehand.
Then you can take the auto-generated lines that we’re about to generate, and stick them back in the pretty-looking .travis.yml file.

To get this doc transfer automated, we need to be able to move things from server to server without any required input, because there is no human in the loop to type passwords or answer prompts.  This will require some ssh key wizardry.  The overall concept is to have an encrypted key already on the Travis server, then to decrypt it, and use it to do a passwordless transfer of the docs from Travis to Webfaction.

	1.	Generate a dedicated SSH key for this job (it is easier to isolate and to revoke it if we ever must).
	2.	Encrypt the private key to make it readable only by Travis CI .
	3.	Copy the public key file into the remote SSH host’s allowed_hosts file.
	4.     Modify the yaml further.
    5.     Commit the modified yaml and encrypted key files into Git.

The commands to do all the above look something like this:

1. To do the generation of the key, run this command at the top level of your project (in my case the blue/ level):
`ssh-keygen -t rsa -b 4096 -f ./travis_deploy_rsa`

-t chooses the type of key to generate, in this case, rss
-b  Specifies the number of bits in the key to create.  For RSA keys, the minimum
    size is 768 bits and the default is 2048 bits. I used 4096 because it was recommended by travis.
-f output into this keyfile, in this case I want the

 It will spit out a private key file and a public key file: travis_deploy_rsa and travis_deploy_rsa.pub.

2.  Again from the same directory, execute the next command to encrypt the PRIVATE key file that we just generated. (NOTE: this is the command that will change your .travis.yml file! ):

`travis encrypt-file travis_deploy_rsa --add —debug`

—add adds the decryption command to the .travis.yml file automatically.
—debug shows all the available output of the command.

So, what does this command do?
*Well, it creates a file `travis_deploy_rsa.enc`, that is an encrypted file. We want to add ONLY this encrypted file to the current repository.  `git add travis_deploy_rsa.enc`  The reason we want to add travis_deploy_rsa.enc to the repo is because the repo gets cloned from github onto the Travis machines, and we want to ensure this file is up there to decrypt.

*Next, the above command edits the  `before install:` section of your .travis.yml file with the instruction to decrypt the key at the appropriate time. This is the part that you might want to just copy back into your original .travis.yml file.

`before_install:
- openssl aes-256-cbc -K $encrypted_67e08862e114_key -iv $encrypted_67e08862e114_iv
  -in travis deploy.rsa.enc -out /tmp/travis_deploy.rsa -d;`

That openssl command above command has several arguments:
aes-256-cbc is a subcommand that just calls out an decryption strategy
-d decrypt
-K/-iv key/iv in hex is the next argument
-in is the file that needs encrypting
-out is the decrypted key file

*Finally, It is SUPPOSED to create and assign two environment variables in the the travis settings for the repository in question.  This was a big stumbling block for me…and it is why I added the `—debug` arg to the `travis encrypt-file` command.
I was executing the correct command, but the person I was signed in as (kmarsteller) and the identity of the repo (OpenMDAO) didn’t match and so those env vars were never created.  Going to the travis-ci.org webpage for blue and going into Settings and using the web interface to add two new env vars is the way around this problem.  But what are the env vars called, and what will their values be?  That’s where —debug comes in:

```
(blue2) GRSLA16081176:blue kmarstel$ travis encrypt-file deploy.rsa --add --debug
** Loading "/Users/kmarstel/.travis/config.yml"
** Timeout::Error: execution expired
** GET "repos/OpenMDAO/blue"
**   took 0.2 seconds
encrypting deploy.rsa for OpenMDAO/blue
storing result as deploy.rsa.enc
storing secure env variables for decryption
** GET "settings/env_vars/?repository_id=1912536"
**   took 0.051 seconds
** POST "settings/env_vars/?repository_id=1912536" "{\"env_var\":{\"public\":false,\"name\":\"encrypted_67e08862e114_key\",\"value\":\"ffacb31708427ca650bb519a5e3294478f22262b5f15267aafc0178340417849\"}}"
**   took 0.064 seconds
** GET "settings/env_vars/?repository_id=1912536"
**   took 0.049 seconds
** POST "settings/env_vars/?repository_id=1912536" "{\"env_var\":{\"public\":false,\"name\":\"encrypted_67e08862e114_iv\",\"value\":\"06fb0c828c00fc07b8dd7fa4d1e04584\"}}"
**   took 0.057 seconds

Make sure to add deploy.rsa.enc to the git repository.
Make sure not to add deploy.rsa to the git repository.
Commit all changes to your .travis.yml.
** Deleting "/Users/kmarstel/.travis/error.log"
** Storing "/Users/kmarstel/.travis/config.yml"
```

The command is attempting to POST those env vars, but they don’t make it. But the name and value are right there in the debug output, so they can easily be copied-and-pasted into the Travis web interface (https://travis-ci.org/<user>/<project>/settings). Creating these env variables must be done, because the openssl decrypt command is going to refer to those env vars in the -K and -iv arguments.

(Travis_Env_Vars.png)

3. To copy the key over to webfaction.  Let’s take a moment to explore what needs to be done on webfaction.


Webfaction Side
1. Need to create a web server application.
2. Need to copy the public key generated above to our Webfaction server to allow passwordless entrance.

1. Go to panel.webfaction.com, click Domains/Websites, then choose the Applications tab.  Click the Add New Application button.  Give your new app an appropriate name, for our example, I chose “bluedocs.” Make the app as type “Static Only (no .htaccess). You then need to click on Websites, choose openmdao_org, choose, “reuse and existing application” and then pic your app and give it a url.  After a moment, a folder will appear on web543, under ~/webapps/<name>, that is accessible at openmdao.org/<url>.  Keep in mind that web543.webfaction.com:webapps/<name> will be your path to copy your docs to.

2. On web543, in the ~/.ssh folder, there is a file called authorized_keys.  Copy the contents of the travis_deploy_rsa.pub as an entry into the authorized_keys file.



Modify the yaml further…

late in the before_install section add this line:
- echo -e "Host web543.webfaction.com\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config

this will turn off a human-prompt by Travis machine “are you willing to accept web543 as a host (yes/no)”

create a new subhead in your addons->apt called ssh_known_hosts, like this:
before_deploy:
- eval "$(ssh-agent -s)";
- chmod 600 /tmp/deploy.rsa;
- ssh-add /tmp/deploy.rsa;

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

Finally, add these sections to the end of your .travis.yml file, after your after_success section:
before_deploy:
- eval "$(ssh-agent -s)";
- chmod 600 /tmp/deploy.rsa;
- ssh-add /tmp/deploy.rsa;

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


The first part makes sure the newly-decrypted key is the right permissions and that the Travis system is aware of it.

The second part is focused on actually deploying the docs.  Note there is a section that makes sure the doc copy only happens on ONE machine (don’t want 4 machines racing to rsync docs!), and only on a certain branch, and only after success.


So to summarize, you need to heavily edit your .travis.yml, create and git add an encrypted key (.enc) file.  So your pull req should be only those files.  The rest of the work is done in the travis-ci.org settings and on your webfaction server.



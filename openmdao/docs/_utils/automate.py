import subprocess
import os
import pipes
import openmdao


def get_tag_info():
    """
    Return the latest git tag, meaning, highest numerically, as a string.
    """
    # using a pattern to only grab tags that are in version format "X.Y.Z"
    git_versions = subprocess.Popen(['git', 'tag', '-l', '*.*.*'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = git_versions.communicate()

    # take the output of git tag -l *.*.*, and split it from one string into a list.
    version_tags = cmd_out.split()

    # use sort to put the versions list in order from lowest to highest
    version_tags.sort(key=lambda s: [int(u) for u in s.split('.')])
    print("The tags in this repo, sorted, are: " + str(version_tags))

    # grab the highest tag that this repo knows about
    latest_tag = version_tags[-1]

    # grab the version of the most recent openmdao release
    om_version = openmdao.__version__
    if (om_version not in version_tags):
        print("The tag " + om_version + " does not exist in this repo!")

    print("LATEST TAG: " + latest_tag + ", OPENMDAO VERSION: " + om_version)
    return latest_tag


def get_commit_info():
    """
    Return the commit number of the most recent git commit as a string.
    """
    git_commit = subprocess.Popen(['git', 'show', '--oneline', '-s'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    commit_cmd_out, commit_cmd_err = git_commit.communicate()
    commit_id = commit_cmd_out.split()[0]
    print("COMMIT ID: " + commit_id)
    return commit_id


def exists_remote(host, path):
    """
    Test if a dir exists at path on a host accessible with SSH.
    """
    status = subprocess.call(
        ['ssh', host, 'test -d {}'.format(pipes.quote(path))])
    if status == 0:
        print ("Remote directory " + path + " exists.")
        return True
    elif status == 1:
        print("Remote directory " + path + " does not exist.")
        return False
    raise Exception('SSH failed')


def automate():
    """
    Perform operations, then set environment variables for later use in conf.py and .travis.yml
    """
    tag = get_tag_info()
    release_path = '/home/openmdao/webapps/twodocversions/' + tag
    remote_host = 'openmdao@web543.webfaction.com'

    # if the remote path exists, release docs have happened already
    # and with commit ID as identifier in docs title. (used in conf.py)
    # finally, set the place for travis to send docs to "latest" .
    if exists_remote(remote_host, release_path):
        commit = get_commit_info()
        os.environ['OPENMDAO_NEW_RELEASE'] = '0'
        os.environ['OPENMDAO_DOC_VERSION'] = commit
        os.environ['OPENMDAO_SERVER_PATH'] = remote_host + ':/home/openmdao/webapps/twodocversions/latest'

    # if the remote path doesn't exist, this is a release. Need
    # to use the tag as identifier in docs title. (used in conf.py)
    # and make a new remote dir.
    else:
        os.environ['OPENMDAO_NEW_RELEASE'] = '1'
        os.environ['OPENMDAO_DOC_VERSION'] = tag
        os.environ['OPENMDAO_SERVER_PATH'] = remote_host + ":" + remote_path

    print("NEW RELEASE? " + os.environ['OPENMDAO_NEW_RELEASE'])
    print("DOC VERSION: " + os.environ['OPENMDAO_DOC_VERSION'])
    print("SERVER PATH: " + os.environ['OPENMDAO_SERVER_PATH'])

if __name__ == "__main__":
    automate()

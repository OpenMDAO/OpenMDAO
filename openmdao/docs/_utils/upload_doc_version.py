import subprocess
import pipes
import os


def get_tag_info():
    """
    Return the latest git tag, meaning, highest numerically, as a string, and the associated commit ID.
    """
    # using a pattern to only grab tags that are in version format "X.Y.Z"
    git_versions = subprocess.Popen(['git', 'tag', '-l', '*.*.*'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = git_versions.communicate()

    cmd_out = cmd_out.decode('utf8')
    # take the output of git tag -l *.*.*, and split it from one string into a list.
    version_tags = cmd_out.split()

    if not version_tags:
        raise Exception('No tags found in repository')

    # use sort to put the versions list in order from lowest to highest
    version_tags.sort(key=lambda s: [int(u) for u in s.split('.')])

    # grab the highest tag that this repo knows about
    latest_tag = version_tags[-1]

    cmd = subprocess.Popen(['git', 'rev-list', '-1', latest_tag, '-s'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()

    cmd_out = cmd_out.decode('utf8')
    commit_id = cmd_out.strip()

    return latest_tag, commit_id


def get_commit_info():
    """
    Return the commit number of the most recent git commit as a string.
    """
    git_commit = subprocess.Popen(['git', 'show', '--pretty=oneline', '-s'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = git_commit.communicate()

    cmd_out = cmd_out.decode('utf8')
    commit_id = cmd_out.split()[0]

    return commit_id


def get_doc_version():
    """
    Returns either a git commit ID, or a X.Y.Z release number,
    and an indicator if this is a release or not
    """
    release_tag, release_commit = get_tag_info()

    current_commit = get_commit_info()

    if current_commit == release_commit:
        return release_tag, 1
    else:
        return current_commit, 0


def upload_doc_version():
    """
    Perform operations, then upload properly-named docs to WebFaction
    """
    name, rel = get_doc_version()
    destination = "openmdao@web543.webfaction.com:/home/openmdao/webapps/twodocversions/"

    # if release, send to version-numbered dir
    if rel:
        destination += name
    # if not release, it's a "daily build," send to latest
    else:
        destination += "latest"

    # execute the rsync to upload docs
    cmd = "rsync -r --delete-after -v _build/html/* " + destination
    status = subprocess.call(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, shell=True)

    if status == 0:
        return True
    else:
        raise Exception('Doc transfer failed.')


if __name__ == "__main__":
    upload_doc_version()

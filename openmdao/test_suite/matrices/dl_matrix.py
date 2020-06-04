"""
Download a Matlab matrix file from sparse.tamu.edu and save it locally.
"""

import sys
import scipy.io
import urllib.request

def download_matfile(group, name, outfile="matrix.out"):
    """
    Downloads a matrix file (matlab format) from sparse.tamu.edu and returns the matrix.
    """
    with open(outfile, "wb") as f:
        url = 'https://sparse.tamu.edu/mat/%s/%s.mat' % (group, name)
        print("Downloading", url)
        f.write(urllib.request.urlopen(url).read())

    dct = scipy.io.loadmat(outfile)
    return dct



if __name__ == '__main__':
    mat = download_matfile(sys.argv[1], sys.argv[2])
    print(mat['Problem'][0][0][0])
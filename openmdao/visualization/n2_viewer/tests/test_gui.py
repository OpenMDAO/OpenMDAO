"""Test N2 GUI with multiple models using Pyppeteer."""
import sys
import unittest
import os
import urllib.request

# Platform detection adapted from Pyppeteer, used for Pyppeteer's
# download of a new Chromium revision than the one that breaks
# on new JavaScript.
def current_platform() -> str:
    """Get current platform name by short string."""
    if sys.platform.startswith('linux'):
        return 'Linux_x64'
    elif sys.platform.startswith('darwin'):
        return 'Mac'
    elif (sys.platform.startswith('win') or
          sys.platform.startswith('msys') or
          sys.platform.startswith('cyg')):
        if sys.maxsize > 2 ** 31 - 1:
            return 'Win64_x64'
        return 'Win'
    return None

def select_chromium_revision(platform):
    """Grab the current Chromium revision for the specified platform for Pyppeteer."""
    if platform is not None:
        url = 'https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o/' + \
            platform + '%2FLAST_CHANGE?alt=media'
    
        webf = urllib.request.urlopen(url)
        chromium_revision = bytearray(next(webf)).decode()

        print(f'If Pyppeteer cannot find Chromium, install rev. {chromium_revision}')
        os.environ['PYPPETEER_CHROMIUM_REVISION'] = chromium_revision
        download_chromium()

# Make sure Python version is at least 3.6, otherwise do nothing.
if sys.version_info > (3, 5):
    from pyppeteer.chromium_downloader import check_chromium, download_chromium
    from appdirs import AppDirs
    import shutil

    if 'PYPPETEER_CHROMIUM_REINSTALL' in os.environ:
        pyppeteer_chromium_dir = AppDirs('pyppeteer').user_data_dir + '/local-chromium'
        if os.path.isdir(pyppeteer_chromium_dir):
            print("Deleting previous pyppeteer Chromium installation.")
            try:
                shutil.rmtree(pyppeteer_chromium_dir)
            except:
                print(f'Deleting old {pyppeteer_chromium_dir} failed.')
        else:
            print(f'{pyppeteer_chromium_dir} does not exist.')

    if check_chromium():
        print("Chromium already installed, pyppeteer will not download.")
    else:
        select_chromium_revision(current_platform())

    from n2_gui_test import n2_gui_test_case

    if __name__ == "__main__":
        unittest.main()

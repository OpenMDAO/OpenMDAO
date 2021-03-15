import configparser
import os
import pathlib
from openmdao.warnings import get_warning_defaults, filter_warnings


class UserPreferences(object):

    def __init__(self):
        self._cfg = configparser.ConfigParser()

        self._path = pathlib.Path(pathlib.Path.home(), '.openmdao.cfg')

        if os.path.exists(self._path):
            self.load()
        else:
            self.reset_defaults()

        self.apply()

    def reset_defaults(self):
        self._cfg = configparser.ConfigParser()

        # The user check preferences

        # The user warning preferences
        self._cfg['warnings'] = get_warning_defaults()

    def apply(self):
        filter_warnings(reset_to_defaults=True, **self._cfg['warnings'])

    def load(self):
        with open(self._path) as f:
            self._cfg.read_file(f)
        print(f'Loaded user preferences from {self._path}')

    def save(self, defaults=False):
        if defaults:
            self.reset_defaults()

        with open(self._path, 'w') as f:
            self._cfg.write(f)

    def __getitem__(self, item):
        return self._cfg[item]


prefs = UserPreferences()


if __name__ == '__main__':
    pass



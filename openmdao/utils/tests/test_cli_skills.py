import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from openmdao.utils.cli_skills import (
    OPENMDAO_PATH_PLACEHOLDER,
    OPENMDAO_DOCS_PLACEHOLDER,
    OPENMDAO_EXAMPLES_PLACEHOLDER,
    _TOOL_SECTION_START,
    _TOOL_SECTION_END,
    _SKILL_PREFIX,
    _REPO_ROOT
)

class TestCmdlineSkills(unittest.TestCase):
    def _run_command(self, cmd):
        cp = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)  # nosec: trusted input
        if (cp.returncode != 0):
            raise RuntimeError(f"Failed to run command {cmd}: ({cp.stderr}).")

    def claude_skills_install_test(self, global_install):

        if global_install:
            self._run_command([sys.executable, '-m', 'openmdao.utils.om', 'skills', 'install', '--global', 'claude'])
            claude_dir = Path.home() / '.claude'
        else:
            self._run_command([sys.executable, '-m', 'openmdao.utils.om', 'skills', 'install', 'claude'])
            claude_dir = _REPO_ROOT / '.claude'

        # 1. CLAUDE.md has been updated with the managed section markers.
        claude_md = claude_dir / 'CLAUDE.md'
        self.assertTrue(claude_md.exists(), 'CLAUDE.md does not exist in .claude directory')
        content = claude_md.read_text(encoding='utf-8')
        self.assertIn(_TOOL_SECTION_START, content)
        self.assertIn(_TOOL_SECTION_END, content)

        # 2. At least one openmdao-builtin-* skill directory was installed.
        skills_dir = claude_dir / 'skills'
        self.assertTrue(skills_dir.exists(), '.claude/skills directory was not created')
        installed = [d for d in skills_dir.iterdir()
                     if d.is_dir() and d.name.startswith(_SKILL_PREFIX)]
        self.assertTrue(installed, 'No OpenMDAO skill directories found in .claude/skills')

        # 3. No placeholder strings remain in any installed markdown file.
        placeholders = [OPENMDAO_PATH_PLACEHOLDER, OPENMDAO_DOCS_PLACEHOLDER,
                        OPENMDAO_EXAMPLES_PLACEHOLDER]
        for md_file in skills_dir.rglob('*.md'):
            text = md_file.read_text(encoding='utf-8')
            for placeholder in placeholders:
                self.assertNotIn(
                    placeholder, text,
                    f"Placeholder '{placeholder}' was not replaced in {md_file}",
                )

    def test_claude_skills_local_install(self):
        self.claude_skills_install_test(False)

    def test_claude_skills_global_install(self):
        self.claude_skills_install_test(True)

    def claude_skills_uninstall_test(self, global_install):
        # install it first so when we uninstall it, it is actually there to uninstall
        if global_install:
            self._run_command([sys.executable, '-m', 'openmdao.utils.om', 'skills', 'install', '--global', 'claude'])
            self._run_command([sys.executable, '-m', 'openmdao.utils.om', 'skills', 'uninstall', '--global', 'claude'])
            claude_dir = Path.home() / '.claude'
        else:
            self._run_command([sys.executable, '-m', 'openmdao.utils.om', 'skills', 'install', 'claude'])
            self._run_command([sys.executable, '-m', 'openmdao.utils.om', 'skills', 'uninstall', 'claude'])
            claude_dir = _REPO_ROOT / '.claude'

        # 1. CLAUDE.md has been edited to remove the content in the managed section markers.
        claude_md = claude_dir / 'CLAUDE.md'
        if claude_md.exists():
            content = claude_md.read_text(encoding='utf-8')
            self.assertNotIn(_TOOL_SECTION_START, content)
            self.assertNotIn(_TOOL_SECTION_END, content)

        # 2. All the openmdao-builtin-* skill directories were removed.
        skills_dir = claude_dir / 'skills'
        if skills_dir.exists():
            remaining = [d for d in skills_dir.iterdir()
                         if d.is_dir() and d.name.startswith(_SKILL_PREFIX)]
            self.assertFalse(remaining, f'OpenMDAO skill directories were not removed: {remaining}')

    def test_claude_skills_local_uninstall(self):
        self.claude_skills_uninstall_test(False)

    def test_claude_skills_global_uninstall(self):
        self.claude_skills_uninstall_test(True)



if __name__ == "__main__":
    unittest.main()

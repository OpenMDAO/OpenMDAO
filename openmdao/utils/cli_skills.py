"""
OpenMDAO Agent Skills installer.

Installs OpenMDAO Claude (and, in future, other-platform) agent skills that
ship *inside* the installed package, so the skills always match the installed
OpenMDAO version. A re-run of the installer after `pip install -U openmdao`
re-syncs the skills.

Design notes
------------
* Skills are bundled in ``openmdao/skills/`` and copied into a tool's skills
  directory on install.
* Skill files may contain ``{{OPENMDAO_PATH}}``, ``{{OPENMDAO_DOCS}}`` and
  ``{{OPENMDAO_EXAMPLES}}`` placeholders, which are rewritten to absolute paths
  on the user's machine at install time.
* ``CLAUDE.md`` is managed via a marker-bracketed section so user-authored
  content is preserved across re-installs.
* Each supported AI tool is described by a ``Tool`` object with a detection
  function and an install strategy, making it easy to add new platforms later.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# This file lives at openmdao/utils/cli_skills.py
_PACKAGE_DIR = Path(__file__).resolve().parent.parent   # .../openmdao/
_REPO_ROOT = _PACKAGE_DIR.parent                        # repo root (for Editable/source checkout)

# Editable/source checkout has build files at the repo root; a wheel install
# does not. Used only for an informational message at the end of install.
_IS_EDITABLE = (_REPO_ROOT / "pyproject.toml").exists() or (_REPO_ROOT / "setup.py").exists()

# Placeholders rewritten to absolute paths at install time.
OPENMDAO_PATH_PLACEHOLDER = "{{OPENMDAO_PATH}}"
OPENMDAO_DOCS_PLACEHOLDER = "{{OPENMDAO_DOCS}}"
OPENMDAO_EXAMPLES_PLACEHOLDER = "{{OPENMDAO_EXAMPLES}}"

# TOOL.md managed-section markers.
_TOOL_SECTION_START = (
    "<!-- OPENMDAO_SECTION_START - Auto-managed by 'openmdao install-skills' -->"
)
_TOOL_SECTION_END = "<!-- OPENMDAO_SECTION_END -->"

# Skill directory naming convention (only dirs with this prefix are installed).
_SKILL_PREFIX = "openmdao-builtin-"

def get_package_path() -> Path:
    """Return the path to the bundled package."""
    return _PACKAGE_DIR

def get_docs_path() -> Path:
    """Return the path to the bundled documentation (Jupyter Book source)."""
    # should work for both editable and wheel installs
    return _PACKAGE_DIR / "docs" / "openmdao_book"

def get_examples_path() -> Path:
    """Return the path to runnable example components."""
    # should work for both editable and wheel installs
    return _PACKAGE_DIR / "test_suite" / "test_examples"

def get_skills_source_dir() -> Path:
    """Return the directory containing bundled skill sources."""
    src = _PACKAGE_DIR / "skills"
    if not src.exists():
        raise FileNotFoundError(
            f"OpenMDAO skills not found at {src}. "
            "Reinstall openmdao or run from a source checkout."
        )
    return src


def replace_path_placeholders(content: str) -> str:
    """Replace all path placeholders in *content* with absolute paths."""
    return (
        content
        .replace(OPENMDAO_PATH_PLACEHOLDER, str(get_package_path()))
        .replace(OPENMDAO_DOCS_PLACEHOLDER, str(get_docs_path()))
        .replace(OPENMDAO_EXAMPLES_PLACEHOLDER, str(get_examples_path()))
    )


def _find_skill_dirs(root: Path) -> list[Path]:
    """
    Return top-level skill directories starting with ``_SKILL_PREFIX`` and containing a SKILL.md.
    """
    if not root.exists():
        return []
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir()
        and d.name.startswith(_SKILL_PREFIX)
        and (d / "SKILL.md").exists()
    )

def _find_skill_dir(root: Path, key: str) -> Path | None:
    """
    Return the top-level skill directory named ``{_SKILL_PREFIX}{key}`` containing a SKILL.md.
    """
    if not root.exists():
        return None
    for d in root.iterdir():
        if d.is_dir() and d.name == f"{_SKILL_PREFIX}{key}" and (d / "SKILL.md").exists():
            return d
    return None

# ---------------------------------------------------------------------------
# Tool definitions and functions
# ---------------------------------------------------------------------------
class Tool:
    """Represents one AI coding tool and how to install skills into it."""

    def __init__(self, use_global):
        """
        Initialize a Tool instance.

        Parameters
        ----------
        use_global : bool
            Install skills globally (e.g. ~/.claude) if True, otherwise install locally.
        """
        self.use_global = use_global

    def detected(self) -> bool:
        """
        Return True if this tool (e.g. Claude) is detected on the current system.

        Returns
        -------
        bool
            True if the tool is present, False otherwise.
        """
        raise NotImplementedError(f"detected method has not been implemented for {self.name}")

    def is_installed(self) -> bool:
        """
        Return True if OpenMDAO skills have been installed for this tool.

        Returns
        -------
        bool
            True if the install path exists and contains at least one
            OpenMDAO built-in skill directory.
        """
        return self.install_path.exists() and any(
            d for d in self.install_path.iterdir()
            if d.is_dir() and d.name.startswith(_SKILL_PREFIX)
        )

    def install(self, skill_dir: Path) -> None:
        """
        Install skills into this tool's install path.

        Parameters
        ----------
        skill_dir : Path
            Source directory containing the skill subdirectory to install.

        Raises
        ------
        RuntimeError
            If the managed section in the main config file cannot be written.
        OSError
            If any file operation fails.
        """
        self.install_path.mkdir(parents=True, exist_ok=True)
        target = self.install_path / skill_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            skill_dir,
            target,
            ignore=shutil.ignore_patterns(".DS_Store", ".gitkeep", "__pycache__"),
        )

        for md in target.rglob("*.md"):
            text = md.read_text(encoding="utf-8")
            new_text = replace_path_placeholders(text)
            if new_text != text:
                md.write_text(new_text, encoding="utf-8")

        self.install_main_file_md()

    def install_main_file_md(self) -> None:
        """
        Install or update OpenMDAO-managed section in the main file for the tool, e.g. CLAUDE.md.

        The OpenMDAO content is wrapped in HTML-comment markers so the section can
        be replaced in place on re-install without disturbing user content.

        Raises
        ------
        RuntimeError
            If no skill directory or template file is found, or if the managed
            section markers are in an invalid order.
        OSError
            If the file cannot be read or written.
        """
        skill_dir = _find_skill_dir(get_skills_source_dir(), self.key)
        if skill_dir is None:
            raise RuntimeError(f"No skill directory found for key '{self.key}'")
        source = skill_dir / self.main_filename
        if not source.exists():
            raise RuntimeError(f"No {self.main_filename} template found at {source}")

        body = replace_path_placeholders(source.read_text(encoding="utf-8"))
        wrapped = f"\n\n{_TOOL_SECTION_START}\n{body}\n{_TOOL_SECTION_END}\n"

        target = self.install_path.parent / self.main_filename
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if _TOOL_SECTION_START in existing and _TOOL_SECTION_END in existing:
                start = existing.find(_TOOL_SECTION_START)
                end = existing.find(_TOOL_SECTION_END) + len(_TOOL_SECTION_END)
                if end <= start:
                    raise RuntimeError(
                        f"Malformed OpenMDAO section in {self.main_filename}: "
                        "END marker appears before START marker"
                    )
                new_content = existing[:start] + wrapped.lstrip("\n") + existing[end:]
                target.write_text(new_content, encoding="utf-8")
            else:
                # File exists but has no managed section yet — append one.
                new_content = existing.rstrip("\n") + wrapped
                target.write_text(new_content, encoding="utf-8")
        else:
            # Create a fresh mainfile for the tool.
            header = '# OpenMDAO Framework Reference\n'
            target.write_text(header + wrapped.lstrip("\n"), encoding="utf-8")

    def uninstall_main_file_md(self) -> None:
        """
        Remove the OpenMDAO-managed section from the specified main file, if present.

        Leaves any user-authored content and the file itself intact.

        Raises
        ------
        RuntimeError
            If the managed section markers are in an invalid order.
        OSError
            If the file cannot be read or written.
        """
        target = self.install_path.parent / self.main_filename
        if not target.exists():
            return

        existing = target.read_text(encoding="utf-8")
        if _TOOL_SECTION_START not in existing or _TOOL_SECTION_END not in existing:
            return

        start = existing.find(_TOOL_SECTION_START)
        end = existing.find(_TOOL_SECTION_END) + len(_TOOL_SECTION_END)
        if end <= start:
            raise RuntimeError(
                f"Malformed OpenMDAO section in {self.main_filename}: "
                "END marker appears before START marker"
            )
        new_content = (existing[:start].rstrip("\n") + "\n"
                    + existing[end:].lstrip("\n")).strip()
        new_content = (new_content + "\n") if new_content else ""
        target.write_text(new_content, encoding="utf-8")

    def uninstall(self) -> None:
        """
        Remove OpenMDAO skills previously installed for this tool.

        Only subdirectories whose names begin with the OpenMDAO skill prefix are
        removed, leaving any unrelated skills the user keeps in the same directory
        untouched.

        Raises
        ------
        RuntimeError
            If the managed section in the main config file is malformed.
        OSError
            If any file operation fails.
        """
        self.uninstall_main_file_md()

        if not self.install_path.exists():
            return
        # don't remove unrelated skills the user may keep in the same dir.
        for d in self.install_path.iterdir():
            if d.is_dir() and d.name.startswith(_SKILL_PREFIX):
                shutil.rmtree(d)

class ClaudeTool(Tool):
    """Represents the Claude Code tool and how to install skills into it."""

    def __init__(self, use_global: bool = False):
        super().__init__(use_global)
        self.key = 'claude'
        self.name = 'Claude Code'
        self.main_filename = 'CLAUDE.md'
        if use_global:
            self.install_path = Path.home() / '.claude' / 'skills'
        else:
            self.install_path = Path.cwd() / '.claude' / 'skills'

    def detected(self) -> bool:
        """
        Return True if this tool is detected on the current system.

        Returns
        -------
        bool
            True if the tool is present, False otherwise.
        """
        return bool(shutil.which('claude'))


def _make_tools(use_global: bool) -> dict[str, Tool]:
    """
    Build the registry of supported AI tools.

    Parameters
    ----------
    use_global : bool
        If True, install paths resolve to the user's home directory instead of
        the current working directory.

    Returns
    -------
    dict of str to Tool
        Mapping of tool key to Tool instance for each supported AI coding tool.
    """
    tools: list[Tool] = [
        ClaudeTool(use_global),
    ]
    return {t.key: t for t in tools}

# ---------------------------------------------------------------------------
# Command execution and parser setup functions
# ---------------------------------------------------------------------------
def cmd_skills_install(args, user_args) -> int:
    """
    Execute the ``install-skills`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

        Relevant attributes: ``use_global`` (bool) and ``tool`` (str) for the tool key to install.
    user_args : list of str
        Extra positional arguments passed through by the OpenMDAO CLI harness
        (unused; reserved for future extension).

    Returns
    -------
    int
        Exit code: 0 on success, 1 on failure.
    """
    tools = _make_tools(args.use_global)
    if not tools:
        print("No OpenMDAO skills are currently supported for installation.", file=sys.stderr)
        return 1

    tool_key = args.tool
    tool = tools.get(tool_key)
    if not tool:
        print(f"Tool '{tool_key}' is not supported by OpenMDAO skills installer.", file=sys.stderr)
        return 1

    skill_dir = _find_skill_dir(get_skills_source_dir(), args.tool)
    if not skill_dir:
        print(f"No OpenMDAO skills found in {get_skills_source_dir()}", file=sys.stderr)
        return 1

    scope = 'global (home directory)' if args.use_global else 'project (current directory)'
    print(f"Installing OpenMDAO skill — scope: {scope}\n")

    print(f"Installing {tool.name} into ({tool.install_path})")
    try:
        tool.install(skill_dir)
    except (RuntimeError, OSError) as e:
        print(f"Failed to install OpenMDAO skills for {tool.name}: {e}", file=sys.stderr)
        return 1

    # Report configured paths so users can verify the placeholder substitution.
    print("Paths configured:")
    print(f"  Package:  {_PACKAGE_DIR}")
    print(f"  Docs:     {get_docs_path()}")
    print(f"  Examples: {get_examples_path()}")
    if not get_docs_path().exists():
        print(
            "\n  WARNING: the docs path above does not exist on disk. If you "
            "installed OpenMDAO from a wheel, the docs may not be packaged; "
            "skills referencing {{OPENMDAO_DOCS}} will point at a missing path.",
            file=sys.stderr,
        )
    if _IS_EDITABLE:
        print("\n  (Detected editable/source install)")

    return 0

def cmd_skills_list(args, user_args) -> int:
    """
    Execute the ``list-skills`` subcommand.

    Prints a table of all registered tools, their project and global install
    paths, and whether skills are currently installed at each location.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (currently unused for this subcommand).
    user_args : list of str
        Extra positional arguments passed through by the OpenMDAO CLI harness
        (unused; reserved for future extension).

    Returns
    -------
    int
        Exit code: always 0.
    """
    skill_dirs = _find_skill_dirs(get_skills_source_dir())
    project_tools = _make_tools(False)
    global_tools = _make_tools(True)

    print(
        f"Available OpenMDAO skills ({len(skill_dirs)}): "
        f"{', '.join(d.name for d in skill_dirs) or '(none)'}\n"
    )

    # Build rows: (name, project_path, global_path, status)
    rows: list[tuple[str, str, str, str]] = []
    for key, pt in project_tools.items():
        gt = global_tools[key]
        proj_path = str(pt.install_path)
        glob_path = str(gt.install_path)
        where: list[str] = []
        if pt.is_installed():
            where.append("project")
        if gt.is_installed():
            where.append("global")
        if where:
            status = "installed (" + ", ".join(where) + ")"
        elif pt.detected():
            status = "detected"
        else:
            status = "-"
        rows.append((pt.name, proj_path, glob_path, status))

    if not rows:
        print("No tools registered.")
        return 0

    # Column widths.
    name_w = max(len("Tool"), max(len(r[0]) for r in rows))
    proj_w = max(len("Project path"), max(len(r[1]) for r in rows))
    glob_w = max(len("Global path"), max(len(r[2]) for r in rows))
    stat_w = max(len("Status"), max(len(r[3]) for r in rows))

    print(
        f"  {'Tool':<{name_w}}  {'Project path':<{proj_w}}  "
        f"{'Global path':<{glob_w}}  {'Status':<{stat_w}}"
    )
    print(f"  {'-' * name_w}  {'-' * proj_w}  {'-' * glob_w}  {'-' * stat_w}")
    for name, proj, glob, status in rows:
        print(f"  {name:<{name_w}}  {proj:<{proj_w}}  {glob:<{glob_w}}  {status:<{stat_w}}")
    print()
    return 0

def cmd_skills_uninstall(args, user_args) -> int:
    """
    Execute the ``uninstall-skills`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

        Relevant attributes: ``use_global`` (bool) and ``tool`` (str) for the tool key to uninstall.
    user_args : list of str
        Extra positional arguments passed through by the OpenMDAO CLI harness
        (unused; reserved for future extension).

    Returns
    -------
    int
        Exit code: 0 on success, 1 if the specified tool is not supported.
    """
    tools = _make_tools(args.use_global)

    tool_key = args.tool
    tool = tools.get(tool_key)
    if not tool:
        print(f"Tool '{tool_key}' is not supported by OpenMDAO skills installer.", file=sys.stderr)
        return 1

    try:
        tool.uninstall()
    except (RuntimeError, OSError) as e:
        print(f"Failed to uninstall OpenMDAO skills for {tool.name}: {e}", file=sys.stderr)
        return 1

    print("\nUninstall complete.")
    return 0

def _setup_skills_install(parser):
    """
    Configure the argument parser for the ``install-skills`` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The subcommand parser to configure.
    """
    parser.add_argument('tool', type=str, help='AI coding tool to install skills for (e.g. claude)')
    parser.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help="Use global (home-directory) paths instead of the current project",
    )

def _setup_skills_uninstall(parser):
    """
    Configure the argument parser for the ``uninstall-skills`` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The subcommand parser to configure.
    """
    parser.add_argument('tool', type=str,
                        help='AI coding tool to uninstall skills for (e.g. claude)')
    parser.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help="Use global (home-directory) paths instead of the current project",
    )

def _setup_skills_list(parser):
    """
    Configure the argument parser for the ``list-skills`` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The subcommand parser to configure.
    """
    parser.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help="Use global (home-directory) paths instead of the current project",
    )


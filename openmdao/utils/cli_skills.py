"""
OpenMDAO Agent Skills installer.

Installs OpenMDAO Claude (and, in future, other-platform) agent skills that
ship *inside* the installed package, so the skills always match the installed
OpenMDAO version. A re-run of the installer after `pip install -U openmdao`
re-syncs the skills.

Design notes
------------
* Skills are bundled at ``openmdao/skills/`` and copied into a tool's skills
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

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# This file lives at openmdao/utils/cli_skills.py
_PACKAGE_DIR = Path(__file__).resolve().parent.parent   # .../openmdao/
_REPO_ROOT = _PACKAGE_DIR.parent                        # repo root (editable only)

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
_SKILL_PREFIX = "openmdao-builtin"

def get_package_path() -> Path:
    """Return the path to the bundled package."""
    return _PACKAGE_DIR

def get_docs_path() -> Path:
    """Return the path to the bundled documentation (Jupyter Book source)."""
    return _PACKAGE_DIR / "docs" / "openmdao_book"


def get_examples_path() -> Path:
    """Return the path to runnable example components."""
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
    """Return top-level skill directories (starting with ``_SKILL_PREFIX`` and containing a SKILL.md file)."""
    if not root.exists():
        return []
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir()
        and d.name.startswith(_SKILL_PREFIX)
        and (d / "SKILL.md").exists()
    )

def _find_skill_dir(root: Path, key) -> list[Path]:
    """Return top-level skill directory (named ``_SKILL_PREFIX}{key}`` with a SKILL.md)."""
    if not root.exists():
        return []
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir()
        and d.name == f"{_SKILL_PREFIX}{key}"
        and (d / "SKILL.md").exists()
    )

# ---------------------------------------------------------------------------
# Tool definitions and functions
# ---------------------------------------------------------------------------
class Tool:
    """Represents one AI coding tool and how to install skills into it."""

    def __init__(self, key, name, install_path, detect_fn, install_fn):
        """
        Initialize a Tool instance.

        Parameters
        ----------
        key : str
            Short identifier for the tool, used as a lookup key.
        name : str
            Human-readable display name for the tool.
        install_path : Path
            Directory where skills are installed for this tool.
        detect_fn : callable
            Zero-argument callable that returns a truthy value when the tool is
            detected on the current system.
        install_fn : callable
            Callable with signature ``(skill_dirs, install_path)`` that copies
            skills into *install_path* and returns the number of skills installed.
        """
        self.key = key
        self.name = name
        self.install_path = install_path
        self.detect_fn = detect_fn
        self.install_fn = install_fn

    def detected(self) -> bool:
        """
        Return True if this tool (e.g. Claude, Codex) is detected on the current system.

        Returns
        -------
        bool
            True if the tool is present, False otherwise.
        """
        return bool(self.detect_fn())

    def is_installed(self) -> bool:
        """
        Return True if OpenMDAO skills have been installed for this tool.

        Returns
        -------
        bool
            True if the install path exists and contains at least one OpenMDAO built-in skill directory.
        """
        return self.install_path.exists() and any(
            d for d in self.install_path.iterdir() if d.is_dir() and d.name.startswith(_SKILL_PREFIX)
        )

    def install(self, skill_dirs: list[Path]) -> int:
        """
        Install skills into this tool's install path.

        Parameters
        ----------
        skill_dirs : list of Path
            Source directories containing skill subdirectories to install.

        Returns
        -------
        int
            Number of skills installed.
        """
        return self.install_fn(skill_dirs, self.install_path)

    def uninstall(self) -> int:
        """
        Remove OpenMDAO skills previously installed for this tool.

        Only subdirectories whose names begin with the OpenMDAO skill prefix are
        removed, leaving any unrelated skills the user keeps in the same directory
        untouched.

        Returns
        -------
        int
            Number of skill directories removed.
        """
        if not self.install_path.exists():
            return 0
        # Only remove the OpenMDAO skill subdirectories we installed, so we
        # don't remove unrelated skills the user may keep in the same dir.
        removed = 0
        for d in self.install_path.iterdir():
            if d.is_dir() and d.name.startswith(_SKILL_PREFIX):
                shutil.rmtree(d)
                removed += 1
        return removed

def _make_tools(*, use_global: bool = False) -> dict[str, Tool]:
    """
    Build the registry of supported AI tools.

    Parameters
    ----------
    use_global : bool, optional
        If True, install paths resolve to the user's home directory instead of
        the current working directory.

    Returns
    -------
    dict of str to Tool
        Mapping of tool key to Tool instance for each supported AI coding tool.
    """
    home = Path.home()
    cwd = Path.cwd()

    # helper function to choose between project and global install paths
    def _which_install_path(project: Path, global_: Path) -> Path:
        return global_ if use_global else project

    tools: list[Tool] = [
        Tool(
            key="claude",
            name="Claude Code",
            install_path=_which_install_path(cwd / ".claude" / "skills", home / ".claude" / "skills"),
            detect_fn=lambda: bool(shutil.which("claude")),
            install_fn=_install_dirs,
        ),
        Tool(
            key="codex",
            name="OpenAI Codex",
            install_path=_which_install_path(cwd / ".agents" / "skills", home / ".agents" / "skills"),
            detect_fn=lambda: bool(shutil.which("codex")),
            install_fn=_install_dirs,
        ),
    ]
    return {t.key: t for t in tools}

# ---- Install strategies ----------------------------------------------------
def _install_dirs(skill_dirs: list[Path], dest: Path) -> int:
    """
    Copy skill directory trees into *dest*, rewriting path placeholders in place.

    This is the install strategy for Claude Code, which expects one subdirectory
    per skill, each containing a ``SKILL.md``. Any existing copy of a skill is
    removed before copying so the install is always a clean replacement. After
    copying, every ``*.md`` file in the tree is scanned and any path placeholders
    (``{{OPENMDAO_PATH}}``, ``{{OPENMDAO_DOCS}}``, ``{{OPENMDAO_EXAMPLES}}``)
    are rewritten to absolute paths on the current machine.

    Parameters
    ----------
    skill_dirs : list of Path
        Source skill directories to install. Each must contain a ``SKILL.md``.
    dest : Path
        Directory into which the skill subdirectories are copied. Created if it
        does not exist.

    Returns
    -------
    int
        Number of skill directories installed.
    """
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for skill_dir in skill_dirs:
        target = dest / skill_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            skill_dir,
            target,
            ignore=shutil.ignore_patterns(".DS_Store", ".gitkeep", "__pycache__"),
        )

        # Rewrite placeholders in every markdown file in the copied tree.
        for md in target.rglob("*.md"):
            text = md.read_text(encoding="utf-8")
            new_text = replace_path_placeholders(text)
            if new_text != text:
                md.write_text(new_text, encoding="utf-8")

        count += 1
    return count


# ---------------------------------------------------------------------------
# CLAUDE.md management
# ---------------------------------------------------------------------------
def install_claude_md(global_install: bool) -> tuple[bool, str]:
    """
    Install or update the OpenMDAO-managed section in CLAUDE.md.

    The OpenMDAO content is wrapped in HTML-comment markers so the section can
    be replaced in place on re-install without disturbing user content.

    Parameters
    ----------
    global_install : bool
        If True, target ``~/.claude/CLAUDE.md``; otherwise target
        ``.claude/CLAUDE.md`` in the current working directory.

    Returns
    -------
    tuple of (bool, str)
        A (success, message) pair. *success* is False only if an unexpected
        error prevented the write; informational outcomes still return True.
    """
    try:
        source = get_skills_source_dir() / "CLAUDE.md"
        if not source.exists():
            return False, f"No CLAUDE.md template found at {source}"

        body = replace_path_placeholders(source.read_text(encoding="utf-8"))
        wrapped = f"\n\n{_TOOL_SECTION_START}\n{body}\n{_TOOL_SECTION_END}\n"

        base = (Path.home() if global_install else Path.cwd()) / ".claude"
        target = base / "CLAUDE.md"
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if _TOOL_SECTION_START in existing and _TOOL_SECTION_END in existing:
                start = existing.find(_TOOL_SECTION_START)
                end = existing.find(_TOOL_SECTION_END) + len(_TOOL_SECTION_END)
                new_content = existing[:start] + wrapped.lstrip("\n") + existing[end:]
                target.write_text(new_content, encoding="utf-8")
                return True, "Updated existing OpenMDAO section in CLAUDE.md"
            else:
                # File exists but has no managed section yet — append one.
                new_content = existing.rstrip("\n") + wrapped
                target.write_text(new_content, encoding="utf-8")
                return True, "Appended OpenMDAO section to existing CLAUDE.md"
        else:
            # Create a fresh CLAUDE.md.
            header = "# OpenMDAO Framework Reference\n"
            target.write_text(header + wrapped.lstrip("\n"), encoding="utf-8")
            return True, "Created new CLAUDE.md with OpenMDAO content"

    except OSError as e:
        return False, f"Error installing CLAUDE.md: {e}"


def uninstall_claude_md(global_install: bool) -> tuple[bool, str]:
    """
    Remove the OpenMDAO-managed section from CLAUDE.md, if present.

    Leaves any user-authored content and the file itself intact.

    Parameters
    ----------
    global_install : bool
        If True, target ``~/.claude/CLAUDE.md``; otherwise target
        ``.claude/CLAUDE.md`` in the current working directory.

    Returns
    -------
    tuple of (bool, str)
        A (success, message) pair. *success* is False only if an unexpected
        error prevented the write.
    """
    try:
        base = (Path.home() if global_install else Path.cwd()) / ".claude"
        target = base / "CLAUDE.md"
        if not target.exists():
            return True, "No CLAUDE.md to clean up"

        existing = target.read_text(encoding="utf-8")
        if _TOOL_SECTION_START not in existing or _TOOL_SECTION_END not in existing:
            return True, "No OpenMDAO section found in CLAUDE.md"

        start = existing.find(_TOOL_SECTION_START)
        end = existing.find(_TOOL_SECTION_END) + len(_TOOL_SECTION_END)
        new_content = (existing[:start].rstrip("\n") + "\n" + existing[end:].lstrip("\n")).strip()
        new_content = (new_content + "\n") if new_content else ""
        target.write_text(new_content, encoding="utf-8")
        return True, "Removed OpenMDAO section from CLAUDE.md"

    except OSError as e:
        return False, f"Error cleaning CLAUDE.md: {e}"


def uninstall_codex_md(global_install: bool) -> tuple[bool, str]:
    """
    Remove the OpenMDAO-managed section from CODEX.md, if present.

    Leaves any user-authored content and the file itself intact.

    Parameters
    ----------
    global_install : bool
        If True, target ``~/.codex/CODEX.md``; otherwise target
        ``.codex/CODEX.md`` in the current working directory.

    Returns
    -------
    tuple of (bool, str)
        A (success, message) pair. *success* is False only if an unexpected
        error prevented the write.
    """
    try:
        base = (Path.home() if global_install else Path.cwd()) / ".codex"
        target = base / "CODEX.md"
        if not target.exists():
            return True, "No CODEX.md to clean up"

        existing = target.read_text(encoding="utf-8")
        if _TOOL_SECTION_START not in existing or _TOOL_SECTION_END not in existing:
            return True, "No OpenMDAO section found in CODEX.md"

        start = existing.find(_TOOL_SECTION_START)
        end = existing.find(_TOOL_SECTION_END) + len(_TOOL_SECTION_END)
        new_content = (existing[:start].rstrip("\n") + "\n" + existing[end:].lstrip("\n")).strip()
        new_content = (new_content + "\n") if new_content else ""
        target.write_text(new_content, encoding="utf-8")
        return True, "Removed OpenMDAO section from CODEX.md"

    except OSError as e:
        return False, f"Error cleaning CODEX.md: {e}"


# ---------------------------------------------------------------------------
# Command execution and parser setup functions
# ---------------------------------------------------------------------------
def cmd_skills_install(args, user_args) -> int:
    """
    Execute the ``install-skills`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Relevant attributes: ``use_global`` (bool) and
        one bool attribute per registered tool key (e.g. ``claude_code``).
    user_args : list of str
        Extra positional arguments passed through by the OpenMDAO CLI harness
        (unused; reserved for future extension).

    Returns
    -------
    int
        Exit code: 0 on success, 1 on failure.
    """
    tools = _make_tools(use_global=args.use_global)

    # TODO this is all skills that are available in the repo that can be installed
    # skill_dirs = _find_skill_dirs(get_skills_source_dir())
    skill_dirs = _find_skill_dir(get_skills_source_dir(), args.tool)
    if not skill_dirs:
        print(f"No OpenMDAO skills found in {get_skills_source_dir()}", file=sys.stderr)
        return 1

    tool_name = args.tool
    tool = tools.get(tool_name)

    scope = "global (home directory)" if args.use_global else "project (current directory)"
    print(f"Installing {len(skill_dirs)} OpenMDAO skill(s) — scope: {scope}\n")

    print(f"Installing into {tool_name} ({tool.install_path})")
    count = tool.install(skill_dirs)
    print(f"  -> {count} skill(s) installed\n")

    # Manage CLAUDE.md (only meaningful for the Claude tool, but harmless to run
    # whenever Claude Code is among the requested tools).

    # TODO do I need this separate code for just Claude?
    if tool.key == "claude":
        print("Updating CLAUDE.md...")
        ok, message = install_claude_md(args.use_global)
        prefix = "  " if ok else "  WARNING: "
        print(f"{prefix}{message}\n", file=sys.stderr if not ok else sys.stdout)
    # TODO uncomment this and make it work for codex
    # if tool.key == "codex":
    #     print("Updating CLAUDE.md...")
    #     ok, message = install_claude_md(args.use_global)
    #     prefix = "  " if ok else "  WARNING: "
    #     print(f"{prefix}{message}\n", file=sys.stderr if not ok else sys.stdout)

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
    project_tools = _make_tools(use_global=False)
    global_tools = _make_tools(use_global=True)

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

def cmd_skills_uninstall(args, user_args: argparse.Namespace) -> int:
    """
    Execute the ``uninstall-skills`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Relevant attributes: ``use_global`` (bool) and
        one bool attribute per registered tool key (e.g. ``claude_code``).
        At least one tool flag must be set; the command prints an error and
        returns 1 if none are provided.

    Returns
    -------
    int
        Exit code: 0 on success, 1 if no tool flags were specified.
    """
    tools = _make_tools(use_global=args.use_global)

    tool_key = args.tool

    tools[tool_key].uninstall()

    if tool_key == "claude":
        ok, message = uninstall_claude_md(args.use_global)
        print(f"  {message}")

    if tool_key == "codex":
        ok, message = uninstall_codex_md(args.use_global)
        print(f"  {message}")

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
    parser.add_argument('tool', type=str, help='AI coding tool to uninstall skills for (e.g. claude)')
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


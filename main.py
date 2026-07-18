"""
MR1 — Entry Point
==================
Run this to start the MR1 multi-agent system.

  python main.py
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_PKG_DIR = _PROJECT_ROOT / "mr1"

_REQUIRED_DIRS = [
    _PKG_DIR / "memory" / "active",
    _PKG_DIR / "memory" / "dumps",
    _PKG_DIR / "memory" / "rag",
    _PKG_DIR / "tasks",
]

_REQUIRED_AGENT_CONFIGS = [
    _PKG_DIR / "agents" / "mr1.yml",
    _PKG_DIR / "agents" / "mrn.yml",
    _PKG_DIR / "agents" / "worker.yml",
    _PKG_DIR / "agents" / "mini" / "mem_dltr.yml",
    _PKG_DIR / "agents" / "mini" / "mem_rtvr.yml",
    _PKG_DIR / "agents" / "mini" / "ctx_pkgr.yml",
    _PKG_DIR / "agents" / "mini" / "com_smrzr.yml",
]

_CONFIG_PATH = _PROJECT_ROOT / "config.yml"


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_claude_installed() -> None:
    """Verify the claude CLI is on PATH."""
    if shutil.which("claude") is None:
        print(
            "ERROR: 'claude' CLI not found on PATH.\n"
            "Install it from: https://claude.ai/code\n"
            "Then re-run: python main.py"
        )
        sys.exit(1)


def _check_claude_authenticated() -> None:
    """
    Run 'claude --version' to confirm the CLI responds.
    A missing auth token causes claude to exit non-zero or print an error;
    we catch that and give a clear message.
    """
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: 'claude --version' timed out. Check your install.")
        sys.exit(1)
    except OSError as e:
        print(f"ERROR: Could not run claude CLI: {e}")
        sys.exit(1)

    if result.returncode != 0:
        print(
            "ERROR: claude CLI returned an error.\n"
            "You may not be authenticated. Run:\n"
            "  claude login\n"
            f"Details: {result.stderr.strip()}"
        )
        sys.exit(1)


def _init_directories() -> None:
    """Create any missing directories in the project layout."""
    for d in _REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def _check_agent_configs() -> None:
    """Ensure all agent YAML definitions are present."""
    missing = [str(p) for p in _REQUIRED_AGENT_CONFIGS if not p.exists()]
    if missing:
        print("ERROR: Missing agent config files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)


def _check_config() -> None:
    """Ensure config.yml exists at the project root."""
    if not _CONFIG_PATH.exists():
        print(
            "WARNING: config.yml not found at project root. "
            "Using default height_limit=4."
        )


def _load_and_validate_configs() -> dict:
    """
    Load all agent YAML configs and return them keyed by agent name.
    Exits if any config is malformed.
    """
    import yaml

    configs = {}
    for path in _REQUIRED_AGENT_CONFIGS:
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            name = cfg["name"]
            configs[name] = cfg
        except Exception as e:
            print(f"ERROR: Failed to load {path}: {e}")
            sys.exit(1)

    return configs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MR1 entry point")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="run the plain-text chat interface",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="run the read-only runtime TUI",
    )
    args = parser.parse_args()

    # Ensure mr1 package is importable from this file's directory.
    sys.path.insert(0, str(_PROJECT_ROOT))

    if args.tui:
        from mr1.tui import main as tui_main

        sys.exit(tui_main())

    print("MR1 — starting up...")

    # 1. Check the claude CLI is present and responding.
    _check_claude_installed()
    _check_claude_authenticated()
    print("  claude CLI OK")

    # 2. Create missing directories.
    _init_directories()
    print("  directories OK")

    # 3. Validate all agent configs are present and loadable.
    _check_agent_configs()
    _check_config()
    configs = _load_and_validate_configs()
    agent_names = ", ".join(sorted(configs))
    print(f"  agent configs OK ({agent_names})")

    print()

    # 4. Hand off to the MR1 orchestrator loop.
    from mr1.mr1 import main as mr1_main
    mr1_main()


if __name__ == "__main__":
    main()

"""Verify agent-engine-runner-shared source has no framework SDK imports.

Uses the ast module to check actual import statements, ignoring comments
and docstrings.
"""

import ast
import pathlib

RUNNER_SHARED_SRC = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "agent_engine_runner_shared"
)

FORBIDDEN_MODULES = [
    "langchain",
    "langgraph",
    "agent_engine_sdk_langgraph",
    "magenta_sdklanggraph",  # open-source-refs:ignore — retired namespace
    "openinference",
]


def _forbidden_modules(node: ast.AST) -> list[str]:
    """Return all forbidden module names from an import node."""
    modules: list[str] = []
    if isinstance(node, ast.ImportFrom) and node.module:
        modules = [node.module]
    elif isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names if alias.name]
    return [m for m in modules if any(m.startswith(p) for p in FORBIDDEN_MODULES)]


def test_no_forbidden_imports():
    violations = []
    for py in sorted(RUNNER_SHARED_SRC.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            for module in _forbidden_modules(node):
                rel = py.relative_to(RUNNER_SHARED_SRC)
                violations.append(f"{rel}:{node.lineno}: {module}")
    assert violations == [], "Forbidden imports in agent-engine-runner-shared:\n" + "\n".join(
        violations
    )

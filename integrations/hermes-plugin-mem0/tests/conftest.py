"""Make the standalone Hermes plugin importable without the Hermes runtime.

The plugin is a Hermes provider: ``__init__.py`` imports ``agent.memory_provider``,
``agent.secret_scope``, ``tools.registry``, ``utils`` and ``hermes_constants`` at module
level. Two separate things need those names present:

1. ``pytest`` treats ``integrations/hermes-plugin-mem0`` as a package and imports its
   ``__init__.py`` in ``Package.setup()`` — *before* any fixture runs. A plain checkout
   has neither the real Hermes packages nor these stubs, so every test errored at setup
   with ``ModuleNotFoundError: No module named 'agent'``.
2. ``test_plugin.py``'s ``plugin`` fixture installs its own richer stubs per test.

So install minimal stand-ins here, and let the fixture override them where it cares.
Nothing is asserted against these: they exist only so the module body can be imported.
Upstream runs this suite inside the Hermes venv, where the real packages win; the
``if not in sys.modules`` guards below keep that case untouched.

``agent`` and ``tools`` need ``__path__`` for dotted leaf imports to resolve.
"""

import json
import sys
import types
from pathlib import Path

_LEAF_STUBS = {
    "agent.memory_provider": {"MemoryProvider": object, "spawn_context_thread": lambda *a, **k: None},
    "agent.secret_scope": {"get_secret": lambda key, default="": default},
    "tools.registry": {"tool_error": lambda message: json.dumps({"error": message})},
    "utils": {
        "read_json_or_empty": lambda path: json.loads(Path(path).read_text()) if Path(path).exists() else {},
        "atomic_json_write": lambda path, value, **kwargs: Path(path).write_text(json.dumps(value)),
        "atomic_write_text": lambda path, value, **kwargs: Path(path).write_text(value),
    },
    "hermes_constants": {"get_hermes_home": lambda: Path(".")},
}

for _package in ("agent", "tools"):
    if _package not in sys.modules:
        _module = types.ModuleType(_package)
        _module.__path__ = []  # marks it a package so dotted leaf imports resolve
        sys.modules[_package] = _module

for _name, _attrs in _LEAF_STUBS.items():
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        _module.__dict__.update(_attrs)
        sys.modules[_name] = _module

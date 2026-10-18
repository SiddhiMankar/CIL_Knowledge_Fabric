"""
Smoke test to verify app.py compiles cleanly and can be inspected without syntax errors.
"""

from pathlib import Path
import py_compile
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_app_compiles_cleanly():
    app_path = PROJECT_ROOT / "app.py"
    assert app_path.exists()
    compiled = py_compile.compile(str(app_path), doraise=True)
    assert compiled is not None

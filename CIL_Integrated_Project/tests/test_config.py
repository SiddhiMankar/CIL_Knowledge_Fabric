"""
Unit tests for config.py path resolution and constants.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def test_paths_exist_or_can_be_created():
    assert config.PROJECT_ROOT.exists()
    assert config.NOTEBOOKS_DIR.exists()
    assert config.ARTIFACTS_DIR.exists()


def test_feature_lists_integrity():
    assert len(config.NUMERICAL_FEATURES) == 5
    assert "Machine failure" in config.EXCLUDE_COLS
    assert "UDI" in config.EXCLUDE_COLS
    assert "Product ID" in config.EXCLUDE_COLS
    assert len(config.FEATURE_COLS_LABEL_ENCODED) == 6
    assert len(config.FEATURE_COLS_ONEHOT) == 7


def test_locate_dataset_contract():
    found, path = config.locate_dataset()
    assert isinstance(found, bool)
    if found:
        assert isinstance(path, Path)
        assert path.exists()
    else:
        assert path is None

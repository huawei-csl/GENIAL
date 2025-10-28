from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.skip(reason="Environment/bootstrap commands are not executed in tests")
def test_setup_docs_present():
    assert Path("docs/setup.md").exists()

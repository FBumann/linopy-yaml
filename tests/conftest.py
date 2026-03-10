"""Shared fixtures for linopy_yaml tests."""

from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def dispatch_yaml() -> Path:
    return EXAMPLES_DIR / "dispatch.yaml"

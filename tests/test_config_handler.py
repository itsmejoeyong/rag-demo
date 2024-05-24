import os
from unittest.mock import patch
from src.config_handler import ConfigHandler

import pytest


@pytest.fixture
def config_handler():
    with patch.dict(os.environ, {"test": "test"}):
        yield ConfigHandler()


def test_get_env(config_handler):
    env = config_handler._get_env("test")
    assert env == "test"

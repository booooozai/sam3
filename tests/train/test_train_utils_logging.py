"""Environment logging must not expose credentials."""

import logging
import os
from unittest.mock import patch

import pytest

from sam3.train.utils.train_utils import _is_sensitive_env_variable, log_env_variables


@pytest.mark.parametrize("name", [
    "OPENAI_API_KEY", "HF_TOKEN", "PASSWORD", "DB_PASSWD",
    "CLIENT_SECRET", "AWS_CREDENTIAL", "AUTH_HEADER", "lowercase_token",
])
def test_sensitive_environment_names_are_detected(name):
    assert _is_sensitive_env_variable(name)


def test_environment_log_redacts_secrets_and_preserves_runtime_settings(caplog):
    environment = {
        "HF_TOKEN": "synthetic-secret-token",
        "DB_PASSWORD": "synthetic-secret-password",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
    }
    with patch.dict(os.environ, environment, clear=True):
        with caplog.at_level(logging.INFO):
            log_env_variables()
    assert "HF_TOKEN=<redacted>" in caplog.text
    assert "DB_PASSWORD=<redacted>" in caplog.text
    assert "synthetic-secret" not in caplog.text
    assert "CUDA_VISIBLE_DEVICES=\n" in caplog.text
    assert "OMP_NUM_THREADS=4" in caplog.text

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from test_tokenizer import FIXTURE_VOCAB_SIZE, _python_corpus
from tokenizer import load_tokenizer
from tokenizer.train import run


def test_run_saves_tokenizer_settings_and_stats(tmp_path):
    corpus = _python_corpus()
    settings = {"vocab_size": FIXTURE_VOCAB_SIZE, "seed": 1234}

    run(iter(corpus), iter(corpus[:10]), out_dir=tmp_path, settings=settings)

    tok = load_tokenizer(tmp_path / "tokenizer.json")
    assert tok.get_vocab_size() == FIXTURE_VOCAB_SIZE

    meta = json.loads((tmp_path / "training_config.json").read_text())
    assert meta["settings"] == settings
    assert meta["train_files"] == len(corpus)
    assert meta["valid_stats"]["files"] == 10
    assert meta["valid_stats"]["chars_per_token"] > 1


def test_run_saves_tokenizer_before_validation_fails(tmp_path):
    corpus = _python_corpus()
    settings = {"vocab_size": FIXTURE_VOCAB_SIZE}

    def broken_valid():
        yield corpus[0]
        raise ConnectionError("stream dropped")

    with pytest.raises(ConnectionError):
        run(iter(corpus), broken_valid(), out_dir=tmp_path, settings=settings)

    assert load_tokenizer(tmp_path / "tokenizer.json").get_vocab_size() == FIXTURE_VOCAB_SIZE
    meta = json.loads((tmp_path / "training_config.json").read_text())
    assert meta["settings"] == settings
    assert meta["train_files"] == len(corpus)
    assert "valid_stats" not in meta


def test_cli_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-m", "tokenizer.train", "--help"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--num-files" in result.stdout

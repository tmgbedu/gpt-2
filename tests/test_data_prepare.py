import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from data import file_sha256, load_tokens
from data.prepare import prepare
from test_tokenizer import FIXTURE_VOCAB_SIZE, _python_corpus
from tokenizer import EOT_ID, train_tokenizer


def _setup(tmp_path):
    tok_path = tmp_path / "tokenizer.json"
    train_tokenizer(_python_corpus(), vocab_size=FIXTURE_VOCAB_SIZE).save(str(tok_path))
    settings = {"seed": 7, "shuffle_buffer": 100, "max_bytes": 100_000, "fim": None,
                "train_revision": "abc", "valid_revision": "def"}
    return tok_path, settings


def test_prepare_writes_both_splits_with_metadata(tmp_path):
    tok_path, settings = _setup(tmp_path)
    corpus = _python_corpus()
    out = tmp_path / "data"

    meta = prepare(iter(corpus), iter(corpus[:20]), tok_path, out,
                   train_tokens=500, val_tokens=100, settings=settings)

    for split in ("train", "val"):
        tokens = load_tokens(out / f"{split}.bin")
        saved = json.loads((out / f"{split}.json").read_text())
        assert saved == meta[split]
        assert saved["token_count"] == len(tokens)
        assert saved["documents"] == int((tokens == EOT_ID).sum())
        assert tokens[-1] == EOT_ID
        assert saved["tokenizer_sha256"] == file_sha256(tok_path)
        assert saved["split"] == split
        for key, value in settings.items():
            assert saved[key] == value
    assert meta["train"]["token_count"] >= 500
    assert meta["val"]["token_count"] >= 100


def test_prepare_skips_files_over_max_bytes(tmp_path):
    tok_path, settings = _setup(tmp_path)
    settings["max_bytes"] = 200
    small, big = "x = 1\n", "y = 2\n" * 100  # 6 and 600 bytes
    meta = prepare(iter([big, small, big, small]), iter([small]), tok_path, tmp_path / "d",
                   train_tokens=10_000, val_tokens=10_000, settings=settings)
    assert meta["train"]["documents"] == 2
    assert meta["train"]["skipped_too_large"] == 2


def test_cli_help_runs_as_installed_module():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-m", "data.prepare", "--help"],
                            capture_output=True, text=True, env=env, cwd=Path(__file__).parent)
    assert result.returncode == 0, result.stderr
    assert "--train-tokens" in result.stdout

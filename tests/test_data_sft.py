import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.sft import (
    ASSISTANT_ID,
    END_ID,
    USER_ID,
    chat_prompt_ids,
    format_example,
    load_sft,
    pack_rows,
    prepare_sft,
    sft_batch,
)
from tokenizer import PAD_ID, load_tokenizer

TOKENIZER = Path(__file__).parents[1] / "artifacts/tokenizer/tokenizer.json"


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(TOKENIZER)


# --- one example: <|user|> instruction <|end|> <|assistant|> response <|end|> ---

def test_format_wraps_the_example_in_chat_tokens(tok):
    ids, mask = format_example(tok, "Add two numbers.", "def add(a, b):\n    return a + b\n")
    assert ids[0] == USER_ID
    assert ids[-1] == END_ID
    a = ids.index(ASSISTANT_ID)
    assert ids[a - 1] == END_ID  # the instruction is closed before the answer starts
    assert tok.decode(ids[1 : a - 1]) == "Add two numbers."
    assert tok.decode(ids[a + 1 : -1]) == "def add(a, b):\n    return a + b\n"


def test_mask_marks_only_the_answer_and_its_end_token(tok):
    ids, mask = format_example(tok, "Say hi.", "print('hi')\n")
    a = ids.index(ASSISTANT_ID)
    assert len(mask) == len(ids)
    assert mask[: a + 1] == [0] * (a + 1)  # instruction and chat tokens: not learned
    assert mask[a + 1 :] == [1] * (len(ids) - a - 1)  # answer + final <|end|>: learned


def test_chat_token_text_inside_an_example_stays_plain_text(tok):
    ids, _ = format_example(tok, "Explain <|end|> please.", "It is <|assistant|> text.")
    assert ids.count(END_ID) == 2  # only the two we add
    assert ids.count(ASSISTANT_ID) == 1


# --- packing examples into fixed-length rows ---

def test_pack_keeps_each_example_whole_in_one_row():
    examples = [([5] * 4, [0, 0, 1, 1]), ([6] * 5, [0, 1, 1, 1, 1]), ([7] * 3, [0, 1, 1])]
    tokens, mask, dropped = pack_rows(examples, row_len=8)
    assert tokens.shape == mask.shape == (2, 8)
    assert tokens[0].tolist() == [5, 5, 5, 5, 7, 7, 7, PAD_ID]  # 3 fits after 4; 5 does not
    assert tokens[1].tolist() == [6, 6, 6, 6, 6, PAD_ID, PAD_ID, PAD_ID]
    assert dropped == 0


def test_padding_is_never_learned_and_answer_tokens_are_all_kept():
    examples = [([5] * 4, [0, 0, 1, 1]), ([6] * 5, [0, 1, 1, 1, 1])]
    tokens, mask, _ = pack_rows(examples, row_len=8)
    assert mask[tokens == PAD_ID].sum() == 0
    assert mask.sum() == 2 + 4


def test_examples_longer_than_a_row_are_dropped():
    tokens, mask, dropped = pack_rows([([5] * 9, [1] * 9), ([6] * 2, [0, 1])], row_len=8)
    assert dropped == 1
    assert tokens.shape == (1, 8)


# --- batches: x = row[:-1], y = row[1:] with -100 where nothing is learned ---

def test_batch_targets_are_next_tokens_with_unlearned_positions_skipped():
    tokens = np.array([[10, 11, 12, 13, 14]], dtype=np.uint16)
    mask = np.array([[0, 0, 1, 1, 0]], dtype=np.uint8)
    x, y = sft_batch(tokens, mask, [0])
    assert x.tolist() == [[10, 11, 12, 13]]
    assert y.tolist() == [[-100, 12, 13, -100]]
    assert x.dtype == y.dtype == torch.int64


# --- writing and reading the dataset ---

def test_prepare_writes_train_and_val_rows_that_load_back(tok, tmp_path):
    rows = [{"instruction": f"Task {i}.", "response": f"x = {i}\n"} for i in range(50)]
    meta = prepare_sft(rows, tok, tmp_path, row_len=64, val_fraction=0.1, seed=0)
    train_tokens, train_mask = load_sft(tmp_path, "train")
    val_tokens, val_mask = load_sft(tmp_path, "val")
    assert train_tokens.shape[1] == val_tokens.shape[1] == 64
    assert meta["train"]["examples"] == 45 and meta["val"]["examples"] == 5
    assert json.loads((tmp_path / "meta.json").read_text()) == meta
    assert train_mask.sum() > 0 and val_mask.sum() > 0


def test_prepare_split_is_reproducible(tok, tmp_path):
    rows = [{"instruction": f"Task {i}.", "response": f"x = {i}\n"} for i in range(30)]
    prepare_sft(rows, tok, tmp_path / "a", row_len=64, val_fraction=0.2, seed=3)
    prepare_sft(rows, tok, tmp_path / "b", row_len=64, val_fraction=0.2, seed=3)
    assert np.array_equal(load_sft(tmp_path / "a", "val")[0], load_sft(tmp_path / "b", "val")[0])


def test_chat_prompt_is_exactly_how_training_examples_begin(tok):
    ids, _ = format_example(tok, "Add two numbers.", "x")
    prompt = chat_prompt_ids(tok, "Add two numbers.")
    assert ids[: len(prompt)] == prompt
    assert prompt[-1] == ASSISTANT_ID

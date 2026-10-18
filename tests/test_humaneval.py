import os
import time

import pytest
import torch

from humaneval import (block_finished, build_program, chat_instruction, evaluate, extract_code, is_chat_model,
                       latest_checkpoint, pass_at_k, run_program)
from model.transformer import Keera, ModelConfig
from sample import generate

PROBLEM = {
    "task_id": "Toy/0",
    "prompt": 'def add(a, b):\n    """Return a + b."""\n',
    "canonical_solution": "    return a + b\n",
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
    "entry_point": "add",
}


# --- pass@k: chance that at least one of k samples passes, given c of n passed ---

def test_pass_at_1_is_the_share_that_passed():
    assert pass_at_k(n=10, c=5, k=1) == pytest.approx(0.5)


def test_pass_at_k_edges():
    assert pass_at_k(n=10, c=0, k=5) == 0.0
    assert pass_at_k(n=10, c=10, k=5) == 1.0
    assert pass_at_k(n=10, c=6, k=5) == 1.0  # only 4 failures: any 5 samples include a pass


def test_pass_at_k_matches_the_formula():
    # 1 - C(n-c, k) / C(n, k) = 1 - C(4, 2) / C(5, 2) = 1 - 6/10
    assert pass_at_k(n=5, c=1, k=2) == pytest.approx(0.4)


# --- building and running a program ---

def test_build_program_joins_prompt_answer_tests_and_check():
    program = build_program(PROBLEM, "    return a + b\n")
    assert program.startswith(PROBLEM["prompt"] + "    return a + b\n")
    assert PROBLEM["test"] in program
    assert program.rstrip().endswith("check(add)")


def test_a_correct_answer_passes():
    assert run_program(build_program(PROBLEM, "    return a + b\n"))


def test_a_wrong_answer_fails():
    assert not run_program(build_program(PROBLEM, "    return a - b\n"))


def test_a_syntax_error_fails():
    assert not run_program(build_program(PROBLEM, "    return a +\n"))


def test_an_endless_loop_is_stopped_by_the_timeout():
    start = time.monotonic()
    assert not run_program(build_program(PROBLEM, "    while True:\n        pass\n"), timeout=1.0)
    assert time.monotonic() - start < 5


def test_programs_run_in_a_throwaway_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_program("open('left_behind.txt', 'w').write('x')\n")
    assert not (tmp_path / "left_behind.txt").exists()


def test_programs_get_no_secrets_from_the_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    assert run_program("import os\nassert 'HF_TOKEN' not in os.environ\n")


# --- the whole loop, with a fake model ---

def test_evaluate_scores_each_problem():
    other = dict(PROBLEM, task_id="Toy/1")
    answers = {"Toy/0": "    return a + b\n", "Toy/1": "    return 0\n"}
    result = evaluate([PROBLEM, other], lambda problem: answers[problem["task_id"]])
    assert result["pass@1"] == pytest.approx(0.5)
    assert [r["passed"] for r in result["results"]] == [True, False]


def test_evaluate_with_several_samples_per_problem():
    tries = iter(["    return a + b\n", "    return 0\n", "    return 0\n", "    return 0\n"])
    result = evaluate([PROBLEM], lambda problem: next(tries), samples=4)
    assert result["pass@1"] == pytest.approx(0.25)
    assert result["results"][0]["passed_count"] == 1


# --- stopping generation early once the function is finished ---

TINY = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, d_ff=256, max_seq_len=16)


def test_generate_stops_when_stop_fn_says_so():
    torch.manual_seed(0)
    model = Keera(TINY).eval()
    out = generate(model, [1, 2], max_new_tokens=20, temperature=0.0, stop_id=None,
                   stop_fn=lambda new: len(new) == 3)
    assert len(out) == 3


def test_block_finished_once_a_new_top_level_line_starts():
    assert not block_finished("    return a + b\n")
    assert not block_finished("    total = 0\n\n    for x in xs:")
    assert block_finished("    return a + b\n\ndef")
    assert block_finished("    return 1\n<|endoftext|>")


# --- chat mode: the fine-tuned model is asked in words and answers with a code block ---

def test_chat_instruction_shows_the_function_to_complete():
    text = chat_instruction(PROBLEM)
    assert PROBLEM["prompt"].strip() in text
    assert "```python" in text


def test_extract_code_takes_the_first_python_block():
    answer = "Here is how:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nTests:\n```python\nassert 0\n```"
    assert extract_code(answer) == "def add(a, b):\n    return a + b\n"


def test_extract_code_accepts_a_bare_fence_or_no_fence():
    assert extract_code("```\nx = 1\n```") == "x = 1\n"
    assert extract_code("x = 1\n") == "x = 1\n"
    assert extract_code("```python\nx = 1\n") == "x = 1\n"  # cut off before the closing fence


def test_a_whole_function_from_chat_mode_passes():
    code = extract_code("```python\ndef add(a, b):\n    return a + b\n```")
    assert run_program(build_program(PROBLEM, "\n" + code))  # redefines add after the prompt


# --- picking the model to test: the newest checkpoint, in chat mode if it was fine-tuned ---

def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))


def test_latest_checkpoint_is_the_most_recently_written_run(tmp_path):
    _touch(tmp_path / "run-2b/checkpoint.pt", 1_000)
    _touch(tmp_path / "sft-1/checkpoint.pt", 2_000)
    _touch(tmp_path / "old/checkpoint.pt", 500)
    assert latest_checkpoint(tmp_path) == tmp_path / "sft-1/checkpoint.pt"


def test_latest_checkpoint_ignores_kept_snapshots(tmp_path):
    _touch(tmp_path / "run-2b/checkpoint.pt", 1_000)
    _touch(tmp_path / "run-2b/step_003500.pt", 9_000)
    assert latest_checkpoint(tmp_path) == tmp_path / "run-2b/checkpoint.pt"


def test_latest_checkpoint_without_any_run_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        latest_checkpoint(tmp_path)


def test_fine_tuned_checkpoints_are_chat_models():
    assert is_chat_model({"train_config": {"data_format": "sft"}})
    assert not is_chat_model({"train_config": {"data_format": "tokens"}})
    assert not is_chat_model({"train_config": {}})  # written before fine-tuning existed

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from model.transformer import Keera, ModelConfig
from sample import generate, read_prompt, trim_to_block

TINY = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, d_ff=256, max_seq_len=16)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Keera(TINY).eval()


def test_greedy_picks_the_most_likely_token_each_time(model):
    prompt = [5, 9, 2]
    out = generate(model, prompt, max_new_tokens=6, temperature=0.0)
    ids = torch.tensor([prompt])
    for _ in range(6):
        logits, _ = model(ids)
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert out == ids[0, 3:].tolist()  # only the new tokens are returned


def test_returns_max_new_tokens(model):
    out = generate(model, [1, 2], max_new_tokens=10, temperature=1.0, stop_id=None,
                   generator=torch.Generator().manual_seed(0))
    assert len(out) == 10


def test_stops_after_the_stop_token(model):
    first = generate(model, [1, 2], max_new_tokens=1, temperature=0.0)[0]
    out = generate(model, [1, 2], max_new_tokens=10, temperature=0.0, stop_id=first)
    assert out == [first]  # included, then generation ends


def test_prompts_longer_than_the_context_are_cropped(model):
    out = generate(model, list(range(1, 40)), max_new_tokens=20, temperature=0.0)
    assert len(out) == 20  # would raise max_seq_len otherwise


def test_top_k_only_samples_from_the_k_most_likely(model):
    prompt = [3, 4]
    logits, _ = model(torch.tensor([prompt]))
    top3 = set(logits[0, -1].topk(3).indices.tolist())
    gen = torch.Generator().manual_seed(0)
    firsts = {generate(model, prompt, max_new_tokens=1, temperature=5.0, top_k=3, generator=gen)[0]
              for _ in range(50)}
    assert firsts <= top3


def test_same_seed_same_output(model):
    a = generate(model, [1], 8, temperature=1.0, generator=torch.Generator().manual_seed(3))
    b = generate(model, [1], 8, temperature=1.0, generator=torch.Generator().manual_seed(3))
    assert a == b


def test_cli_help_runs_as_installed_module():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-m", "sample", "--help"],
                            capture_output=True, text=True, env=env, cwd=Path(__file__).parent)
    assert result.returncode == 0, result.stderr
    assert "--prompt" in result.stdout


# --- trimming a completion to the code block it started ---

def test_trim_stops_at_the_next_top_level_line():
    out = "    return n * 2\n\ndef other():\n    pass\n"
    assert trim_to_block(out) == "    return n * 2\n"


def test_trim_keeps_a_top_level_first_line_and_its_body():
    out = 'if __name__ == "__main__":\n    main()\n\nprint("again")\n'
    assert trim_to_block(out) == 'if __name__ == "__main__":\n    main()\n'


def test_trim_keeps_the_rest_of_the_prompt_line():
    assert trim_to_block("[x * x for x in range(10)]\nprint(squares)\n") == "[x * x for x in range(10)]\n"


def test_trim_does_not_cut_at_else_elif_except_finally():
    out = "    x = 1\nelse:\n    x = 2\nexcept ValueError:\n    pass\nfinally:\n    done()\nnext_thing()\n"
    assert trim_to_block(out) == "    x = 1\nelse:\n    x = 2\nexcept ValueError:\n    pass\nfinally:\n    done()\n"


def test_trim_cuts_at_the_end_of_text_token():
    assert trim_to_block("    return 1\n<|endoftext|>import os\n") == "    return 1\n"


def test_trim_leaves_an_unfinished_block_alone():
    assert trim_to_block("    total = 0\n    for x in xs:\n        total +=") == "    total = 0\n    for x in xs:\n        total +="


# --- reading a prompt typed in the terminal ---

def test_read_prompt_ends_at_an_empty_line():
    lines = iter(["def f(n):", '    """Double n."""', "", "ignored"])
    assert read_prompt(lambda: next(lines)) == 'def f(n):\n    """Double n."""\n'


def test_read_prompt_returns_none_at_end_of_input():
    def eof():
        raise EOFError
    assert read_prompt(eof) is None

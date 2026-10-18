from collections.abc import Callable

import torch

from model.transformer import Keera
from tokenizer import EOT_ID


@torch.no_grad()
def generate(
    model: Keera,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    stop_id: int | None = EOT_ID,
    generator: torch.Generator | None = None,
    stop_fn: Callable[[list[int]], bool] | None = None,
) -> list[int]:
    """Continue `prompt_ids` one token at a time; returns only the new tokens.

    temperature 0 = always the most likely token; higher = more random.
    top_k keeps only the k most likely tokens before sampling.
    Stops early after `stop_id` (end of file by default), or once stop_fn(new tokens) is True.
    """
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=device)
    new: list[int] = []
    for _ in range(max_new_tokens):
        context = ids[:, -model.cfg.max_seq_len :]  # the model only sees its context window
        logits, _ = model(context)
        logits = logits[0, -1].float()
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = logits.topk(min(top_k, logits.numel())).values[-1]
                logits[logits < kth] = float("-inf")
            probs = torch.softmax(logits, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        new.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == stop_id or (stop_fn is not None and stop_fn(new)):
            break
    return new


EOT_TEXT = "<|endoftext|>"
CONTINUES_BLOCK = ("else", "elif", "except", "finally")


def trim_to_block(completion: str) -> str:
    """Cut a completion where the code block it started ends.

    The block ends at the next line that starts at column 0 (a new def, class,
    statement...), except else/elif/except/finally, which belong to the block.
    The first non-blank line is always kept, and the end-of-file token ends it too.
    """
    cut = EOT_TEXT in completion
    completion = completion.split(EOT_TEXT)[0]
    kept: list[str] = []
    seen_code = False
    for line in completion.splitlines(keepends=True):
        top_level = line.strip() and not line[0].isspace()
        if seen_code and top_level and not line.startswith(CONTINUES_BLOCK):
            cut = True
            break
        kept.append(line)
        seen_code = seen_code or bool(line.strip())
    text = "".join(kept)
    return text.rstrip() + "\n" if cut else text


def read_prompt(read_line=input) -> str | None:
    """Read typed lines until an empty line; None when input ends (Ctrl-D)."""
    lines: list[str] = []
    while True:
        try:
            line = read_line()
        except EOFError:
            return "\n".join(lines) + "\n" if lines else None
        if line == "" and lines:
            return "\n".join(lines) + "\n"
        if line != "":
            lines.append(line)

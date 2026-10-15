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
) -> list[int]:
    """Continue `prompt_ids` one token at a time; returns only the new tokens.

    temperature 0 = always the most likely token; higher = more random.
    top_k keeps only the k most likely tokens before sampling.
    Stops early after `stop_id` (end of file by default).
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
        if nxt == stop_id:
            break
    return new

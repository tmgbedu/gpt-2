"""Ask the newest model something.

    uv run python tinker.py                                   # the PROMPT below
    uv run python tinker.py "Write a function that reverses a string."

Uses the run whose checkpoint.pt was written last (now: checkpoints/sft-1). A fine-tuned
model gets the prompt as a chat question; a pretrained one just continues the text.
"""

import sys

import torch

from data.sft import END_ID, chat_prompt_ids
from humaneval import is_chat_model, latest_checkpoint
from model.transformer import Keera, ModelConfig
from sample import generate
from tokenizer import load_tokenizer
from trainer import load_checkpoint

PROMPT = "Write a Python function that returns the n-th Fibonacci number."
TEMPERATURE = 0.2  # 0 = always the most likely token; higher = more varied
MAX_NEW_TOKENS = 400

prompt = sys.argv[1] if len(sys.argv) > 1 else PROMPT
path = latest_checkpoint("checkpoints")
ckpt = load_checkpoint(path)
model = Keera(ModelConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model"])
model.eval()
tok = load_tokenizer("artifacts/tokenizer/tokenizer.json")

chat = is_chat_model(ckpt)
ids = chat_prompt_ids(tok, prompt) if chat else tok.encode(prompt).ids
print(f"# {path} | step {ckpt['step']} | {'chat' if chat else 'continue'}\n")
print(prompt, end="\n\n" if chat else "")  # a continuation follows the prompt directly
new = generate(
    model,
    ids,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    top_k=50,
    stop_id=END_ID if chat else 0,
    generator=torch.Generator().manual_seed(0),
)
print(tok.decode(new))

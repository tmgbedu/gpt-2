"""Write code with a trained checkpoint.

    uv run python -m sample --prompt "def fibonacci(n):"
    uv run python -m sample --prompt-file start.py --samples 3 --temperature 0.6
"""

import argparse
from pathlib import Path

import torch

from model.transformer import Keera, ModelConfig
from sample import generate
from tokenizer import load_tokenizer
from trainer import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/smoke/checkpoint.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--prompt", default="def ", help="code to continue (\\n and \\t are understood)")
    parser.add_argument("--prompt-file", type=Path, help="read the prompt from a file instead")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = always the most likely token")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu keeps the GPU free for training")
    args = parser.parse_args()

    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt.encode().decode("unicode_escape")
    tok = load_tokenizer(args.tokenizer)
    ckpt = load_checkpoint(args.checkpoint)
    model = Keera(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    print(f"# {args.checkpoint} | step {ckpt['step']} | {ckpt['tokens']:,} tokens trained\n")

    prompt_ids = tok.encode(prompt).ids
    gen = torch.Generator().manual_seed(args.seed)
    for i in range(args.samples):
        new = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k, generator=gen)
        print(f"# --- sample {i + 1} ---")
        print(prompt + tok.decode(new, skip_special_tokens=False))


if __name__ == "__main__":
    main()

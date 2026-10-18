"""Write code with a trained checkpoint.

    uv run python -m sample --prompt "def fibonacci(n):"
    uv run python -m sample --prompt-file start.py --samples 3 --temperature 0.6
    uv run python -m sample -i        # type prompts one after another; Enter twice runs one
    uv run python -m sample --chat --checkpoint checkpoints/sft-1/checkpoint.pt --prompt "Write a function that reverses a string."
"""

import argparse
from pathlib import Path

import torch

from model.transformer import Keera, ModelConfig
from data.sft import END_ID, chat_prompt_ids
from sample import generate, read_prompt, trim_to_block
from tokenizer import load_tokenizer
from trainer import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/run-2b/checkpoint.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--prompt", default="def ", help="code to continue (\\n and \\t are understood)")
    parser.add_argument("--prompt-file", type=Path, help="read the prompt from a file instead")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = always the most likely token")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu keeps the GPU free for training")
    parser.add_argument("-i", "--interactive", action="store_true", help="keep asking for prompts")
    parser.add_argument("--full", action="store_true", help="don't stop where the function ends")
    parser.add_argument("--chat", action="store_true", help="fine-tuned model: the prompt is an instruction")
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer)
    ckpt = load_checkpoint(args.checkpoint)
    model = Keera(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    print(f"# {args.checkpoint} | step {ckpt['step']} | {ckpt['tokens']:,} tokens trained\n")

    gen = torch.Generator().manual_seed(args.seed)

    def complete(prompt: str) -> None:
        for i in range(args.samples):
            print(f"# --- sample {i + 1} ---")
            if args.chat:
                new = generate(model, chat_prompt_ids(tok, prompt.strip()), args.max_new_tokens,
                               args.temperature, args.top_k, stop_id=END_ID, generator=gen)
                print(tok.decode(new))
                continue
            new = generate(model, tok.encode(prompt).ids, args.max_new_tokens, args.temperature, args.top_k,
                           generator=gen)
            text = tok.decode(new, skip_special_tokens=False)
            print(prompt + (text if args.full else trim_to_block(text)))

    if not args.interactive:
        complete(args.prompt_file.read_text() if args.prompt_file
                 else args.prompt.encode().decode("unicode_escape"))
        return
    print("Type or paste code, then press Enter on an empty line. Ctrl-D quits.\n")
    while (prompt := read_prompt(lambda: input(">>> "))) is not None:
        complete(prompt)
        print()


if __name__ == "__main__":
    main()

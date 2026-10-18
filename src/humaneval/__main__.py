"""Score a checkpoint on HumanEval (164 problems; ~3-5% expected for ~100M raw-code models).

    uv run python -m humaneval                         # greedy pass@1, final 2B model
    uv run python -m humaneval --limit 10              # quick look at the first 10
    uv run python -m humaneval --samples 10 --temperature 0.8
    uv run python -m humaneval --chat --checkpoint checkpoints/sft-1/checkpoint.pt   # fine-tuned model

Model-written code runs on this machine, each program in its own process with a timeout.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset

from data.sft import END_ID, chat_prompt_ids
from humaneval import block_finished, chat_instruction, evaluate, extract_code
from model.transformer import Keera, ModelConfig
from sample import generate, trim_to_block
from tokenizer import load_tokenizer
from trainer import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/run-2b/checkpoint.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--limit", type=int, help="only the first N problems")
    parser.add_argument("--samples", type=int, default=1, help="answers per problem")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (use > 0 with --samples)")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--chat", action="store_true",
                        help="fine-tuned model: ask in words, take the first code block of the answer")
    parser.add_argument("--out", type=Path, help="results file (default: next to the checkpoint)")
    args = parser.parse_args()

    problems = list(load_dataset("openai/openai_humaneval", split="test"))[: args.limit]
    tok = load_tokenizer(args.tokenizer)
    ckpt = load_checkpoint(args.checkpoint)
    model = Keera(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    out = args.out or args.checkpoint.with_name(f"humaneval-{args.checkpoint.stem}.jsonl")
    print(f"# {args.checkpoint} | step {ckpt['step']} | {len(problems)} problems x {args.samples} samples"
          f"{' | chat' if args.chat else ''}\n")

    def complete_chat(problem: dict) -> str:
        stop = lambda new: tok.decode(new).count("```") >= 2  # noqa: E731  first code block closed
        new = generate(model, chat_prompt_ids(tok, chat_instruction(problem)), args.max_new_tokens,
                       args.temperature, args.top_k, stop_id=END_ID, generator=gen, stop_fn=stop)
        return "\n" + extract_code(tok.decode(new))  # a whole function, defined again after the prompt

    def complete(problem: dict) -> str:
        if args.chat:
            return complete_chat(problem)
        stop = lambda new: block_finished(tok.decode(new, skip_special_tokens=False))  # noqa: E731
        new = generate(model, tok.encode(problem["prompt"]).ids, args.max_new_tokens, args.temperature,
                       args.top_k, generator=gen, stop_fn=stop)
        return trim_to_block(tok.decode(new, skip_special_tokens=False))

    solved = 0
    start = time.monotonic()
    with out.open("w") as f:
        def report(result: dict) -> None:
            nonlocal solved
            solved += result["passed_count"] > 0
            f.write(json.dumps(result) + "\n")
            f.flush()
            mark = "✅" if result["passed_count"] else "❌"
            print(f"{mark} {result['task_id']:<14} {result['passed_count']}/{result['samples']} "
                  f"| solved so far {solved} | {time.monotonic() - start:,.0f}s")

        score = evaluate(problems, complete, samples=args.samples, on_result=report)["pass@1"]
    print(f"\npass@1 = {score:.1%} ({solved}/{len(problems)} problems solved at least once) -> {out}")


if __name__ == "__main__":
    main()

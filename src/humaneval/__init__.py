"""HumanEval: 164 Python problems with hidden tests; the score is the share the model solves.

Each problem is a function header + docstring. The model writes the body, then
prompt + body + tests + check(function) runs in its own process. Exit code 0 = passed.
"""

import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from sample import trim_to_block

Problem = dict[str, str]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k samples passes, when c of n samples passed.

    1 - C(n-c, k) / C(n, k): one minus the chance that all k picks are failures.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def build_program(problem: Problem, completion: str) -> str:
    return f"{problem['prompt']}{completion}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"


def run_program(program: str, timeout: float = 10.0) -> bool:
    """Run untrusted code in a fresh process: throwaway folder, no environment secrets, time limit.

    This is not a full sandbox: the code can still read your files and use the network.
    """
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "program.py"
        path.write_text(program)
        try:
            result = subprocess.run([sys.executable, "-I", str(path)], cwd=folder, timeout=timeout,
                                    env={"PATH": os.environ.get("PATH", "")}, capture_output=True)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def evaluate(problems: list[Problem], complete: Callable[[Problem], str], samples: int = 1,
             on_result: Callable[[dict], None] | None = None) -> dict:
    """Ask `complete` for `samples` bodies per problem, run each, and average pass@1."""
    results = []
    for problem in problems:
        completions = [complete(problem) for _ in range(samples)]
        passed = [run_program(build_program(problem, c)) for c in completions]
        result = {"task_id": problem["task_id"], "passed": passed[0], "passed_count": sum(passed),
                  "samples": samples, "completions": completions}
        results.append(result)
        if on_result:
            on_result(result)
    score = sum(pass_at_k(samples, r["passed_count"], 1) for r in results) / len(results)
    return {"pass@1": score, "results": results}


def block_finished(text: str) -> bool:
    """True once the generated text has moved past the function body."""
    return len(trim_to_block(text)) < len(text.rstrip())


def chat_instruction(problem: Problem) -> str:
    """How a fine-tuned (chat) model is asked: in words, like its training instructions."""
    return f"Complete the following Python function.\n\n```python\n{problem['prompt'].strip()}\n```"


def extract_code(answer: str) -> str:
    """The first ``` code block of an answer (also one cut off before its closing fence)."""
    match = re.search(r"```[^\n]*\n(.*?)(?:```|\Z)", answer, re.DOTALL)
    return match.group(1) if match else answer


def latest_checkpoint(root: str | Path = "checkpoints") -> Path:
    """The run whose checkpoint.pt was written last (step_*.pt snapshots are not runs)."""
    runs = list(Path(root).glob("*/checkpoint.pt"))
    if not runs:
        raise FileNotFoundError(f"no checkpoint found in {root}/*/checkpoint.pt")
    return max(runs, key=lambda p: p.stat().st_mtime)


def is_chat_model(ckpt: dict) -> bool:
    """Fine-tuned on instructions (data_format "sft"): ask in words instead of giving code to continue."""
    return ckpt.get("train_config", {}).get("data_format") == "sft"

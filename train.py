from datasets import load_dataset

dataset = load_dataset(
    "codeparrot/codeparrot-clean-train",
    split="train",
    streaming=True,
)

for example in dataset.take(3):
    print(example["content"])

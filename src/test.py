from datasets import load_dataset
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

dataset = load_dataset("vectara/open_ragbench")
print(dataset)

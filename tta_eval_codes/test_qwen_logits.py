from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)

# Find token IDs for "good" and "poor"
# Qwen tokenizer might prefix a space or not. Let's see.
good_ids = processor.tokenizer.encode("good", add_special_tokens=False)
poor_ids = processor.tokenizer.encode("poor", add_special_tokens=False)
good_ids_space = processor.tokenizer.encode(" good", add_special_tokens=False)
poor_ids_space = processor.tokenizer.encode(" poor", add_special_tokens=False)

print(f"good: {good_ids}, poor: {poor_ids}")
print(f" good: {good_ids_space},  poor: {poor_ids_space}")

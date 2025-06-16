import torch
import re
import polars as pl
from datasets import Dataset, NamedSplit, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# tokenizer.add_tokens(['<think>', '</think>', '<answer>', '</answer>'])
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
print(model.get_input_embeddings())
# model.resize_token_embeddings(len(tokenizer))
# print(model.get_input_embeddings())


inputs = tokenizer("who is the president of america in year 2015?", return_tensors="pt")
outputs = model.generate(**inputs)
print(outputs)
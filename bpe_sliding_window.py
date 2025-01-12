""""
Title: bpe_sliding_window.py
Author: Ann Hagan
Date: 2025-1-11
Purpose: Creating target input pairs for a language model using a sliding window approach.
"""
#Need to install pip install tiktoken.
#The tokenizer is a Python implementation of the byte pair encoding algorithm.

from importlib.metadata import version
import tiktoken


#instantiate the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

#encode the entire text with tiktoken
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

#lets remove the first 50 tokens just for demo purposes to change the text
enc_sample = enc_text[50:]

#We are going to have the code slide over the text, and predict the next token for each window
#We will use a window size of 4 tokens
context_size = 4 

#Easiest way is to have x variables which is the input and y variable which is the output
x = enc_sample[:context_size]
y = enc_sample[1:context_size]
print(f"x: {x}")
print(f"y: {y}")

#because we are processing one word at a time, we can predict the next word based on the input tokens.
#left of the arrow is the input token to the LLM, right side is the target token ID based on training set.
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"context: {context}, -----> desired: {desired}")
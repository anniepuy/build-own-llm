""""
Title: byte_pair_enconding.py
Author: Ann Hagan
Date: 2025-1-11
Purpose: Tokenizer that uses byte pair encoding to tokenize text. The cool thing about byte pair encoding
is that instead of replacing unknown words with <|unk|>, it can split the unknown word into known subwords and then 
assign tokens based on the subwords or characters. This enables it to handle unknown words better than the previous tokenizers.
So cool!
"""
#Need to install pip install tiktoken.
#The tokenizer is a Python implementation of the byte pair encoding algorithm.

from importlib.metadata import version
import tiktoken

print(version('tiktoken'))

#instantiate the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

#Test the tokenizer using similar encode methods as SimpleTokenizer2
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terrances"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

#Decode the integers
strings = tokenizer.decode(integers)
print(strings)
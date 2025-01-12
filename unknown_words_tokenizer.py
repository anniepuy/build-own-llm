""""
Title: tokenizing_text.py
Author: Ann Hagan
Date: 2025-1-11
Purpose: Tokenizer class that replaces unknown words with <|unk|> and adds <|endoftext|> to the end of the text.
"""

import urllib.request
import re

#Step 1: Import the text
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

#load the text using Pythons standard file reading utiliites
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of characters: ", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f"Total number of tokens: {len(preprocessed)}")
print(f"First 30 tokens: {preprocessed[:30]}")

#modify the vocubular to include special tokens of <unk> and <|endoftext|>
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))

#print the last five entries
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
    if i >= 4:
        break

#now implement it as a class
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\']|--)', r'\1', text)
        return text
    
#Test the tokenizer class V2
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
tokenizer = SimpleTokenizerV2(vocab)
print(f"This is the result of the SimpleTokenizerV2: {tokenizer.encode(text)}")
print(f"Decoded text: {tokenizer.decode(tokenizer.encode(text))}")
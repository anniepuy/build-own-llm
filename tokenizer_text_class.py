""""
Title: tokenizing_text_class.py
Author: Ann Hagan
Date: 2025-1-11
Purpose: Using the baseline tokenizer class and applying to a new text.
"""
import re

#Step 1: Import the text
# Open the file in read mode
with open('wind_turbines.txt', 'r') as f:
    # Read the entire file
    raw_text = f.read()

# Print the content
print(raw_text[:99])
print("Total number of characters: ", len(raw_text))

#Step 2: Split the text for better tokenization
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f"Total number of tokens: {len(preprocessed)}")
print(f"First 30 tokens: {preprocessed[:30]}")

#Step 3: Build a vocabulary, removes dupliqates and assigns an integer to each unique word
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(f"Vocabulary size: {vocab_size}")

#Print the first 51 integers and their corresponding words (note THIS IS SO COOL!)
vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        #Creates an inverse vocubulary that maps tokenIDs back to the original token text
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(f'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids
    
    def decode(self, ids):
        #Converts the token IDs back to text
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s([,.:;?_!"()\']|--)', r'\1', text)
        return text
            
#Test the tokenizer
tokenizer = SimpleTokenizerV1(vocab)
text = raw_text
ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")
print(f"Decoded text: {tokenizer.decode(ids)}")

#Note, in order for the tokenizer to work, the dictionary must be built from the text. 

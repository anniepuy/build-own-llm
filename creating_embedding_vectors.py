""""
Title: creating_embedding_vectors.py
Author: Ann Hagan
Date: 2025-1-12
Purpose: After the vocabulary is created from the corpus of text, and the byte pair encoding is applied to the text, 
and the token IDs are converted to PyTorch tensors, the next step is to create the embedding vectors. Essentially the embedding 
layer is a loopup operation that gets teh rows from the layer's weight matrix based on the token IDs.
"""
#Small scale example from the "Build an LLM"

import torch

#Example input tokens with IDs 1, 2, 3, 4, 5
input_ids = torch.tensor([1, 2, 3, 4, 5])

#small sample of 6 word vocabulary
vocab_size = 6
#goal out put tokens with the embedding vector size of 3
output_dim = 3

#Now we will instantiate the PyTorch embedding layer
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(f"Weighted matrix result: {embedding_layer.weight}")
print("Each row is one of the possible 6 vocabulary tokens, and the columns are the three embedding dimensions")

#now apply it to a tokeID to obtain the embedding vector
#This output matches the fourth row of the weight matrix
print(embedding_layer(torch.tensor([3])))
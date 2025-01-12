""""
Title: positional_embedding_vectors.py
Author: Ann Hagan
Date: 2025-1-12
Purpose: To apply the attention mechanism to the embeddings. This way
the LLM is aware of the position of the tokens in the input sequence - this 
gives LLM's the ability to have semantic understanding of the text.
There are two types of positional embeddings: 
1. Absolute positional embeddings
2. relative positional embeddings 
"""

#For this example, we will use input tokens of 256-dmensional embeddings
#and the token IDs are created by a BPE tokenizer with vocabulary size of 50,257

#using PyTorch built in Dataset and DataLoader classes 

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    #return total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)
    
    #returns a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

#Data loader to load the inputs in batches via PyTorch DataLoader
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                        stride=128, shuffle=True, drop_last=True,
                        num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        #drops last batch if it is shorter than batch_size to reduce loss spikes in training
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

#Test the data loader
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
vocab_size = 50257
output_dim = 256  
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter) 
print("Token IDs: \n", inputs)
print("\nInputs tensor shape: \n", inputs.shape)
print("The tensor shape means there are 8 text sampels with four tokens each.")

#Next use the embedding layer to embed token IDs ijnto 256-dimensional vectors
token_embeddings = token_embedding_layer(inputs)
print("The token embeddings shape means each token ID is now embedded as a 256-dimenstional vector")
print(token_embeddings.shape)

#Now lets use the GPT's aboso;ute positional embeddings which has the same
#embedding dimensions as the token embedding layer
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Absolute positional embeddings shape: \n", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("Input embeddings shape: \n", input_embeddings.shape)

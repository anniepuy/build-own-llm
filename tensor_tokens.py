""""
Title: tensor_tokens.py
Author: Ann Hagan
Date: 2025-1-12
Purpose: fFinal task before turning tokens into embeddings. This script will iterate over theinput dataset and return inputs/targets as PyTorch tensors.
"""

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

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)


#The first tensors stores the input tokenIS and second tensor stores the target token IDs

#Second batch's tokens are shifted by one position, this is the effect of stride=1
second_batch = next(data_iter)
print(second_batch)

#now lests runs it with max_length2 and stride=2
dataloader2 = create_dataloader_v1(raw_text, batch_size=1, max_length=2, stride=2, shuffle=False)
data_iter2 = iter(dataloader2)
first_batch2 = next(data_iter2)
print(f"First batch with max_length = 2, stride = 2: {first_batch2}")
second_batch2 = next(data_iter2)
print(f"Second batch with max_length =2, stride = 2: {second_batch2}")

#now lests runs it with max_length6 and stride=1, batch_size=2
#Realistically, one wants to keep the max_length and stride the same to prevent overfitting
dataloader3 = create_dataloader_v1(raw_text, batch_size=2, max_length=6, stride=1, shuffle=False)
data_iter3 = iter(dataloader3)
first_batch3 = next(data_iter3)
print(f"First batch with max_length = 6, stride = 1, batch_size =2 : {first_batch3}")
second_batch3 = next(data_iter3)
print(f"Second batch with max_length =6, stride = 1, batch_size=2: {second_batch3}")
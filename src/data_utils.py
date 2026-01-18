import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# customized dataset class
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, context_lim, stride):
        super().__init__()

        self.input_ids = []
        self.output_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids)-context_lim, stride):
            context_chunk = token_ids[i: i+context_lim]
            target_chunk = token_ids[i+1: i+context_lim+1]
            self.input_ids.append(torch.tensor(context_chunk))
            self.output_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.output_ids[index]
    
# new and improved version of of dataset class

class GPTDatasetV2(Dataset):
    def __init__(self, txt, tokenizer, context_len, stride):
        super().__init__()

        self.input_ids = []
        self.output_ids = []
        self.context_len = context_len
        self.stride = stride

        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        self.data = torch.tensor(token_ids, dtype=torch.long)
        print(f"Loaded {len(self.data)} tokens.")

    def __len__(self):
        return (len(self.data) - self.context_len) // self.stride

    def __getitem__(self, idx):
        start_idx = idx * self.stride

        # Slice the tensor on the fly (Zero RAM overhead)
        # Input:  0 -> 1024
        # Target: 1 -> 1025
        chunk = self.data[start_idx : start_idx + self.context_len + 1]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y

# function to create dataloader 
def create_dataloader(txt, GPTDataset, context_lim=256, stride=128, 
                      batch_size=2, shuffle=True,
                      drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, context_lim, stride)
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

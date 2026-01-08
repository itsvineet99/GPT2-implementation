import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# customized dataset class
class GPTDataset(Dataset):
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

# function to create dataloader 
def create_dataloader(txt, context_lim=256, stride=128, 
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

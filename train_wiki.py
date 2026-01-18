'''
    Here we train our same gpt2 124m implementation on bigger dataset
    with optimized training script including mixed-precision training
    and gradient accumulation training. 
'''

import torch
import math
from datasets import load_dataset # download datasets using pip
import os
import time
import tiktoken
from src import (GPTConfig, GPTModel, GPTDatasetV1, GPTDatasetV2,
                 create_dataloader, calc_loss_loader,
                 calc_loss_batch, evaluate_model,
                 generate_and_print_sample, plot_losses,
                 generate, text_to_token_ids,
                 token_ids_to_text)

def prepare_raw_data():
    print("1. Downloading dataset...")
    # Load the Parquet version 
    dataset = load_dataset("rahular/simple-wikipedia", split="train")

    # Split into Train (90%) and Validation (10%)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"   Train Size: {len(split_dataset['train'])} articles")
    print(f"   Val Size:   {len(split_dataset['test'])} articles")

    # Extract Raw Text and Save to Files
    def save_to_file(data_split, filename):
        # Join all articles with a newline character
        # distinct articles are separated by '\n<|endoftext|>\n' to help the model know where one ends
        # But for simple GPT-2, just '\n\n' is often enough.
        # Let's use standard newline for simplicity.
        raw_text = "\n\n".join(data_split['text'])

        with open(filename, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"   Saved {filename} ({len(raw_text)/1024/1024:.2f} MB)")
        return raw_text

    print("3. Saving raw text to disk...")
    train_text = save_to_file(split_dataset['train'], "data/wiki/train.txt")
    val_text = save_to_file(split_dataset['test'], "data/wiki/val.txt")

    return train_text, val_text

# Execute the function
train_raw_text, val_raw_text = prepare_raw_data()

# training function 
def train_model(model, num_epochs, optimizer, tokenizer, train_loader, val_loader,
                start_context, device, eval_freq, eval_iter,gradient_accumulation_steps, 
                # arguments for Learning Rate Schedule
                warmup_steps, max_steps, min_lr, max_lr,
                #set up global step
                global_step=-1):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0

    # Define the learning rate scheduler helper function
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        
        if it > max_steps:
            return min_lr
        
        # Cosine decay
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # mixed precision training (use fp16)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss = calc_loss_batch(input_batch, target_batch, model, device)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward() # use scaler cause we are using autocast

            is_accum_complete = (batch_idx + 1) % gradient_accumulation_steps == 0

            if is_accum_complete:

                iter_num = global_step + 1
                lr = get_lr(iter_num) # Determine the learning rate for this specific step
                # Apply the calculated LR to the optimizer
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                scaler.unscale_(optimizer) #unscalre optimizer before stepping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            tokens_seen += input_batch.numel()

            if is_accum_complete and global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                # Added current LR to print statement for debugging
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1} (step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}, "
                      f"LR {current_lr:.6f}")

        print('\n')
        generate_and_print_sample(start_context, model, tokenizer, device)
        print('\n')

    return train_losses, val_losses, track_tokens_seen, global_step

# set device and initialize tokenizer
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("device: ", device)

# setting up mixed precision
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# initialize model
torch.manual_seed(53)
cfg = GPTConfig().GPT_124M()
model = GPTModel(cfg)
model.to(device)

model = torch.compile(model) # Free speedup (roughly 20%)

# set up epoch num
total_num_epoch = 3
# setting up numper of epochs you will run for current session 
# cause we will be using continual learning to traing on bigger dataset 
num_epoch_sesh = 1 

# gradient accumulation stratedy 
batch_size = 4
gradient_accumulation_steps = 8

# creating dataloaders
train_dataloader = create_dataloader(
    txt=train_raw_text,
    GPTDataset=GPTDatasetV2,
    context_lim=cfg.context_length,
    stride=cfg.context_length,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2
)

val_dataloader = create_dataloader(
    txt=val_raw_text,
    GPTDataset=GPTDatasetV2,
    context_lim=cfg.context_length,
    stride=cfg.context_length,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=2
)

# Create groups safely (checking requires_grad)
# this sets weight decay for biases and layernorms to 0 cause we don't need 
# any weight decay for them.
param_groups = [
    {'params': [p for p in model.parameters() if p.dim() >= 2 and p.requires_grad], 'weight_decay': 0.1},
    {'params': [p for p in model.parameters() if p.dim() < 2 and p.requires_grad], 'weight_decay': 0.0}
]

# Initialize optimizer, tokenizer and scaler
optimizer = torch.optim.AdamW(
    param_groups,
    lr=3e-4,
    betas=(0.9, 0.95),
    fused=torch.cuda.is_available()
)
tokenizer = tiktoken.get_encoding('gpt2')
scaler = torch.amp.GradScaler('cuda')

# calculate Max Steps, warmup steps
steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
max_steps = steps_per_epoch * total_num_epoch

warmup_steps = int(max_steps * 0.05)
warmup_steps = max(1, warmup_steps)

print(f"Total optimization steps: {max_steps}")
print(f"Warmup steps: {warmup_steps}")

# Cosine-decay learning rate
max_lr = 3e-4
min_lr = max_lr * 0.1

# start training and recording time
start_time = time.perf_counter()

train_losses, val_losses, tokens_seen, global_step = train_model(
    model, num_epoch_sesh, optimizer, tokenizer, train_dataloader,
    val_dataloader, start_context="Every effort moves you",
    device=device, eval_freq=50, eval_iter=50,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,max_steps=max_steps,
    min_lr=min_lr, max_lr=max_lr
)

end_time = time.perf_counter()
total_time = (end_time - start_time) / 60

print(f"total time required to train thsi model: {total_time:.2f} minutes.\n")

# plot losses 
epochs_tensor = torch.linspace(0, num_epoch_sesh, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


model.eval()

# sampling to see how good model performance is
torch.manual_seed(53)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    context_size=cfg.context_length,
    max_new_tokens=20,
    top_k=50,
    temperature=2,
    eos_id=50257
)

print("Output text (by fully trained model):\n",
    token_ids_to_text(token_ids, tokenizer))


# save model checkpoint for continual training
checkpoint = {
    'epoch': num_epoch_sesh,  # Save next epoch to start from
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # Important for mixed precision
    'global_step': global_step,
}
torch.save(checkpoint, 'checkpoints/gpt2(1).pth')
print('Checkpoint saved!')


# profiler function to evaluate memory performances 
import torch.profiler as profiler
from torch.cuda.amp import GradScaler
from torch.amp import autocast

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=2, warmup=2, active=5),  # Profile 5 batches after warmup
    on_trace_ready=profiler.tensorboard_trace_handler('./log'),
    profile_memory=True,
    record_shapes=True
) as prof:

    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        if batch_idx >= 9:  # 2 wait + 2 warmup + 5 active = 9 batches
            break

        inputs, labels = inputs.to(device), labels.to(device)

        with torch.amp.autocast('cuda'):
            loss = calc_loss_batch(inputs, labels, model, device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        prof.step()  # Signal profiler to move to next step

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

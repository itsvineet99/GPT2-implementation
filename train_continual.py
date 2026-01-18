import torch
import tiktoken
import time
import math
from src import (GPTConfig, GPTModel, GPTDatasetV1, GPTDatasetV2,
                 create_dataloader, calc_loss_loader,
                 calc_loss_batch, evaluate_model,
                 generate_and_print_sample, plot_losses,
                 generate, text_to_token_ids,
                 token_ids_to_text)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# Setting up mixed precision
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Initialize model architecture 
torch.manual_seed(53)
cfg = GPTConfig().GPT_124M()
model = GPTModel(cfg)
model.to(device)
model = torch.compile(model)

#setting up batch size and accumulation step
batch_size = 4
gradient_accumulation_steps = 8

# loading ras text from saved files 
with open("data/wiki/train.txt", 'r') as f:
    train_raw_text = f.read()

with open("data/wiki/val.txt", 'r') as f:
    val_raw_text = f.read()

# load dataloaders
train_dataloader = create_dataloader(
    txt=train_raw_text,
    context_lim=cfg.context_length,
    stride=cfg.context_length,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2
)

val_dataloader = create_dataloader(
    txt=val_raw_text,
    context_lim=cfg.context_length,
    stride=cfg.context_length,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=2
)

# Initialize optimizer (same architecture as before)
param_groups = [
    {'params': [p for p in model.parameters() if p.dim() >= 2 and p.requires_grad], 'weight_decay': 0.1},
    {'params': [p for p in model.parameters() if p.dim() < 2 and p.requires_grad], 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(
    param_groups,
    lr=6e-4,  # This will be overwritten by checkpoint
    betas=(0.9, 0.95),
    fused=torch.cuda.is_available()
)

# Initialize scaler and tokenizer
scaler = torch.amp.GradScaler('cuda')
tokenizer = tiktoken.get_encoding('gpt2')

# --- LOAD CHECKPOINT ---
checkpoint_path = 'checkpoints/gpt2(1).pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
start_epoch = checkpoint['epoch']
global_step = checkpoint['global_step']

print(f"Checkpoint loaded! Resuming from epoch {start_epoch}, global step {global_step}")

# Calculate learning rate schedule parameters
total_num_epoch = 3
remaining_epochs = total_num_epoch - start_epoch
num_epoch_sesh = 1 # how many epochs you want to run in this session

steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
max_steps = steps_per_epoch * total_num_epoch
warmup_steps = int(max_steps * 0.05)
warmup_steps = max(1, warmup_steps)

max_lr = 3e-4
min_lr = max_lr * 0.1

print(f"Total optimization steps: {max_steps}")
print(f"Warmup steps: {warmup_steps}")
print(f"Remaining epochs: {remaining_epochs}")

# --- MODIFIED TRAIN_MODEL TO ACCEPT START VALUES ---
def train_model_resume(model, num_epochs, optimizer, tokenizer, train_loader, val_loader,
                       start_context, device, eval_freq, eval_iter,
                       gradient_accumulation_steps, warmup_steps, max_steps, min_lr, max_lr,
                       start_epoch=0, start_global_step=global_step):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = start_global_step

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        optimizer.zero_grad()

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

            is_accum_complete = (batch_idx + 1) % gradient_accumulation_steps == 0

            if is_accum_complete:
                iter_num = global_step + 1
                lr = get_lr(iter_num)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1} (step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}, "
                      f"LR {current_lr:.6f}")

        print('\n')
        generate_and_print_sample(start_context, model, tokenizer, device)
        print('\n')

    return train_losses, val_losses, track_tokens_seen, global_step

# Resume training
start_time = time.perf_counter()

train_losses, val_losses, tokens_seen, global_step = train_model_resume(
    model, num_epoch_sesh, optimizer, tokenizer, train_dataloader,
    val_dataloader, start_context="Every effort moves you",
    device=device, eval_freq=50, eval_iter=50,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps, max_steps=max_steps,
    min_lr=min_lr, max_lr=max_lr,
    start_epoch=start_epoch, start_global_step=global_step
)

end_time = time.perf_counter()
total_time = (end_time - start_time) / 60

print(f"total time required to train this model: {total_time:.2f} minutes.\n")

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

# Save final checkpoint
checkpoint = {
    'epoch': start_epoch + num_epoch_sesh,  # Save next epoch to start from
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # Important for mixed precision
    'global_step': global_step,
}

torch.save(checkpoint, 'checkpoints/gpt2(2).pth')
print('Checkpoint saved!')

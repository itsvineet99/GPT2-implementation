import torch
import time
import tiktoken
from src import (GPTConfig, GPTModel, 
                 create_dataloader, calc_loss_loader,
                 calc_loss_batch, evaluate_model,
                 generate_and_print_sample)


# training script
def train_model(model, epoch, optimizer,
                tokenizer, train_loader, 
                val_loader, start_context, 
                device, eval_freq, eval_iter):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(epoch):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch,
                                model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader,
                                                    val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"epoch {epoch+1} (step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"val loss {val_loss:.3f}")
                
        print('\n')
        generate_and_print_sample(start_context, model, tokenizer, device)
        print('\n')
    
    return train_losses, val_losses, track_tokens_seen

def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # reading data from file source
    file_path = "data/the-verdict.txt"
    with open(file_path, 'r') as file:
        raw_text = file.read()

    # split data for training and validation 
    train_ratio = 0.9
    splitting_idx = int(len(raw_text) * train_ratio)
    train_data = raw_text[:splitting_idx]
    val_data = raw_text[splitting_idx:]

    # creating dataloaders
    train_dataloader = create_dataloader(
        txt=train_data,
        context_lim=256,
        stride=256,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_dataloader = create_dataloader(
        txt=val_data,
        context_lim=256,
        stride=256,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    # initialize model
    torch.manual_seed(53)
    model = GPTModel(GPTConfig)
    model.to(device)

    #initialize optimizer and tokenizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0004,
        weight_decay=0.1
    )
    tokenizer = tiktoken.get_encoding('gpt2')

    num_epoch = 30

    start_time = time.perf_counter()

    train_losses, val_losses, tokens_seen = train_model(
        model, num_epoch, optimizer, tokenizer, train_dataloader,
        val_dataloader, start_context="Every effort moves you",
        device=device, eval_freq=5, eval_iter=5 
    )

    end_time = time.perf_counter()
    total_time = (end_time - start_time) / 60

    print(f"total time required to train thsi model: {total_time:.2f} minutes.")

if __name__ == "__main__":
    main()

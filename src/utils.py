import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# model generates text, given input ids
def generate_simple_text(idx, model, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        new_idx = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, new_idx), dim=1)

    return idx

# convert text into tokens ids (also tensor) and vice versa
def text_to_token_ids(text, tokenizer):
    tokens = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    tokens = torch.tensor(tokens).unsqueeze(0)
    return tokens

def token_ids_to_text(idx, tokenizer):
    text = tokenizer.decode(idx.squeeze(0).tolist())
    return text


# loss calculation functions
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        # This effectively treats every single token in the entire 
        # batch as an independent classification problem.
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    
    # working of for loop on dataloader 
    # link : https://x.com/x0_vineet/status/2012765474346418668?s=20
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches

# evaluation function to find loss on both train and validation set
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, 
                                      device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model,
                                     device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# sampling function i.e generates and prints text using model, 
# given a starting context

def generate_and_print_sample(start_context, model, 
                              tokenizer, device):
    model.eval()
    token_ids = text_to_token_ids(start_context, tokenizer).to(device)
    context_size = model.pos_emb.weight.shape[0]
    with torch.no_grad():
        token_ids = generate_simple_text(
            token_ids, model, max_new_tokens=50,context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# plotting losses 
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny() #1
    ax2.plot(tokens_seen, train_losses, alpha=0) #2
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

# sampling function with some sampling techniques
def generate(model, idx, context_size, max_new_tokens, 
             top_k=None, temperature=0.0, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_logits = top_logits[:, -1]
            logits = torch.where(
                logits < min_logits,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits/temperature
            probas = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probas, num_samples=1)
        else:
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)

        if next_idx == eos_id:
            break  

        idx = torch.cat((idx, next_idx), dim=1)
    
    return idx

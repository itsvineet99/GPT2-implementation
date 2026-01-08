import torch

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

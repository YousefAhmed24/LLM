import os
import torch
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from Model import GPTLanguageModel, block_size

load_dotenv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
HF_TOKEN = os.getenv("HF_TOKEN")

batch_size = int(os.getenv("PRETRAIN_BATCH_SIZE", 16))
learning_rate = float(os.getenv("PRETRAIN_LR", 3e-4))
max_steps = int(os.getenv("PRETRAIN_MAX_STEPS", 10000))
eval_interval = int(os.getenv("PRETRAIN_EVAL_INTERVAL", 500))
eval_iters = int(os.getenv("PRETRAIN_EVAL_ITERS", 50))
grad_accum_steps = int(os.getenv("GRAD_ACCUM_STEPS", 4))

torch.manual_seed(1337)

# 1. Tokenizer & Dataset Streaming
print("Loading tokenizer and dataset streams...")
enc = tiktoken.get_encoding("gpt2")

# Stream both splits independently
train_stream = load_dataset("roneneldan/TinyStories", streaming=True, split="train", token=HF_TOKEN)
val_stream = load_dataset("roneneldan/TinyStories", streaming=True, split="validation", token=HF_TOKEN)

train_iter = iter(train_stream)
val_iter = iter(val_stream)

# Separate buffers so evaluation batches do not corrupt training sequence flow
buffers = {
    'train': [],
    'val': []
}

def get_batch(split='train'):
    global train_iter, val_iter
    
    tokens_needed = batch_size * (block_size + 1)
    current_iter = train_iter if split == 'train' else val_iter
    buf = buffers[split]
    
    while len(buf) < tokens_needed:
        try:
            example = next(current_iter)
            buf.extend(enc.encode(example['text']))
        except StopIteration:
            # Refresh the stream if exhausted
            if split == 'train':
                train_iter = iter(train_stream)
                current_iter = train_iter
            else:
                val_iter = iter(val_stream)
                current_iter = val_iter
                
    chunk = buf[:tokens_needed]
    buffers[split] = buf[tokens_needed:]
    
    data_tensor = torch.tensor(chunk, dtype=torch.long).view(batch_size, block_size + 1)
    x = data_tensor[:, :-1].contiguous().to('cuda')
    y = data_tensor[:, 1:].contiguous().to('cuda')
    
    return x, y

# 2. Evaluation Metric Function
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 3. Model Initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Initializing model on GPU...")
m = GPTLanguageModel().to(device)

# Instantiate optimizer before loading the state dict
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Updated fallback to resume perfectly from the recent Colab disconnect
start_step = 18000

if os.path.exists('gpt_weights.pt'):
    # weights_only=False is required to load integer steps and optimizer tensors
    checkpoint = torch.load('gpt_weights.pt', map_location=device, weights_only=False)
    
    # Check if this is the new comprehensive dictionary format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        m.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"Loaded full checkpoint. Resuming training from step {start_step}...")
    # Fallback for a weights-only file
    else:
        m.load_state_dict(checkpoint)
        print(f"Loaded legacy weights-only checkpoint. Counter starting at {start_step}...")
else:
    print("No checkpoint found. Starting from scratch...")
    
# 4. Training Loop 
print(f"Starting {max_steps} step training loop with AMP...")
print(f"Effective batch size: {batch_size * grad_accum_steps} sequences per weight update.")

# Use start_step in the range parameters
for step in range(start_step, max_steps):
    # Evaluate Validation Loss
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss(m)
        print(f"Step {step:05d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
        # Comprehensive interval save
        torch.save({
            'step': step,
            'model_state_dict': m.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 'gpt_weights.pt')
        
    # Gradient Accumulation Micro-Steps
    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch('train')
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = m(xb, yb)
            # Scale the loss mathematically so it averages out over the 4 steps
            loss = loss / grad_accum_steps
            
        loss.backward()
        
    # Once 4 micro-steps are complete, update the weights and flush the memory
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# 5. Final Checkpoint
torch.save({
    'step': step,
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'gpt_weights.pt')
print("Training complete. Full state saved to 'gpt_weights.pt'.")
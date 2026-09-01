import os
import torch
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from Model import GPTLanguageModel, block_size

load_dotenv()

# Hyperparameters
batch_size = int(os.getenv("JFLEG_BATCH_SIZE", 8))
learning_rate = float(os.getenv("JFLEG_LR", 3e-5))
max_steps = int(os.getenv("JFLEG_MAX_STEPS", 1500))
eval_interval = int(os.getenv("JFLEG_EVAL_INTERVAL", 150))
eval_iters = int(os.getenv("JFLEG_EVAL_ITERS", 20))
grad_accum_steps = int(os.getenv("JFLEG_GRAD_ACCUM", 4))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
enc = tiktoken.get_encoding("gpt2")
eot_token = enc.eot_token

# 1. Load JFLEG Dataset
print("Loading JFLEG dataset from Hugging Face...")
jfleg = load_dataset("jhu-clsp/jfleg")
train_data = jfleg['validation']  # JFLEG uses validation/test splits
eval_data = jfleg['test']

def format_data(split_data):
    pairs = []
    for item in split_data:
        src = item['sentence'].strip()
        # Take the first human-annotated reference correction
        tgt = item['corrections'][0].strip()
        if src and tgt:
            pairs.append((src, tgt))
    return pairs

train_pairs = format_data(train_data)
eval_pairs = format_data(eval_data)

def get_batch(split='train'):
    data = train_pairs if split == 'train' else eval_pairs
    x_list, y_list = [], []

    while len(x_list) < batch_size:
        idx = torch.randint(len(data), (1,)).item()
        corrupted, clean = data[idx]

        prompt_str = f"### Input:\n{corrupted}\n\n### Correction:\n"
        target_str = f"{clean}<|endoftext|>"

        prompt_ids = enc.encode(prompt_str)
        target_ids = enc.encode(target_str, allowed_special={"<|endoftext|>"})

        full_seq = prompt_ids + target_ids
        if len(full_seq) > block_size + 1:
            continue

        x_seq = full_seq[:-1]
        y_seq = full_seq[1:]

        # Loss masking on prompt tokens
        mask_len = len(prompt_ids) - 1
        masked_y = [-100] * mask_len + y_seq[mask_len:]

        # Pad to block_size
        pad_len = block_size - len(x_seq)
        x_seq = x_seq + [eot_token] * pad_len
        masked_y = masked_y + [-100] * pad_len

        x_list.append(x_seq)
        y_list.append(masked_y)

    x_tensor = torch.tensor(x_list, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y_list, dtype=torch.long, device=device)
    return x_tensor, y_tensor

# 2. Load Base Model
print("Loading base checkpoint 'gpt_weights.pt'...")
m = GPTLanguageModel().to(device)
checkpoint = torch.load('gpt_weights.pt', map_location=device, weights_only=True)
if 'model_state_dict' in checkpoint:
    m.load_state_dict(checkpoint['model_state_dict'])
else:
    m.load_state_dict(checkpoint)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = m(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# 3. Training Loop
print(f"Starting JFLEG Fine-Tuning ({max_steps} steps)...")
for step in range(max_steps):
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        print(f"Step {step:04d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch('train')
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = m(xb, yb)
            loss = loss / grad_accum_steps
        loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

torch.save(m.state_dict(), 'grammar_gpt_weights.pt')
print("Fine-tuning complete. Checkpoint saved to 'grammar_gpt_weights.pt'.")
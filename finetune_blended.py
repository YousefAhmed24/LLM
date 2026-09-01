import os
import torch
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from Model import GPTLanguageModel, block_size

load_dotenv()

# 1. Hyperparameters (Increased for larger dataset)
batch_size = int(os.getenv("GEC_BATCH_SIZE", 8))
learning_rate = float(os.getenv("GEC_LR", 3e-5))
max_steps = int(os.getenv("GEC_MAX_STEPS", 8000))
eval_interval = int(os.getenv("GEC_EVAL_INTERVAL", 250))
eval_iters = int(os.getenv("GEC_EVAL_ITERS", 20))
grad_accum_steps = int(os.getenv("GEC_GRAD_ACCUM", 4))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
enc = tiktoken.get_encoding("gpt2")
eot_token = enc.eot_token

# 2. Blend Datasets
print("Downloading and blending JFLEG and C4_200M datasets...")
jfleg = load_dataset("jhu-clsp/jfleg", split="validation")
c4_subset = load_dataset("agentlans/grammar-correction", split="train[:20000]")

train_pairs = []

# Format JFLEG
for item in jfleg:
    src = item['sentence'].strip()
    tgt = item['corrections'][0].strip()
    if src and tgt:
        train_pairs.append((src, tgt))

# Format C4_200M
for item in c4_subset:
    src = item['input'].strip()
    tgt = item['output'].strip()
    if src and tgt:
        train_pairs.append((src, tgt))

print(f"Total blended training pairs ready: {len(train_pairs)}")

# 3. Batch Generator 
def get_batch():
    x_list, y_list = [], []
    while len(x_list) < batch_size:
        idx = torch.randint(len(train_pairs), (1,)).item()
        corrupted, clean = train_pairs[idx]

        prompt_str = f"### Input:\n{corrupted}\n\n### Correction:\n"
        target_str = f"{clean}<|endoftext|>"

        prompt_ids = enc.encode(prompt_str)
        target_ids = enc.encode(target_str, allowed_special={"<|endoftext|>"})

        full_seq = prompt_ids + target_ids
        if len(full_seq) > block_size + 1:
            continue

        x_seq = full_seq[:-1]
        y_seq = full_seq[1:]

        # Mask the prompt so it only learns from the correction
        mask_len = len(prompt_ids) - 1
        masked_y = [-100] * mask_len + y_seq[mask_len:]

        pad_len = block_size - len(x_seq)
        x_seq = x_seq + [eot_token] * pad_len
        masked_y = masked_y + [-100] * pad_len

        x_list.append(x_seq)
        y_list.append(masked_y)

    x_tensor = torch.tensor(x_list, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y_list, dtype=torch.long, device=device)
    return x_tensor, y_tensor

# 4. Model Setup
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
    m.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = m(x, y)
        losses[k] = loss.item()
    m.train()
    return losses.mean()

# 5. Training Loop
print(f"Starting Blended Fine-Tuning ({max_steps} steps)...")
for step in range(max_steps):
    if step % eval_interval == 0 or step == max_steps - 1:
        val_loss = estimate_loss()
        print(f"Step {step:04d} | Blended Dataset Loss: {val_loss:.4f}")

    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = m(xb, yb)
            loss = loss / grad_accum_steps
        loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

torch.save(m.state_dict(), 'grammar_gpt_weights.pt')
print("Fine-tuning complete. Checkpoint saved to 'grammar_gpt_weights.pt'.")
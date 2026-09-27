# Grammar Correction GPT

A ~45M parameter GPT model I built from scratch in PyTorch. It's pre-trained on TinyStories and then fine-tuned to correct grammatical errors in English sentences.

No HuggingFace model classes — the entire architecture (attention, MLP, transformer blocks) is written by hand in `Model.py`.

## Architecture

Standard GPT-2 style decoder-only transformer with pre-layer normalization:

- 6 transformer blocks, 512 embedding dim, 6 attention heads
- 256 token context window, GPT-2 BPE tokenizer (50,257 vocab via `tiktoken`)
- GELU activations, 4x MLP expansion
- FlashAttention via PyTorch's `scaled_dot_product_attention`
- Weight tying between token embeddings and the output projection

## Training

### Pre-training (`train.py`)

Trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset using streaming. Standard next-token prediction with AdamW (lr=3e-4), bfloat16 mixed precision, and gradient accumulation (effective batch size of 64). Saves full checkpoints (model + optimizer + step counter) to `gpt_weights.pt`.

### Fine-tuning for grammar correction

There are two fine-tuning scripts — pick whichever fits your needs:

**`finetune_synthetic.py`** — Fine-tunes on [JFLEG](https://huggingface.co/datasets/jhu-clsp/jfleg) only (~750 sentence pairs). Quick to run but limited data.

**`finetune_blended.py`** — Blends JFLEG with 20k pairs from [agentlans/grammar-correction](https://huggingface.co/datasets/agentlans/grammar-correction). Runs for 8k steps and generally produces better results.

Both use the same prompt format:
```
### Input:
{ungrammatical sentence}

### Correction:
{corrected sentence}<|endoftext|>
```

The loss is masked so the model only learns to generate the correction, not echo the prompt. Output goes to `grammar_gpt_weights.pt`.

## Results

Some examples from the interactive agent:

| Input | Output |
|---|---|
| She go to school yesterday. | She went to school yesterday. |
| I has a big dog. | I have a big dog. |
| The childs are playing. | The children are playing. |
| He don't knows the answer. | He doesn't know the answer. |

It handles subject-verb agreement, tense errors, pluralization, and auxiliary verb mistakes reasonably well.

## Usage

### Install dependencies

```bash
pip install torch tiktoken datasets python-dotenv
```

### Run the grammar correction agent

```bash
python agent.py
```

It loads `grammar_gpt_weights.pt` and gives you an interactive prompt to type sentences.

### Train from scratch

```bash
# pre-train on TinyStories
python train.py

# fine-tune (pick one)
python finetune_synthetic.py
python finetune_blended.py
```

### Configuration

Hyperparameters are read from environment variables (or a `.env` file). Defaults are sensible so you don't need to set anything to get started — check the top of each script to see what's configurable.

## Files

```
Model.py                 - GPT architecture
train.py                 - pre-training on TinyStories
finetune_synthetic.py    - fine-tune on JFLEG only
finetune_blended.py      - fine-tune on JFLEG + synthetic data
agent.py                 - interactive grammar correction CLI
gpt_weights.pt           - pre-trained checkpoint (~583 MB)
grammar_gpt_weights.pt   - fine-tuned checkpoint (~195 MB)
```
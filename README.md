# Grammar Correction GPT

A decoder-only transformer built from scratch in PyTorch and fine-tuned for English grammatical error correction. The project implements the model architecture directly—without Hugging Face model classes—then provides scripts for pre-training, task-specific fine-tuning, and interactive inference.

## Highlights

- Approximately 45 million parameters
- Custom causal self-attention, MLP, transformer block, and generation code
- GPT-2 BPE tokenization through `tiktoken`
- PyTorch scaled dot-product attention with causal masking
- Pre-training on the TinyStories dataset
- Prompt-masked fine-tuning on grammar-correction pairs
- Interactive command-line correction agent

## Architecture

The default configuration is a GPT-2-style, pre-layer-normalized transformer:

| Setting | Value |
|---|---:|
| Transformer blocks | 6 |
| Embedding dimension | 512 |
| Attention heads | 6 |
| Context length | 256 tokens |
| Vocabulary | 50,257 tokens |
| MLP expansion | 4× |
| Dropout | 0.1 |

Token embeddings are tied to the output projection. Generation uses temperature scaling and top-k sampling.

## Training pipeline

### 1. Pre-training

`train.py` streams [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and trains with next-token prediction, AdamW, bfloat16 autocasting, and gradient accumulation. Full checkpoints—including the model, optimizer, and step—are written to `gpt_weights.pt`.

### 2. Grammar fine-tuning

Two alternatives are included:

- `finetune_synthetic.py`: fine-tunes on [JFLEG](https://huggingface.co/datasets/jhu-clsp/jfleg)
- `finetune_blended.py`: combines JFLEG with 20,000 examples from [agentlans/grammar-correction](https://huggingface.co/datasets/agentlans/grammar-correction)

Both scripts use this prompt format:

```text
### Input:
{ungrammatical sentence}

### Correction:
{corrected sentence}<|endoftext|>
```

The prompt tokens are masked from the loss so that optimization focuses on the correction. The resulting task checkpoint is saved as `grammar_gpt_weights.pt`.

## Example outputs

| Input | Generated correction |
|---|---|
| `She go to school yesterday.` | `She went to school yesterday.` |
| `I has a big dog.` | `I have a big dog.` |
| `The childs are playing.` | `The children are playing.` |
| `He don't knows the answer.` | `He doesn't know the answer.` |

These examples illustrate the saved model's behavior; this small model can still produce incorrect or incomplete corrections.

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended for training.

```bash
pip install torch tiktoken datasets python-dotenv
```

Optional settings can be placed in a `.env` file. Configuration includes model dimensions, batch sizes, learning rates, step counts, evaluation intervals, gradient accumulation, sampling temperature, and top-k.

## Usage

Run the included fine-tuned checkpoint through the interactive agent:

```bash
python agent.py
```

Train the full pipeline:

```bash
python train.py
python finetune_blended.py
```

For the smaller JFLEG-only experiment, replace the second command with:

```bash
python finetune_synthetic.py
```

## Repository structure

```text
Model.py                 Transformer architecture and generation
train.py                 TinyStories pre-training
finetune_synthetic.py    JFLEG fine-tuning
finetune_blended.py      Blended grammar-correction fine-tuning
agent.py                 Interactive correction CLI
gpt_weights.pt           Base-language-model checkpoint
grammar_gpt_weights.pt   Fine-tuned grammar checkpoint
```

## Limitations

- The training scripts are designed around CUDA and bfloat16 execution.
- The 256-token context window is intended for short correction prompts.
- Evaluation is currently qualitative; the repository does not include a benchmark score on a held-out grammatical-error-correction test set.
- Model checkpoints are large and may be slow to download or load on CPU-only machines.
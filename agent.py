import os
import torch
import tiktoken
from dotenv import load_dotenv
from Model import GPTLanguageModel

load_dotenv()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")
temperature = float(os.getenv("TEMPERATURE", 0.2))
top_k = int(os.getenv("TOP_K", 40))

print(f"Booting up Agent on {device}...")

model = GPTLanguageModel().to(device)
try:
    model.load_state_dict(torch.load('grammar_gpt_weights.pt', map_location=device, weights_only=True))
    print("Grammar Corrector Agent ready.")
except FileNotFoundError:
    print("Error: 'grammar_gpt_weights.pt' not found. Run finetune_jfleg.py first.")
    exit(1)

model.eval()

def correct_text(raw_sentence):
    prompt = f"### Input:\n{raw_sentence.strip()}\n\n### Correction:\n"
    input_tokens = enc.encode(prompt)
    context = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=40, temperature=temperature, top_k=top_k)
        
    output_text = enc.decode(generated[0].tolist())
    
    if "### Correction:\n" in output_text:
        correction = output_text.split("### Correction:\n")[-1]
        
        # Force a hard stop at standard sentence boundaries or the eot_token
        for stop_char in ["<|endoftext|>", "\n", "###"]:
            if stop_char in correction:
                correction = correction.split(stop_char)[0]
                
        return correction.strip()
    return output_text

if __name__ == '__main__':
    while True:
        text = input("Sentence to correct (or 'exit'): ")
        if text.lower() == 'exit':
            break
        if text.strip():
            print(f"Fixed: {correct_text(text)}\n")
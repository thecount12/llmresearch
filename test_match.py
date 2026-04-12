import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

m = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float16)

input_ids = torch.tensor([[14990]])  # same single-token prompt as hello_plain.tok
with torch.no_grad():
    logits = model(input_ids).logits[0, -1]
next_id = int(logits.argmax())
print("greedy next id:", next_id, repr(tok.convert_ids_to_tokens(next_id)))


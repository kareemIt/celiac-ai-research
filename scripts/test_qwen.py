"""Load Qwen in 4-bit on the local GPU and run one inference. No hosted APIs."""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights must already be local; proves no network model use
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bitsandbytes as bnb
import torch

from src.models.qwen import BNB_CONFIG, MODEL_ID, chat_prompt, load_qwen_4bit

gib = lambda b: f"{b / 2**30:.2f} GiB"

t0 = time.perf_counter()
model, tok = load_qwen_4bit()
load_s = time.perf_counter() - t0

print(f"Model:            {MODEL_ID} ({type(model).__name__})")
# Params4bit stores two 4-bit weights per uint8 element, so numel() is half the real count.
n_params = sum(p.numel() * (2 if isinstance(p, bnb.nn.Params4bit) else 1) for p in model.parameters())
print(f"Parameters:       {n_params:,} (text model; vision tower not loaded)")
print(f"Quantization:     {BNB_CONFIG.to_dict()}")
print(f"Load time:        {load_s:.1f}s")
print(f"GPU mem (loaded): allocated {gib(torch.cuda.memory_allocated())}, reserved {gib(torch.cuda.memory_reserved())}")

prompt = "In two sentences, what is tissue transglutaminase (TG2) and why is it relevant to celiac disease?"
inputs = tok(chat_prompt(tok, prompt), return_tensors="pt").to("cuda")
# First call JIT-compiles/autotunes the Triton (fla) kernels; time it separately from steady-state speed.
t0 = time.perf_counter()
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(f"Warm-up (JIT):    {time.perf_counter() - t0:.1f}s")
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
torch.cuda.synchronize()
gen_s = time.perf_counter() - t0
new = out[0, inputs["input_ids"].shape[1]:]

print(f"Prompt:           {prompt}")
print(f"Output:           {tok.decode(new, skip_special_tokens=True).strip()}")
print(f"Inference:        {len(new)} tokens in {gen_s:.1f}s ({len(new) / gen_s:.1f} tok/s)")
print(f"Peak GPU mem:     {gib(torch.cuda.max_memory_allocated())}")
assert len(new) > 0
print("PASS")

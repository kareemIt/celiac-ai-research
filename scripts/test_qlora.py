"""QLoRA smoke test: real 4-bit fine-tuning on SYNTHETIC data to validate infrastructure only.

The synthetic task (made-up compound codes; prefix ZQ -> 'alpha', KX -> 'beta') has no scientific meaning.
The base model can't know the rule, so held-out accuracy rising shows the adapter learned it.
"""
import gc
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bitsandbytes as bnb
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from src.models.qwen import chat_prompt, load_qwen_4bit

OUT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "qlora_smoke")
STEPS, ACCUM, LR, MAX_LEN = 30, 4, 2e-4, 64
gib = lambda b: f"{b / 2**30:.2f} GiB"

random.seed(0)
torch.manual_seed(0)


def example(n):
    even = n % 2 == 0
    return f"Synthetic compound {'ZQ' if even else 'KX'}-{n}. Class? Answer alpha or beta.", "alpha" if even else "beta"


def encode(tok, q, a):
    p = tok(chat_prompt(tok, q), add_special_tokens=False)["input_ids"]
    t = tok(a + "<|im_end|>", add_special_tokens=False)["input_ids"]  # Qwen chat turns end with <|im_end|>
    ids = (p + t)[:MAX_LEN]
    labels = ([-100] * len(p) + t)[:MAX_LEN]  # loss on the answer only
    return torch.tensor([ids], device="cuda"), torch.tensor([labels], device="cuda")


def predict(model, tok, ns):
    model.eval()
    res = []
    for n in ns:
        q, a = example(n)
        x = tok(chat_prompt(tok, q), return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**x, max_new_tokens=4, do_sample=False)
        res.append((n, a, tok.decode(o[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()))
    return res


def accuracy(res):
    return sum(pred == a for _, a, pred in res) / len(res)  # exact match: also checks the model stops


model, tok = load_qwen_4bit()
print(f"Base loaded: GPU allocated {gib(torch.cuda.memory_allocated())}")

held_out = [random.randint(10_000, 99_999) for _ in range(10)]
base_acc = accuracy(predict(model, tok, held_out))

# Frozen 4-bit backbone + LoRA. Skips peft's prepare_model_for_kbit_training on purpose:
# it upcasts the 248k-vocab embeddings/lm_head to fp32 (~8 GB), which would not fit in 16 GB.
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False
lcfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules="all-linear"
)
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
trainable = [p for p in model.parameters() if p.requires_grad]
assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n), "backbone not frozen"

probe_name, probe = next((n, p) for n, p in model.named_parameters() if "lora_B" in n)
probe_before = probe.detach().clone()

opt = bnb.optim.PagedAdamW8bit(trainable, lr=LR)
train = [example(random.randint(0, 9_999)) for _ in range(STEPS * ACCUM)]

model.train()
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
losses = []
for step in range(STEPS):
    total = 0.0
    for q, a in train[step * ACCUM:(step + 1) * ACCUM]:
        ids, labels = encode(tok, q, a)
        loss = model(input_ids=ids, labels=labels).loss / ACCUM
        loss.backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    losses.append(total)
    if step % 5 == 0 or step == STEPS - 1:
        print(f"step {step:3d}  loss {total:.4f}")
torch.cuda.synchronize()
train_s = time.perf_counter() - t0
train_peak = torch.cuda.max_memory_allocated()

delta = (probe.detach() - probe_before).abs().max().item()
assert delta > 0, "LoRA weights did not change"
trained_acc = accuracy(predict(model, tok, held_out))

model.save_pretrained(OUT)
saved = probe.detach().cpu().clone()
del model, opt
gc.collect()
torch.cuda.empty_cache()

base, tok = load_qwen_4bit()
reloaded = PeftModel.from_pretrained(base, OUT)
re_probe = dict(reloaded.named_parameters())[probe_name].detach().cpu()
assert torch.equal(saved, re_probe), "reloaded adapter differs from saved one"
res = predict(reloaded, tok, held_out)
reload_acc = accuracy(res)

print("\n=== QLoRA smoke test (synthetic data, infrastructure check only) ===")
print(f"Optimizer steps:     {STEPS} x accum {ACCUM} (batch 1), max_len {MAX_LEN}, lr {LR}")
print(f"Loss:                {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"LoRA update:         max |dW| on {probe_name} = {delta:.3e}")
print(f"Training time:       {train_s:.1f}s ({train_s / STEPS:.2f}s/step)")
print(f"Peak VRAM training:  {gib(train_peak)}")
print(f"Adapter saved:       {os.path.abspath(OUT)}")
print(f"Reload check:        weights identical after reload")
print(f"Held-out synthetic accuracy: base {base_acc:.0%} -> trained {trained_acc:.0%} -> reloaded {reload_acc:.0%}")
for n, a, pred in res[:4]:
    print(f"  #{n}: expected {a!r}, got {pred!r}")
assert losses[-1] < losses[0], "loss did not decrease"
assert reload_acc >= 0.9, "reloaded adapter did not learn the synthetic rule"
print("PASS")

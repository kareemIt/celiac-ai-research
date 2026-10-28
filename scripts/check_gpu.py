"""Verify PyTorch can see the GPU and run a real training step on it."""
import sys
import time

import pynvml
import torch

if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda.is_available() is False")

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print(f"NVML GPU:          {pynvml.nvmlDeviceGetName(h)}")
print(f"NVML driver:       {pynvml.nvmlSystemGetDriverVersion()}")

props = torch.cuda.get_device_properties(0)
print(f"torch GPU:         {props.name}")
print(f"Total VRAM:        {props.total_memory / 2**30:.2f} GiB")
print(f"Compute cap:       {props.major}.{props.minor}")
print(f"torch / CUDA:      {torch.__version__} / {torch.version.cuda}")
print(f"cuda.is_available: {torch.cuda.is_available()}")

dev = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()

# Tensor math, checked against CPU.
a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
gpu = (a.to(dev) @ b.to(dev)).cpu()
assert torch.allclose(gpu, a @ b, rtol=1e-3, atol=1e-2), "GPU matmul disagrees with CPU"
print("Tensor matmul:     OK (matches CPU)")

# Tiny MLP: forward, backward, optimizer step; loss must drop.
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 1)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(1024, 256, device=dev)
y = x[:, :8].sum(1, keepdim=True)
before = model[0].weight.detach().clone()

t0 = time.perf_counter()
losses = []
for _ in range(200):
    loss = torch.nn.functional.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
torch.cuda.synchronize()

assert not torch.equal(before, model[0].weight), "weights did not change"
assert losses[-1] < losses[0] * 0.5, f"loss did not drop: {losses[0]:.4f} -> {losses[-1]:.4f}"
print(f"Training:          OK, loss {losses[0]:.4f} -> {losses[-1]:.4f} in 200 steps ({time.perf_counter() - t0:.2f}s)")
print(f"Peak GPU memory:   {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
print("PASS")

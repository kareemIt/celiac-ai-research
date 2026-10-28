"""Load the project's Qwen checkpoint in 4-bit NF4 on the GPU."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3.5-9B"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_qwen_4bit(model_id=MODEL_ID):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; refusing to run Qwen on CPU")
    tok = AutoTokenizer.from_pretrained(model_id)
    # device_map={"": 0} puts every layer on GPU 0: an OOM fails loudly instead of offloading to CPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=BNB_CONFIG, device_map={"": 0}, dtype=torch.bfloat16
    )
    off_gpu = {p.device.type for p in model.parameters()} - {"cuda"}
    if off_gpu:
        raise RuntimeError(f"model parameters landed on {off_gpu}, expected all on cuda")
    return model, tok


def chat_prompt(tok, text):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

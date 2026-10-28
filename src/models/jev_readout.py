"""Training-free Jev-style decision readout, following LLM-as-Jev (Li & Wagle, arXiv 2610.02076):
numbered options in the prompt, assistant turn prefilled with "Best answer: [", each option k scored as
the summed log-prob of the suffix "k]", softmax over options. No text is generated or parsed.
"""
import torch

from src.models.qwen import chat_prompt

PREFILL = "Best answer: ["


def build_prompt(tok, state, question, options):
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options, 1))
    text = f"{state}\n\nQuestion: {question}\nOptions:\n{opts}\nAnswer with the number of the best option."
    return chat_prompt(tok, text) + PREFILL


@torch.no_grad()
def option_logprobs(model, tok, prompt, n_options):
    """Log-prob of each option suffix 'k]' given the prompt, as a float tensor [n_options]."""
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    sufs = [tok(f"{k}]", add_special_tokens=False)["input_ids"] for k in range(1, n_options + 1)]
    # shortcut: equal-length suffixes (true for options 1-9) let all options share one batch;
    # >9 options needs the paper's prefix-trie scoring.
    assert len({len(s) for s in sufs}) == 1, "options >9 not supported"
    ids = torch.tensor([p + s for s in sufs], device="cuda")
    logp = model(input_ids=ids).logits[:, len(p) - 1:-1].float().log_softmax(-1)  # positions predicting suffix
    idx = torch.tensor(sufs, device="cuda")
    return logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1).sum(-1).cpu()

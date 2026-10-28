"""KL anchors from LLM-as-Jev (Li & Wagle, arXiv 2610.02076, sec. 3.4): three KL terms pulling the
fine-tuned model back towards the frozen reference p0, so that training the decision head does not
overwrite the base model's distributions.

    L_anchor = L_mass + L_out + L_pos          (paper eq. 3)
    L        = L_tree + lambda * L_anchor      (paper eq. 4, default lambda = 1)

  L_mass   binary KL between reference and model on legal-vs-illegal probability mass at a trie node
  L_out    KL over the renormalized illegal-exit-token distribution at that node
  L_pos    KL over full next-token distributions at context positions, each weighted 0.5

Every term is KL(p0 || p_theta) and is truncated to the reference's top-64 tokens, as in the paper.
shortcut: the paper samples three off-path trie nodes beside the gold path; a two-option,
single-digit decision has exactly one node, so there are no off-path nodes to sample. Revisit if the
option set ever grows past one token.
"""
import torch

TOPK = 64


def kl_top(ref_logits, logits, k=TOPK):
    """KL(p0 || p) restricted to the reference's top-k tokens, both renormalized onto that support.
    Trailing dim is the vocabulary; any leading dims are kept."""
    k = min(k, ref_logits.shape[-1])
    idx = ref_logits.topk(k, dim=-1).indices
    lp0 = ref_logits.gather(-1, idx).log_softmax(-1)
    lp = logits.gather(-1, idx).log_softmax(-1)
    return (lp0.exp() * (lp0 - lp)).sum(-1)


def anchor_loss(logits, ref_logits, decision, context, option_ids):
    """Scalar L_anchor for one example.

    logits, ref_logits: [positions, vocab] over the kept window, model and frozen reference.
    decision: window index of the position whose next token is the option digit.
    context:  window indices L_pos anchors (reply start, and the position after the closing bracket).
    option_ids: vocabulary ids of the legal option tokens at the decision node.
    """
    z, z0 = logits[decision].float(), ref_logits[decision].float()
    m = z.softmax(-1)[option_ids].sum().clamp(1e-6, 1 - 1e-6)
    m0 = z0.softmax(-1)[option_ids].sum().clamp(1e-6, 1 - 1e-6)
    l_mass = m0 * (m0 / m).log() + (1 - m0) * ((1 - m0) / (1 - m)).log()

    illegal = torch.ones(z.shape[-1], dtype=torch.bool, device=z.device)
    illegal[option_ids] = False
    l_out = kl_top(z0[illegal], z[illegal])

    l_pos = 0.5 * kl_top(ref_logits[context].float(), logits[context].float()).sum()
    return l_mass + l_out + l_pos

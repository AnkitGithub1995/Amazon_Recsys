"""Contrastive (InfoNCE) + calibrated BCE losses for LightGCN.

Replaces BPR with a dual objective:

  L = lam_nce * InfoNCE  +  lam_bce * CalibratedBCE

- InfoNCE (ranking/recall): softmax over [pos, sampled negs] with logQ correction
  so the sampled softmax approximates full-catalogue softmax (Bengio & Senecal;
  the "sampled softmax" bias correction). Drives retrieval quality.

- Calibrated BCE (pCTR for eCPM): BCE on the affine head sigmoid(a*dot + b).
  This head's a,b act as learned Platt scaling so outputs are true click
  probabilities. logQ / prior term keeps the base rate honest under sampling.
"""
import torch
import torch.nn.functional as F


def info_nce_loss(s_pos, s_neg, logq_pos, logq_neg, tau=0.1, correct_logq=True,
                  target=None):
    """
    s_pos:     (B,)      dot score user . pos_item
    s_neg:     (B, N)    dot scores user . neg_items
    logq_pos:  (B,)      log sampling prob of the positive item
    logq_neg:  (B, N)    log sampling prob of the negative items
    tau:       temperature
    correct_logq: subtract logq so popular items aren't unfairly penalised as
                  negatives (sampled-softmax correction). Turn OFF for uniform negs.
    target:    optional (B,) class indices. In THIS layout the logits are built as
               [pos | neg_1 ... neg_N], so the positive is column 0 BY CONSTRUCTION
               -- which item sits in that slot is decided by the data (the sampled
               (user, pos_item) pair). Hence the default target is 0. If you build
               the candidate list differently, pass the true index per row.
               For a fully data-driven diagonal target, use info_nce_inbatch_loss.

    Returns scalar InfoNCE loss.
    """
    if correct_logq:
        pos_logits = (s_pos - logq_pos).unsqueeze(1)   # (B,1)
        neg_logits = s_neg - logq_neg                  # (B,N)
    else:
        pos_logits = s_pos.unsqueeze(1)
        neg_logits = s_neg
    logits = torch.cat([pos_logits, neg_logits], dim=1) / tau  # (B, 1+N)
    if target is None:
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def info_nce_inbatch_loss(user_vecs, pos_item_vecs, logq=None, tau=0.1,
                          extra_neg_vecs=None, extra_logq=None):
    """
    In-batch-negatives InfoNCE where the target COMES FROM THE DATA, not a constant.

    For a batch of B aligned (user_i, pos_item_i) pairs we form the full
    similarity matrix S = U @ P^T of shape (B, B). Row i should match column i
    (its own paired positive); every other column is another user's positive and
    acts as a negative. The correct class per row is therefore its diagonal index:

        target = arange(B)

    i.e. the label is defined by how the dataset paired users to items, which is
    exactly the "label from the dataset" semantics. Optionally append shared extra
    (e.g. popularity-sampled) negatives, and apply logQ correction on the item
    (column) axis so popular positives-as-negatives are not over-penalised.

    user_vecs:      (B, d)
    pos_item_vecs:  (B, d)   item paired with each user (same order)
    logq:           (B,) log-sampling-prob of each in-batch item (column), optional
    extra_neg_vecs: (M, d)   extra shared negatives, optional
    extra_logq:     (M,) log-sampling-prob of extra negatives, optional
    """
    B = user_vecs.size(0)
    logits = user_vecs @ pos_item_vecs.t()             # (B, B)
    if logq is not None:
        logits = logits - logq.unsqueeze(0)            # correct along item columns
    if extra_neg_vecs is not None:
        extra = user_vecs @ extra_neg_vecs.t()         # (B, M)
        if extra_logq is not None:
            extra = extra - extra_logq.unsqueeze(0)
        logits = torch.cat([logits, extra], dim=1)     # (B, B+M)
    logits = logits / tau
    target = torch.arange(B, device=user_vecs.device)  # data-driven diagonal
    return F.cross_entropy(logits, target)


def calibrated_bce_loss(model, s_pos, s_neg, logq_pos=None, logq_neg=None,
                        prior_correction=False):
    """
    BCE on the calibration head sigmoid(a*dot + b).
    Positives -> 1, sampled negatives -> 0.

    prior_correction: if True, add logq to the logit of sampled negatives so the
    head learns the TRUE base rate rather than the sampled one (Elkan-style).
    On AmazonBook (no observed negatives) this is approximate; on H&M with real
    impressions it becomes exact and is the key to eCPM-correct pCTR.

    Returns (loss, pos_prob_mean, neg_prob_mean) for logging.
    """
    pos_logit = model.calibrated_logit(s_pos)            # (B,)
    neg_logit = model.calibrated_logit(s_neg)            # (B,N)

    if prior_correction and logq_neg is not None:
        # Undo the oversampling of popular negatives at the probability level.
        neg_logit = neg_logit + logq_neg

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logit, torch.ones_like(pos_logit))
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logit, torch.zeros_like(neg_logit))
    loss = pos_loss + neg_loss

    with torch.no_grad():
        pos_p = torch.sigmoid(pos_logit).mean()
        neg_p = torch.sigmoid(neg_logit).mean()
    return loss, pos_p, neg_p


def combined_loss(model, users, items, u_idx, ipos_idx, ineg_idx,
                  logq_pos, logq_neg, tau=0.1, lam_nce=1.0, lam_bce=1.0,
                  l2_reg=1e-4, correct_logq=True, prior_correction=False):
    """
    Full dual objective on cached graph embeddings.

    model:  LightGCN (provides calibrated_logit + embeddings for reg)
    users:  (num_users, d) cached propagated user embeddings
    items:  (num_items, d) cached propagated item embeddings
    *_idx:  batch index tensors on the same device
    Returns (total_loss, logs dict).
    """
    u = users[u_idx]                     # (B, d)
    ip = items[ipos_idx]                 # (B, d)
    ineg = items[ineg_idx]               # (B, N, d)

    s_pos = (u * ip).sum(-1)                          # (B,)
    s_neg = torch.einsum("bd,bnd->bn", u, ineg)       # (B, N)

    nce = info_nce_loss(s_pos, s_neg, logq_pos, logq_neg, tau=tau,
                        correct_logq=correct_logq)
    bce, pos_p, neg_p = calibrated_bce_loss(
        model, s_pos, s_neg, logq_pos, logq_neg, prior_correction=prior_correction)

    # L2 on the *initial* embeddings (LightGCN standard regularisation)
    reg = (model.user_emb(u_idx).pow(2).sum()
           + model.item_emb(ipos_idx).pow(2).sum()
           + model.item_emb(ineg_idx).pow(2).sum()) / u_idx.size(0)

    total = lam_nce * nce + lam_bce * bce + l2_reg * reg
    logs = {
        "loss": float(total.detach()),
        "nce": float(nce.detach()),
        "bce": float(bce.detach()),
        "reg": float(reg.detach()),
        "pos_prob": float(pos_p),
        "neg_prob": float(neg_p),
        "cal_scale": float(model.cal_scale.detach()),
        "cal_bias": float(model.cal_bias.detach()),
    }
    return total, logs

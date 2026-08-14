# import de librerias
import torch


def cox_ph_loss(
    risk_scores: torch.Tensor,  # (B,)  log-risk scores
    durations: torch.Tensor,  # (B,)  time from landmark to event/censor
    events: torch.Tensor,  # (B,)  1 = lung cancer, 0 = censored/other
    weights: torch.Tensor | None = None,  # (B,)  optional importance weights
) -> torch.Tensor:
    """
    Negative Cox partial log-likelihood.

    Reference: Cox (1972) "Regression models and life tables", JRSS-B.
    """
    # ── TODO 6 ── Cox partial log-likelihood ──────────────────────────────────
    # Implement the negative Cox partial log-likelihood.
    #
    # Step 1 — Sort by descending duration.
    #   After sorting, the risk set for patient i is exactly the prefix [0..i].
    #   Apply the same permutation to risk_scores, events, and weights.
    #
    durations, indices = torch.sort(durations, descending=True)
    risk_scores = risk_scores[indices]
    events = events[indices]
    if weights is not None:
        weights = weights[indices]
    #
    # Step 2 — Compute log of the cumulative risk-set sum (log-sum-exp trick).
    #   Hint: torch.cumsum on exp(risk_scores), then torch.log.
    #   Subtract max(risk_scores) before exp and add it back after log to avoid overflow.
    #
    max_risk = torch.max(risk_scores)
    exp_risk = torch.exp(risk_scores - max_risk)
    cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)
    log_cumsum_exp = torch.log(cumsum_exp_risk) + max_risk
    #
    # Step 3 — Per-patient log-likelihood contribution.
    #   For each patient: risk_score_i minus the log of the cumulative sum up to i.
    #   Only event patients (events == 1) contribute to the loss.
    #
    log_likelihood = risk_scores - log_cumsum_exp
    #
    # Step 4 — Normalise and negate.
    #   If weights are provided, weight each event patient's contribution
    #   by its sample weight before summing; normalise by the total weight.
    #   Otherwise normalise by the number of events.
    #   Return the negated scalar loss (we minimise, so we negate the log-likelihood).
    #
    if weights is not None:
        sum_weights = torch.sum(events * weights)
        if sum_weights == 0:
            loss = (risk_scores * 0.0).sum()
        else:
            loss = -torch.sum(log_likelihood * events * weights) / sum_weights
    else:
        sum_events = torch.sum(events)
        if sum_events == 0:
            loss = (risk_scores * 0.0).sum()
        else:
            loss = -torch.sum(log_likelihood * events) / sum_events
    return loss

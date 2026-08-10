from loss import cox_ph_loss


def run_raw_loss_epoch_icd(loader, model, device, optimizer=None):
    """
    One forward pass over the dataset.

    Returns (mean_loss, c_index).
    If optimizer is not None, also runs backward + step (training mode).

    Batch format: (labs, masks, deltas, lengths, sex_male, time_since_index,
                   durations, events, landmarks, weights, ids, flat_codes, offsets).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_risks, all_dur, all_ev = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            (labs, masks, deltas, lengths, sex_male,
             time_since_index, durations, events, landmarks, weights, ids, flat_codes, offsets) = batch

            labs             = labs.to(device)
            masks            = masks.to(device)
            deltas           = deltas.to(device)
            lengths          = lengths.to(device)
            sex_male         = sex_male.to(device)
            time_since_index = time_since_index.to(device)
            durations        = durations.to(device)
            events           = events.to(device)
            weights          = weights.to(device)
            flat_codes       = flat_codes.to(device)
            offsets          = offsets.to(device)

            # ── TODO 7 ── Forward pass, loss, and backward ───────────────────
            # 1. Call the model to get risk_scores.
            #    Signature: model(labs, masks, deltas, lengths, sex_male, time_since_index)
            #    Returns (last_risk, all_risks, h_states) — you only need the first.
            (risk_scores,_,_)= model(labs, masks, deltas, lengths, sex_male, time_since_index, flat_codes, offsets)
            #
            # 2. Compute the loss with cox_ph_loss(risk_scores, durations, events, weights).
            loss= cox_ph_loss(risk_scores, durations, events, weights)
            #
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            #

            total_loss += loss.item()
            all_risks.extend(risk_scores.detach().cpu().numpy())
            all_dur.extend(durations.cpu().numpy())
            all_ev.extend(events.cpu().numpy())

    mean_loss = total_loss / max(len(loader), 1)

    # ── TODO 8 ── C-index ────────────────────────────────────────────────────
    # Compute the concordance index (C-index) from the accumulated arrays:
    #   all_risks  — model log-risk scores  (higher = more risk)
    #   all_dur    — time from landmark to event or censoring
    #   all_ev     — event indicator (1 = lung cancer, 0 = censored)
    #
    # Use lifelines.utils.concordance_index(event_times, predicted_scores, event_observed).
    #
    # Think before you code:
    #   - concordance_index expects *higher* predicted_scores to mean *longer* survival.
    #     Our model outputs log-risk: higher risk → shorter survival.
    #     How should you transform all_risks before passing it?
    #   - The function expects event_observed as a boolean array.
    #
    ci = concordance_index(np.array(all_dur), -np.array(all_risks), np.array(all_ev, dtype=bool))
    return mean_loss, ci

from torch.utils.data import WeightedRandomSampler

# Event oversampling: give event patients EVENT_OVERSAMPLE× more draw probability
sample_weights = [EVENT_OVERSAMPLE if f == 1 else 1.0 for f in train_dataset_icd.event_flags]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset_icd, batch_size=BATCH_SIZE, sampler=sampler,
                          collate_fn=raw_pad_collate_with_icd)
val_loader   = DataLoader(val_dataset_icd,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=raw_pad_collate_with_icd)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

best_val_loss = float("inf")
best_state    = None
no_improve    = 0

for epoch in range(1, MAX_EPOCHS + 1):

    # ── TODO 5 ── Training step ───────────────────────────────────────────────
    # Call run_raw_loss_epoch in training mode to run one full epoch and get
    # (train_loss, train_ci). Replace the two placeholder lines below.
    #
    # Hint: check the function signature above — what argument switches it
    #       between training and evaluation mode?
    #
    train_loss, train_ci = run_raw_loss_epoch_icd(train_loader, model, device, optimizer=optimizer)  # replace with the real call
    # ─────────────────────────────────────────────────────────────────────────

    val_loss, val_ci = run_raw_loss_epoch_icd(val_loader, model, device, optimizer=None)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  "
          f"train_loss={train_loss:.4f}  train_ci={train_ci:.4f}  "
          f"val_loss={val_loss:.4f}  val_ci={val_ci:.4f}")

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state    = copy.deepcopy(model.state_dict())
        no_improve    = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  → early stopping at epoch {epoch}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

print(f"\nBest val loss: {best_val_loss:.4f}")


def compute_baseline_hazard(risks, durations, events, horizon):
    """Breslow estimator: H0(horizon) from training data."""
    risks     = np.asarray(risks,     dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    events    = np.asarray(events,    dtype=np.float64)

    max_r   = risks.max()
    shifted = risks - max_r
    H0 = 0.0
    for t in np.unique(durations[events == 1]):
        if t > horizon:
            continue
        d        = np.sum((durations == t) & (events == 1))
        risk_set = durations >= t
        denom    = np.sum(np.exp(shifted[risk_set]))
        if denom > 0:
            H0 += d / denom
    return H0, max_r


def collect_raw_outputs(loader, model, device):
    """
    Collect model outputs over a full loader.

    Returns (risks, durations, events, landmarks) as numpy arrays.
    """
    model.eval()
    risks, durs, evs, lms = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            (labs, masks, deltas, lengths, sex_male,
             time_since_index, durations, events, landmarks, weights, ids, flat_codes, offsets) = batch
            r, _, _ = model(
                labs.to(device), masks.to(device), deltas.to(device),
                lengths.to(device), sex_male.to(device), time_since_index.to(device), flat_codes.to(device), offsets.to(device)
            )
            risks.extend(r.cpu().numpy())
            durs.extend(durations.numpy())
            evs.extend(events.numpy())
            lms.extend(landmarks.numpy())
    return (np.array(risks), np.array(durs),
            np.array(evs, dtype=float), np.array(lms))
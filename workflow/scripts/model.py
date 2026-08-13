import numpy as np
import torch
import torch.nn as nn


class TimeAwareGRUD(nn.Module):
    """
    GRU-D: GRU with temporal decay for irregular, partially-observed time series.

    forward() returns (last_risk, all_risks, h_states):
    - last_risk   (B,)      — risk score at each patient's final observed timestep
    - all_risks   (B, T)    — risk score at every timestep
    - h_states    (B, T, H) — hidden states at every timestep
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int = 0,
        mean_x: np.ndarray | None = None,
        embed_dim: int = 0,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        embed_static_icd: bool = False,  # <-- NUEVO PARÁMETRO DE INICIALIZACIÓN
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embed_static_icd = embed_static_icd  # <-- Guardamos la configuración

        if mean_x is None:
            mean_x = np.zeros(input_dim, dtype=np.float32)
        self.register_buffer("mean_x", torch.tensor(mean_x, dtype=torch.float32))

        # Optional ICD-10 embedding
        if embed_dim > 0 and vocab_size > 0:
            self.embedding = nn.EmbeddingBag(
                vocab_size, embed_dim, mode="mean", padding_idx=0
            )
        else:
            self.embedding = None

        # Decay networks: scalar gap → per-feature decay rates
        self.decay_x = nn.Linear(1, input_dim)
        self.decay_h = nn.Linear(1, hidden_dim)

        # Ajuste dinámico de dimensiones de entrada de la celda GRU
        if self.embed_static_icd:
            # Caso Estático: El embedding no entra en la secuencia temporal de la GRU
            gru_input_dim = input_dim + input_dim + 2
        else:
            # Caso Secuencial: El embedding se concatena en cada timestep
            gru_input_dim = input_dim + input_dim + 2 + embed_dim

        self.gru_cell = nn.GRUCell(gru_input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Ajuste dinámico de la capa final de salida (Linear classifier)
        if self.embed_static_icd and embed_dim > 0 and vocab_size > 0:
            # Si es estático, la entrada será la concatenación de h_T e icd_vec (hidden_dim + embed_dim)
            self.linear = nn.Linear(hidden_dim + embed_dim, 1)
        else:
            # Si no, la entrada es solo h_T (hidden_dim)
            self.linear = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x,  # (B, T, D)  padded lab values
        mask,  # (B, T, D)  observation masks
        deltas,  # (B, T)     inter-observation gaps in years
        lengths,  # (B,)       true sequence lengths
        sex_male,  # (B,)
        time_since_index=None,  # (B, T)  optional
        flat_codes=None,  # EmbeddingBag input  (extension)
        offsets=None,  # EmbeddingBag offsets (extension)
        embed_static_icd=None,
    ):
        batch_size, max_len, _ = x.shape
        device = x.device

        # Si no se especifica en forward, heredamos la configuración del __init__
        if embed_static_icd is None:
            embed_static_icd = self.embed_static_icd

        # Procesamiento del Embedding de diagnósticos
        bag_embeds = None
        if self.embedding is not None and flat_codes is not None:
            if embed_static_icd:
                # Retorna un único vector agregado por paciente: (B, embed_dim)
                bag_embeds = self.embedding(flat_codes, offsets)
            else:
                # Retorna un vector por cada timestep de la secuencia: (B, T, embed_dim)
                bag_embeds = self.embedding(flat_codes, offsets).view(
                    batch_size, max_len, -1
                )

        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        x_last = self.mean_x.unsqueeze(0).repeat(batch_size, 1)  # (B, D)
        outputs = []
        h_states = []

        for t in range(max_len):
            delta_t = deltas[:, t : t + 1]  # (B, 1)
            x_t = x[:, t, :]  # (B, D)  — 0 where lab not observed
            m_t = mask[:, t, :]  # (B, D)

            # ── TODO 1 ── Lab-value decay
            gamma_x = torch.exp(-torch.clamp(self.decay_x(delta_t), min=0.0))

            # ── TODO 2 ── Hidden-state decay
            gamma_h = torch.exp(-torch.clamp(self.decay_h(delta_t), min=0.0))

            # ── TODO 3 ── Impute missing values
            x_hat = m_t * x_t + (1.0 - m_t) * (
                gamma_x * x_last + (1.0 - gamma_x) * self.mean_x
            )

            # Update x_last
            x_last = m_t * x_t + (1.0 - m_t) * x_last

            # ── TODO 4 ── Decay the hidden state
            h_t = gamma_h * h_t

            # ── GRU cell update ──
            sex_t = sex_male[:, None]  # (B, 1)
            time_t = (
                time_since_index[:, t : t + 1]
                if time_since_index is not None
                else torch.zeros(batch_size, 1, device=device)
            )
            parts = [x_hat, m_t, sex_t, time_t]

            # Solo añadimos el embedding al paso temporal si no es estático
            if bag_embeds is not None and not embed_static_icd:
                parts.append(bag_embeds[:, t, :])

            input_t = torch.cat(parts, dim=-1)
            h_t = self.gru_cell(input_t, h_t)

            # ── Salida Linear Head ──
            if embed_static_icd and bag_embeds is not None:
                # Para el caso estático, concatenamos h_t con el embedding global
                h_t_out = torch.cat([h_t, bag_embeds], dim=-1)
            else:
                h_t_out = h_t

            outputs.append(self.linear(self.dropout(h_t_out)))
            h_states.append(h_t)

        all_risks = torch.stack(outputs, dim=1).squeeze(-1)  # (B, T)
        h_states = torch.stack(h_states, dim=1)  # (B, T, H)

        # Risk score at each patient's last valid (non-padded) timestep
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=device)
        last_risk = all_risks[batch_idx, last_idx]  # (B,)

        return last_risk, all_risks, h_states

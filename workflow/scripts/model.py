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


class SimpleLSTM(nn.Module):
    """LSTM básica sin decaimento para baseline."""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int = 0,
        mean_x: np.ndarray | None = None,
        embed_dim: int = 0,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

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

        lstm_input_dim = input_dim + input_dim + 1 + embed_dim
        self.lstm_cell = nn.LSTMCell(lstm_input_dim, hidden_dim)
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
    ):
        batch_size, max_len, _ = x.shape
        device = x.device

        # Procesamiento del Embedding de diagnósticos
        bag_embeds = None
        if self.embedding is not None and flat_codes is not None:
            # Retorna un vector por cada timestep de la secuencia: (B, T, embed_dim)
            bag_embeds = self.embedding(flat_codes, offsets).view(
                batch_size, max_len, -1
            )

        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        outputs = []
        h_states = []

        for t in range(max_len):
            x_t = x[:, t, :]  # (B, D)  — 0 where lab not observed
            m_t = mask[:, t, :]  # (B, D)

            # Impute missing values with the mean
            x_hat = m_t * x_t + (1.0 - m_t) * self.mean_x

            # ── LSTM cell update ──
            sex_t = sex_male[:, None]  # (B, 1)
            parts = [x_hat, m_t, sex_t]

            # Solo añadimos el embedding al paso temporal si no es estático
            if bag_embeds is not None:
                parts.append(bag_embeds[:, t, :])

            input_t = torch.cat(parts, dim=-1)
            h_t, c_t = self.lstm_cell(input_t, (h_t, c_t))

            outputs.append(self.linear(self.dropout(h_t)))
            h_states.append(h_t)

        all_risks = torch.stack(outputs, dim=1).squeeze(-1)  # (B, T)
        h_states = torch.stack(h_states, dim=1)  # (B, T, H)

        # Risk score at each patient's last valid (non-padded) timestep
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=device)
        last_risk = all_risks[batch_idx, last_idx]  # (B,)

        return last_risk, all_risks, h_states


# Arquitectura STraTS

# Definimos CVE y FusionAtt tal y como lo hacen en el repositorio del paper


class CVE(nn.Module):
    """
    Continuous Value Embedding module (MLP 2 capas: 1 -> sqrt(d) -> d).
    """

    def __init__(self, hid_dim: int):
        super().__init__()
        int_dim = max(int(np.sqrt(hid_dim)), 1)
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        # x: (bsz, max_len)
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x, self.W1) + self.b1[None, None, :]  # (bsz, max_len, int_dim)
        x = self.activation(x)
        x = torch.matmul(x, self.W2)  # (bsz, max_len, hid_dim)
        return x


class FusionAtt(nn.Module):
    """
    Fusion Self-Attention pooling module (pondera tripletas observadas para obtener e^T).
    """

    def __init__(self, hid_dim: int):
        super().__init__()
        int_dim = hid_dim
        self.W = nn.Parameter(torch.empty(hid_dim, int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(int_dim, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x, mask):
        # x: (bsz, max_len, hid_dim)
        # mask: (bsz, max_len) — 1.0 para tokens reales, 0.0 para padding
        att = torch.matmul(x, self.W) + self.b[None, None, :]  # (bsz, max_len, int_dim)
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]  # (bsz, max_len)
        att = att + (1.0 - mask.float()) * -1e9  # Enmascarar padding
        att = torch.softmax(att, dim=-1)  # (bsz, max_len)
        return att


# Implementamos STraTS adaptandolo para que guarde similitud con los argumentos de entrada y las salidas que espera nuestro pipeline.


class STraTS(nn.Module):
    """
    Modelo STraTS completo para predicción de supervivencia.
    """

    def __init__(
        self,
        vocab_size: int,
        lab_dim: int,
        hid_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.2,
        demo_dim: int = 8,
        hidden_dim: int | None = None,
    ):
        if hidden_dim is not None:
            hid_dim = hidden_dim
        super().__init__()
        self.vocab_size = vocab_size
        self.lab_dim = lab_dim
        self.hid_dim = hid_dim

        # 1. Embeddings de tripletas
        self.time_cve = CVE(hid_dim)
        self.feature_emb = nn.Embedding(vocab_size, hid_dim)
        self.val_cve = CVE(hid_dim)
        self.dropout = nn.Dropout(dropout)

        # 2. Bloques de Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 3. Fusion Attention Pooling
        self.fusion_att = FusionAtt(hid_dim)

        # 4. Demographics Embedding (sex_male)
        self.demo_fc = nn.Linear(1, demo_dim)

        # 5. Cabeza de predicción de riesgo
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_dim + demo_dim, 1),
        )

    def forward(
        self,
        values,  # (B, N_max) float
        features,  # (B, N_max) long
        times,  # (B, N_max) float (edad exacta del evento)
        time_since_index,  # (B, N_max) float
        padding_mask,  # (B, N_max) bool (True for padding)
        lengths,  # (B,) long
        sex_male,  # (B,) float
        **kwargs,
    ):
        # 1. Embeddings de los componentes de la tripleta (t, f, v)
        t_emb = self.time_cve(times)  # (B, N_max, hid_dim)
        f_emb = self.feature_emb(features)  # (B, N_max, hid_dim)
        v_emb = self.val_cve(values)  # (B, N_max, hid_dim)

        # Para diagnósticos ICD-10 (features >= lab_dim), anulamos el embedding continuo
        is_lab = (features < self.lab_dim).unsqueeze(-1).float()
        v_emb = (
            v_emb * is_lab
        )  # No tienen sentido los valores de continuo para los diagnosticos que se representan como valores binarios.

        triplet_emb = self.dropout(t_emb + f_emb + v_emb)  # (B, N_max, hid_dim)

        # 2. Transformer Encoder con máscara de padding
        out = self.transformer_encoder(
            triplet_emb, src_key_padding_mask=padding_mask
        )  # (B, N_max, hid_dim)

        # 3. Fusion Self-Attention pooling -> e^T
        real_mask = (~padding_mask).float()
        att_weights = self.fusion_att(out, real_mask)  # (B, N_max)
        e_T = torch.sum(out * att_weights.unsqueeze(-1), dim=1)  # (B, hid_dim)

        # 4. Demographics embedding e^d
        e_d = torch.tanh(self.demo_fc(sex_male.unsqueeze(-1)))  # (B, demo_dim)

        # 5. Concatenación [e^T, e^d] y proyección a riesgo
        e = torch.cat([e_T, e_d], dim=-1)  # (B, hid_dim + demo_dim)
        last_risk = self.head(e).squeeze(-1)  # (B,)

        all_risks = None  # Como transformer se ejecuta de manera paralela, no tiene pasos temporales definidos, por lo tanto devolvemos None
        h_states = out  # out ya son los estados de cada observacion, es decir, h_t

        return last_risk, all_risks, h_states

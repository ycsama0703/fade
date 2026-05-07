"""
Deep learning survival models.

D1  DeepSurv     — MLP risk encoder + Cox partial-likelihood loss
D2  LSTMSurv     — LSTM over raw IC series + Cox partial-likelihood loss
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Shared Cox partial-likelihood loss
# ---------------------------------------------------------------------------

def cox_partial_likelihood_loss(
    log_risk: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Breslow approximation of the Cox partial likelihood."""
    order = torch.argsort(durations, descending=True)
    log_risk = log_risk[order]
    events   = events[order].float()
    log_cumsum = torch.logcumsumexp(log_risk, dim=0)
    loss = -torch.mean((log_risk - log_cumsum) * events)
    return loss


def concordance_index_torch(
    log_risk: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> float:
    from lifelines.utils import concordance_index as _ci
    # higher log_risk = higher hazard = shorter survival → negate for CI
    return float(_ci(durations, -log_risk, events))


# ---------------------------------------------------------------------------
# D1: DeepSurv
# ---------------------------------------------------------------------------

class _DeepSurvNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeepSurv:
    def __init__(
        self,
        hidden_dims: list[int] = (64, 32),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 64,
        patience: int = 20,
        random_state: int = 42,
    ):
        self.hidden_dims  = list(hidden_dims)
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.patience     = patience
        self.random_state = random_state
        self.net_: _DeepSurvNet | None = None
        self.feature_columns_: list[str] = []
        self.x_mean_: np.ndarray | None = None
        self.x_std_:  np.ndarray | None = None

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        df = pd.get_dummies(X.copy(), drop_first=True).astype(float)
        df = df.reindex(columns=self.feature_columns_, fill_value=0)
        arr = df.values.astype(np.float32)
        return (arr - self.x_mean_) / (self.x_std_ + 1e-8)

    def fit(
        self,
        X: pd.DataFrame,
        durations: pd.Series,
        events: pd.Series,
        X_val: pd.DataFrame | None = None,
        dur_val: pd.Series | None = None,
        evt_val: pd.Series | None = None,
    ):
        torch.manual_seed(self.random_state)
        df = pd.get_dummies(X.copy(), drop_first=True).astype(float)
        self.feature_columns_ = list(df.columns)
        arr = df.values.astype(np.float32)
        self.x_mean_ = arr.mean(0)
        self.x_std_  = arr.std(0)
        arr = (arr - self.x_mean_) / (self.x_std_ + 1e-8)

        self.net_ = _DeepSurvNet(arr.shape[1], self.hidden_dims, self.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        Xt = torch.tensor(arr)
        dt = torch.tensor(durations.values, dtype=torch.float32)
        et = torch.tensor(events.values,    dtype=torch.float32)

        best_val_ci = -1.0
        best_state  = None
        no_improve  = 0

        for epoch in range(self.epochs):
            self.net_.train()
            # Full-batch (small dataset)
            opt.zero_grad()
            loss = cox_partial_likelihood_loss(self.net_(Xt), dt, et)
            loss.backward()
            opt.step()

            # Early stopping on validation C-index
            if X_val is not None and (epoch + 1) % 5 == 0:
                self.net_.eval()
                with torch.no_grad():
                    Xv = torch.tensor(self._prep(X_val))
                    risk_v = self.net_(Xv).numpy()
                val_ci = concordance_index_torch(risk_v, dur_val.values, evt_val.values)
                if val_ci > best_val_ci:
                    best_val_ci = val_ci
                    best_state  = {k: v.clone() for k, v in self.net_.state_dict().items()}
                    no_improve  = 0
                else:
                    no_improve += 5
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def _risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            Xp = torch.tensor(self._prep(X))
            return self.net_(Xp).numpy()

    def predict_median_survival(self, X: pd.DataFrame) -> np.ndarray:
        # We don't have a baseline hazard → return negative risk as proxy.
        # For consistent units, we'll calibrate in evaluation via concordance only.
        return -self._risk_scores(X)

    def concordance_index(self, X, durations, events) -> float:
        risk = self._risk_scores(X)
        return concordance_index_torch(risk, np.asarray(durations), np.asarray(events))


# ---------------------------------------------------------------------------
# D2: LSTMSurv
# ---------------------------------------------------------------------------

class _LSTMSurvNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x: (B, T, 1)  lengths: (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = self.drop(h_n[-1])          # last layer hidden state
        return self.head(h).squeeze(-1)


class LSTMSurv:
    """
    LSTM over the raw IC time series (first `seq_len` months post-discovery).

    Bypasses manual feature engineering: the model learns its own representation
    of early IC dynamics directly from the sequence.
    """

    def __init__(
        self,
        seq_len: int = 12,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        epochs: int = 200,
        patience: int = 30,
        random_state: int = 42,
    ):
        self.seq_len      = seq_len
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.patience     = patience
        self.random_state = random_state
        self.net_: _LSTMSurvNet | None = None

    @staticmethod
    def _build_sequences(
        factor_ids: list[str],
        ic_panel: pd.DataFrame,
        discovery_dates: dict[str, str],
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        seqs    : (N, seq_len) float32 — padded IC sequences
        lengths : (N,) int64   — actual (unpadded) length per factor
        """
        seqs, lengths = [], []
        for fid in factor_ids:
            grp = ic_panel[ic_panel["factor_id"] == fid].copy()
            grp["date"] = pd.to_datetime(grp["date"])
            if fid in discovery_dates:
                disc = pd.Timestamp(discovery_dates[fid])
                grp  = grp[grp["date"] >= disc]
            ic = grp.sort_values("date")["ic"].dropna().values[:seq_len]
            L  = len(ic)
            pad = np.zeros(seq_len, dtype=np.float32)
            if L > 0:
                pad[:L] = ic.astype(np.float32)
            seqs.append(pad)
            lengths.append(max(L, 1))
        return np.stack(seqs), np.array(lengths, dtype=np.int64)

    def fit(
        self,
        factor_ids_tr: list[str],
        ic_panel: pd.DataFrame,
        discovery_dates: dict[str, str],
        durations: pd.Series,
        events: pd.Series,
        factor_ids_val: list[str] | None = None,
        dur_val: pd.Series | None = None,
        evt_val: pd.Series | None = None,
    ):
        torch.manual_seed(self.random_state)

        seqs_tr, lens_tr = self._build_sequences(
            factor_ids_tr, ic_panel, discovery_dates, self.seq_len
        )
        Xs = torch.tensor(seqs_tr).unsqueeze(-1)   # (N, T, 1)
        Ls = torch.tensor(lens_tr)
        dt = torch.tensor(durations.values, dtype=torch.float32)
        et = torch.tensor(events.values,    dtype=torch.float32)

        self.net_ = _LSTMSurvNet(self.hidden_dim, self.num_layers, self.dropout)
        opt = torch.optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_ci = -1.0
        best_state  = None
        no_improve  = 0

        for epoch in range(self.epochs):
            self.net_.train()
            opt.zero_grad()
            log_risk = self.net_(Xs, Ls)
            loss = cox_partial_likelihood_loss(log_risk, dt, et)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
            opt.step()

            if factor_ids_val is not None and (epoch + 1) % 5 == 0:
                self.net_.eval()
                with torch.no_grad():
                    sv, lv = self._build_sequences(
                        factor_ids_val, ic_panel, discovery_dates, self.seq_len
                    )
                    rv = self.net_(
                        torch.tensor(sv).unsqueeze(-1),
                        torch.tensor(lv),
                    ).numpy()
                val_ci = concordance_index_torch(rv, dur_val.values, evt_val.values)
                if val_ci > best_val_ci:
                    best_val_ci = val_ci
                    best_state  = {k: v.clone() for k, v in self.net_.state_dict().items()}
                    no_improve  = 0
                else:
                    no_improve += 5
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def _risk_scores(
        self,
        factor_ids: list[str],
        ic_panel: pd.DataFrame,
        discovery_dates: dict[str, str],
    ) -> np.ndarray:
        self.net_.eval()
        seqs, lens = self._build_sequences(factor_ids, ic_panel, discovery_dates, self.seq_len)
        with torch.no_grad():
            return self.net_(
                torch.tensor(seqs).unsqueeze(-1),
                torch.tensor(lens),
            ).numpy()

    def predict_median_survival(
        self,
        factor_ids: list[str],
        ic_panel: pd.DataFrame,
        discovery_dates: dict[str, str],
    ) -> np.ndarray:
        return -self._risk_scores(factor_ids, ic_panel, discovery_dates)

    def concordance_index(
        self,
        factor_ids: list[str],
        ic_panel: pd.DataFrame,
        discovery_dates: dict[str, str],
        durations,
        events,
    ) -> float:
        risk = self._risk_scores(factor_ids, ic_panel, discovery_dates)
        return concordance_index_torch(risk, np.asarray(durations), np.asarray(events))

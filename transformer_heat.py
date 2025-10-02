# =============================================================================
# Transformer Forecasting for the 1D Heat Equation u_t = alpha * u_xx
# =============================================================================
# This script:
#   1) Generates training/validation data from a stable explicit FD solver.
#   2) Trains a small Transformer to predict the next H time steps from the
#      previous T steps (sequence-to-sequence in time).
#   3) Runs physics-oriented checks: boundary conditions, PDE residual,
#      and energy (L2) monotonicity.
#
# Rationale (what correctness means here):
#   - The FD "teacher" integrates u_t = alpha u_xx with Dirichlet BCs using a
#     stable time step (CFL). This is the ground-truth.
#   - The Transformer is causal and learns a time-advancement operator \mathcal{T}.
#   - After training, predictions should respect: small BC error, small PDE residual,
#     decreasing L2 norm in time (diffusion), and better MSE than a trivial baseline.
#
# Design choices:
#   - We treat each *time slice* (a length-nx vector) as a "token".
#   - The encoder consumes T tokens; the decoder emits H tokens with a causal mask.
#   - We clamp boundaries to 0 during training & rollout to enforce BCs explicitly.
#     (You can disable this clamp to study learned BC behavior; BC error will grow.)
#
# Verifiability:
#   - The script prints numbers for MSE, PDE residual, BC error, and L2 trend.
#   - You can change seeds, sizes, and see consistent qualitative behavior.
# =============================================================================

import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Reproducibility and utilities
# ----------------------------
def set_seed(seed: int = 1234) -> None:
    """Set RNG seeds for reproducible runs (best-effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# =============================================================================
# Ground-Truth Simulator: Explicit Finite-Difference for u_t = alpha u_xx
# =============================================================================

def make_initial_condition(nx: int, kind: str = "sines") -> np.ndarray:
    """
    Construct a random initial condition on [0,1] with Dirichlet BCs u(0)=u(1)=0.

    Parameters
    ----------
    nx : int
        Number of spatial grid points (including boundaries).
    kind : {'sines', 'gaussians'}
        Two families of smooth ICs that decay under diffusion.

    Returns
    -------
    u0 : (nx,) float32
        Initial field vector with endpoints set to exactly zero.
    """
    x = np.linspace(0.0, 1.0, nx)
    u = np.zeros_like(x, dtype=np.float32)

    if kind == "sines":
        # Sum a few sine modes (exactly satisfy BCs)
        modes = np.random.randint(1, 5)
        for _ in range(modes):
            k = np.random.randint(1, 6)               # frequency
            amp = np.random.uniform(-1.0, 1.0)        # amplitude
            u += amp * np.sin(k * math.pi * x)
    elif kind == "gaussians":
        # One or two Gaussian bumps; force endpoints to zero
        bumps = np.random.randint(1, 3)
        for _ in range(bumps):
            mu = np.random.uniform(0.2, 0.8)
            sig = np.random.uniform(0.03, 0.12)
            amp = np.random.uniform(-1.0, 1.0)
            u += amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    else:
        raise ValueError("Unknown IC kind. Use 'sines' or 'gaussians'.")

    # Enforce Dirichlet BCs exactly
    u[0] = 0.0
    u[-1] = 0.0
    return u.astype(np.float32)


def simulate_heat_equation(
    alpha: float,
    nx: int = 64,
    nt: int = 128,
    dt: float = None,
    ic_kind: str = "sines",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Explicit FD time-march for u_t = alpha * u_xx on x in [0,1] with BCs u(0)=u(1)=0.

    Scheme (Forward Euler in time, centered second derivative in space):
        u^{n+1}_i = u^n_i + dt * alpha * (u^n_{i+1} - 2 u^n_i + u^n_{i-1}) / dx^2

    Stability (CFL) condition:
        dt <= dx^2 / (2*alpha)

    We choose dt = 0.4 * dx^2 / (2*alpha) if the user doesn't provide dt, which
    is safely within the stability bound.

    Returns
    -------
    U  : (nt, nx) float32
         Time sequence of fields; U[0] is the initial condition.
    x  : (nx,) float64
         Spatial grid (uniform).
    dt : float
         Time step actually used.
    dx : float
         Spatial grid spacing.
    """
    x = np.linspace(0.0, 1.0, nx)
    dx = float(x[1] - x[0])

    # Pick a conservative stable step if not provided
    if dt is None:
        dt = 0.4 * dx * dx / (2.0 * alpha + 1e-12)

    # Allocate time stack
    U = np.zeros((nt, nx), dtype=np.float32)
    U[0] = make_initial_condition(nx, ic_kind)

    # March in time
    for n in range(nt - 1):
        u = U[n]
        unew = u.copy()

        # Discrete Laplacian in the interior (vectorized)
        lap = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)

        # Update interior points
        unew[1:-1] = u[1:-1] + dt * alpha * lap

        # Enforce Dirichlet BCs every step
        unew[0] = 0.0
        unew[-1] = 0.0

        U[n + 1] = unew

    return U, x, dt, dx


# =============================================================================
# Dataset: windows of (past T steps -> future H steps)
# =============================================================================

class HeatSequenceDataset(Dataset):
    """
    For each sample:
      Input  X: (T, nx)  — past frames
      Target Y: (H, nx)  — future frames
      Also returns alpha, dt, dx (scalars) for physics diagnostics.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        nx: int = 64,
        nt: int = 160,
        T: int = 16,
        H: int = 8,
        alpha_range: Tuple[float, float] = (0.05, 0.25),
        ic_kind: str = "sines",
    ):
        self.examples = []
        self.T = T
        self.H = H

        for _ in range(n_samples):
            alpha = float(np.random.uniform(*alpha_range))
            U, x, dt, dx = simulate_heat_equation(alpha, nx=nx, nt=nt, ic_kind=ic_kind)

            # Random window where we have T past and H future steps available
            t0 = np.random.randint(0, nt - (T + H))
            X = U[t0 : t0 + T]
            Y = U[t0 + T : t0 + T + H]

            self.examples.append(
                (X.astype(np.float32), Y.astype(np.float32), alpha, float(dt), float(dx))
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        X, Y, alpha, dt, dx = self.examples[idx]
        # Return torch tensors; alpha/dt/dx as 0D tensors for broadcasting later
        return (
            torch.from_numpy(X),
            torch.from_numpy(Y),
            torch.tensor(alpha),
            torch.tensor(dt),
            torch.tensor(dx),
        )


# =============================================================================
# Model: Time-Transformer (encoder-decoder over time axis)
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017).
    We apply it along the time dimension to give the model a sense of order.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (S, B, D) — sequence length S, batch B, features D
        return x + self.pe[: x.shape[0]].unsqueeze(1)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Upper-triangular boolean mask for causal decoding:
    True entries are masked (not attended).
    """
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()


class HeatTransformer(nn.Module):
    """
    Minimal encoder-decoder Transformer for time series of spatial fields.

    Mapping:
        src_seq: (B, T, nx)  ->  d_model  ->  encoder
        tgt_seq: (B, H, nx)  ->  d_model  ->  decoder (causal mask)
        output:  (B, H, nx)

    Where each token is an entire spatial field (length nx).
    """

    def __init__(
        self,
        nx: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nx = nx
        self.in_proj = nn.Linear(nx, d_model)   # embed field -> token
        self.out_proj = nn.Linear(d_model, nx)  # token -> field
        self.pos_enc = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=False,  # we will pass (S,B,D)
        )

    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        src_seq : (B, T, nx)
        tgt_seq : (B, H, nx)  (teacher forcing or previous predictions)

        Returns
        -------
        out : (B, H, nx)  predicted future fields
        """
        # Project to embedding space
        src = self.in_proj(src_seq)  # (B,T,d)
        tgt = self.in_proj(tgt_seq)  # (B,H,d)

        # Transformer expects (S,B,D)
        src = src.transpose(0, 1)    # (T,B,d)
        tgt = tgt.transpose(0, 1)    # (H,B,d)

        # Add positional encodings in time
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)

        # Causal mask to prevent access to future target positions
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)  # (H,B,d)
        out = out.transpose(0, 1)                            # (B,H,d)
        return self.out_proj(out)                            # (B,H,nx)


# =============================================================================
# Physics metrics: PDE residual, boundary error, L2 trend
# =============================================================================

def pde_residual_norm(pred: torch.Tensor, alpha: torch.Tensor, dt: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """
    Compute || u_t - alpha u_xx ||_2 over the predicted window.
    - pred : (B,H,nx)
    - alpha, dt, dx are 0D tensors (per-batch scalars)
    Notes:
      * Uses forward difference for u_t inside predicted horizon (requires H >= 2).
      * Uses centered second difference for u_xx with Dirichlet ends ignored (interior only).
    """
    B, H, nx = pred.shape
    if H < 2:
        return torch.full((1,), float("nan"), device=pred.device)

    # Time derivative on the first H-1 predicted intervals
    ut = (pred[:, 1:, :] - pred[:, :-1, :]) / dt.view(-1, 1, 1)  # (B,H-1,nx)

    # Spatial Laplacian on the matching H-1 frames
    lap = torch.zeros_like(ut)
    u = pred[:, :-1, :]  # (B,H-1,nx)
    lap[:, :, 1:-1] = (u[:, :, 2:] - 2.0 * u[:, :, 1:-1] + u[:, :, :-2]) / (dx.view(-1, 1, 1) ** 2)

    resid = ut - alpha.view(-1, 1, 1) * lap
    # L2 norm over all space-time samples, averaged over batch
    return torch.linalg.vector_norm(resid.reshape(B, -1), ord=2, dim=1).mean()


def boundary_error(pred: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute boundary violation:
        (|u(x=0)| + |u(x=1)|)/2 averaged over (B,H)
    """
    left = pred[..., 0]
    right = pred[..., -1]
    return (left.abs().mean() + right.abs().mean()) / 2.0


def l2_norm_trend(pred: torch.Tensor) -> torch.Tensor:
    """
    L2 norm over x for each predicted time slice, averaged over batch.
    For diffusion with Dirichlet BCs, this should be **non-increasing** in time.
    Returns a vector of length H (one per predicted step).
    """
    # pred: (B,H,nx) -> compute sqrt(mean(u^2) over x) per (B,H), then mean over B
    l2_per_bh = torch.sqrt(torch.mean(pred ** 2, dim=-1))  # (B,H)
    return l2_per_bh.mean(dim=0)                           # (H,)


def l2_is_nonincreasing(seq: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check that seq[t+1] - seq[t] <= tol for all t (allow tiny numerical noise)."""
    return bool(torch.all((seq[1:] - seq[:-1]) <= tol))


# =============================================================================
# Training
# =============================================================================

def train_model():
    """
    Train the Transformer on synthetic FD data.
    Adds boundary clamps during training/rollout to hard-enforce BCs
    (removing them will show higher BC errors but similar qualitative behavior).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Grid/time configs (modest sizes for quick runs)
    nx = 64   # number of spatial points
    nt = 200  # time length in FD teacher used to build windows
    T = 16    # history length
    H = 8     # forecast horizon

    # Datasets with different IC families to reduce overfitting
    train_ds = HeatSequenceDataset(2000, nx=nx, nt=nt, T=T, H=H, alpha_range=(0.05, 0.25), ic_kind="sines")
    val_ds   = HeatSequenceDataset( 250, nx=nx, nt=nt, T=T, H=H, alpha_range=(0.05, 0.25), ic_kind="gaussians")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # Model
    model = HeatTransformer(nx=nx, d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.1).to(device)

    # Optimizer, scheduler, and loss
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    criterion = nn.MSELoss()

    best_val = float("inf")
    epochs = 20

    for ep in range(1, epochs + 1):
        # -------------------------
        # Training epoch
        # -------------------------
        model.train()
        tr_mse = []
        tr_res = []
        tr_bc  = []

        for X, Y, alpha, dt, dx in train_loader:
            # Shapes: X=(B,T,nx), Y=(B,H,nx)
            X = X.to(device)
            Y = Y.to(device)
            alpha = alpha.to(device)
            dt = dt.to(device)
            dx = dx.to(device)

            # Hard-enforce boundary conditions on teacher data to remove drift
            X[..., 0]  = 0.0
            X[..., -1] = 0.0
            Y[..., 0]  = 0.0
            Y[..., -1] = 0.0

            # Teacher forcing: give decoder the last src frame, then H-1 true targets
            dec_in = torch.cat([X[:, -1:, :], Y[:, :-1, :]], dim=1)  # (B,H,nx)

            # Forward pass
            pred = model(X, dec_in)

            # Hard clamp model output to satisfy BCs (optional but recommended)
            pred[..., 0]  = 0.0
            pred[..., -1] = 0.0

            loss = criterion(pred, Y)

            # Standard optimization
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Physics diagnostics (detach to avoid backprop)
            with torch.no_grad():
                tr_mse.append(loss.item())
                tr_res.append(pde_residual_norm(pred, alpha, dt, dx).item())
                tr_bc.append(boundary_error(pred).item())

        scheduler.step()

        # -------------------------
        # Validation epoch
        # -------------------------
        model.eval()
        va_mse = []
        va_res = []
        va_bc  = []
        with torch.no_grad():
            for X, Y, alpha, dt, dx in val_loader:
                X = X.to(device)
                Y = Y.to(device)
                alpha = alpha.to(device)
                dt = dt.to(device)
                dx = dx.to(device)

                X[..., 0]  = 0.0
                X[..., -1] = 0.0
                Y[..., 0]  = 0.0
                Y[..., -1] = 0.0

                dec_in = torch.cat([X[:, -1:, :], Y[:, :-1, :]], dim=1)
                pred = model(X, dec_in)
                pred[..., 0]  = 0.0
                pred[..., -1] = 0.0

                va_mse.append(criterion(pred, Y).item())
                va_res.append(pde_residual_norm(pred, alpha, dt, dx).item())
                va_bc.append(boundary_error(pred).item())

        # Aggregate epoch stats
        m_tr_mse = float(np.mean(tr_mse))
        m_tr_res = float(np.mean(tr_res))
        m_tr_bc  = float(np.mean(tr_bc))
        m_va_mse = float(np.mean(va_mse))
        m_va_res = float(np.mean(va_res))
        m_va_bc  = float(np.mean(va_bc))

        print(f"[Ep {ep:02d}] "
              f"train: MSE={m_tr_mse:.3e}, PDEres={m_tr_res:.3e}, BCerr={m_tr_bc:.3e} | "
              f"val: MSE={m_va_mse:.3e}, PDEres={m_va_res:.3e}, BCerr={m_va_bc:.3e}")

        # Save best by val MSE (you could also consider PDEres as a criterion)
        if m_va_mse < best_val:
            best_val = m_va_mse
            torch.save(
                {"state_dict": model.state_dict(), "nx": nx, "T": T, "H": H},
                "best_heat_transformer.pt",
            )

    return model, (nx, T, H)


# =============================================================================
# Rollout and Baseline
# =============================================================================

def rollout(model: nn.Module, X0: torch.Tensor, H_roll: int) -> torch.Tensor:
    """
    Autoregressively generate H_roll steps from a history buffer X0 = (B,T,nx).
    At each step, we feed the *last* frame as the decoder input (H=1 case).
    We clamp boundaries at each emitted step to enforce BCs strictly.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        B, T, nx = X0.shape

        # Ensure BCs on the provided history
        X = X0.clone().to(device)
        X[..., 0]  = 0.0
        X[..., -1] = 0.0

        preds = []
        last = X[:, -1:, :]  # (B,1,nx)

        for _ in range(H_roll):
            out = model(X, last)       # (B,1,nx)
            out[..., 0]  = 0.0
            out[..., -1] = 0.0
            preds.append(out)

            # Slide the window: drop oldest, append newest prediction
            X = torch.cat([X[:, 1:, :], out], dim=1)
            last = out

        return torch.cat(preds, dim=1)  # (B,H_roll,nx)


def persistence_baseline(X_hist: torch.Tensor, H: int) -> torch.Tensor:
    """
    Trivial baseline: predict the next H frames as an exact copy of the last history frame.
    """
    last = X_hist[:, -1:, :]
    return last.repeat(1, H, 1)


# =============================================================================
# Sanity Suite (verifications you can read as numbers)
# =============================================================================

def sanity_suite(model, nx: int, T: int, H: int) -> None:
    """
    Build a fresh test trajectory and report:
      - MSE vs FD ground truth
      - PDE residual on predictions
      - Boundary error
      - L2 trend and the non-increasing check
      - Comparison vs a persistence baseline

    These are *quantitative* and hence verifiable.
    """
    # Create a test trajectory from the FD teacher (IC different from training)
    alpha = 0.12
    U, x, dt, dx = simulate_heat_equation(alpha, nx=nx, nt=220, ic_kind="gaussians")

    # Select a window
    t0 = 40
    X_hist = torch.from_numpy(U[t0 : t0 + T]).unsqueeze(0).float()   # (1,T,nx)
    Y_true = torch.from_numpy(U[t0 + T : t0 + T + H]).unsqueeze(0).float()  # (1,H,nx)

    # Model rollout
    pred = rollout(model, X_hist, H_roll=H)

    # Metrics
    mse_test = torch.mean((pred - Y_true) ** 2).item()
    resid_test = pde_residual_norm(
        pred,
        alpha=torch.tensor([alpha]),
        dt=torch.tensor([dt]),
        dx=torch.tensor([dx]),
    ).item()
    berr_test = boundary_error(pred).item()
    l2_vec = l2_norm_trend(pred).cpu()

    # Baseline
    base = persistence_baseline(X_hist, H)
    base_mse = torch.mean((base - Y_true) ** 2).item()

    print("\n================= SANITY SUITE =================")
    print(f"TEST MSE               : {mse_test:.3e}")
    print(f"TEST PDE residual norm : {resid_test:.3e}")
    print(f"TEST boundary error    : {berr_test:.3e}  (should be ~0 with hard clamp)")
    print("TEST L2 norm trajectory:", ", ".join(f"{v:.3e}" for v in l2_vec))
    print("L2 non-increasing?     :", l2_is_nonincreasing(l2_vec))
    print(f"Baseline MSE (persist) : {base_mse:.3e}")
    print("Model beats baseline?  :", mse_test < base_mse)
    print("================================================\n")

    # (Optional) quick visualization (commented out to keep script headless)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.title("Initial frame (last of history)")
        plt.plot(x, X_hist[0, -1].numpy(), label="input last")
        plt.xlabel("x"); plt.ylabel("u"); plt.legend()

        plt.subplot(1,2,2); plt.title(f"t+{H}: Transformer vs FD")
        plt.plot(x, Y_true[0, -1].numpy(), label="FD truth")
        plt.plot(x, pred[0, -1].numpy(), '--', label="Transformer")
        plt.xlabel("x"); plt.ylabel("u"); plt.legend()
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("Plot skipped (matplotlib not available or headless env):", e)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # 1) Train
    model, (nx, T, H) = train_model()

    # 2) Verify via quantitative diagnostics
    sanity_suite(model, nx, T, H)

    # Notes for further verification:
    #   * Change basis size nx or horizon H; the trained model should still beat the baseline.
    #   * Remove the hard boundary clamp and observe BC error increase.
    #   * Increase training data / model size: PDE residual and MSE should drop further.

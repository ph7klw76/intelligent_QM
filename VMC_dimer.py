"""
Tiny Variational Monte Carlo (VMC) demo on a 2-site/2-qubit "dimer" with exactly one electron.
----------------------------------------------------------------------------------------------

Model
-----
- Configuration space is constrained to exactly one '1' across two bits: x ∈ {(1,0), (0,1)}.
- Probabilistic (autoregressive) ansatz: P(10) = p, P(01) = 1 - p.
- Real, non-negative wavefunction amplitudes: Ψ(10) = sqrt(p), Ψ(01) = sqrt(1 - p).

Hamiltonian
-----------
H = -(X1 X2 + Y1 Y2)/2.
Restricted to the one-electron subspace, H flips |10⟩ ↔ |01⟩ with matrix element -1:
    H|10⟩ = -|01⟩,  H|01⟩ = -|10⟩.

Exact Energy for this 1-parameter ansatz
----------------------------------------
E(p) = ⟨Ψ|H|Ψ⟩ = -2 * sqrt(p * (1 - p)), minimized at p = 0.5 with E_min = -1.

Estimator & Optimization
------------------------
- Local energy: E_loc(x) = (HΨ)(x) / Ψ(x).
  Since H only flips the two states with coefficient -1,
    E_loc(10) = -Ψ(01)/Ψ(10) = -sqrt((1-p)/p),
    E_loc(01) = -Ψ(10)/Ψ(01) = -sqrt(p/(1-p)).

- Gradient (score-function / REINFORCE) on log Ψ:
    dE/dp = E_{x~P} [ (E_loc(x) - b) * ∂_p log Ψ(x) ],
  with a variance-reducing baseline b (we use the batch mean energy).
  Because Ψ = sqrt(P), log Ψ = 0.5 * log P, so ∂_p log Ψ = 0.5 * ∂_p log P.
  For our policy: ∂_p log P(10) = 1/p, ∂_p log P(01) = -1/(1-p).

We iteratively:
  1) Sample x ~ P,
  2) Compute E_loc(x) and the gradient factor,
  3) Take one SGD step on p (clipped to (0,1)).
The plots show energy descent and p → 0.5.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# Global RNG (fixed seed for reproducibility).
rng = np.random.default_rng(0)


def sample_bitstring(p: float) -> tuple[int, int]:
    """
    Autoregressive sample respecting the "one electron" mask (exactly one '1'):
      - Draw x1 ~ Bernoulli(p)
      - Force x2 = 1 - x1 so that (x1, x2) ∈ {(1,0), (0,1)}.

    Parameters
    ----------
    p : float
        Probability that the first bit is 1, i.e., P(10) = p.

    Returns
    -------
    (x1, x2) : tuple[int, int]
        A valid masked bitstring (1,0) or (0,1).
    """
    x1 = 1 if rng.random() < p else 0
    x2 = 0 if x1 == 1 else 1
    return (x1, x2)


def psi_amplitude(p: float, x: tuple[int, int]) -> float:
    """
    Real, non-negative amplitude (zero phase) from sqrt-probability:
        Ψ(10) = sqrt(p),  Ψ(01) = sqrt(1 - p).

    Any other configuration is masked out (amplitude 0) by construction.

    Parameters
    ----------
    p : float
        Policy parameter (P(10) = p).
    x : (int, int)
        Bitstring (x1, x2).

    Returns
    -------
    float
        Wavefunction amplitude Ψ(x).
    """
    if x == (1, 0):
        return np.sqrt(p)
    elif x == (0, 1):
        return np.sqrt(1 - p)
    else:
        return 0.0  # not reachable here due to the mask


def local_energy(p: float, x: tuple[int, int]) -> float:
    """
    Local energy E_loc(x) = (HΨ)(x) / Ψ(x) for H = -(X1X2 + Y1Y2)/2.

    In the one-electron subspace, H flips 10 <-> 01 with total coefficient -1,
    hence:
        E_loc(10) = -Ψ(01)/Ψ(10) = -sqrt((1-p)/p),
        E_loc(01) = -Ψ(10)/Ψ(01) = -sqrt(p/(1-p)).

    Parameters
    ----------
    p : float
        Policy parameter (P(10) = p).
    x : (int, int)
        Bitstring (x1, x2).

    Returns
    -------
    float
        Local energy at x.
    """
    if x == (1, 0):
        num = psi_amplitude(p, (0, 1))
        den = psi_amplitude(p, (1, 0))
        return -(num / den)
    else:  # x == (0, 1)
        num = psi_amplitude(p, (1, 0))
        den = psi_amplitude(p, (0, 1))
        return -(num / den)


def logP_grad(p: float, x: tuple[int, int]) -> float:
    """
    ∂_p log P(x) for the masked AR policy with only two outcomes:
        P(10) = p,  P(01) = 1 - p.

    Therefore:
        ∂_p log P(10) =  +1/p,
        ∂_p log P(01) =  -1/(1-p).

    Parameters
    ----------
    p : float
        Policy parameter (P(10) = p).
    x : (int, int)
        Sampled configuration.

    Returns
    -------
    float
        ∂_p log P(x).
    """
    if x == (1, 0):
        return 1.0 / p
    else:
        return -1.0 / (1.0 - p)


def vmc_step(p: float, batch: int = 2000, lr: float = 0.05) -> tuple[float, float]:
    """
    Perform one stochastic VMC-like update on p using a score-function gradient.

    We estimate dE/dp = E_{x~P}[(E_loc(x) - b) * ∂_p log Ψ(x)].
    Since log Ψ = 0.5 * log P, we use (0.5 * ∂_p log P(x)) in the estimator.
    The baseline b is chosen as the batch mean of E_loc (variance reduction).

    Parameters
    ----------
    p : float
        Current policy parameter.
    batch : int, optional
        Number of Monte Carlo samples for the gradient estimate.
    lr : float, optional
        Learning rate for the parameter update.

    Returns
    -------
    p_new : float
        Updated (and clipped) parameter in (0, 1).
    baseline : float
        Batch mean energy (used as the baseline b, also a crude energy estimate).
    """
    # 1) Sample configurations x ~ P
    samples = [sample_bitstring(p) for _ in range(batch)]

    # 2) Compute local energies and the variance-reduction baseline
    Eloc = np.array([local_energy(p, x) for x in samples])
    baseline = Eloc.mean()

    # 3) Score-function factors (0.5 from log Ψ = 0.5 log P)
    grads = np.array([0.5 * logP_grad(p, x) for x in samples])

    # 4) Monte Carlo estimate of dE/dp
    dE_dp = np.mean((Eloc - baseline) * grads)

    # 5) SGD step on p with clipping to keep p strictly inside (0,1)
    p_new = float(np.clip(p - lr * dE_dp, 1e-6, 1 - 1e-6))

    return p_new, baseline


# ----------------------------- Main experiment -----------------------------

# Start from an imbalanced policy: P(10) = 0.8, P(01) = 0.2.
p = 0.80

energies = []  # Monte Carlo energy estimates (mean of local energies per step)
ps = []        # Parameter trajectory
steps = 60     # Number of optimization iterations

for _ in range(steps):
    p, E = vmc_step(p, batch=4000, lr=0.05)
    energies.append(E)
    ps.append(p)

# Closed-form exact energy E(p) for comparison at the visited p's:
# E(p) = -2 * sqrt(p * (1 - p)) with minimum -1 at p = 0.5.
E_exact = [-2.0 * np.sqrt(pp * (1 - pp)) for pp in ps]

# Print final status. The ground-state target is p = 0.5 with E = -1.0.
print(f"Final p ~ {p:.4f}  (target for ground state is 0.5)")
print(
    f"Final energy estimate ~ {energies[-1]:.4f}  ;  "
    f"exact E(p) ~ {E_exact[-1]:.4f}  ;  exact minimum = -1.0000"
)

# ----------------------------- Visualization ------------------------------

# Plot energy vs iteration (Monte Carlo estimate and exact curve at those p's)
plt.figure()
plt.plot(energies, label="Monte Carlo E estimate")
plt.plot(E_exact, label="Exact E(p)")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("VMC energy descent on 2-orbital dimer")
plt.legend()
plt.tight_layout()
plt.show()

# Plot parameter p vs iteration (policy converging toward p = 0.5)
plt.figure()
plt.plot(ps)
plt.xlabel("Iteration")
plt.ylabel("p = P(x1=1) = P(10)")
plt.title("Autoregressive policy converging to bonding state (p → 0.5)")
plt.tight_layout()
plt.show()

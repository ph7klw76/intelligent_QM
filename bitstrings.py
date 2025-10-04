# Autoregressive sampler on bitstrings with a fixed electron number (N_so=6 qubits, N_e=3)
# + a simple local-energy estimator for a few Jordan–Wigner hopping terms.
#
# The model is intentionally tiny:
#   - P(x) is factorized autoregressively: P(x)=∏_i P(x_i | x_<i)
#   - At each step we *mask* choices so that exactly N_e ones are placed by the end
#   - We keep amplitudes real: Ψ(x)=sqrt(P(x))  (phase=0), so Ψ(x')/Ψ(x)=sqrt(P(x')/P(x))
#
# Hamiltonian (toy): two hoppings (1<->2) and (3<->4) with strengths t12=1.0, t34=0.7
# Using correct JW form: H_hop(p,q) = -t/2 * (X_p Z_{p+1..q-1} X_q + Y_p Z_{p+1..q-1} Y_q)
# On computational basis, this connects states where bits at p,q differ (10<->01).
# The matrix element carries a parity sign (-1)^(# of ones between p and q).
#
# We'll:
#   (1) define an autoregressive policy with base logits (here a simple linear ramp)
#   (2) sample a batch of bitstrings with *exact* enforcement of N_e
#   (3) compute local energy for each sample via JW hopping rules
#   (4) print a few examples and the batch-averaged energy

import numpy as np
from math import comb

rng = np.random.default_rng(42)

N = 6         # number of spin-orbitals (qubits)
Ne = 3        # total electrons (ones)
terms = [     # (p, q, t) with 0-based indices and p<q
    (0, 1, 1.0),   # hop 1<->2
    (2, 3, 0.7),   # hop 3<->4
]

# --- Autoregressive policy parameters (tiny, hand-chosen logits for demonstration) ---
# Base logits favor early ones a bit; masking will enforce exact count.
base_logits = np.array([0.6, 0.5, 0.4, 0.6, 0.3, 0.5])  # you can tweak these
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def masked_step_prob(i, x_prefix, ones_used, N=N, Ne=Ne):
    """Return Bernoulli prob p_i for x_i=1 given prefix and masks enforcing exactly Ne ones."""
    remaining_positions = N - i
    remaining_ones = Ne - ones_used

    # Hard mask logic
    if remaining_ones <= 0:
        return 0.0
    if remaining_ones >= remaining_positions:
        return 1.0

    # Otherwise use base policy (can depend on prefix; we use fixed logits here for simplicity)
    return sigmoid(base_logits[i])

def sample_autoregressive(batch=10):
    """Sample a batch of bitstrings with exact Ne enforcement."""
    X = []
    Ps = []  # store P(x) for each sample (product of conditionals)
    for _ in range(batch):
        x = np.zeros(N, dtype=int)
        p_prod = 1.0
        ones_used = 0
        for i in range(N):
            p1 = masked_step_prob(i, x[:i], ones_used)
            # draw x_i ∈ {0,1}
            xi = 1 if rng.random() < p1 else 0
            # if impossible (can happen only due to float edge cases), force mask
            if p1 == 0.0: xi = 0
            if p1 == 1.0: xi = 1
            x[i] = xi
            p_prod *= (p1 if xi==1 else (1.0 - p1))
            ones_used += xi
        # sanity: ensure exactly Ne ones
        assert x.sum() == Ne, f"masking failed: {x} has {x.sum()} ones"
        X.append(x)
        Ps.append(p_prod)
    return np.array(X, dtype=int), np.array(Ps)

def parity_between(x, p, q):
    """Number of ones between p and q (exclusive)."""
    if p > q: p, q = q, p
    return int(np.sum(x[p+1:q]) % 2)

def jw_hop_local_contrib(x, p, q, t, P):
    """
    Local-energy contribution for a single JW hopping term between p<->q, strength t.
    For basis state x, if bits differ at (p,q), the connected state x' is x with both bits flipped.
    Matrix element magnitude is t, with sign (-1)^(parity_between).
    Local energy adds:  (-t) * sgn * Ψ(x')/Ψ(x)  , where sgn = (-1)^(parity).
    With Ψ = sqrt(P), ratio = sqrt(P(x')/P(x)).
    """
    if x[p] == x[q]:
        return 0.0  # no coupling
    # build x'
    x_prime = x.copy()
    x_prime[p] ^= 1
    x_prime[q] ^= 1

    # compute conditional probability P(x) and P(x') exactly from our AR model (with masks)
    def prob_of_bitstring(x_bits):
        ones_used = 0
        p_prod = 1.0
        for i, xi in enumerate(x_bits):
            p1 = masked_step_prob(i, x_bits[:i], ones_used)
            p_prod *= (p1 if xi==1 else (1.0 - p1))
            ones_used += xi
        return p_prod

    Px  = prob_of_bitstring(x)
    Px2 = prob_of_bitstring(x_prime)

    if Px == 0:
        return 0.0  # should not happen due to masking; safe-guard

    sgn = -1.0 if (parity_between(x, p, q) % 2 == 1) else 1.0
    # The full JW off-diagonal coefficient for (XZX + YZY)/2 sums to ±1; with prefactor -t/2 -t/2 we get -t*sgn.
    return (-t) * sgn * np.sqrt(Px2 / Px)

def local_energy(x):
    """Sum of contributions from all hopping terms."""
    E = 0.0
    for (p, q, t) in terms:
        E += jw_hop_local_contrib(x, p, q, t, None)
    return E

# ---- Demo run ----
X, P = sample_autoregressive(batch=12)
Eloc = np.array([local_energy(x) for x in X])

print("Sampled bitstrings x (N=6, Ne=3) and their local energies (toy Hamiltonian):")
for i, (x, e) in enumerate(zip(X, Eloc)):
    print(f"{i:2d}: x={''.join(map(str,x))}   E_loc={e:+.4f}")

print("\nBatch average of E_loc (variational energy estimator): ", Eloc.mean())

# Show that masking is exact: all samples have exactly Ne ones
unique_counts = np.unique(np.sum(X, axis=1))
print("\nCheck #electrons (ones) in each sample:", unique_counts)

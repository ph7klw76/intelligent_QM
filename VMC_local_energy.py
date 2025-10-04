# Illustrations of VMC local energy and neural autoregressive sampling on tiny spin-orbital toys
#
# Example 1: Two configurations {A,B} with a 2x2 hopping Hamiltonian; shows E_loc and variational energy.
# Example 2: Three spin-orbitals, exactly two electrons (masking), adjacent Jordan–Wigner hops, E_loc per bitstring,
#            and the variational energy. Also scans a policy parameter to show how energy changes.
#
# No external data; pure numpy/matplotlib. You can tweak the numbers at the top of each example.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

# -------------------------
# Example 1: 2-state toy
# -------------------------
# Hamiltonian H = [[0, -t], [-t, 0]]
t = 1.0
H = np.array([[0.0, -t],
              [-t, 0.0]])

# Choose real amplitudes Psi(A), Psi(B)
psi_A = 0.8
psi_B = 0.6
P_A = psi_A**2
P_B = psi_B**2
Z = P_A + P_B
P_A /= Z
P_B /= Z

# Local energies via ratios
E_loc_A = H[0,1] * (psi_B / psi_A)
E_loc_B = H[1,0] * (psi_A / psi_B)
E_var_2state = P_A * E_loc_A + P_B * E_loc_B

df2 = pd.DataFrame({
    "state": ["A","B"],
    "Psi": [psi_A, psi_B],
    "P = |Psi|^2 / Z": [P_A, P_B],
    "E_loc": [E_loc_A, E_loc_B]
})
display_dataframe_to_user("Example 1: Two-state toy — amplitudes, probabilities, local energies", df2)

print("Example 1 (2-state toy)")
print("------------------------")
print(df2.to_string(index=False))
print(f"\nVariational energy (average of E_loc): E = {E_var_2state:.6f}")
print("Exact ground-state energy for this H is -t = -1.0; the model will move toward Psi(A)=Psi(B) to reach it.\n")

# --------------------------------------------
# Example 2: 3 spin-orbitals, 2 electrons (mask)
# --------------------------------------------
# Allowed bitstrings: 110, 101, 011
states = [(1,1,0),(1,0,1),(0,1,1)]

# Autoregressive conditionals (you can tweak):
# Step 1: P(x1=1) = p1
# Step 2: if x1=1 -> P(x2=1|x1=1) = p2_if1 ; if x1=0 -> P(x2=1|x1=0) = p2_if0
p1 = 0.7
p2_if1 = 0.6
p2_if0 = 0.4

def ar_probs_3orb_2el(p1, p2_if1, p2_if0):
    P = {}
    P[(1,1,0)] = p1 * p2_if1 * 1.0        # x3 forced to 0 by mask
    P[(1,0,1)] = p1 * (1.0 - p2_if1) * 1.0  # x3 forced to 1
    P[(0,1,1)] = (1.0 - p1) * p2_if0 * 1.0  # x3 forced to 1
    Z = sum(P.values())
    for k in P: P[k] /= Z
    return P

P = ar_probs_3orb_2el(p1, p2_if1, p2_if0)
Psi = {x: np.sqrt(P[x]) for x in states}  # real amplitudes; phase=0

# Jordan–Wigner adjacent hops (no parity sign for adjacency)
# H terms: (1<->2) with t12=1.0, (2<->3) with t23=0.8
t12, t23 = 1.0, 0.8
terms_adjacent = [ (0,1,t12), (1,2,t23) ]  # 0-based indices

def jw_adj_flip(x, p, q):
    if x[p] == x[q]: return None
    x_prime = list(x)
    x_prime[p] ^= 1
    x_prime[q] ^= 1
    return tuple(x_prime)

def local_energy_bitstring(x, Psi, terms):
    E = 0.0
    for (p,q,tij) in terms:
        x_prime = jw_adj_flip(x, p, q)
        if x_prime is not None:
            E += (-tij) * (Psi[x_prime] / Psi[x])  # adjacent -> no parity sign
    return E

# Compute E_loc(x) and variational energy
E_loc_vals = {x: local_energy_bitstring(x, Psi, terms_adjacent) for x in states}
E_var = sum(P[x] * E_loc_vals[x] for x in states)

df3 = pd.DataFrame({
    "x (bitstring)": [''.join(map(str,x)) for x in states],
    "P(x)": [P[x] for x in states],
    "|Psi(x)|": [Psi[x] for x in states],
    "E_loc(x)": [E_loc_vals[x] for x in states],
})
display_dataframe_to_user("Example 2: 3-orbital, 2-electron toy — probs, amplitudes, local energies", df3)

print("Example 2 (3 orbitals, 2 electrons)")
print("------------------------------------")
print(df3.to_string(index=False))
print(f"\nVariational energy  ⟨E_loc⟩  = {E_var:.6f}\n")

# Optional: scan p1 to show how energy changes with the first-step AR policy
p1_grid = np.linspace(0.1, 0.9, 17)
E_curve = []
for p1v in p1_grid:
    Pv = ar_probs_3orb_2el(p1v, p2_if1, p2_if0)
    Psiv = {x: np.sqrt(Pv[x]) for x in states}
    Elocv = {x: local_energy_bitstring(x, Psiv, terms_adjacent) for x in states}
    E_curve.append(sum(Pv[x] * Elocv[x] for x in states))

plt.figure()
plt.plot(p1_grid, E_curve)
plt.xlabel("p1 = P(x1=1)")
plt.ylabel("Variational energy  ⟨E_loc⟩")
plt.title("Energy vs first-step autoregressive policy (mask enforces 2 electrons)")
plt.show()

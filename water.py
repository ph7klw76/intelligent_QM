# h2o_excited_vmc_fixed_es.py
# AR-NNQS + VMC for the first singlet excited state of H2O
# with (1) strong/low-variance overlap (EMA), (2) ES seeding (HOMO->LUMO),
# (3) tightened training protocol and RHF sanity checks.

import argparse, math, random, time
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_EVERY = 50
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


# -------------------- small helpers --------------------
def pc(x: int) -> int:
    try:
        return x.bit_count()
    except AttributeError:
        c = 0
        while x:
            x &= (x - 1); c += 1
        return c

def det_to_bits(occ: int, n: int) -> np.ndarray:
    return np.array([(occ >> i) & 1 for i in range(n)], dtype=np.int64)

def bits_to_det(bits: np.ndarray) -> int:
    occ = 0
    for i, b in enumerate(bits):
        if b: occ |= (1 << i)
    return occ

def occ_list_from_int(occ: int, n: int) -> np.ndarray:
    return np.array([i for i in range(n) if (occ >> i) & 1], dtype=np.int64)


# -------------------- FCIDUMP I/O --------------------
@dataclass
class FCIDump:
    norb: int
    nelec_alpha: int
    nelec_beta: int
    e_core: float
    h1: np.ndarray                       # (norb, norb)
    g2: np.ndarray                       # (norb, norb, norb, norb) chemists' (pq|rs)

def parse_fcidump(path: str,
                  nelec_alpha: Optional[int]=None,
                  nelec_beta: Optional[int]=None) -> FCIDump:
    with open(path, "r") as f:
        lines = f.readlines()

    header: Dict[str, str] = {}
    in_hdr = False
    for raw in lines:
        s = raw.strip()
        if not in_hdr and s.startswith("&"):
            in_hdr = True
        if in_hdr:
            for tok in s.replace(",", " ").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    header[k.strip().upper()] = v.strip()
            if s == "/" or s.upper().startswith("&END"):
                break

    ints, maxorb = [], 0
    for raw in lines:
        sp = raw.strip().split()
        if len(sp) < 5:
            continue
        try:
            v = float(sp[0]); i, j, k, l = map(int, sp[1:5])
        except ValueError:
            continue
        ints.append((v, i, j, k, l))
        maxorb = max(maxorb, i, j, k, l)

    norb = int(header.get("NORB", maxorb))
    h1 = np.zeros((norb, norb), dtype=np.float64)
    g2 = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    ecore = 0.0

    for v, i, j, k, l in ints:
        if i == j == k == l == 0:
            ecore = v
        elif k == 0 and l == 0:
            h1[i-1, j-1] = v
        else:
            g2[i-1, j-1, k-1, l-1] = v

    if (nelec_alpha is None) or (nelec_beta is None):
        if "NELEC" in header:
            Ne  = int(header["NELEC"])
            Ms2 = int(header.get("MS2", "0"))
            if Ms2 != 0:
                raise ValueError("MS2 != 0; pass --nalpha/--nbeta explicitly.")
            nelec_alpha = Ne // 2
            nelec_beta  = Ne - nelec_alpha
        else:
            raise ValueError("Provide --nalpha/--nbeta or include NELEC/MS2 in FCIDUMP header.")

    return FCIDump(norb, nelec_alpha, nelec_beta, ecore, h1, g2)

def energy_order_spatial(fci: FCIDump) -> FCIDump:
    """Sort spatial orbitals by 1e diagonal as a crude MO order proxy."""
    eps = np.diag(fci.h1).copy()
    perm = np.argsort(eps)
    return replace(
        fci,
        h1=fci.h1[np.ix_(perm, perm)],
        g2=fci.g2[np.ix_(perm, perm, perm, perm)]
    )

def freeze_core_and_truncate_spatial(fci: FCIDump, ncore_sp: int, nvirt_sp: int) -> FCIDump:
    """(Optional) form a spatial active space; mean-field core shift folded into e_core."""
    n_sp = fci.norb
    if ncore_sp == 0 and nvirt_sp == 0:
        return fci
    if n_sp <= (ncore_sp + nvirt_sp):
        raise ValueError("Active space would be empty.")
    C = np.arange(0, ncore_sp, dtype=int)
    A = np.arange(ncore_sp, n_sp - nvirt_sp, dtype=int)

    h = fci.h1; g = fci.g2
    ecore_shift = 0.0
    if len(C) > 0:
        ecore_shift += 2.0 * np.sum(np.diag(h)[C])
        Gcc = g[np.ix_(C, C, C, C)]
        coul = np.einsum('ccpp->', Gcc)
        exch = np.einsum('cpcp->', Gcc)
        ecore_shift += 2.0 * coul - exch
        hA = h[np.ix_(A, A)].copy()
        term1 = 2.0 * np.einsum('pqcc->pq', g[np.ix_(A, A, C, C)]) if len(C) else 0.0
        term2 = np.einsum('pcqc->pq', g[np.ix_(A, C, A, C)]) if len(C) else 0.0
        hA += (term1 - term2)
    else:
        hA = h[np.ix_(A, A)].copy()
    gA = g[np.ix_(A, A, A, A)].copy()

    return FCIDump(norb=len(A),
                   nelec_alpha=fci.nelec_alpha - ncore_sp,
                   nelec_beta=fci.nelec_beta - ncore_sp,
                   e_core=fci.e_core + ecore_shift,
                   h1=hA, g2=gA)

def expand_spatial_to_spin(fci: FCIDump) -> FCIDump:
    n_sp = fci.norb
    n_so = 2 * n_sp
    h1 = np.zeros((n_so, n_so), dtype=np.float64)
    h1[0::2, 0::2] = fci.h1
    h1[1::2, 1::2] = fci.h1
    g2 = np.zeros((n_so, n_so, n_so, n_so), dtype=np.float64)
    g2[0::2, 0::2, 0::2, 0::2] = fci.g2   # αα,αα
    g2[0::2, 1::2, 0::2, 1::2] = fci.g2   # αβ,αβ
    g2[1::2, 0::2, 1::2, 0::2] = fci.g2   # βα,βα
    g2[1::2, 1::2, 1::2, 1::2] = fci.g2   # ββ,ββ
    return replace(fci, norb=n_so, h1=h1, g2=g2)

def rhf_energy_from_spatial(fci: FCIDump, nocc_sp: int) -> float:
    """Closed-shell RHF expression using *spatial* integrals."""
    occ = list(range(nocc_sp))
    E = fci.e_core
    for i in occ:
        E += 2.0 * fci.h1[i, i]
    for i in occ:
        for j in occ:
            J = fci.g2[i, i, j, j]      # (ii|jj)
            K = fci.g2[i, j, j, i]      # (ij|ji)
            E += 2.0 * J - K
    return float(E)


# -------------------- Slater–Condon (spin-orbital) --------------------
def excitation_sign_np(occ: int, r: int, a: int) -> int:
    if r == a: return 1
    if ((occ >> r) & 1) == 0: return 0
    if ((occ >> a) & 1) == 1: return 0
    left, right = (r, a) if r < a else (a, r)
    mask_between = ((1 << right) - 1) ^ ((1 << (left+1)) - 1)
    n_between = pc(occ & mask_between)
    return -1 if (n_between & 1) else +1

def double_excitation_sign_np(occ: int, r: int, s: int, a: int, b: int) -> int:
    if len({r, s, a, b}) < 4: return 0
    s1 = excitation_sign_np(occ, r, a)
    if s1 == 0: return 0
    occ1 = occ ^ (1<<r) ^ (1<<a)
    s2 = excitation_sign_np(occ1, s, b)
    return s1 * s2

def diag_energy_np(occ: int, occs: np.ndarray,
                   h1: np.ndarray, g2: np.ndarray, ecore: float) -> float:
    # spin-orbital diagonal matrix element
    E = ecore
    for p in occs:
        E += h1[p, p]
    for p in occs:
        for q in occs:
            E += 0.5 * (g2[p, q, p, q] - g2[p, q, q, p])
    return float(E)

def single_me_np(occ: int, r: int, a: int, h1: np.ndarray, g2: np.ndarray) -> float:
    if r == a: return 0.0
    occs = occ_list_from_int(occ, h1.shape[0])
    val = h1[a, r]
    for j in occs:
        if j == r: continue
        val += g2[a, j, r, j] - g2[a, j, j, r]
    return float(val)

def double_me_np(r: int, s: int, a: int, b: int, g2: np.ndarray) -> float:
    return float(g2[a, b, r, s] - g2[a, b, s, r])

def connected_np(occ: int, h1: np.ndarray, g2: np.ndarray, ecore: float) -> Tuple[List[int], List[complex]]:
    n = h1.shape[0]
    occs = occ_list_from_int(occ, n)
    out_occ: List[int] = []
    out_me:  List[complex] = []
    # diagonal
    out_occ.append(occ); out_me.append(diag_energy_np(occ, occs, h1, g2, ecore))
    # singles
    virs = [i for i in range(n) if ((occ >> i) & 1) == 0]
    for r in occs:
        for a in virs:
            sgn = excitation_sign_np(occ, r, a)
            if sgn == 0: continue
            me = sgn * single_me_np(occ, r, a, h1, g2)
            if me != 0.0:
                out_occ.append(occ ^ (1<<r) ^ (1<<a))
                out_me.append(me)
    # doubles
    for i, r in enumerate(occs):
        for s in occs[i+1:]:
            for j, a in enumerate(virs):
                for b in virs[j+1:]:
                    sgn = double_excitation_sign_np(occ, r, s, a, b)
                    if sgn == 0: continue
                    me = sgn * double_me_np(r, s, a, b, g2)
                    if me != 0.0:
                        out_occ.append(occ ^ (1<<r) ^ (1<<s) ^ (1<<a) ^ (1<<b))
                        out_me.append(me)
    return out_occ, out_me


# -------------------- Autoregressive NNQS --------------------
def choose_amp_dtype(tag: str) -> Optional[torch.dtype]:
    if tag == "off":  return None
    if tag == "bf16": return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if tag == "fp16": return torch.float16
    return None

class ARCore(nn.Module):
    def __init__(self, n_orb: int, hidden: int, depth: int, nhead: int,
                 amp_dtype: Optional[torch.dtype], grad_ckpt: bool):
        super().__init__()
        self.n = n_orb
        self.embed = nn.Embedding(3, hidden)  # tokens {unk=0, 0=1, 1=2}
        self.pos = nn.Parameter(torch.randn(self.n, hidden) * 0.01)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, batch_first=True, dropout=0.0)
             for _ in range(depth)]
        )
        self.amp_dtype = amp_dtype
        self.grad_ckpt = grad_ckpt

    def encode(self, tok: torch.Tensor) -> torch.Tensor:
        # defensive: avoid invalid embedding indices
        if torch.any(tok < 0) or torch.any(tok > 2):
            raise RuntimeError(f"Embed index out of range: [{int(tok.min())}, {int(tok.max())}]")
        # AMP
        if self.amp_dtype is not None and tok.is_cuda:
            ctx = torch.cuda.amp.autocast(dtype=self.amp_dtype)
        else:
            class Dummy:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            ctx = Dummy()
        with ctx:
            h = self.embed(tok) + self.pos.unsqueeze(0)
            for lyr in self.layers:
                if self.grad_ckpt:
                    # simple reentrant-less checkpoint wrapper
                    h = torch.utils.checkpoint.checkpoint(lambda x, l=lyr: l(x), h, use_reentrant=False)
                else:
                    h = lyr(h)
        return h

class ARAmplitude(ARCore):
    def __init__(self, n_orb, hidden, depth, nhead, amp_dtype, grad_ckpt):
        super().__init__(n_orb, hidden, depth, nhead, amp_dtype, grad_ckpt)
        self.proj = nn.Linear(self.layers[0].linear1.in_features, 2)

    @torch.no_grad()
    def sample(self, batch: int, n_alpha: int, n_beta: int, device="cpu") -> torch.Tensor:
        x = torch.zeros((batch, self.n), dtype=torch.long, device=device)
        tok = torch.zeros((batch, self.n), dtype=torch.long, device=device)
        remain_a = torch.full((batch,), n_alpha, dtype=torch.long, device=device)
        remain_b = torch.full((batch,), n_beta,  dtype=torch.long, device=device)
        for i in range(self.n):
            h = self.encode(tok)
            logits = self.proj(h[:, i, :].to(torch.float32))
            probs  = F.softmax(logits, dim=-1)

            is_alpha  = (i % 2 == 0)
            left_same = ((self.n - i + (0 if is_alpha else 1)) // 2)
            rem = remain_a if is_alpha else remain_b

            p1 = probs[:, 1].clone(); p0 = probs[:, 0].clone()
            p1[rem == 0]         = 0.0
            p0[rem == left_same] = 0.0
            norm = p0 + p1 + 1e-12
            p1 = p1 / norm

            bit = torch.bernoulli(p1).long()
            x[:, i]   = bit
            tok[:, i] = bit + 1
            if is_alpha: remain_a -= bit
            else:        remain_b -= bit
        return x

    def log_amp_masked(self, x01: torch.Tensor) -> torch.Tensor:
        B, N = x01.shape
        tok  = torch.zeros_like(x01)
        logp = torch.zeros(B, device=x01.device, dtype=torch.float32)
        placed_a = torch.zeros(B, dtype=torch.long, device=x01.device)
        placed_b = torch.zeros(B, dtype=torch.long, device=x01.device)
        total_a  = x01[:, 0::2].sum(dim=1)
        total_b  = x01[:, 1::2].sum(dim=1)
        for i in range(N):
            h = self.encode(tok)
            logits = self.proj(h[:, i, :].to(torch.float32))
            probs  = F.softmax(logits, dim=-1)
            is_alpha  = (i % 2 == 0)
            left_same = ((N - i + (0 if is_alpha else 1)) // 2)
            remain = (total_a - placed_a) if is_alpha else (total_b - placed_b)

            p1 = probs[:, 1].clone(); p0 = probs[:, 0].clone()
            p1[remain == 0]         = 0.0
            p0[remain == left_same] = 0.0
            norm = p0 + p1 + 1e-12
            p0 /= norm; p1 /= norm

            bit = x01[:, i].long()
            p_choose = torch.where(bit == 1, p1, p0).clamp_min(1e-20)
            logp = logp + torch.log(p_choose)

            tok[:, i] = bit + 1
            if is_alpha: placed_a += bit
            else:        placed_b += bit
        return 0.5 * logp  # amplitude = sqrt(prob)

class ARPhase(ARCore):
    def __init__(self, n_orb, hidden, depth, nhead, amp_dtype, grad_ckpt):
        super().__init__(n_orb, hidden, depth, nhead, amp_dtype, grad_ckpt)
        d = self.layers[0].linear1.in_features
        self.head = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
    def phi(self, x01: torch.Tensor) -> torch.Tensor:
        tok = torch.zeros_like(x01); tok[x01==0] = 1; tok[x01==1] = 2
        h = self.encode(tok)
        pooled = h.mean(dim=1).to(torch.float32)
        return self.head(pooled).squeeze(-1)

class NNQS(nn.Module):
    def __init__(self, n_orb, hidden=96, depth=8, nhead=4,
                 amp_dtype: Optional[torch.dtype]=None, grad_ckpt: bool=False):
        super().__init__()
        self.amp  = ARAmplitude(n_orb, hidden, depth, nhead, amp_dtype, grad_ckpt)
        self.phas = ARPhase(n_orb, hidden, depth, nhead, amp_dtype, grad_ckpt)
    @torch.no_grad()
    def sample(self, batch, n_alpha, n_beta, device="cpu"):
        return self.amp.sample(batch, n_alpha, n_beta, device)
    def log_psi_complex(self, x01: torch.Tensor) -> torch.Tensor:
        return self.amp.log_amp_masked(x01).to(torch.cfloat) + 1j * self.phas.phi(x01).to(torch.cfloat)


# -------------------- Local energy --------------------
def batched_local_energy(dets: np.ndarray, model: NNQS, fci: FCIDump,
                         device: str, cap_dlog: float=40.0) -> torch.Tensor:
    """Return complex E_loc for each determinant in `dets`."""
    h1, g2, ecore = fci.h1, fci.g2, fci.e_core
    n = h1.shape[0]

    # connectivity + union
    conn: List[Tuple[int, List[int], List[complex]]] = []
    all_states = set()
    for D in dets:
        occs, mes = connected_np(int(D), h1, g2, ecore)
        conn.append((int(D), occs, mes))
        for o in occs: all_states.add(int(o))
    all_states = np.fromiter(all_states, dtype=np.int64)

    Xbits = torch.tensor(np.stack([det_to_bits(o, n) for o in all_states], axis=0),
                         dtype=torch.long, device=device)
    with torch.no_grad():
        logpsi_all = model.log_psi_complex(Xbits)  # (M,)
    idx: Dict[int, int] = { int(all_states[i]): i for i in range(len(all_states)) }

    Eloc = torch.zeros(len(dets), dtype=torch.cfloat, device=device)
    for i, (D, occs, mes) in enumerate(conn):
        logD = logpsi_all[idx[D]]
        s = 0j
        for k, Dp in enumerate(occs):
            dlog = logpsi_all[idx[int(Dp)]] - logD
            # clip real part to avoid overflow in exp
            dlog = torch.clamp(torch.real(dlog), min=-cap_dlog, max=cap_dlog) + 1j*torch.imag(dlog)
            s += mes[k] * torch.exp(dlog)
        Eloc[i] = s
    return Eloc


# -------------------- Strong/low-variance overlap (EMA) --------------------
class OverlapMeter:
    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.value: Optional[torch.Tensor] = None
    @torch.no_grad()
    def update(self, gs: NNQS, es: NNQS, n_alpha: int, n_beta: int,
               device: str, nsamp: int = 4096) -> torch.Tensor:
        nsamp = max(nsamp, 512)
        XA = gs.sample(nsamp//2, n_alpha, n_beta, device=device)
        XB = es.sample(nsamp//2, n_alpha, n_beta, device=device)
        X  = torch.cat([XA, XB], dim=0)
        logA = gs.log_psi_complex(X)
        logB = es.log_psi_complex(X)
        qa = torch.exp(2.0*torch.real(logA))
        qb = torch.exp(2.0*torch.real(logB))
        q  = 0.5*qa + 0.5*qb + 1e-30
        w  = torch.exp(logA + logB) / q  # unbiased estimator for <A|B>
        val = torch.mean(w)
        self.value = val if (self.value is None) else (self.decay*self.value + (1-self.decay)*val)
        return self.value


# -------------------- spin-count enforcement & ES seeding --------------------
@torch.no_grad()
def enforce_spin_counts(x01: torch.Tensor, n_alpha: int, n_beta: int) -> torch.Tensor:
    dev = x01.device
    X = x01.detach().to('cpu').numpy().astype(np.int64, copy=True)
    B, N = X.shape
    a_idx = np.arange(0, N, 2, dtype=np.int64)
    b_idx = np.arange(1, N, 2, dtype=np.int64)
    for b in range(B):
        ca = int(X[b, a_idx].sum())
        if ca > n_alpha:
            ones = np.where(X[b, a_idx]==1)[0][: (ca - n_alpha)]
            X[b, a_idx[ones]] = 0
        elif ca < n_alpha:
            zeros = np.where(X[b, a_idx]==0)[0][: (n_alpha - ca)]
            X[b, a_idx[zeros]] = 1
        cb = int(X[b, b_idx].sum())
        if cb > n_beta:
            ones = np.where(X[b, b_idx]==1)[0][: (cb - n_beta)]
            X[b, b_idx[ones]] = 0
        elif cb < n_beta:
            zeros = np.where(X[b, b_idx]==0)[0][: (n_beta - cb)]
            X[b, b_idx[zeros]] = 1
    return torch.tensor(X, dtype=torch.long, device=dev)

@torch.no_grad()
def seed_single_ph(x01: torch.Tensor, homo_sp: int, lumo_sp: int) -> torch.Tensor:
    """Force exactly one particle-hole excitation (HOMO->LUMO);  even idx=α, odd idx=β."""
    X = x01.clone()
    B, N = X.shape
    for b in range(B):
        if random.random() < 0.5:
            i_occ = 2*homo_sp;     i_vir = 2*lumo_sp      # α
        else:
            i_occ = 2*homo_sp + 1; i_vir = 2*lumo_sp + 1  # β
        X[b, i_occ] = 0
        X[b, i_vir] = 1
    return X


# -------------------- training --------------------
def diag_of_batch(X_bits: torch.Tensor, fci: FCIDump) -> float:
    """Mean diagonal matrix element over a batch (sanity/debug)."""
    h1, g2, ecore = fci.h1, fci.g2, fci.e_core
    vals = []
    for b in range(X_bits.shape[0]):
        D = bits_to_det(X_bits[b].tolist())
        occs = occ_list_from_int(D, fci.norb)
        vals.append(diag_energy_np(D, occs, h1, g2, ecore))
    return float(np.mean(vals))

def train_state(model: NNQS,
                fci: FCIDump,
                n_alpha: int, n_beta: int,
                iters: int = 1000, batch_unique: int = 256, lr: float = 1e-3,
                device: str = "cpu",
                # overlap/orthogonality
                ortho_model: Optional[NNQS] = None,
                ortho_lambda: float = 0.0,
                overlap_nsamp: int = 4096,
                # ES seeding (HOMO->LUMO, in spatial order)
                seed_es_ph_iters: int = 0,
                homo_sp: Optional[int] = None, lumo_sp: Optional[int] = None,
                # numerics
                cap_dlog: float = 40.0,
                grad_clip: float = 1.0):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ovm = OverlapMeter(decay=0.9) if (ortho_model is not None and ortho_lambda > 0.0) else None

    for it in range(1, iters+1):
        # sampling (+ optional ES seeding for initial iterations)
        with torch.no_grad():
            X = model.sample(batch_unique, n_alpha, n_beta, device=device)
            if seed_es_ph_iters > 0 and it <= seed_es_ph_iters and homo_sp is not None and lumo_sp is not None:
                X = seed_single_ph(X, homo_sp=homo_sp, lumo_sp=lumo_sp)
            X = enforce_spin_counts(X, n_alpha, n_beta)

        dets = np.array([bits_to_det(x.cpu().numpy()) for x in X], dtype=np.int64)
        Eloc = batched_local_energy(dets, model, fci, device=device, cap_dlog=cap_dlog)  # complex
        Emean = torch.real(Eloc).mean().item()

        # complex REINFORCE with baseline
        baseline = torch.real(Eloc).mean().detach()
        logψ = model.log_psi_complex(X)               # complex with grad
        weights = (Eloc.detach() - baseline)          # detached complex weights
        loss_core = torch.real((-2.0) * torch.conj(weights) * logψ).mean()

        # strong, low-variance overlap penalty (EMA of large-sample estimate)
        ortho_term = torch.tensor(0.0, device=device)
        if ovm is not None:
            ov = ovm.update(ortho_model, model, n_alpha, n_beta, device, nsamp=overlap_nsamp)
            ortho_term = ortho_lambda * (torch.real(ov)**2 + torch.imag(ov)**2)

        loss = loss_core + ortho_term

        opt.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if it % LOG_EVERY == 0 or it == 1:
            diag_b = diag_of_batch(X, fci)
            print(f"[iter {it:4d}]  Re(E)={Emean:+.8f}  Im(E_loc)~{float(torch.abs(torch.imag(Eloc)).mean()):.2e}  "
                  f"ortho_pen={float(ortho_term):.3e}  diag_batch~{diag_b:.6f}")

    with torch.no_grad():
        X = model.sample(4*batch_unique, n_alpha, n_beta, device=device)
        X = enforce_spin_counts(X, n_alpha, n_beta)
        dets = np.array([bits_to_det(x.cpu().numpy()) for x in X], dtype=np.int64)
        Eloc = batched_local_energy(dets, model, fci, device=device, cap_dlog=cap_dlog)
        Efinal = torch.real(Eloc).mean().item()
    return Efinal


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcidump", type=str, required=True)
    ap.add_argument("--fcidump_type", choices=["spatial", "spin"], default="spatial",
                    help="Most FCIDUMPs are spatial (MO). Choose 'spin' only if already spin-orbital.")
    ap.add_argument("--nalpha", type=int, required=True)
    ap.add_argument("--nbeta",  type=int, required=True)

    # model/training
    ap.add_argument("--iters_gs", type=int, default=800)
    ap.add_argument("--iters_es", type=int, default=1200)
    ap.add_argument("--batch",    type=int, default=256)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--hidden",   type=int, default=96)
    ap.add_argument("--depth",    type=int, default=8)
    ap.add_argument("--nhead",    type=int, default=4)
    ap.add_argument("--amp",      choices=["off","bf16","fp16"], default="bf16")
    ap.add_argument("--grad_ckpt", type=int, default=1)

    # orthogonality & seeding
    ap.add_argument("--ortho_lambda", type=float, default=10.0)
    ap.add_argument("--overlap_nsamp", type=int, default=4096)
    ap.add_argument("--seed_es_ph_iters", type=int, default=400,
                    help="How many ES iterations to force a single HOMO->LUMO p-h before free training.")

    # active space (spatial only)
    ap.add_argument("--ncore", type=int, default=0)
    ap.add_argument("--nvirt", type=int, default=0)

    # numerics
    ap.add_argument("--cap_dlog", type=float, default=40.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed",   type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    amp_dtype = choose_amp_dtype(args.amp)
    use_ckpt  = bool(args.grad_ckpt)

    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if (cuda_avail and args.device.startswith("cuda")) else "N/A"
    print("=== Runtime config ===")
    print(f"device requested : {args.device}")
    print(f"CUDA available   : {cuda_avail}")
    print(f"GPU in use       : {gpu_name if args.device.startswith('cuda') and cuda_avail else 'CPU only'}")
    print(f"LOG_EVERY        : {LOG_EVERY}")
    print(f"AMP dtype        : {str(amp_dtype) if amp_dtype else 'disabled'}")
    print(f"grad checkpoint  : {use_ckpt}")
    print("===============================")

    # --- parse FCIDUMP & sanity (RHF) ---
    fci = parse_fcidump(args.fcidump, nelec_alpha=args.nalpha, nelec_beta=args.nbeta)
    print(f"Loaded FCIDUMP: norb(spatial?)={fci.norb}, Nα={fci.nelec_alpha}, Nβ={fci.nelec_beta}, E_core={fci.e_core:+.6f}")

    if args.fcidump_type == "spatial":
        fci = energy_order_spatial(fci)
        if args.ncore or args.nvirt:
            fci = freeze_core_and_truncate_spatial(fci, args.ncore, args.nvirt)
        # RHF sanity on spatial integrals
        nocc_sp = min(fci.nelec_alpha, fci.nelec_beta)
        E_rhf = rhf_energy_from_spatial(fci, nocc_sp)
        print(f"[sanity] RHF-like diagonal energy (spatial) ≈ {E_rhf:+.6f} Hartree")
        # expand to spin-orbitals
        fci = expand_spatial_to_spin(fci)
        print(f"[info] expanded to spin orbitals: norb={fci.norb}")
        # HOMO/LUMO indices for seeding (spatial indices!)
        HOMO_sp = nocc_sp - 1
        LUMO_sp = nocc_sp
    else:
        if args.ncore or args.nvirt:
            raise ValueError("ncore/nvirt are only for spatial FCIDUMP.")
        assert fci.norb % 2 == 0, "Spin-orbital FCIDUMP must have even norb."
        # Do not try to compute RHF with spin-orbital tensor here
        HOMO_sp = None; LUMO_sp = None

    print(f"Spin-orbital problem: norb={fci.norb}, Nα={fci.nelec_alpha}, Nβ={fci.nelec_beta}\n")

    # --- build models ---
    model_gs = NNQS(fci.norb, hidden=args.hidden, depth=args.depth, nhead=args.nhead,
                    amp_dtype=amp_dtype, grad_ckpt=use_ckpt).to(args.device)
    model_es = NNQS(fci.norb, hidden=args.hidden, depth=args.depth, nhead=args.nhead,
                    amp_dtype=amp_dtype, grad_ckpt=use_ckpt).to(args.device)

    # --- train GS ---
    print("=== Train ground state (singlet sector via masks) ===")
    E_gs = train_state(model_gs, fci, fci.nelec_alpha, fci.nelec_beta,
                       iters=args.iters_gs, batch_unique=args.batch, lr=args.lr,
                       device=args.device,
                       ortho_model=None, ortho_lambda=0.0,
                       overlap_nsamp=0, seed_es_ph_iters=0,
                       cap_dlog=args.cap_dlog, grad_clip=args.grad_clip)
    print(f"Estimated ground-state energy (Hartree): {E_gs:+.8f}\n")

    # --- freeze GS, train ES with strong orthogonality + seeding ---
    print("=== Train first singlet excited state (orthogonal to GS) ===")
    for p in model_gs.parameters():
        p.requires_grad_(False)

    E_es = train_state(model_es, fci, fci.nelec_alpha, fci.nelec_beta,
                       iters=args.iters_es, batch_unique=args.batch, lr=args.lr,
                       device=args.device,
                       ortho_model=model_gs, ortho_lambda=args.ortho_lambda,
                       overlap_nsamp=args.overlap_nsamp,
                       seed_es_ph_iters=args.seed_es_ph_iters,
                       homo_sp=HOMO_sp, lumo_sp=LUMO_sp,
                       cap_dlog=args.cap_dlog, grad_clip=args.grad_clip)
    print(f"Estimated first singlet excited-state energy (Hartree): {E_es:+.8f}")
    print(f"Estimated excitation energy (Hartree): {E_es - E_gs:+.8f}  "
          f"(≈ {(E_es - E_gs)*27.2114:+.3f} eV)")

if __name__ == "__main__":
    main()

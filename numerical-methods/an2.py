import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Spatial Grid Setup
# ==========================================
L = 8.0          # Expanded boundaries to contain higher states
N = 1000         # Grid resolution
x = np.linspace(-L, L, N)
h = x[1] - x[0]

def solve_pt_symmetric_dense(alpha):
    """
    Constructs and solves the non-Hermitian Hamiltonian on the spatial grid.
    """
    V = 0.5 * x**2 + 1j * alpha * x**3
    main_diag = np.full(N, 1.0 / h**2, dtype=complex) + V
    off_diag = np.full(N - 1, -1.0 / (2.0 * h**2), dtype=complex)

    H = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    evals, evecs = np.linalg.eig(H)
    
    # Sort by real part of eigenvalues to maintain consistent indexing
    idx = np.argsort(np.real(evals))
    return evals[idx], evecs[:, idx]

# ==========================================
# 2. Extracting the States
# ==========================================
print("Solving Unbroken Phase (alpha = 0.05)...")
evals_unbroken, evecs_unbroken = solve_pt_symmetric_dense(0.05)

print("Solving Broken Phase (alpha = 0.22)...")
evals_broken, evecs_broken = solve_pt_symmetric_dense(0.22)

# We target a pair of higher excited states that we know break symmetry 
# at this specific coupling strength.
state_A_idx = 12
state_B_idx = 13

# Unbroken Wavefunctions
psi_un_A = evecs_unbroken[:, state_A_idx]
psi_un_B = evecs_unbroken[:, state_B_idx]

# Broken Wavefunctions
psi_br_A = evecs_broken[:, state_A_idx]
psi_br_B = evecs_broken[:, state_B_idx]

# Calculate Probability Densities (|psi|^2)
prob_un_A = np.abs(psi_un_A)**2
prob_un_B = np.abs(psi_un_B)**2

prob_br_A = np.abs(psi_br_A)**2
prob_br_B = np.abs(psi_br_B)**2

# Normalize the densities so they peak at 1.0 for clean plotting
prob_un_A /= np.max(prob_un_A)
prob_un_B /= np.max(prob_un_B)

prob_br_A /= np.max(prob_br_A)
prob_br_B /= np.max(prob_br_B)

# Background Potential (Real part, scaled down for visual reference)
V_real_scaled = (0.5 * x**2) / 15.0

# ==========================================
# 3. Plotting the Results
# ==========================================
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- Top Panel: Unbroken Phase ---
ax1.plot(x, V_real_scaled, color='gray', linestyle=':', label='V(x) / 15')
ax1.plot(x, prob_un_A, color='cyan', linewidth=2, label=rf'$|\psi_{{{state_A_idx}}}|^2$')
ax1.plot(x, prob_un_B, color='gold', linewidth=2, label=rf'$|\psi_{{{state_B_idx}}}|^2$')

ax1.fill_between(x, prob_un_A, alpha=0.15, color='cyan')
ax1.fill_between(x, prob_un_B, alpha=0.15, color='gold')

ax1.set_title(r"Unbroken $\mathcal{PT}$ Phase ($\alpha = 0.05$) — Symmetric Distribution", color='springgreen', fontsize=13)
ax1.set_ylabel(r"$|\psi(x)|^2$")
ax1.set_ylim(-0.05, 1.2)
ax1.grid(True, alpha=0.2)
ax1.legend(loc='upper right', framealpha=0.3)

# --- Bottom Panel: Broken Phase ---
ax2.plot(x, V_real_scaled, color='gray', linestyle=':', label='V(x) / 15')
ax2.plot(x, prob_br_A, color='cyan', linewidth=2, label=rf'$|\psi_{{{state_A_idx}}}|^2$ (Loss side)')
ax2.plot(x, prob_br_B, color='hotpink', linewidth=2, label=rf'$|\psi_{{{state_B_idx}}}|^2$ (Gain side)')

# Fill underneath the broken curves to emphasize the spatial separation
ax2.fill_between(x, prob_br_A, alpha=0.3, color='cyan')
ax2.fill_between(x, prob_br_B, alpha=0.3, color='hotpink')

ax2.set_title(r"Broken $\mathcal{PT}$ Phase ($\alpha = 0.22$) — Localised / Skewed", color='tomato', fontsize=13)
ax2.set_xlabel("Position (x)")
ax2.set_ylabel(r"$|\psi(x)|^2$")
ax2.set_ylim(-0.05, 1.2)
ax2.grid(True, alpha=0.2)
ax2.legend(loc='upper right', framealpha=0.3)

plt.tight_layout()
plt.show()
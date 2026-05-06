import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Spatial Grid Setup
# ==========================================
L = 8.0          # Expanded boundaries slightly to contain higher excited states
N = 1000         # Grid resolution 
x = np.linspace(-L, L, N)
h = x[1] - x[0]

def solve_pt_symmetric_dense(alpha):
    """
    Constructs the dense non-Hermitian Hamiltonian and computes eigenvalues.
    """
    V = 0.5 * x**2 + 1j * alpha * x**3

    main_diag = np.full(N, 1.0 / h**2, dtype=complex) + V
    off_diag = np.full(N - 1, -1.0 / (2.0 * h**2), dtype=complex)

    H = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    evals, evecs = np.linalg.eig(H)
    return evals, evecs

# ==========================================
# 2. Phase Transition Sweep (Finding the EP)
# ==========================================
# Pushed alpha to 0.6 to give plenty of room to see the bifurcation
alpha_values = np.linspace(0.0, 0.6, 120)

# Instead of deeply bound states (0-3), we target a higher cluster (10-13).
# They interact with the complex cubic term much earlier.
start_idx = 10
num_levels = 4
end_idx = start_idx + num_levels

real_energies = np.zeros((len(alpha_values), num_levels))
imag_energies = np.zeros((len(alpha_values), num_levels))

print("Assembling dense matrices and tracking states...")

tracked_evals = None

for i, alpha in enumerate(alpha_values):
    evals, _ = solve_pt_symmetric_dense(alpha)
    
    if i == 0:
        # Step 0: Establish the baseline by sorting by real part
        idx = np.argsort(np.real(evals))
        tracked_evals = evals[idx][start_idx:end_idx]
    else:
        # Step i > 0: Nearest-neighbor tracking in the complex plane
        # This prevents the `argsort` from randomly swapping coalesced states past the EP
        new_tracked = np.zeros_like(tracked_evals)
        available_evals = list(evals)
        
        for j, prev_val in enumerate(tracked_evals):
            # Find the eigenvalue closest to the one from the previous alpha step
            distances = [np.abs(e - prev_val) for e in available_evals]
            best_match_idx = np.argmin(distances)
            new_tracked[j] = available_evals[best_match_idx]
            
            # Remove to prevent two tracks from snapping to the same state
            available_evals.pop(best_match_idx)
            
        tracked_evals = new_tracked

    # Sanitize the numerical noise from the eigensolver
    im_part = np.imag(tracked_evals)
    im_part[np.abs(im_part) < 1e-10] = 0.0

    # Store results
    real_energies[i, :] = np.real(tracked_evals)
    imag_energies[i, :] = im_part

# ==========================================
# 3. Plotting the Results
# ==========================================
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = ['cyan', 'gold', 'hotpink', 'springgreen']

# Left Plot: Real part
for n in range(num_levels):
    state_label = f'$E_{{{start_idx + n}}}$'
    ax1.plot(alpha_values, real_energies[:, n], color=colors[n], 
             linewidth=2.5, label=state_label)
    
ax1.set_title("Real Part of Eigenvalues", fontsize=14)
ax1.set_xlabel(r"Coupling Strength ($\alpha$)", fontsize=12)
ax1.set_ylabel(r"Re($E$)", fontsize=12)
ax1.grid(True, alpha=0.2)
ax1.legend(loc='upper left')

# Right Plot: Imaginary part
for n in range(num_levels):
    state_label = f'$E_{{{start_idx + n}}}$'
    ax2.plot(alpha_values, imag_energies[:, n], color=colors[n], 
             linewidth=2.5, label=state_label)

ax2.set_title("Imaginary Part of Eigenvalues", fontsize=14)
ax2.set_xlabel(r"Coupling Strength ($\alpha$)", fontsize=12)
ax2.set_ylabel(r"Im($E$)", fontsize=12)
ax2.grid(True, alpha=0.2)
ax2.legend(loc='upper left')

# Locate EP for the lowest state in our tracked cluster
broken_indices = np.where(np.abs(imag_energies[:, 0]) > 0.0)[0]
if len(broken_indices) > 0:
    ep_alpha = alpha_values[broken_indices[0]]
    for ax in (ax1, ax2):
        ax.axvline(ep_alpha, color='red', linestyle='--', alpha=0.7)
    ax1.text(ep_alpha + 0.01, ax1.get_ylim()[1] * 0.9, "Exceptional Point", color='red', fontsize=11)

plt.suptitle(r"$\mathcal{PT}$-Symmetry Spontaneous Breaking Phase Transition", fontsize=16)
plt.tight_layout()
plt.show()